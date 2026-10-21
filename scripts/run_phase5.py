#!/usr/bin/env python3
"""Phase 5 runner: guidance ablation and official SiT checkpoints (post-thesis).

Exploratory, NOT pre-registered. Reuses the Phase-3 confirmatory pipeline
(scripts/run_phase3_grid.py, which stays untouched) with two additions:

  1. classifier-free guidance: the attack (src.attacks.pgd_latent already
     supports cfg_scale) and the measurement run on the SAME guided
     trajectory. The hooks see the duplicated batch [cond; uncond]; the
     conditional half is the primary substrate (file layout identical to
     Phase 3, so scripts/analyze_phase3.load_leg reads it unchanged), the
     unconditional half is persisted next to it as *_uncond.pt.
  2. official SiT checkpoints (nyu-visionx/SiT-collections on Hugging Face,
     plain state dicts; Linear interpolant + velocity prediction, same VAE,
     same 32x32x4 latents as our training) via a two-branch loader.

Seeds: same anchoring as Phase 3 (SEED_BASE 21260705, eps-value bands), so a
leg at (model, cfg) pairs seed by seed with every other leg, including the
Phase-3 legs at cfg 1. Labels i % 1000, ids 0..n-1.

Defaults: eps 0.05 only, 40 PGD iterations, step eps/4, random start, 25
Euler steps, fp32, SVD on CPU, no NFE-transfer legs.

Output under <out-dir>/<slug>[.<tag>]/:
  benign.pt, eps_<eps>.pt, meta.json (+ benign_uncond.pt, eps_<eps>_uncond.pt
  when cfg != 1; transient partial.pt every --ckpt-every-seeds for resume).

Usage (always via nohup for the long legs):
  .venv/bin/python -u scripts/run_phase5.py --model SiT-B/2 --cfg-scale 1.5 \
      --out-dir experiments/phase5_cfg
  .venv/bin/python -u scripts/run_phase5.py --model SiT-XL/2-official \
      --batch-size 10 --out-dir experiments/phase5_pretrained
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
for _p in (str(REPO_ROOT), str(SIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lpips as lpips_lib
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from src.attacks.pgd_latent import (MODEL_REGISTRY, NUM_CLASSES, euler_step,
                                    pgd_latent)
from src.evaluation.attention_metrics import reduce_snapshot, stack_steps
from src.evaluation.fid import VAE_REPO
from src.utils.attention_hooks import AttentionCollector
from scripts.run_phase2_pilot import cat_stacked, decode_images, file_md5
from scripts.run_phase3_grid import (CKPTS as P3_CKPTS, EPS_INDEX,
                                     MODEL_KEY as P3_MODEL_KEY, SEED_BASE,
                                     make_pgd_init, make_rademacher, make_z)

OFFICIAL = {
    "SiT-S/2-official": ("SiT-S/2", "checkpoints/pretrained/SiT-S-2-256.pt"),
    "SiT-B/2-official": ("SiT-B/2", "checkpoints/pretrained/SiT-B-2-256.pt"),
    "SiT-L/2-official": ("SiT-L/2", "checkpoints/pretrained/SiT-L-2-256.pt"),
    "SiT-XL/2-official": ("SiT-XL/2", "checkpoints/pretrained/SiT-XL-2-256.pt"),
}
HF_REVISION = "9a43ddf08f80ac7f3525208ce72943e98fe98343"  # nyu-visionx/SiT-collections

CKPTS = {**P3_CKPTS, **{k: v[1] for k, v in OFFICIAL.items()}}
MODEL_KEY = {**P3_MODEL_KEY, **{k: v[0] for k, v in OFFICIAL.items()}}
SLUG = {k: k.replace("/", "-") for k in CKPTS}  # "SiT-XL/2-official" -> "SiT-XL-2-official"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--eps-grid", type=float, nargs="+", default=[0.05])
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--n-steps-attack", type=int, default=40)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    ap.add_argument("--ckpt-every-seeds", type=int, default=100)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def validate(args) -> None:
    for e in args.eps_grid:
        assert e in EPS_INDEX, f"eps {e} not in the frozen EPS_INDEX {EPS_INDEX}"
    assert len(set(args.eps_grid)) == len(args.eps_grid)
    assert args.cfg_scale >= 1.0, "cfg_scale < 1 makes no sense here"
    assert 1 <= args.n_seeds <= 10_000 and args.batch_size >= 1


def load_model_any(name: str, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Phase-3 loader semantics (fp32, eval, frozen, null label row present),
    accepting both our training checkpoints ({"ema": ...}) and the official
    SiT files (plain state dict). strict=True: every key must match."""
    model = MODEL_REGISTRY[name](input_size=32, num_classes=NUM_CLASSES,
                                 class_dropout_prob=0.1).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = blob["ema"] if isinstance(blob, dict) and "ema" in blob else blob
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def measure_attention_cfg(model, z, y, n_steps: int, device, cfg_scale: float):
    """Phase-3 measure_attention generalized to guidance.

    At cfg 1 this is bit-for-bit the Phase-3 pass (same euler_step call, same
    collector, same reduction). At cfg != 1 euler_step duplicates the batch
    as [cond; uncond]; both halves are reduced, the conditional half first.
    Returns (stacked_cond, stacked_uncond | None, z0).
    """
    col = AttentionCollector(model, store_dtype=torch.float32, store_device="cpu")
    B = z.shape[0]
    cond_steps, uncond_steps = [], []
    with col, torch.inference_mode():
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y, cfg_scale)
            snap = col.snapshot()
            if cfg_scale != 1.0:
                for v in snap.values():
                    assert v.shape[0] == 2 * B, v.shape
                cond_steps.append(reduce_snapshot({k: v[:B] for k, v in snap.items()}))
                uncond_steps.append(reduce_snapshot({k: v[B:] for k, v in snap.items()}))
            else:
                cond_steps.append(reduce_snapshot(snap))
    unc = stack_steps(uncond_steps) if uncond_steps else None
    return stack_steps(cond_steps), unc, x.clone()


def _blank_state(eps_grid):
    return {
        "attn_ben": None, "attn_ben_unc": None, "z0_ben": [], "z_T": [], "y": [],
        "per_eps": {e: {"attn_rand": None, "attn_pgd": None,
                        "attn_rand_unc": None, "attn_pgd_unc": None,
                        "z0_rand": [], "z0_pgd": [], "z_T_adv": [],
                        "lpips_pgd": [], "lpips_rand": [], "l2_pgd": [], "l2_rand": [],
                        "pgd_curves": []} for e in eps_grid},
        "next_seed": 0,
    }


def run_leg(model, vae, lpips_net, args, device, out_dir: Path) -> dict:
    n, bs, cfg = args.n_seeds, args.batch_size, args.cfg_scale
    eps_grid = list(args.eps_grid)
    partial = out_dir / "partial.pt"
    fp = {"model": args.model, "cfg_scale": cfg, "eps_grid": eps_grid, "n_seeds": n,
          "batch_size": bs, "n_steps_attack": args.n_steps_attack,
          "n_steps_ode": args.n_steps_ode, "seed_base": SEED_BASE}
    if partial.exists():
        blob = torch.load(partial, weights_only=False)
        assert blob["fingerprint"] == fp, \
            f"partial.pt config mismatch; refusing to resume.\n got {blob['fingerprint']}\n want {fp}"
        st = blob["state"]
        print(f"[p5 {args.model} cfg{cfg}] resuming from seed {st['next_seed']}/{n}", flush=True)
    else:
        st = _blank_state(eps_grid)

    t0 = time.perf_counter()
    ckpt_every = max(1, args.ckpt_every_seeds // bs)
    batches_done = 0
    for bi in range(st["next_seed"], n, bs):
        ids = list(range(bi, min(bi + bs, n)))
        z = make_z(ids, device)
        y = torch.tensor([i % 1000 for i in ids], device=device)

        s_ben, s_ben_unc, z0_ben = measure_attention_cfg(model, z, y, args.n_steps_ode, device, cfg)
        st["attn_ben"] = cat_stacked(st["attn_ben"], s_ben)
        if s_ben_unc is not None:
            st["attn_ben_unc"] = cat_stacked(st["attn_ben_unc"], s_ben_unc)
        img_ben = decode_images(vae, z0_ben)
        st["z0_ben"].append(z0_ben.cpu()); st["z_T"].append(z.cpu()); st["y"].append(y.cpu())

        for eps in eps_grid:
            pe = st["per_eps"][eps]
            d = make_rademacher(z, eps, ids, device)
            s_r, s_r_unc, z0_r = measure_attention_cfg(model, z + d, y, args.n_steps_ode, device, cfg)
            pe["attn_rand"] = cat_stacked(pe["attn_rand"], s_r)
            z_adv, info = pgd_latent(
                model, z, y, eps=eps, n_steps_attack=args.n_steps_attack,
                n_steps_ode=args.n_steps_ode, cfg_scale=cfg,
                delta_init=make_pgd_init(z, eps, ids, device),
            )
            s_p, s_p_unc, z0_p = measure_attention_cfg(model, z_adv, y, args.n_steps_ode, device, cfg)
            pe["attn_pgd"] = cat_stacked(pe["attn_pgd"], s_p)
            if s_r_unc is not None:
                pe["attn_rand_unc"] = cat_stacked(pe["attn_rand_unc"], s_r_unc)
                pe["attn_pgd_unc"] = cat_stacked(pe["attn_pgd_unc"], s_p_unc)
            with torch.inference_mode():
                pe["lpips_pgd"].append(lpips_net(img_ben, decode_images(vae, z0_p)).flatten().cpu())
                pe["lpips_rand"].append(lpips_net(img_ben, decode_images(vae, z0_r)).flatten().cpu())
            pe["l2_pgd"].append((z0_p - z0_ben).flatten(1).norm(dim=1).cpu())
            pe["l2_rand"].append((z0_r - z0_ben).flatten(1).norm(dim=1).cpu())
            pe["z0_pgd"].append(z0_p.cpu()); pe["z0_rand"].append(z0_r.cpu())
            pe["z_T_adv"].append(z_adv.cpu()); pe["pgd_curves"].append(info["losses"])

        st["next_seed"] = bi + len(ids)
        batches_done += 1
        done = st["next_seed"]
        rate = (time.perf_counter() - t0) / (batches_done * bs)
        print(f"[p5 {args.model} cfg{cfg}] {done}/{n} seeds  ({rate:.1f} s/seed, "
              f"eta {(n - done) * rate / 60:.0f} min, "
              f"peak {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB)", flush=True)
        if batches_done % ckpt_every == 0 and done < n:
            torch.save({"fingerprint": fp, "state": st}, partial)
            print(f"[p5 {args.model} cfg{cfg}] partial saved @ {done} seeds", flush=True)

    torch.save({"attn": st["attn_ben"], "z0": torch.cat(st["z0_ben"]),
                "z_T": torch.cat(st["z_T"]), "y": torch.cat(st["y"])}, out_dir / "benign.pt")
    if st["attn_ben_unc"] is not None:
        torch.save({"attn": st["attn_ben_unc"], "half": "uncond"}, out_dir / "benign_uncond.pt")
    for eps in eps_grid:
        pe = st["per_eps"][eps]
        torch.save({
            "attn_rand": pe["attn_rand"], "attn_pgd": pe["attn_pgd"],
            "z0_rand": torch.cat(pe["z0_rand"]), "z0_pgd": torch.cat(pe["z0_pgd"]),
            "z_T_adv": torch.cat(pe["z_T_adv"]),
            "lpips": {"pgd": torch.cat(pe["lpips_pgd"]), "rand": torch.cat(pe["lpips_rand"])},
            "l2_out": {"pgd": torch.cat(pe["l2_pgd"]), "rand": torch.cat(pe["l2_rand"])},
            "pgd_loss_curves": pe["pgd_curves"], "eps": eps, "cfg_scale": cfg,
        }, out_dir / f"eps_{eps}.pt")
        if pe["attn_rand_unc"] is not None:
            torch.save({"attn_rand": pe["attn_rand_unc"], "attn_pgd": pe["attn_pgd_unc"],
                        "eps": eps, "cfg_scale": cfg, "half": "uncond"},
                       out_dir / f"eps_{eps}_uncond.pt")
    partial.unlink(missing_ok=True)
    rate = (time.perf_counter() - t0) / (batches_done * bs) if batches_done else float("nan")
    print(f"[p5 {args.model} cfg{cfg}] saved to {out_dir}", flush=True)
    return {"s_per_seed_this_session": round(rate, 2), "seeds_this_session": batches_done * bs}


def main() -> int:
    args = parse_args()
    validate(args)
    device = torch.device("cuda")
    ckpt_path = REPO_ROOT / CKPTS[args.model]
    slug = SLUG[args.model] + (f".{args.tag}" if args.tag else "")
    out_dir = REPO_ROOT / args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "meta.json").exists():
        print(f"[p5] {out_dir} already complete (meta.json present); nothing to do", flush=True)
        return 0

    print(f"=== phase5 | model={args.model} cfg={args.cfg_scale} eps={args.eps_grid} "
          f"n={args.n_seeds} bs={args.batch_size} iters={args.n_steps_attack} "
          f"ode={args.n_steps_ode} seed_base={SEED_BASE} ===", flush=True)

    model = load_model_any(MODEL_KEY[args.model], ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[p5] loaded {args.model} from {ckpt_path.name}: {n_params / 1e6:.1f} M params", flush=True)
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    extra = run_leg(model, vae, lpips_net, args, device, out_dir)
    t_total = time.perf_counter() - t0

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    meta = {"phase": 5, "model": args.model, "model_key": MODEL_KEY[args.model],
            "cfg_scale": args.cfg_scale, "args": vars(args), "ckpt": str(ckpt_path),
            "ckpt_md5": file_md5(ckpt_path), "n_params": n_params,
            "seed_base": SEED_BASE, "eps_index": EPS_INDEX, "git_commit": commit,
            "torch": torch.__version__,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
            "wall_seconds": round(t_total, 1), **extra,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.model in OFFICIAL:
        meta["hf_repo"] = "nyu-visionx/SiT-collections"
        meta["hf_revision"] = HF_REVISION
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"peak VRAM {meta['peak_vram_gb']:.2f} GB, done in {t_total / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
