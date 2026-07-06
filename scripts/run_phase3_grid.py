#!/usr/bin/env python3
"""Phase 3 grid runner: PGD attention fingerprint across an eps grid, DiT vs UNet.

Confirmatory experiment (experiments/phase3_main/PREREG.md, tag phase3-prereg).
Reuses the pilot's pure helpers (measure_attention, cat_stacked, decode_images)
and changes what the PREREG requires:
  - disjoint seeds: SEED_BASE = 21260705 (pilot used 20260705; +1e6 clears the
    largest offset band, asserted at startup);
  - eps grid {0.01,0.02,0.05,0.1}, n=500, 40 PGD iters, K=1 random draw;
  - benign measured ONCE per (model, seed) and reused across all eps;
  - class-distinct labels 0..n-1;
  - full per-layer x per-step x per-head substrate persisted per branch;
  - optional NFE-transfer (50/100, forward-only) at the primary eps.

Seed anchoring (audit fix): random and PGD-init seeds key off the eps VALUE via a
frozen EPS_INDEX map and off the per-SEED id, NOT off the eps position in the
grid or the batch composition. So DiT, UNet and the DiT@FID95 control at the same
(eps, seed) get identical z_T, Rademacher direction, and PGD start regardless of
which eps grid or batch size each run uses -- the paired comparison the PREREG
declares holds across all three legs.

Incremental checkpointing: the run saves partial.pt every ~100 seeds and resumes
from it (same config only), so a blackout/OOM mid-run loses at most ~100 seeds.

Differential FID is NOT run here (PREREG drops it from the confirmatory set).

Output under experiments/phase3_main/<model-slug>[.<tag>]/:
  benign.pt, eps_<eps>.pt, nfe<NFE>_eps<eps>.pt, meta.json (+ transient partial.pt)

Usage:
  uv run python scripts/run_phase3_grid.py --model SiT-B/2
  uv run python scripts/run_phase3_grid.py --model UNet-B
  uv run python scripts/run_phase3_grid.py --model SiT-B/2-FID95 --eps-grid 0.05
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
from diffusers import AutoencoderKL

from src.attacks.pgd_latent import load_model, pgd_latent, rademacher_delta
from src.evaluation.fid import SCALING_FACTOR, VAE_REPO
from scripts.run_phase2_pilot import (cat_stacked, decode_images, file_md5,
                                      measure_attention)

CKPTS = {
    "SiT-B/2": "experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt",
    "UNet-B": "experiments/20260611-UNet-B-cosine-6p4M/checkpoints/step_06400000_final.pt",
    "SiT-B/2-FID95": "experiments/20260518-SiT-B-2-b64/checkpoints/step_00150000.pt",
}
MODEL_KEY = {"SiT-B/2": "SiT-B/2", "UNet-B": "UNet-B", "SiT-B/2-FID95": "SiT-B/2"}
SLUG = {"SiT-B/2": "SiT-B-2", "UNet-B": "UNet-B", "SiT-B/2-FID95": "SiT-B-2-FID95"}

SEED_BASE = 21260705
PILOT_BASE = 20260705
# Frozen eps->index so seeds key off the eps VALUE, not its position in the grid.
EPS_INDEX = {0.01: 0, 0.02: 1, 0.05: 2, 0.1: 3}
Z_BAND = 0                # z_T seed i           -> BASE + i
RAND_BAND = 100_000       # rand (eps e, seed i) -> BASE + 100_000 + 10_000*EPS_INDEX[e] + i
PGD_BAND = 200_000        # pgd init (eps e, i)  -> BASE + 200_000 + 10_000*EPS_INDEX[e] + i
BAND_STRIDE = 10_000      # per-eps sub-band width; must exceed n_seeds
CKPT_EVERY_SEEDS = 100


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--eps-grid", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--primary-eps", type=float, default=0.05)
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--n-steps-attack", type=int, default=40)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    ap.add_argument("--nfe-extra", type=int, nargs="*", default=[50, 100])
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--ckpt-every-seeds", type=int, default=100,
                    help="dump partial.pt every ~this many seeds (blackout resilience)")
    ap.add_argument("--out-dir", type=str, default="experiments/phase3_main")
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args()


def validate(args) -> None:
    eps_grid = list(args.eps_grid)
    assert len(set(eps_grid)) == len(eps_grid), f"duplicate eps in grid: {eps_grid}"
    for e in eps_grid:
        assert e in EPS_INDEX, f"eps {e} not in the frozen EPS_INDEX {EPS_INDEX}"
    assert args.primary_eps in eps_grid, \
        f"primary_eps {args.primary_eps} not in grid {eps_grid} (NFE-transfer would no-op)"
    assert args.cfg_scale == 1.0, \
        "cfg_scale != 1.0 unsupported: measure_attention samples at cfg=1, so the " \
        "attack would optimize a trajectory different from the one measured"
    assert args.batch_size >= 1
    # band separation: rand sub-bands must not spill into PGD_BAND, and seeds
    # must fit inside a sub-band.
    assert args.n_seeds <= BAND_STRIDE, f"n_seeds {args.n_seeds} > band stride {BAND_STRIDE}"
    max_eps_idx = max(EPS_INDEX[e] for e in eps_grid)
    assert RAND_BAND + BAND_STRIDE * max_eps_idx + args.n_seeds <= PGD_BAND, \
        "rand band spills into PGD band"
    max_off = PGD_BAND + BAND_STRIDE * max_eps_idx + args.n_seeds
    assert max_off < 1_000_000, f"offset band {max_off} >= 1e6 breaks pilot disjointness"
    # disjoint from pilot: +1e6 base gap with all offsets < 1e6
    assert SEED_BASE >= PILOT_BASE + 1_000_000, "base gap < 1e6"


def seed_gen(offset: int, device) -> torch.Generator:
    assert 0 <= offset < 1_000_000, f"offset {offset} outside the disjoint band"
    return torch.Generator(device=device).manual_seed(SEED_BASE + offset)


def make_z(ids: list[int], device) -> torch.Tensor:
    return torch.cat([
        torch.randn(1, 4, 32, 32, device=device, generator=seed_gen(Z_BAND + i, device))
        for i in ids
    ])


def make_rademacher(z, eps, ids, device) -> torch.Tensor:
    off = RAND_BAND + BAND_STRIDE * EPS_INDEX[eps]
    return torch.cat([
        rademacher_delta(z[j:j + 1], eps, generator=seed_gen(off + i, device))
        for j, i in enumerate(ids)
    ])


def make_pgd_init(z, eps, ids, device) -> torch.Tensor:
    """Per-seed PGD random start in U(-eps,eps); batch-invariant, eps-value-anchored."""
    off = PGD_BAND + BAND_STRIDE * EPS_INDEX[eps]
    return torch.cat([
        torch.empty(1, 4, 32, 32, device=device).uniform_(
            -eps, eps, generator=seed_gen(off + i, device))
        for j, i in enumerate(ids)
    ])


@torch.inference_mode()
def measure_nfe(model, z, y, nfe, device):
    return measure_attention(model, z, y, nfe, device)[0]


def _blank_state(eps_grid, nfe_extra, primary_eps):
    return {
        "attn_ben": None, "z0_ben": [], "z_T": [], "y": [],
        "per_eps": {e: {"attn_rand": None, "attn_pgd": None, "z0_rand": [], "z0_pgd": [],
                        "z_T_adv": [], "lpips_pgd": [], "lpips_rand": [],
                        "l2_pgd": [], "l2_rand": [], "pgd_curves": []} for e in eps_grid},
        "nfe": {(nfe, primary_eps): {"ben": None, "rand": None, "pgd": None}
                for nfe in nfe_extra},
        "next_seed": 0,
    }


def run_grid(model, vae, lpips_net, args, device, out_dir: Path) -> None:
    n, bs = args.n_seeds, args.batch_size
    eps_grid = list(args.eps_grid)
    partial = out_dir / "partial.pt"
    fp = {"model": args.model, "eps_grid": eps_grid, "n_seeds": n, "batch_size": bs,
          "n_steps_attack": args.n_steps_attack, "n_steps_ode": args.n_steps_ode,
          "nfe_extra": list(args.nfe_extra), "seed_base": SEED_BASE}

    if partial.exists():
        blob = torch.load(partial, weights_only=False)
        assert blob["fingerprint"] == fp, \
            f"partial.pt config mismatch; refusing to resume.\n got {blob['fingerprint']}\n want {fp}"
        st = blob["state"]
        print(f"[p3 {args.model}] resuming from seed {st['next_seed']}/{n}", flush=True)
    else:
        st = _blank_state(eps_grid, list(args.nfe_extra), args.primary_eps)

    t0 = time.perf_counter()
    ckpt_every = max(1, args.ckpt_every_seeds // bs)
    batches_done = 0
    for bi in range(st["next_seed"], n, bs):
        ids = list(range(bi, min(bi + bs, n)))
        z = make_z(ids, device)
        y = torch.tensor([i % 1000 for i in ids], device=device)

        stacked_ben, z0_ben = measure_attention(model, z, y, args.n_steps_ode, device)
        st["attn_ben"] = cat_stacked(st["attn_ben"], stacked_ben)
        img_ben = decode_images(vae, z0_ben)
        st["z0_ben"].append(z0_ben.cpu()); st["z_T"].append(z.cpu()); st["y"].append(y.cpu())

        for eps in eps_grid:
            pe = st["per_eps"][eps]
            d = make_rademacher(z, eps, ids, device)
            stacked_r, z0_r = measure_attention(model, z + d, y, args.n_steps_ode, device)
            pe["attn_rand"] = cat_stacked(pe["attn_rand"], stacked_r)
            z_adv, info = pgd_latent(
                model, z, y, eps=eps, n_steps_attack=args.n_steps_attack,
                n_steps_ode=args.n_steps_ode, cfg_scale=args.cfg_scale,
                delta_init=make_pgd_init(z, eps, ids, device),
            )
            stacked_p, z0_p = measure_attention(model, z_adv, y, args.n_steps_ode, device)
            pe["attn_pgd"] = cat_stacked(pe["attn_pgd"], stacked_p)
            with torch.inference_mode():
                pe["lpips_pgd"].append(lpips_net(img_ben, decode_images(vae, z0_p)).flatten().cpu())
                pe["lpips_rand"].append(lpips_net(img_ben, decode_images(vae, z0_r)).flatten().cpu())
            pe["l2_pgd"].append((z0_p - z0_ben).flatten(1).norm(dim=1).cpu())
            pe["l2_rand"].append((z0_r - z0_ben).flatten(1).norm(dim=1).cpu())
            pe["z0_pgd"].append(z0_p.cpu()); pe["z0_rand"].append(z0_r.cpu())
            pe["z_T_adv"].append(z_adv.cpu()); pe["pgd_curves"].append(info["losses"])

            if eps == args.primary_eps:
                for nfe in args.nfe_extra:
                    s = st["nfe"][(nfe, eps)]
                    s["ben"] = cat_stacked(s["ben"], measure_nfe(model, z, y, nfe, device))
                    s["rand"] = cat_stacked(s["rand"], measure_nfe(model, z + d, y, nfe, device))
                    s["pgd"] = cat_stacked(s["pgd"], measure_nfe(model, z_adv, y, nfe, device))

        st["next_seed"] = bi + len(ids)
        batches_done += 1
        done = st["next_seed"]
        rate = (time.perf_counter() - t0) / (batches_done * bs)  # this session
        print(f"[p3 {args.model}] {done}/{n} seeds  ({rate:.1f} s/seed, "
              f"eta {(n - done) * rate / 60:.0f} min)", flush=True)
        if batches_done % ckpt_every == 0 and done < n:
            torch.save({"fingerprint": fp, "state": st}, partial)
            print(f"[p3 {args.model}] partial saved @ {done} seeds", flush=True)

    # final write
    torch.save({"attn": st["attn_ben"], "z0": torch.cat(st["z0_ben"]),
                "z_T": torch.cat(st["z_T"]), "y": torch.cat(st["y"])}, out_dir / "benign.pt")
    for eps in eps_grid:
        pe = st["per_eps"][eps]
        torch.save({
            "attn_rand": pe["attn_rand"], "attn_pgd": pe["attn_pgd"],
            "z0_rand": torch.cat(pe["z0_rand"]), "z0_pgd": torch.cat(pe["z0_pgd"]),
            "z_T_adv": torch.cat(pe["z_T_adv"]),
            "lpips": {"pgd": torch.cat(pe["lpips_pgd"]), "rand": torch.cat(pe["lpips_rand"])},
            "l2_out": {"pgd": torch.cat(pe["l2_pgd"]), "rand": torch.cat(pe["l2_rand"])},
            "pgd_loss_curves": pe["pgd_curves"], "eps": eps,
        }, out_dir / f"eps_{eps}.pt")
    for (nfe, eps), s in st["nfe"].items():
        torch.save({"eps": eps, "nfe": nfe, "ben": s["ben"], "rand": s["rand"],
                    "pgd": s["pgd"]}, out_dir / f"nfe{nfe}_eps{eps}.pt")
    partial.unlink(missing_ok=True)
    print(f"[p3 {args.model}] saved to {out_dir}", flush=True)


def main() -> int:
    args = parse_args()
    validate(args)
    device = torch.device("cuda")
    ckpt_path = REPO_ROOT / CKPTS[args.model]
    slug = SLUG[args.model] + (f".{args.tag}" if args.tag else "")
    out_dir = REPO_ROOT / args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== phase3 grid | model={args.model} eps={args.eps_grid} n={args.n_seeds} "
          f"iters={args.n_steps_attack} ode={args.n_steps_ode} nfe_extra={args.nfe_extra} "
          f"seed_base={SEED_BASE} ===", flush=True)

    model = load_model(MODEL_KEY[args.model], ckpt_path, device)
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    run_grid(model, vae, lpips_net, args, device, out_dir)
    t_total = time.perf_counter() - t0

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model": args.model, "args": vars(args), "ckpt": str(ckpt_path),
                   "ckpt_md5": file_md5(ckpt_path), "seed_base": SEED_BASE,
                   "eps_index": EPS_INDEX, "git_commit": commit, "torch": torch.__version__,
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
                   "wall_seconds": round(t_total, 1),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    print(f"peak VRAM {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB, "
          f"done in {t_total / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
