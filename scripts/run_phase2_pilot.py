#!/usr/bin/env python3
"""Phase 2 pilot runner: three branches (benign / random / PGD) per seed.

Protocol from docs/phase2_plan.md and the tracked pre-registration in
experiments/phase2_pilot/PREREG.md: for each shared seed, three branches --
benign, z_T + Rademacher delta (equal-budget control, K draws), z_T + PGD
delta -- run on BOTH models with identical seeds, labels, and attack
hyperparameters. fp32 model-side; VAE decode in fp16 (the Phase 1 FID
protocol, same for both models).

Two parts, run separately per model (resumable, nohup-friendly):

* --part ab   (default n=100 seeds): the paired attention comparison. Full
  metric substrate (all layers x all 25 ODE steps x heads, reduced on CPU
  right after each step so the raw maps never pile up), plus the LPIPS and
  latent-L2 output-damage readouts. The Rademacher control averages K=3
  independent draws per seed.
* --part fid  (default n=1000 seeds, labels = one per class): the Metric C
  substrate. Saves per-sample InceptionV3 features per branch (benign /
  rand x1 / PGD); FID + bootstrap CI are computed downstream in the
  notebook. No attention capture here.

Outputs under experiments/phase2_pilot/<model-slug>[.<tag>]/:
  ab_results.pt / fid_features.pt + <part>_meta.json (config, seed ledger,
  ckpt md5, git commit).

Seed ledger (all torch.Generator("cuda").manual_seed(SEED_BASE + offset)):
  AB   z_T seed i          -> BASE + i                 (i in 0..n-1)
  AB   Rademacher draw k,i -> BASE + 100_000 + 1000*k + i
  AB   PGD random start    -> BASE + 200_000 + batch_index
  FID  z_T seed j          -> BASE + 300_000 + j       (j in 0..n-1)
  FID  Rademacher j        -> BASE + 400_000 + j
  FID  PGD random start    -> BASE + 500_000 + batch_index
Labels: AB y_i = (10*i) % 1000 (100 distinct classes); FID y_j = j % 1000
(exactly class-balanced at n=1000). Identical for both models by construction.

Usage:
    uv run python scripts/run_phase2_pilot.py --model SiT-B/2 --part ab
    uv run python scripts/run_phase2_pilot.py --model UNet-B --part fid
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.attacks.pgd_latent import (euler_step, load_model, pgd_latent,
                                    rademacher_delta)
from src.evaluation.attention_metrics import reduce_snapshot, stack_steps
from src.evaluation.fid import (SCALING_FACTOR, VAE_REPO, euler_ode_sample,
                                inception_features, load_inception)
from src.utils.attention_hooks import AttentionCollector

CKPTS = {
    "SiT-B/2": "experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt",
    "UNet-B": "experiments/20260611-UNet-B-cosine-6p4M/checkpoints/step_06400000_final.pt",
}
SEED_BASE = 20260705
AB_Z, AB_RAND, AB_PGD = 0, 100_000, 200_000
FID_Z, FID_RAND, FID_PGD = 300_000, 400_000, 500_000


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--part", required=True, choices=["ab", "fid"])
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="default: 100 (ab) / 1000 (fid)")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n-steps-attack", type=int, default=20)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    ap.add_argument("--k-rand", type=int, default=3,
                    help="Rademacher draws per seed (ab part)")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default="experiments/phase2_pilot")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the output subdir (e.g. 'smoke')")
    return ap.parse_args()


def seed_gen(offset: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(SEED_BASE + offset)


def make_z(offsets: list[int], device) -> torch.Tensor:
    """One z_T per seed offset, each from its own generator (batch-invariant)."""
    return torch.cat([
        torch.randn(1, 4, 32, 32, device=device, generator=seed_gen(o, device))
        for o in offsets
    ])


def make_rademacher(z: torch.Tensor, eps: float, offsets: list[int],
                    device) -> torch.Tensor:
    """Per-seed Rademacher deltas (batch-invariant draws)."""
    return torch.cat([
        rademacher_delta(z[j:j + 1], eps, generator=seed_gen(o, device))
        for j, o in enumerate(offsets)
    ])


def measure_attention(model, z, y, n_steps, device):
    """Non-differentiable measurement pass: snapshot -> reduce after each step.

    CPU store + CPU SVD reduction (svdvals batched on ROCm is ~49x slower).
    Returns (stacked metrics dict layer->metric->(T,B,H), z0).
    """
    col = AttentionCollector(model, store_dtype=torch.float32,
                             store_device="cpu")
    steps = []
    with col, torch.inference_mode():
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y)
            steps.append(reduce_snapshot(col.snapshot()))
    return stack_steps(steps), x.clone()


def cat_stacked(acc: dict | None, new: dict) -> dict:
    """Concatenate two stacked-metric dicts along the sample axis (dim=1)."""
    if acc is None:
        return new
    for name in acc:
        for k in acc[name]:
            if k == "n_tokens":
                continue
            acc[name][k] = torch.cat([acc[name][k], new[name][k]], dim=1)
    return acc


@torch.inference_mode()
def decode_images(vae, latents: torch.Tensor) -> torch.Tensor:
    """(B,4,32,32) latents -> images in [-1,1] fp32 (fp16 decode, Phase 1 protocol)."""
    imgs = vae.decode(latents.to(torch.float16) / SCALING_FACTOR).sample
    return imgs.float().clamp(-1, 1)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_meta(out_dir: Path, part: str, args, ckpt_path: Path,
               t_total: float) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    meta = {
        "part": part,
        "args": vars(args),
        "ckpt": str(ckpt_path),
        "ckpt_md5": file_md5(ckpt_path),
        "seed_base": SEED_BASE,
        "git_commit": commit,
        "torch": torch.__version__,
        "wall_seconds": round(t_total, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / f"{part}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_ab(model, vae, args, device, out_dir: Path) -> None:
    n = args.n_seeds
    lpips_net = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    branches = ["ben", "pgd"] + [f"rand{k}" for k in range(args.k_rand)]
    attn = {b: None for b in branches}
    z0 = {b: [] for b in branches}
    lp = {b: [] for b in branches if b != "ben"}
    l2 = {b: [] for b in branches if b != "ben"}
    z_T_all, z_T_adv_all, y_all = [], [], []
    pgd_curves = []

    t0 = time.perf_counter()
    seed_ids = list(range(n))
    for bi in range(0, n, args.batch_size):
        ids = seed_ids[bi:bi + args.batch_size]
        batch_idx = bi // args.batch_size
        z = make_z([AB_Z + i for i in ids], device)
        y = torch.tensor([(10 * i) % 1000 for i in ids], device=device)

        stacked, z0_ben = measure_attention(model, z, y, args.n_steps_ode, device)
        attn["ben"] = cat_stacked(attn["ben"], stacked)
        img_ben = decode_images(vae, z0_ben)

        z_adv, info = pgd_latent(
            model, z, y, eps=args.eps, n_steps_attack=args.n_steps_attack,
            n_steps_ode=args.n_steps_ode, cfg_scale=args.cfg_scale,
            generator=seed_gen(AB_PGD + batch_idx, device),
        )
        stacked, z0_pgd = measure_attention(model, z_adv, y, args.n_steps_ode, device)
        attn["pgd"] = cat_stacked(attn["pgd"], stacked)
        with torch.inference_mode():
            lp["pgd"].append(lpips_net(img_ben, decode_images(vae, z0_pgd))
                             .flatten().cpu())
        l2["pgd"].append((z0_pgd - z0_ben).flatten(1).norm(dim=1).cpu())
        pgd_curves.append(info["losses"])

        for k in range(args.k_rand):
            d = make_rademacher(z, args.eps,
                                [AB_RAND + 1000 * k + i for i in ids], device)
            stacked, z0_r = measure_attention(model, z + d, y,
                                              args.n_steps_ode, device)
            attn[f"rand{k}"] = cat_stacked(attn[f"rand{k}"], stacked)
            with torch.inference_mode():
                lp[f"rand{k}"].append(lpips_net(img_ben, decode_images(vae, z0_r))
                                      .flatten().cpu())
            l2[f"rand{k}"].append((z0_r - z0_ben).flatten(1).norm(dim=1).cpu())
            z0[f"rand{k}"].append(z0_r.cpu())

        z0["ben"].append(z0_ben.cpu())
        z0["pgd"].append(z0_pgd.cpu())
        z_T_all.append(z.cpu())
        z_T_adv_all.append(z_adv.cpu())
        y_all.append(y.cpu())

        done = bi + len(ids)
        rate = (time.perf_counter() - t0) / done
        print(f"[ab {args.model}] {done}/{n} seeds  "
              f"({rate:.1f} s/seed, eta {(n - done) * rate / 60:.0f} min)",
              flush=True)

    out = {
        "attn": attn,
        "z0": {b: torch.cat(v) for b, v in z0.items()},
        "z_T": torch.cat(z_T_all),
        "z_T_adv": torch.cat(z_T_adv_all),
        "y": torch.cat(y_all),
        "lpips": {b: torch.cat(v) for b, v in lp.items()},
        "l2_out": {b: torch.cat(v) for b, v in l2.items()},
        "pgd_loss_curves": pgd_curves,
    }
    torch.save(out, out_dir / "ab_results.pt")
    print(f"[ab {args.model}] saved {out_dir / 'ab_results.pt'}", flush=True)


def run_fid(model, vae, args, device, out_dir: Path) -> None:
    n = args.n_seeds
    inception = load_inception(device)

    feats = {b: [] for b in ("ben", "rand", "pgd")}
    z0_store = {b: [] for b in ("ben", "rand", "pgd")}
    y_all = []

    def featurize(z0_batch: torch.Tensor) -> torch.Tensor:
        imgs01 = (decode_images(vae, z0_batch) + 1) / 2
        return inception_features(inception, imgs01).cpu()

    t0 = time.perf_counter()
    seed_ids = list(range(n))
    for bi in range(0, n, args.batch_size):
        ids = seed_ids[bi:bi + args.batch_size]
        batch_idx = bi // args.batch_size
        z = make_z([FID_Z + j for j in ids], device)
        y = torch.tensor([j % 1000 for j in ids], device=device)

        z0_ben = euler_ode_sample(model, z, y, n_steps=args.n_steps_ode)

        d = make_rademacher(z, args.eps, [FID_RAND + j for j in ids], device)
        z0_rand = euler_ode_sample(model, z + d, y, n_steps=args.n_steps_ode)

        z_adv, _ = pgd_latent(
            model, z, y, eps=args.eps, n_steps_attack=args.n_steps_attack,
            n_steps_ode=args.n_steps_ode, cfg_scale=args.cfg_scale,
            generator=seed_gen(FID_PGD + batch_idx, device),
        )
        z0_pgd = euler_ode_sample(model, z_adv, y, n_steps=args.n_steps_ode)

        for b, z0b in (("ben", z0_ben), ("rand", z0_rand), ("pgd", z0_pgd)):
            feats[b].append(featurize(z0b))
            z0_store[b].append(z0b.cpu())
        y_all.append(y.cpu())

        done = bi + len(ids)
        rate = (time.perf_counter() - t0) / done
        print(f"[fid {args.model}] {done}/{n} seeds  "
              f"({rate:.1f} s/seed, eta {(n - done) * rate / 60:.0f} min)",
              flush=True)

    out = {
        "feats": {b: torch.cat(v) for b, v in feats.items()},
        "z0": {b: torch.cat(v) for b, v in z0_store.items()},
        "y": torch.cat(y_all),
    }
    torch.save(out, out_dir / "fid_features.pt")
    print(f"[fid {args.model}] saved {out_dir / 'fid_features.pt'}", flush=True)


def main() -> int:
    args = parse_args()
    if args.n_seeds is None:
        args.n_seeds = 100 if args.part == "ab" else 1000

    device = torch.device("cuda")
    ckpt_path = REPO_ROOT / CKPTS[args.model]
    slug = args.model.replace("/", "-") + (f".{args.tag}" if args.tag else "")
    out_dir = REPO_ROOT / args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== phase2 pilot | model={args.model} part={args.part} "
          f"n={args.n_seeds} batch={args.batch_size} eps={args.eps} "
          f"iters={args.n_steps_attack} ode={args.n_steps_ode} "
          f"cfg={args.cfg_scale} ===", flush=True)

    model = load_model(args.model, ckpt_path, device)
    vae = AutoencoderKL.from_pretrained(
        VAE_REPO, torch_dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    if args.part == "ab":
        run_ab(model, vae, args, device, out_dir)
    else:
        run_fid(model, vae, args, device, out_dir)
    t_total = time.perf_counter() - t0

    print(f"peak VRAM: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB",
          flush=True)
    write_meta(out_dir, args.part, args, ckpt_path, t_total)
    print(f"done in {t_total / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
