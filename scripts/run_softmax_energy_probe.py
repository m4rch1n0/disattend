#!/usr/bin/env python3
"""Measure attention softmax energy on the saved Phase-3 branches (exploratory).

Forward-only probe of the softmax-denominator idea (Masi, arXiv 2407.06315):
does PGD move the row energy logsumexp_j(QK^T/sqrt(d))_ij of the attention
logits, the quantity the post-softmax map is invariant to? Reuses the
persisted confirmatory substrate at the primary eps, so no new attack runs
and the branches are exactly the grid's:
  benign  = z_T from benign.pt
  pgd     = z_T_adv from eps_<eps>.pt
  rand    = z_T + seed-anchored Rademacher delta (regenerated bit-identical)

For each model and branch, a 25-step Euler forward with EnergyCollector
yields a (layer x step x head) row-energy substrate; nothing N x N is stored.
Fidelity self-check per model before measuring: the collector's recomputed
softmax must match the verified AttentionCollector on a live forward.

Output under experiments/energy_probe/<slug>/: energy_eps<eps>.pt, meta.json.
One file per model, written when that model completes (a crash costs at most
one model; just relaunch). Runtime is about an hour, launch via nohup:
  nohup .venv/bin/python -u scripts/run_softmax_energy_probe.py \
      > experiments/energy_probe/run.log 2>&1 &
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

import torch

from src.attacks.pgd_latent import euler_step, load_model
from src.evaluation.softmax_energy import (ENERGY_KEYS, EnergyCollector,
                                           stack_energy_steps)
from src.utils.attention_hooks import AttentionCollector
from scripts.run_phase2_pilot import file_md5
from scripts.run_phase3_grid import CKPTS, MODEL_KEY, SLUG, make_rademacher

PHASE3_DIR = REPO_ROOT / "experiments/phase3_main"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["SiT-B/2", "UNet-B", "SiT-B/2-FID95"],
                    choices=list(CKPTS.keys()))
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    ap.add_argument("--out-dir", type=str, default="experiments/energy_probe")
    ap.add_argument("--skip-self-check", action="store_true")
    return ap.parse_args()


@torch.inference_mode()
def measure_energy(model, z, y, n_steps, device):
    col = EnergyCollector(model)
    steps = []
    with col:
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y)
            steps.append(col.snapshot())
    return stack_energy_steps(steps)


def cat_energy(acc, new):
    if acc is None:
        return new
    for name in acc:
        for k in ENERGY_KEYS:
            acc[name][k] = torch.cat([acc[name][k], new[name][k]], dim=1)
    return acc


@torch.inference_mode()
def self_check(model, z, y, device) -> float:
    """One Euler step with both collectors: recomputed softmaxes must agree."""
    ec = EnergyCollector(model, keep_softmax=True)
    ac = AttentionCollector(model, store_dtype=torch.float32, store_device="cpu")
    with ec, ac:
        euler_step(model, z, torch.zeros((), device=device),
                   torch.tensor(1.0 / 25, device=device), y)
        ec.snapshot()
        maps = ac.snapshot()
    worst = max((ec.softmax_maps[n] - maps[n]).abs().max().item() for n in maps)
    assert worst < 1e-5, f"energy-vs-attention softmax mismatch: {worst:.2e}"
    return worst


def run_model(name: str, args, device) -> None:
    slug = SLUG[name]
    src = PHASE3_DIR / slug
    ben = torch.load(src / "benign.pt", map_location="cpu", weights_only=False)
    eps_blob = torch.load(src / f"eps_{args.eps}.pt", map_location="cpu",
                          weights_only=False)
    n = args.n_seeds
    z_T = ben["z_T"][:n]
    y_all = ben["y"][:n]
    z_adv = eps_blob["z_T_adv"][:n]
    linf = (z_adv - z_T).abs().max().item()
    assert linf <= args.eps * 1.001, f"z_T_adv linf {linf} exceeds eps {args.eps}"

    ckpt_path = REPO_ROOT / CKPTS[name]
    model = load_model(MODEL_KEY[name], ckpt_path, device)

    if not args.skip_self_check:
        worst = self_check(model, z_T[:2].to(device), y_all[:2].to(device), device)
        print(f"[energy {name}] softmax fidelity check ok (worst {worst:.1e})",
              flush=True)

    out = {b: None for b in ("ben", "rand", "pgd")}
    t0 = time.perf_counter()
    for bi in range(0, n, args.batch_size):
        ids = list(range(bi, min(bi + args.batch_size, n)))
        z = z_T[ids[0]:ids[-1] + 1].to(device)
        y = y_all[ids[0]:ids[-1] + 1].to(device)
        d = make_rademacher(z, args.eps, ids, device)
        za = z_adv[ids[0]:ids[-1] + 1].to(device)
        for branch, zz in (("ben", z), ("rand", z + d), ("pgd", za)):
            out[branch] = cat_energy(
                out[branch], measure_energy(model, zz, y, args.n_steps_ode, device))
        done = ids[-1] + 1
        rate = (time.perf_counter() - t0) / done
        print(f"[energy {name}] {done}/{n} seeds  ({rate:.1f} s/seed, "
              f"eta {(n - done) * rate / 60:.0f} min)", flush=True)

    out_dir = REPO_ROOT / args.out_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ben": out["ben"], "rand": out["rand"], "pgd": out["pgd"],
                "eps": args.eps, "n_seeds": n,
                "n_steps_ode": args.n_steps_ode},
               out_dir / f"energy_eps{args.eps}.pt")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    with open(out_dir / f"meta_eps{args.eps}.json", "w") as f:
        json.dump({"model": name, "args": vars(args), "ckpt": str(ckpt_path),
                   "ckpt_md5": file_md5(ckpt_path), "git_commit": commit,
                   "source_substrate": str(src),
                   "wall_seconds": round(time.perf_counter() - t0, 1),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())}, f, indent=2)
    print(f"[energy {name}] saved to {out_dir}", flush=True)
    del model
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    print(f"=== energy probe | models={args.models} eps={args.eps} "
          f"n={args.n_seeds} ode={args.n_steps_ode} ===", flush=True)
    for name in args.models:
        run_model(name, args, device)
    print("ENERGY PROBE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
