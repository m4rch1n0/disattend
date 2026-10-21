#!/usr/bin/env python3
"""Phase 5 energy probe: softmax row-energy (logsumexp of the attention
logits) on a saved Phase-5 leg, forward only, with guidance when the leg was
run with guidance. Mirrors scripts/run_softmax_energy_probe.py (Phase 3) and
reuses its reduction and self-check; the only additions are the cfg-aware
Euler step and the [cond; uncond] split.

Branches, exactly the leg's: benign z_T, pgd z_T_adv from eps_<eps>.pt, rand
= z_T + the seed-anchored Rademacher delta (regenerated bit-identical).
Writes <leg>/energy_eps<eps>.pt (conditional half; keys ben/rand/pgd) and,
at cfg != 1, <leg>/energy_eps<eps>_uncond.pt, plus energy_meta_eps<eps>.json.

  .venv/bin/python -u scripts/run_phase5_energy.py --model SiT-B/2 \
      --leg experiments/phase5_cfg/SiT-B-2 --cfg-scale 1.5
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

from src.attacks.pgd_latent import euler_step
from src.evaluation.softmax_energy import EnergyCollector, stack_energy_steps
from scripts.run_phase2_pilot import file_md5
from scripts.run_phase3_grid import make_rademacher
from scripts.run_phase5 import CKPTS, MODEL_KEY, load_model_any
from scripts.run_softmax_energy_probe import cat_energy, self_check


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--leg", required=True, help="Phase-5 leg dir with benign.pt and eps_<eps>.pt")
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    return ap.parse_args()


@torch.inference_mode()
def measure_energy_cfg(model, z, y, n_steps, device, cfg_scale):
    col = EnergyCollector(model)
    B = z.shape[0]
    cond, unc = [], []
    with col:
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y, cfg_scale)
            snap = col.snapshot()
            if cfg_scale != 1.0:
                cond.append({n: {k: (v[:B] if k != "n_tokens" else v) for k, v in d.items()}
                             for n, d in snap.items()})
                unc.append({n: {k: (v[B:] if k != "n_tokens" else v) for k, v in d.items()}
                            for n, d in snap.items()})
            else:
                cond.append(snap)
    return stack_energy_steps(cond), (stack_energy_steps(unc) if unc else None)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    leg = REPO_ROOT / args.leg
    out_file = leg / f"energy_eps{args.eps}.pt"
    if out_file.exists():
        print(f"[p5 energy] {out_file} exists; nothing to do", flush=True)
        return 0
    meta_leg = json.load(open(leg / "meta.json"))
    assert meta_leg["model"] == args.model and float(meta_leg["cfg_scale"]) == args.cfg_scale, \
        f"leg meta {meta_leg['model']} cfg {meta_leg['cfg_scale']} != args"
    ben = torch.load(leg / "benign.pt", map_location="cpu", weights_only=False)
    blob = torch.load(leg / f"eps_{args.eps}.pt", map_location="cpu", weights_only=False)
    z_T, y_all, z_adv = ben["z_T"], ben["y"], blob["z_T_adv"]
    n = z_T.shape[0]
    assert (z_adv - z_T).abs().max().item() <= args.eps * 1.001

    ckpt_path = REPO_ROOT / CKPTS[args.model]
    model = load_model_any(MODEL_KEY[args.model], ckpt_path, device)
    worst = self_check(model, z_T[:2].to(device), y_all[:2].to(device), device)
    print(f"[p5 energy {args.model} cfg{args.cfg_scale}] softmax fidelity ok ({worst:.1e})", flush=True)

    out = {b: None for b in ("ben", "rand", "pgd")}
    out_u = {b: None for b in ("ben", "rand", "pgd")}
    t0 = time.perf_counter()
    for bi in range(0, n, args.batch_size):
        ids = list(range(bi, min(bi + args.batch_size, n)))
        z = z_T[ids[0]:ids[-1] + 1].to(device)
        y = y_all[ids[0]:ids[-1] + 1].to(device)
        d = make_rademacher(z, args.eps, ids, device)
        za = z_adv[ids[0]:ids[-1] + 1].to(device)
        for branch, zz in (("ben", z), ("rand", z + d), ("pgd", za)):
            c, u = measure_energy_cfg(model, zz, y, args.n_steps_ode, device, args.cfg_scale)
            out[branch] = cat_energy(out[branch], c)
            if u is not None:
                out_u[branch] = cat_energy(out_u[branch], u)
        done = ids[-1] + 1
        rate = (time.perf_counter() - t0) / done
        print(f"[p5 energy {args.model} cfg{args.cfg_scale}] {done}/{n} seeds  ({rate:.1f} s/seed, "
              f"eta {(n - done) * rate / 60:.0f} min)", flush=True)

    common = {"eps": args.eps, "n_seeds": n, "n_steps_ode": args.n_steps_ode,
              "cfg_scale": args.cfg_scale}
    torch.save({**out, **common}, out_file)
    if out_u["ben"] is not None:
        torch.save({**out_u, **common, "half": "uncond"}, leg / f"energy_eps{args.eps}_uncond.pt")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    with open(leg / f"energy_meta_eps{args.eps}.json", "w") as f:
        json.dump({"model": args.model, "args": vars(args), "ckpt": str(ckpt_path),
                   "ckpt_md5": file_md5(ckpt_path), "git_commit": commit,
                   "wall_seconds": round(time.perf_counter() - t0, 1),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  f, indent=2)
    print(f"[p5 energy] saved {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
