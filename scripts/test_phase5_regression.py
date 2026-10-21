#!/usr/bin/env python3
"""Phase 5 test 1: the new runner at cfg 1 must reproduce the frozen Phase-3
substrate for the same seed ids (same model, same batch size, same torch).

Compares, for the first n seeds: benign z0, z_T, attacked z_T_adv, and the
per-cell erank_rv / entropy of every layer and step, plus the per-seed
late-window erank contrast rand - pgd on the N=256 locus. Exact replay is
expected on the benign branch (inference only); the PGD branch is 40 sign
steps, so a last-bit difference can in principle flip a sign and diverge:
the contrast is what the analysis consumes, so that is the number that must
agree, and the per-cell maxima are reported for the record.

  .venv/bin/python scripts/test_phase5_regression.py \
      --ref experiments/phase3_main/SiT-B-2 --new experiments/phase5_tests/SiT-B-2.reg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.analyze_phase3 import agg_scalar, late_window


def slice_stacked(st: dict, n: int) -> dict:
    return {name: {k: (v[:, :n] if k != "n_tokens" else v) for k, v in d.items()}
            for name, d in st.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tol-benign", type=float, default=1e-4)
    ap.add_argument("--tol-contrast", type=float, default=1e-3)
    args = ap.parse_args()
    ref, new = Path(args.ref), Path(args.new)
    rb = torch.load(ref / "benign.pt", map_location="cpu", weights_only=False)
    nb = torch.load(new / "benign.pt", map_location="cpu", weights_only=False)
    re_ = torch.load(ref / f"eps_{args.eps}.pt", map_location="cpu", weights_only=False)
    ne = torch.load(new / f"eps_{args.eps}.pt", map_location="cpu", weights_only=False)
    n = nb["z_T"].shape[0]
    print(f"comparing first {n} seeds")

    def maxdiff(a, b):
        return (a[:n].double() - b[:n].double()).abs().max().item()

    print(f"z_T        max|diff| {maxdiff(rb['z_T'], nb['z_T']):.2e}")
    print(f"y          max|diff| {maxdiff(rb['y'], nb['y']):.2e}")
    print(f"z0 benign  max|diff| {maxdiff(rb['z0'], nb['z0']):.2e}")
    print(f"z_T_adv    max|diff| {maxdiff(re_['z_T_adv'], ne['z_T_adv']):.2e}")
    print(f"z0 pgd     max|diff| {maxdiff(re_['z0_pgd'], ne['z0_pgd']):.2e}")
    print(f"z0 rand    max|diff| {maxdiff(re_['z0_rand'], ne['z0_rand']):.2e}")

    worst = {}
    for branch, ra, na in (("ben", rb["attn"], nb["attn"]),
                           ("rand", re_["attn_rand"], ne["attn_rand"]),
                           ("pgd", re_["attn_pgd"], ne["attn_pgd"])):
        assert list(ra.keys()) == list(na.keys()), "layer sets differ"
        w = 0.0
        for name in ra:
            for k in ("erank_rv", "entropy", "flatness_ratio"):
                w = max(w, (ra[name][k][:, :n].double() - na[name][k].double()).abs().max().item())
        worst[branch] = w
        print(f"attn {branch:4s} per-cell max|diff| {w:.2e}")

    steps = late_window(ra[next(iter(ra))]["erank_rv"].shape[0])
    c_ref = (agg_scalar(slice_stacked(re_["attn_rand"], n), "erank_rv", steps)
             - agg_scalar(slice_stacked(re_["attn_pgd"], n), "erank_rv", steps))
    c_new = (agg_scalar(ne["attn_rand"], "erank_rv", steps)
             - agg_scalar(ne["attn_pgd"], "erank_rv", steps))
    d = np.abs(c_ref - c_new)
    print(f"late-window erank contrast: ref mean {c_ref.mean():.5f}, new mean {c_new.mean():.5f}, "
          f"per-seed max|diff| {d.max():.2e}")
    ok = worst["ben"] <= args.tol_benign and d.max() <= args.tol_contrast
    print("REGRESSION TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
