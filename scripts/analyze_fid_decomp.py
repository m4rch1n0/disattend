#!/usr/bin/env python3
"""Decompose the pilot's Diff-FID drop: mean shift vs covariance term.

The pilot showed PGD LOWERS FID (analysis.json: ddFID about -16/-18, CI all
negative), which killed differential FID as a candidate metric. This script
persists the decomposition behind that reading. FID splits exactly into

    FID(g) = ||mu_r - mu_g||^2  +  Tr(S_r) + Tr(S_g) - 2 Tr((S_r S_g)^1/2)
             [mean term]           [covariance term]

so the drop FID(ben) - FID(pgd) attributes exactly to the two terms. The
share of the mean term tells whether the attack moves the batch's Inception
statistics toward the reference mean (a displacement artifact) rather than
improving realism; Tr(S_g) per branch checks diversity is not collapsing.

Reads experiments/phase2_pilot/<slug>/fid_features.pt and the repo FID
reference stats. Writes experiments/phase2_pilot/fid_decomp.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

SLUGS = {"SiT-B/2": "SiT-B-2", "UNet-B": "UNet-B"}
REF = REPO_ROOT / "data/imagenet_latents/fid_ref_stats.pt"


def stats_of(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def fid_terms(mu_r, sigma_r, sr_half, tr_sigma_r, feats) -> dict:
    mu_g, sigma_g = stats_of(feats)
    mean_term = float(((mu_r - mu_g) ** 2).sum())
    m = sr_half @ sigma_g @ sr_half
    covmean_tr = float(np.sqrt(np.maximum(np.linalg.eigvalsh(m), 0)).sum())
    cov_term = float(tr_sigma_r + np.trace(sigma_g) - 2 * covmean_tr)
    return {"fid": mean_term + cov_term, "mean_term": mean_term,
            "cov_term": cov_term, "tr_sigma_g": float(np.trace(sigma_g))}


def main() -> int:
    ref = torch.load(REF, weights_only=True)
    mu_r = ref["mu"].numpy() if torch.is_tensor(ref["mu"]) else ref["mu"]
    sigma_r = ref["sigma"].numpy() if torch.is_tensor(ref["sigma"]) else ref["sigma"]
    from scipy.linalg import sqrtm
    sr_half = sqrtm(sigma_r).real
    tr_sigma_r = float(np.trace(sigma_r))

    out = {"note": "pilot substrate, n=1000 per branch; terms in FID units. "
                   "mean_share = (mean_term(ben) - mean_term(pgd)) / "
                   "(FID(ben) - FID(pgd))"}
    for model, slug in SLUGS.items():
        blob = torch.load(REPO_ROOT / f"experiments/phase2_pilot/{slug}/fid_features.pt",
                          map_location="cpu", weights_only=False)
        terms = {b: fid_terms(mu_r, sigma_r, sr_half, tr_sigma_r,
                              blob["feats"][b].numpy())
                 for b in ("ben", "rand", "pgd")}
        drop = terms["ben"]["fid"] - terms["pgd"]["fid"]
        mean_share = (terms["ben"]["mean_term"] - terms["pgd"]["mean_term"]) / drop
        div_change = (terms["pgd"]["tr_sigma_g"] / terms["ben"]["tr_sigma_g"] - 1)
        out[model] = {
            "terms": terms,
            "fid_drop_ben_minus_pgd": float(drop),
            "mean_term_share_of_drop": float(mean_share),
            "diversity_change_pgd_vs_ben_pct": float(100 * div_change),
            "diversity_change_rand_vs_ben_pct": float(
                100 * (terms["rand"]["tr_sigma_g"] / terms["ben"]["tr_sigma_g"] - 1)),
        }
        print(f"{model}: FID ben {terms['ben']['fid']:.2f} -> pgd "
              f"{terms['pgd']['fid']:.2f} (drop {drop:.2f}); mean-term share "
              f"{100 * mean_share:.0f}%; diversity Tr(Sigma) {100 * div_change:+.1f}%")
    path = REPO_ROOT / "experiments/phase2_pilot/fid_decomp.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
