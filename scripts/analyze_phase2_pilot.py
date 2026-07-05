#!/usr/bin/env python3
"""Analyze the Phase 2 pilot: triplet gate, LPIPS check, Diff-FID, election.

Implements the pre-registered analysis (experiments/phase2_pilot/PREREG.md):

* Metrics A/B aggregated per sample on the N=256 locus (and full profile as
  context), from the ab_results.pt attention substrate.
* Triplet gate per metric: D_i then PASS iff mean(D) >= 2*SE(D) AND a
  one-sided Wilcoxon signed-rank agrees (p < 0.05).
    - A (entropy, two-sided effect): D_i = |shift_PGD,i| - |shift_rand,i|.
    - B (rank, drop direction):      D_i = drop_PGD,i - drop_rand,i
                                          = M_rand,i - M_PGD,i.
* LPIPS efficacy check: mean LPIPS(PGD)/LPIPS(rand); fallback flagged if <1.5.
* Metric C (Diff-FID): FID of each branch vs the 50k reference, deltas, and
  the ddFID contrast (dFID_pgd - dFID_rand) with a paired bootstrap 95% CI
  (PASS iff the CI excludes 0 on the positive side).
* Primary-metric election: highest |effect size| among passing metrics, with
  the pre-registered tie-break toward Metric B.

Reads experiments/phase2_pilot/<slug>/{ab_results.pt,fid_features.pt}. Writes
experiments/phase2_pilot/analysis.json and prints a comparison table. All CPU.

Usage:
    uv run python scripts/analyze_phase2_pilot.py            # both models
    uv run python scripts/analyze_phase2_pilot.py --models SiT-B/2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy import stats

from src.evaluation.attention_metrics import METRIC_KEYS, aggregate_per_sample
from src.evaluation.fid import compute_stats, fid_from_stats

SLUGS = {"SiT-B/2": "SiT-B-2", "UNet-B": "UNet-B"}
N_BOOT = 200  # >=200 per PREREG; each resample is a 2048x2048 matrix sqrt (~seconds)
LPIPS_FALLBACK_RATIO = 1.5


# ---------- Metrics A/B: per-sample scalars from the attention substrate ----------


def branch_scalars(attn_branch: dict, token_filter: int | None) -> dict[str, np.ndarray]:
    """One aggregated scalar per sample per metric for one branch: (n,) arrays."""
    agg = aggregate_per_sample(attn_branch, token_filter=token_filter)
    return {k: agg[k].numpy() for k in METRIC_KEYS}


def rand_branch_mean(ab: dict, token_filter: int | None,
                     k_rand: int) -> dict[str, np.ndarray]:
    """Mean over the K Rademacher draws of the per-sample aggregated scalar."""
    per_draw = [branch_scalars(ab["attn"][f"rand{k}"], token_filter)
                for k in range(k_rand)]
    return {m: np.mean([d[m] for d in per_draw], axis=0) for m in METRIC_KEYS}


def gate_one_sided(D: np.ndarray) -> dict:
    """mean(D) >= 2*SE(D) AND one-sided Wilcoxon (H1: median > 0)."""
    n = len(D)
    mean, sd = float(D.mean()), float(D.std(ddof=1))
    se = sd / np.sqrt(n)
    two_se_pass = mean >= 2 * se
    # Wilcoxon one-sided; guard the degenerate all-zero case.
    if np.allclose(D, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(D, alternative="greater",
                                 zero_method="wilcox").pvalue)
    return {
        "mean": mean, "se": se, "cohens_d": (mean / sd if sd > 0 else 0.0),
        "two_se_pass": bool(two_se_pass), "wilcoxon_p": p,
        "wilcoxon_pass": bool(p < 0.05),
        "pass": bool(two_se_pass and p < 0.05), "n": n,
    }


def analyze_ab(ab: dict, token_filter: int | None) -> dict:
    """Metrics A and B triplet gates + descriptive shifts on one locus."""
    k_rand = sum(1 for key in ab["attn"] if key.startswith("rand"))
    ben = branch_scalars(ab["attn"]["ben"], token_filter)
    pgd = branch_scalars(ab["attn"]["pgd"], token_filter)
    rnd = rand_branch_mean(ab, token_filter, k_rand)

    out: dict = {"n_layers": int(aggregate_per_sample(
        ab["attn"]["ben"], token_filter=token_filter)["n_layers"]),
        "k_rand": k_rand}

    # Metric A (entropy): two-sided effect -> gate on |shift|.
    shift_pgd = pgd["entropy"] - ben["entropy"]
    shift_rnd = rnd["entropy"] - ben["entropy"]
    D_A = np.abs(shift_pgd) - np.abs(shift_rnd)
    out["entropy"] = {
        "shift_pgd_mean": float(shift_pgd.mean()),
        "shift_rand_mean": float(shift_rnd.mean()),
        "abs_shift_pgd_mean": float(np.abs(shift_pgd).mean()),
        "abs_shift_rand_mean": float(np.abs(shift_rnd).mean()),
        "gate": gate_one_sided(D_A),
    }

    # Metric B (both rank defs): drop direction -> D = M_rand - M_PGD.
    for key in ("flatness_ratio", "erank_rv"):
        drop_pgd = ben[key] - pgd[key]
        drop_rnd = ben[key] - rnd[key]
        D_B = drop_pgd - drop_rnd  # == rnd[key] - pgd[key]
        out[key] = {
            "drop_pgd_mean": float(drop_pgd.mean()),
            "drop_rand_mean": float(drop_rnd.mean()),
            "benign_mean": float(ben[key].mean()),
            "gate": gate_one_sided(D_B),
        }
    return out


def analyze_lpips(ab: dict) -> dict:
    lp_pgd = ab["lpips"]["pgd"].numpy()
    rand_keys = [k for k in ab["lpips"] if k.startswith("rand")]
    lp_rand = np.mean([ab["lpips"][k].numpy() for k in rand_keys], axis=0)
    l2_pgd = ab["l2_out"]["pgd"].numpy()
    l2_rand = np.mean([ab["l2_out"][k].numpy() for k in rand_keys], axis=0)
    ratio = float(lp_pgd.mean() / lp_rand.mean()) if lp_rand.mean() > 0 else float("inf")
    return {
        "lpips_pgd_mean": float(lp_pgd.mean()),
        "lpips_rand_mean": float(lp_rand.mean()),
        "lpips_ratio": ratio,
        "l2_out_pgd_mean": float(l2_pgd.mean()),
        "l2_out_rand_mean": float(l2_rand.mean()),
        "pass": bool(ratio >= LPIPS_FALLBACK_RATIO),
        "fallback_triggered": bool(ratio < LPIPS_FALLBACK_RATIO),
    }


# ---------- Metric C: Diff-FID with a paired bootstrap CI ----------


def _fid_vs_ref(feats: np.ndarray, mu_r: np.ndarray, sigma_r: np.ndarray) -> float:
    """Canonical (pytorch-fid) FID for the reported point estimates."""
    mu, sigma = compute_stats(feats)
    return fid_from_stats(mu_r, sigma_r, mu, sigma)


def _fid_fast(feats: np.ndarray, mu_r: np.ndarray, sigma_r: np.ndarray,
              sr_half: np.ndarray, tr_sigma_r: float) -> float:
    """Symmetric-form FID for the bootstrap loop (identical to _fid_vs_ref).

    Tr(sqrtm(s_r s_g)) = Tr(sqrtm(s_r^.5 s_g s_r^.5)); the argument is
    symmetric PSD so its matrix-sqrt trace = sum(sqrt(eigvalsh(.))). eigvalsh
    on a symmetric 2048x2048 matrix is far faster than sqrtm (9.4s) or eigvals
    on the non-symmetric product (~35s under bootstrap degeneracy). s_r^.5 is
    precomputed once."""
    mu_g, sigma_g = compute_stats(feats)
    m = sr_half @ sigma_g @ sr_half
    covmean_tr = np.sqrt(np.maximum(np.linalg.eigvalsh(m), 0)).sum()
    return float(((mu_r - mu_g) ** 2).sum()
                 + tr_sigma_r + np.trace(sigma_g) - 2 * covmean_tr)


def analyze_fid(fid_data: dict, ref_path: Path) -> dict:
    ref = torch.load(ref_path, weights_only=True)
    mu_r = ref["mu"].numpy() if torch.is_tensor(ref["mu"]) else ref["mu"]
    sigma_r = ref["sigma"].numpy() if torch.is_tensor(ref["sigma"]) else ref["sigma"]

    feats = {b: fid_data["feats"][b].numpy() for b in ("ben", "rand", "pgd")}
    n = feats["ben"].shape[0]
    point = {b: _fid_vs_ref(feats[b], mu_r, sigma_r) for b in feats}
    d_pgd = point["pgd"] - point["ben"]
    d_rand = point["rand"] - point["ben"]
    dd = d_pgd - d_rand

    from scipy.linalg import sqrtm
    sr_half = sqrtm(sigma_r).real  # once; symmetric-form FID uses it every boot
    tr_sigma_r = float(np.trace(sigma_r))
    rng = np.random.default_rng(20260705)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)  # paired resample (same idx across branches)
        f = {br: _fid_fast(feats[br][idx], mu_r, sigma_r, sr_half, tr_sigma_r)
             for br in feats}
        boots[b] = (f["pgd"] - f["ben"]) - (f["rand"] - f["ben"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": n,
        "fid_ben": point["ben"], "fid_rand": point["rand"], "fid_pgd": point["pgd"],
        "dFID_pgd": d_pgd, "dFID_rand": d_rand, "ddFID": float(dd),
        "ddFID_ci95": [float(lo), float(hi)],
        "ddFID_boot_sd": float(boots.std(ddof=1)),
        "effect_size": float(dd / boots.std(ddof=1)) if boots.std() > 0 else 0.0,
        "pass": bool(lo > 0),
    }


def elect_primary(model_result: dict) -> dict:
    """Highest |effect size| among passing metrics; tie-break -> Metric B."""
    cands = []
    ab = model_result["ab_n256"]
    for key in ("entropy", "flatness_ratio", "erank_rv"):
        g = ab[key]["gate"]
        if g["pass"]:
            cands.append((key, abs(g["cohens_d"]), key != "entropy"))
    fid = model_result.get("fid")
    if fid and fid["pass"]:
        cands.append(("diff_fid", abs(fid["effect_size"]), False))
    if not cands:
        return {"primary": None, "reason": "no metric passed the gate"}
    cands.sort(key=lambda c: c[1], reverse=True)
    top_es = cands[0][1]
    within = [c for c in cands if top_es - c[1] <= 0.15 * top_es]
    b_pref = [c for c in within if c[2]]
    chosen = (max(b_pref, key=lambda c: c[1]) if b_pref else cands[0])[0]
    return {"primary": chosen,
            "candidates": {c[0]: round(c[1], 4) for c in cands},
            "reason": ("tie-break to Metric B" if b_pref and chosen != cands[0][0]
                       else "highest effect size")}


def fmt_gate(g: dict) -> str:
    return (f"mean={g['mean']:+.4f} 2SE={2 * g['se']:.4f} d={g['cohens_d']:+.2f} "
            f"W_p={g['wilcoxon_p']:.3f} -> {'PASS' if g['pass'] else 'fail'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["SiT-B/2", "UNet-B"])
    ap.add_argument("--pilot-dir", default="experiments/phase2_pilot")
    ap.add_argument("--ref-stats", default="data/imagenet_latents/fid_ref_stats.pt")
    args = ap.parse_args()

    pilot_dir = REPO_ROOT / args.pilot_dir
    results: dict = {}
    for model in args.models:
        slug = SLUGS[model]
        d = pilot_dir / slug
        ab_path, fid_path = d / "ab_results.pt", d / "fid_features.pt"
        if not ab_path.exists():
            print(f"[skip {model}] no {ab_path}")
            continue
        ab = torch.load(ab_path, weights_only=False)
        r: dict = {
            "ab_n256": analyze_ab(ab, token_filter=256),
            "ab_full": analyze_ab(ab, token_filter=None),
            "lpips": analyze_lpips(ab),
        }
        if fid_path.exists():
            r["fid"] = analyze_fid(torch.load(fid_path, weights_only=False),
                                   REPO_ROOT / args.ref_stats)
        r["election"] = elect_primary(r)
        results[model] = r

    for model, r in results.items():
        print(f"\n{'=' * 70}\n{model}\n{'=' * 70}")
        ab = r["ab_n256"]
        print(f"[N=256 locus, {ab['n_layers']} layers, K={ab['k_rand']} rand draws]")
        print(f"  A entropy   : {fmt_gate(ab['entropy']['gate'])}")
        print(f"      |shift| PGD={ab['entropy']['abs_shift_pgd_mean']:.4f} "
              f"rand={ab['entropy']['abs_shift_rand_mean']:.4f}")
        for key in ("flatness_ratio", "erank_rv"):
            print(f"  B {key:14s}: {fmt_gate(ab[key]['gate'])}")
            print(f"      drop PGD={ab[key]['drop_pgd_mean']:+.4f} "
                  f"rand={ab[key]['drop_rand_mean']:+.4f} "
                  f"(benign={ab[key]['benign_mean']:.4f})")
        lp = r["lpips"]
        print(f"  LPIPS check : PGD={lp['lpips_pgd_mean']:.4f} "
              f"rand={lp['lpips_rand_mean']:.4f} ratio={lp['lpips_ratio']:.2f} "
              f"-> {'OK' if lp['pass'] else 'FALLBACK (image-space objective)'}")
        print(f"  L2 output   : PGD={lp['l2_out_pgd_mean']:.2f} "
              f"rand={lp['l2_out_rand_mean']:.2f}")
        if "fid" in r:
            f = r["fid"]
            print(f"  C Diff-FID  : ddFID={f['ddFID']:+.3f} "
                  f"CI95=[{f['ddFID_ci95'][0]:+.3f},{f['ddFID_ci95'][1]:+.3f}] "
                  f"-> {'PASS' if f['pass'] else 'fail'}")
            print(f"      FID ben={f['fid_ben']:.2f} rand={f['fid_rand']:.2f} "
                  f"pgd={f['fid_pgd']:.2f} (n={f['n']})")
        print(f"  PRIMARY     : {r['election']['primary']} "
              f"({r['election']['reason']}; {r['election'].get('candidates', {})})")

    out_path = pilot_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
