#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, outside the PREREG): attack detection from erank.

Turns the Phase-3 finding into an operating point: can a white-box monitor
that measures the late-window N=256 erank_rv of its own attention flag an
attacked seed, with no benign reference for the incoming sample? Uses the
persisted Phase-3 substrate only (no new runs). Not pre-registered; reported
in the thesis as an exploratory section, clearly separated from the
confirmatory family.

Detector: score = per-sample late-window N=256 erank_rv aggregate (the
Phase-3 primary metric); flag when score < tau. Threshold-free skill = AUC
(Mann-Whitney). Operating points: tau at 5% FPR on the deployer's own benign
distribution. Cross-model transfer: tau expressed in benign z-units
(z = (x - mu_ben) / sd_ben, both known to whoever owns the model), fitted on
one model and applied to the other -- the equivalence result (co-primary A
bounded below SESOI) suggests the *response* is backbone-comparable; raw
erank levels are not (benign 0.47 vs 0.26), hence the z-unit convention.

Writes experiments/phase3_main/detector.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy import stats

from analyze_phase3 import EPS_GRID, PRIMARY_EPS, agg_scalar, late_window, load_leg

BASE = REPO_ROOT / "experiments/phase3_main"
FPR_TARGET = 0.05
N_BOOT = 2000
SEED = 21260705


def auc_low(ben: np.ndarray, att: np.ndarray) -> float:
    """AUC of the 'flag when low' detector: P(att < ben) + .5 P(=)."""
    u = stats.mannwhitneyu(ben, att, alternative="two-sided").statistic
    return float(u / (len(ben) * len(att)))


def auc_ci(ben: np.ndarray, att: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n_b, n_a = len(ben), len(att)
    vals = [auc_low(ben[rng.integers(0, n_b, n_b)], att[rng.integers(0, n_a, n_a)])
            for _ in range(N_BOOT)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def roc_points(ben: np.ndarray, att: np.ndarray) -> dict:
    taus = np.quantile(np.concatenate([ben, att]), np.linspace(0, 1, 201))
    return {"fpr": [float((ben < t).mean()) for t in taus],
            "tpr": [float((att < t).mean()) for t in taus]}


def scores(leg: dict, steps: list[int]) -> dict[str, np.ndarray]:
    out = {"ben": agg_scalar(leg["ben"]["attn"], "erank_rv", steps)}
    for e in leg["eps"]:
        out[f"pgd@{e}"] = agg_scalar(leg["eps"][e]["attn_pgd"], "erank_rv", steps)
        out[f"rand@{e}"] = agg_scalar(leg["eps"][e]["attn_rand"], "erank_rv", steps)
    return out


def main() -> int:
    steps = late_window(25)
    legs = {"dit": load_leg(BASE / "SiT-B-2", EPS_GRID, with_nfe=False),
            "unet": load_leg(BASE / "UNet-B", EPS_GRID, with_nfe=False),
            "dit95": load_leg(BASE / "SiT-B-2-FID95", [PRIMARY_EPS], with_nfe=False)}
    sc = {m: scores(leg, steps) for m, leg in legs.items()}

    res: dict = {"metric": "late-window N=256 erank_rv (Phase-3 primary)",
                 "fpr_target": FPR_TARGET, "auc": {}, "roc": {},
                 "operating_points": {}, "transfer": {}}

    # threshold-free skill per model per eps, PGD and the Rademacher specificity control
    for m in sc:
        ben = sc[m]["ben"]
        for key in [k for k in sc[m] if k != "ben"]:
            att = sc[m][key]
            res["auc"][f"{m}:{key}"] = {"auc": auc_low(ben, att),
                                        "ci95": auc_ci(ben, att)}
        res["roc"][m] = roc_points(ben, sc[m][f"pgd@{PRIMARY_EPS}"])

    # same-model operating point at 5% FPR (tau = 5th pct of own benign)
    for m in sc:
        ben = sc[m]["ben"]
        tau = float(np.quantile(ben, FPR_TARGET))
        pts = {"tau": tau, "fpr": float((ben < tau).mean())}
        for key in [k for k in sc[m] if k != "ben"]:
            pts[f"tpr:{key}"] = float((sc[m][key] < tau).mean())
        res["operating_points"][m] = pts

    # cross-model transfer of the z-unit threshold (fit on A, deploy on B)
    def z(m, key):
        mu, sd = sc[m]["ben"].mean(), sc[m]["ben"].std(ddof=1)
        return (sc[m][key] - mu) / sd
    for src, dst in (("dit", "unet"), ("unet", "dit"), ("dit", "dit95")):
        tau_z = float(np.quantile(z(src, "ben"), FPR_TARGET))
        entry = {"tau_z": tau_z,
                 "fpr_on_dst": float((z(dst, "ben") < tau_z).mean())}
        for key in [k for k in sc[dst] if k.startswith("pgd") or k.startswith("rand")]:
            entry[f"tpr:{key}"] = float((z(dst, key) < tau_z).mean())
        res["transfer"][f"{src}->{dst}"] = entry

    with open(BASE / "detector.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"{'cell':24s} {'AUC':>6s}  ci95")
    for k, v in res["auc"].items():
        print(f"{k:24s} {v['auc']:6.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("\noperating points (tau at 5% FPR on own benign):")
    for m, v in res["operating_points"].items():
        tprs = "  ".join(f"{k.split(':')[1]}={v[k]:.2f}" for k in v if k.startswith("tpr"))
        print(f"  {m:6s} tau={v['tau']:.3f}  {tprs}")
    print("\nz-unit threshold transfer:")
    for k, v in res["transfer"].items():
        tprs = "  ".join(f"{kk.split(':')[1]}={v[kk]:.2f}" for kk in v if kk.startswith("tpr"))
        print(f"  {k:12s} tau_z={v['tau_z']:+.2f} fpr={v['fpr_on_dst']:.3f}  {tprs}")
    print(f"\nwrote {BASE / 'detector.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
