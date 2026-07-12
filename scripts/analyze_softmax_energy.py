#!/usr/bin/env python3
"""Analyze the softmax-energy probe (exploratory, outside the PREREG).

Question (Masi, arXiv 2407.06315): in classifiers untargeted PGD lowers the
EBM energy -logsumexp(logits), which means it RAISES the softmax denominator;
does the same happen to attention rows when the attack goes through the
sampler? Descriptive statistics only, same machinery
as the confirmatory analyzer (paired per-seed contrasts against the
Rademacher control, late window, N=256 locus).

Energy = mean row logsumexp of attention logits (nats). Also decomposed as
row_max + log tail mass (lse - max), to see whether a move comes from the
peak logit or from the rest of the row.

Beyond the primary-eps blocks, persists: the energy dose-response over every
energy_eps*.pt found (contrast, d, detector AUC per eps, Rademacher-control
AUC), and the link to the rank fingerprint at the primary eps (per-seed
energy-vs-erank correlation, energy-only and combined z-sum detector AUC).
Detector caveat, stated once for all AUC entries: thresholds and scores are
in-sample (no held-out split); the out-of-sample evidence is the z-unit
threshold transfer in detector.json.

Reads  experiments/energy_probe/<slug>/energy_eps*.pt
       experiments/phase3_main/<slug>/ (erank substrate, for the rank link)
Writes experiments/energy_probe/energy_analysis.json
       experiments/energy_probe/figures/energy_probe.(png|pdf)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from analyze_phase3 import (DIT_ALL, UNET_ALL, agg_scalar, late_window,
                            load_leg, member_stats)
from src.evaluation.softmax_energy import aggregate_energy

BASE = REPO_ROOT / "experiments/energy_probe"
P3 = REPO_ROOT / "experiments/phase3_main"
EPS = 0.05
SLUGS = {"dit": "SiT-B-2", "unet": "UNet-B", "dit95": "SiT-B-2-FID95"}
C = {"dit": "#0072B2", "unet": "#D55E00", "dit95": "#56B4E9"}
LBL = {"dit": "DiT-B/2", "unet": "UNet-B", "dit95": "DiT-B/2 @ FID95"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.edgecolor": "#666666",
})


def agg(blob, branch, key, steps):
    return aggregate_energy(blob[branch], key, steps).numpy()


def tail_key(blob, branch, steps):
    lse = aggregate_energy(blob[branch], "row_lse", steps).numpy()
    mx = aggregate_energy(blob[branch], "row_max", steps).numpy()
    return lse - mx


def block(blob, steps) -> dict:
    """Benign level + paired shifts and contrast for one model and window."""
    out = {}
    for key in ("row_lse", "row_max"):
        ben = agg(blob, "ben", key, steps)
        pgd = agg(blob, "pgd", key, steps)
        rnd = agg(blob, "rand", key, steps)
        out[key] = {
            "benign_mean": float(ben.mean()),
            "shift_pgd": member_stats(pgd - ben),
            "shift_rand": member_stats(rnd - ben),
            "contrast_pgd_vs_rand": member_stats(pgd - rnd),
        }
    tb, tp, tr = (tail_key(blob, b, steps) for b in ("ben", "pgd", "rand"))
    out["log_tail_mass"] = {
        "benign_mean": float(tb.mean()),
        "contrast_pgd_vs_rand": member_stats(tp - tr),
    }
    return out


def per_layer_shift(blob, layers, steps) -> dict[str, float]:
    out = {}
    for name in layers:
        p = blob["pgd"][name]["row_lse"][steps].mean(dim=(0, 2)).double().numpy()
        r = blob["rand"][name]["row_lse"][steps].mean(dim=(0, 2)).double().numpy()
        out[name] = float((p - r).mean())
    return out


def per_step_curve(blob, n_steps) -> tuple[np.ndarray, np.ndarray]:
    mus, errs = [], []
    for s in range(n_steps):
        c = agg(blob, "pgd", "row_lse", [s]) - agg(blob, "rand", "row_lse", [s])
        mus.append(c.mean())
        errs.append(1.96 * c.std(ddof=1) / np.sqrt(len(c)))
    return np.array(mus), np.array(errs)


def main() -> int:
    blobs = {m: torch.load(BASE / s / f"energy_eps{EPS}.pt", map_location="cpu",
                           weights_only=False) for m, s in SLUGS.items()}
    n_steps = blobs["dit"]["n_steps_ode"]
    steps = late_window(n_steps)

    res = {"exploratory": True, "eps": EPS,
           "note": "energy = mean row logsumexp of attention logits (nats), "
                   "N=256 locus; contrast is PGD vs the equal-budget "
                   "Rademacher control, paired per seed",
           "models": {}}
    for m, blob in blobs.items():
        res["models"][m] = {
            "late": block(blob, steps),
            "full": block(blob, None),
            "per_layer_late_shift": per_layer_shift(
                blob, DIT_ALL if m != "unet" else UNET_ALL, steps),
        }

    # --- dose-response over every measured eps (energy as detector) ---
    def auc_high(x0, x1):
        """P(x1 > x0): AUC of a flag-when-high detector."""
        u = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
        return float(u / (len(x0) * len(x1)))

    def auc_ci(x0, x1, n_boot=2000, seed=21260705):
        rng = np.random.default_rng(seed)
        vals = [auc_high(x0[rng.integers(0, len(x0), len(x0))],
                         x1[rng.integers(0, len(x1), len(x1))])
                for _ in range(n_boot)]
        return [float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    res["dose_response"] = {}
    for m, slug in SLUGS.items():
        per_eps = {}
        for f in sorted((BASE / slug).glob("energy_eps*.pt")):
            blob = torch.load(f, map_location="cpu", weights_only=False)
            b = agg(blob, "ben", "row_lse", steps)
            p = agg(blob, "pgd", "row_lse", steps)
            r = agg(blob, "rand", "row_lse", steps)
            c = p - r
            per_eps[str(blob["eps"])] = {
                "contrast_mean": float(c.mean()),
                "d": float(c.mean() / c.std(ddof=1)),
                "auc_pgd": auc_high(b, p), "auc_pgd_ci95": auc_ci(b, p),
                "auc_rand": auc_high(b, r),
            }
        res["dose_response"][m] = per_eps

    # --- link to the rank fingerprint at the primary eps ---
    res["rank_link"] = {"note": "erank flags low, energy flags high; combined "
                                "= sum of benign-normalized z-scores"}
    for m, slug in SLUGS.items():
        leg = load_leg(P3 / slug, [EPS], with_nfe=False)
        er_b = agg_scalar(leg["ben"]["attn"], "erank_rv", steps)
        er_p = agg_scalar(leg["eps"][EPS]["attn_pgd"], "erank_rv", steps)
        er_r = agg_scalar(leg["eps"][EPS]["attn_rand"], "erank_rv", steps)
        en_b, en_p, en_r = (agg(blobs[m], br, "row_lse", steps)
                            for br in ("ben", "pgd", "rand"))
        c_en, c_rank = en_p - en_r, er_r - er_p
        pe = stats.pearsonr(c_en, c_rank)
        z = lambda x, ref: (x - ref.mean()) / ref.std(ddof=1)  # noqa: E731
        s_b = z(en_b, en_b) - z(er_b, er_b)
        s_p = z(en_p, en_b) - z(er_p, er_b)
        res["rank_link"][m] = {
            "pearson": float(pe.statistic), "pearson_p": float(pe.pvalue),
            "spearman": float(stats.spearmanr(c_en, c_rank).statistic),
            "auc_energy": auc_high(en_b, en_p),
            "auc_erank": auc_high(er_p, er_b),
            "auc_combined_zsum": auc_high(s_b, s_p),
        }

    with open(BASE / "energy_analysis.json", "w") as f:
        json.dump(res, f, indent=2)

    # figure: per-step contrast curves + per-layer late profiles
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2),
                             gridspec_kw={"width_ratios": [1.5, 1, 0.6]})
    ax = axes[0]
    t_axis = (np.arange(n_steps) + 1) / n_steps
    for m in ("dit", "unet", "dit95"):
        mu, err = per_step_curve(blobs[m], n_steps)
        ax.plot(t_axis, mu, color=C[m], lw=2)
        ax.fill_between(t_axis, mu - err, mu + err, color=C[m], alpha=0.15, lw=0)
        ax.annotate(LBL[m], (t_axis[-1], mu[-1]), xytext=(4, 0),
                    textcoords="offset points", color=C[m], fontsize=8.5,
                    va="center", fontweight="bold")
    ax.axvspan(0.72, 1.0, color="#000000", alpha=0.05, lw=0)
    ax.axhline(0, color="#BBBBBB", lw=1)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("ODE time t")
    ax.set_ylabel("row energy contrast, nats\n(PGD vs random, N=256 locus)")
    ax.grid(axis="x", visible=False)

    for ax, m, layers in ((axes[1], "dit", DIT_ALL), (axes[2], "unet", UNET_ALL)):
        prof = res["models"][m]["per_layer_late_shift"]
        x = np.arange(len(layers))
        ax.bar(x, [prof[n] for n in layers], width=0.72, color=C[m],
               edgecolor="white", linewidth=0.8)
        if m == "dit":
            prof95 = res["models"]["dit95"]["per_layer_late_shift"]
            ax.bar(x, [prof95[n] for n in layers], width=0.72, fill=False,
                   edgecolor=C["dit95"], linewidth=1.4)
            ax.set_xticks(x[::2])
            ax.set_xticklabels([str(i) for i in range(0, len(layers), 2)],
                               fontsize=8)
            ax.set_xlabel("block")
            ax.set_title(f"{LBL['dit']} (outline: @ FID95)", fontsize=9.5,
                         color=C[m])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(["in.4", "in.5", "out.6", "out.7", "out.8"],
                               fontsize=8)
            ax.set_title(LBL[m], fontsize=9.5, color=C[m])
        ax.axhline(0, color="#BBBBBB", lw=1)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylabel("late-window shift, nats")

    (BASE / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(BASE / "figures" / f"energy_probe.{ext}", bbox_inches="tight")
    plt.close(fig)

    for m in ("dit", "unet", "dit95"):
        r = res["models"][m]["late"]["row_lse"]
        c = r["contrast_pgd_vs_rand"]
        print(f"{LBL[m]:16s} benign {r['benign_mean']:7.3f} nats | "
              f"contrast {c['mean']:+.3f} (d={c['mean'] / c['sd']:+.2f}, "
              f"p={c['p']:.1e})")
        mx = res["models"][m]["late"]["row_max"]["contrast_pgd_vs_rand"]
        tl = res["models"][m]["late"]["log_tail_mass"]["contrast_pgd_vs_rand"]
        print(f"{'':16s} decomposition: peak {mx['mean']:+.3f}, "
              f"log tail mass {tl['mean']:+.3f}")
    print("\ndose-response (eps: contrast | d | AUC pgd | AUC rand):")
    for m, per_eps in res["dose_response"].items():
        row = "  ".join(f"{e}: {v['contrast_mean']:+.3f}|{v['d']:.2f}|"
                        f"{v['auc_pgd']:.3f}|{v['auc_rand']:.3f}"
                        for e, v in sorted(per_eps.items(), key=lambda kv: float(kv[0])))
        print(f"  {m:6s} {row}")
    print("\nrank link @ primary eps:")
    for m in SLUGS:
        r = res["rank_link"][m]
        print(f"  {m:6s} pearson {r['pearson']:+.2f}  AUC energy {r['auc_energy']:.3f} "
              f"erank {r['auc_erank']:.3f} combined {r['auc_combined_zsum']:.3f}")
    print(f"\nwrote {BASE / 'energy_analysis.json'} and figures/energy_probe.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
