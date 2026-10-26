#!/usr/bin/env python3
"""Paper figure for the post-thesis Phase 5 checks (guidance and official
SiT checkpoints). Reads the Phase-5 substrate through the Phase-3 loaders,
exactly as scripts/analyze_phase5.py does; nothing plotted differs from
experiments/phase5_*/analysis.json.

Writes (into paper/figures/):
  fig_phase5.(png|pdf)   (a) per-block profiles of the four official sizes
                         at cfg 1 on normalized depth, XL at cfg 1.5 dashed;
                         (b) late contrast c with 95% CI per model and cfg.
  phase5_curves.json     the plotted values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_phase3 import PRIMARY_EPS, late_window, load_leg, per_layer_contrast
from analyze_phase5 import contrast, layers_of

P3 = REPO_ROOT / "experiments/phase3_main"
P5C = REPO_ROOT / "experiments/phase5_cfg"
P5P = REPO_ROOT / "experiments/phase5_pretrained"
OUT = REPO_ROOT / "paper/figures"
STEPS = late_window(25)

SIZE_C = {"S": "#CC79A7", "B": "#0072B2", "L": "#009E73", "XL": "#E69F00"}
C_OURS = {"dit": "#0072B2", "unet": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.edgecolor": "#666666",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
})


def ci95(x: np.ndarray) -> float:
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def panel_letter(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=10.5, fontweight="bold")


def main() -> int:
    legs = {
        "S": load_leg(P5P / "SiT-S-2-official", [PRIMARY_EPS], False),
        "B": load_leg(P5P / "SiT-B-2-official", [PRIMARY_EPS], False),
        "L": load_leg(P5P / "SiT-L-2-official", [PRIMARY_EPS], False),
        "XL": load_leg(P5P / "SiT-XL-2-official", [PRIMARY_EPS], False),
        "XL15": load_leg(P5P / "SiT-XL-2-official.cfg15", [PRIMARY_EPS], False),
        "dit": load_leg(P3 / "SiT-B-2", [PRIMARY_EPS], False),
        "unet": load_leg(P3 / "UNet-B", [PRIMARY_EPS], False),
        "dit15": load_leg(P5C / "SiT-B-2", [PRIMARY_EPS], False),
        "unet15": load_leg(P5C / "UNet-B", [PRIMARY_EPS], False),
    }
    curves = {"profiles": {}, "contrast": {}}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 3.2),
                                     gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) profiles on normalized depth
    for key, ls, lw in (("S", "-", 1.6), ("B", "-", 1.6), ("L", "-", 1.6),
                        ("XL", "-", 2.0), ("XL15", (0, (4, 2.5)), 1.6)):
        leg = legs[key]
        layers, _ = layers_of(leg)
        d = leg["eps"][PRIMARY_EPS]
        pl = per_layer_contrast(d["attn_rand"], d["attn_pgd"], STEPS, layers)
        mu = np.array([pl[n].mean() for n in layers])
        ci = np.array([ci95(pl[n]) for n in layers])
        x = np.arange(len(layers)) / (len(layers) - 1)
        size = key.replace("15", "")
        label = f"SiT-{size}/2 ({len(layers)} blocks)" if key != "XL15" \
            else "SiT-XL/2, cfg 1.5"
        ax_a.plot(x, mu, color=SIZE_C[size], ls=ls, lw=lw, marker="o", ms=2.6,
                  label=label)
        ax_a.fill_between(x, mu - ci, mu + ci, color=SIZE_C[size],
                          alpha=0.10 if key != "XL15" else 0.0, lw=0)
        curves["profiles"][key] = {"layers": layers, "contrast_mean": mu.tolist(),
                                   "ci95_half": ci.tolist()}
    ax_a.axvline(0.5, color="#999999", lw=0.9, ls=(0, (4, 3)))
    ax_a.text(0.25, 0.97, "input half", transform=ax_a.transAxes, ha="center",
              va="top", fontsize=8, color="#4d4d4d")
    ax_a.text(0.75, 0.97, "deep half", transform=ax_a.transAxes, ha="center",
              va="top", fontsize=8, color="#4d4d4d")
    ax_a.set_xlabel("relative depth (block index / last block)")
    ax_a.set_ylabel("per-block collapse contrast $c$\n(late window, cfg 1 unless noted)")
    ax_a.set_ylim(bottom=0)
    ax_a.grid(axis="x", visible=False)
    ax_a.legend(fontsize=7.8, frameon=False, loc="upper left",
                bbox_to_anchor=(0.0, 0.93), ncol=1)
    panel_letter(ax_a, "(a)")

    # (b) contrast per model, cfg 1 filled vs cfg 1.5 hollow
    rows = [("DiT-B/2\n(ours)", "dit", "dit15", C_OURS["dit"]),
            ("UNet-B\n(ours)", "unet", "unet15", C_OURS["unet"]),
            ("SiT-S/2", "S", None, SIZE_C["S"]),
            ("SiT-B/2", "B", None, SIZE_C["B"]),
            ("SiT-L/2", "L", None, SIZE_C["L"]),
            ("SiT-XL/2", "XL", "XL15", SIZE_C["XL"])]
    xs = np.arange(len(rows))
    for i, (lbl, k1, k15, col) in enumerate(rows):
        c1 = contrast(legs[k1])
        ax_b.errorbar(xs[i] - (0.16 if k15 else 0.0), c1.mean(), yerr=ci95(c1),
                      fmt="o", color=col, ms=6, capsize=2.5, elinewidth=1.0)
        curves["contrast"][k1] = {"mean": float(c1.mean()), "ci95_half": float(ci95(c1)),
                                  "d": float(c1.mean() / c1.std(ddof=1)), "cfg": 1.0}
        if k15:
            c15 = contrast(legs[k15])
            ax_b.errorbar(xs[i] + 0.16, c15.mean(), yerr=ci95(c15), fmt="o",
                          color=col, mfc="white", mew=1.4, ms=6, capsize=2.5,
                          elinewidth=1.0)
            curves["contrast"][k15] = {"mean": float(c15.mean()),
                                       "ci95_half": float(ci95(c15)),
                                       "d": float(c15.mean() / c15.std(ddof=1)), "cfg": 1.5}
    ax_b.errorbar([], [], fmt="o", color="#4d4d4d", ms=6, label="cfg 1.0")
    ax_b.errorbar([], [], fmt="o", color="#4d4d4d", mfc="white", mew=1.4, ms=6,
                  label="cfg 1.5")
    ax_b.axvline(1.5, color="#999999", lw=0.9, ls=(0, (4, 3)))
    ax_b.text(0.17, 0.97, "trained here", transform=ax_b.transAxes, ha="center",
              va="top", fontsize=8, color="#4d4d4d")
    ax_b.text(0.67, 0.97, "official SiT checkpoints", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=8, color="#4d4d4d")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax_b.set_ylabel("late collapse contrast $c$")
    ax_b.set_ylim(0, 0.05)
    ax_b.grid(axis="x", visible=False)
    ax_b.legend(fontsize=8, frameon=False, loc="lower left")
    panel_letter(ax_b, "(b)")

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_phase5.{ext}", bbox_inches="tight")
    with open(OUT / "phase5_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print("  fig fig_phase5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
