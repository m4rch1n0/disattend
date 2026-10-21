#!/usr/bin/env python3
"""Paper print versions of the phase 3 result figures (English labels).

Copy of scripts/make_ch05_figures.py re-targeted at paper/figures/ with every
rendered string translated to English and one extra combined figure
(dose-response + temporal side by side). Reads the persisted phase3 substrate
via the frozen analyzer's loaders exactly like make_phase3_figures.py (nothing
recomputed differs from analysis.json).
Writes (into paper/figures/):
  fig_dose_response.(png|pdf)
  fig_temporal.(png|pdf)
  fig_dose_temporal.(png|pdf)   (a) dose-response + (b) temporal, one figure
  fig_localization.(png|pdf)
  fig_detector.(png|pdf)
  fig_energy.(png|pdf)
  curves.json   the plotted values (means + CI95 halves)
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

from analyze_phase3 import (DIT_ALL, DIT_IN, EPS_GRID, PRIMARY_EPS, UNET_ALL,
                            UNET_IN, agg_scalar, erank_contrast, late_window,
                            load_leg, per_layer_contrast, share_profile)

BASE = REPO_ROOT / "experiments/phase3_main"
OUT = REPO_ROOT / "paper/figures"
C = {"dit": "#0072B2", "unet": "#D55E00", "dit95": "#56B4E9"}
LBL = {"dit": "DiT-B/2", "unet": "UNet-B", "dit95": "DiT-B/2 @ FID95"}
YLBL = "collapse contrast $c$\n(random erank $-$ attacked erank)"

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


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {name}")


def panel_letter(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=10.5, fontweight="bold")


def draw_dose_response(ax, legs, steps, curves):
    for m in ("dit", "unet"):
        mu = [erank_contrast(legs[m], e, steps).mean() for e in EPS_GRID]
        err = [ci95(erank_contrast(legs[m], e, steps)) for e in EPS_GRID]
        curves["dose_response"][m] = {"eps": EPS_GRID,
                                      "contrast_mean": [float(v) for v in mu],
                                      "ci95_half": [float(v) for v in err]}
        ax.errorbar(EPS_GRID, mu, yerr=err, color=C[m], lw=2, marker="o", ms=5,
                    capsize=2.5, label=LBL[m])
        ax.annotate(LBL[m], (EPS_GRID[-1], mu[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C[m], fontsize=9,
                    va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(EPS_GRID)
    ax.set_xticklabels([str(e) for e in EPS_GRID])
    ax.set_xlim(0.0085, 0.16)
    ax.set_xlabel(r"$\epsilon$  ($L_\infty$ budget on $z_T$)")
    ax.set_ylabel(YLBL)
    ax.grid(axis="x", visible=False)


def draw_temporal(ax, legs, curves):
    t_axis = (np.arange(25) + 1) / 25
    for m in ("dit", "unet"):
        d = legs[m]["eps"][PRIMARY_EPS]
        per_step = []
        for s in range(25):
            r = agg_scalar(d["attn_rand"], "erank_rv", [s])
            p = agg_scalar(d["attn_pgd"], "erank_rv", [s])
            per_step.append(r - p)
        mu = np.array([c.mean() for c in per_step])
        err = np.array([ci95(c) for c in per_step])
        curves["temporal"][m] = {"t": [float(v) for v in t_axis],
                                 "contrast_mean": [float(v) for v in mu],
                                 "ci95_half": [float(v) for v in err]}
        ax.plot(t_axis, mu, color=C[m], lw=2)
        ax.fill_between(t_axis, mu - err, mu + err, color=C[m], alpha=0.18, lw=0)
        ax.annotate(LBL[m], (t_axis[-1], mu[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C[m], fontsize=9,
                    va="center", fontweight="bold")
    ax.axvspan(0.72, 1.0, color="#000000", alpha=0.05, lw=0)
    ax.text(0.86, ax.get_ylim()[1] * 0.06, "late window\n($t \\geq 0.72$)",
            ha="center", fontsize=8, color="#4d4d4d")
    ax.set_xlabel("sampler time $t$ (noise to data)")
    ax.set_ylabel(YLBL)
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", visible=False)


def dose_response(legs, steps, curves):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    draw_dose_response(ax, legs, steps, curves)
    save(fig, "fig_dose_response")


def temporal(legs, curves):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    draw_temporal(ax, legs, curves)
    save(fig, "fig_temporal")


def dose_temporal(legs, steps, curves):
    """(a) dose-response and (b) temporal profile side by side, one figure."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    draw_dose_response(ax_a, legs, steps, curves)
    draw_temporal(ax_b, legs, curves)
    panel_letter(ax_a, "(a)")
    panel_letter(ax_b, "(b)")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_dose_temporal")


def localization(legs, steps, curves):
    """Print version of F2 (per-layer profiles, three panels)."""
    profs, cis, keys, s_note = {}, {}, {}, {}
    for m, layers, inp in (("dit", DIT_ALL, DIT_IN), ("dit95", DIT_ALL, DIT_IN),
                           ("unet", UNET_ALL, UNET_IN)):
        d = legs[m]["eps"][PRIMARY_EPS]
        pl = per_layer_contrast(d["attn_rand"], d["attn_pgd"], steps, layers)
        profs[m] = [pl[n].mean() for n in layers]
        cis[m] = [ci95(pl[n]) for n in layers]
        keys[m] = layers
        s_note[m] = f"S = {share_profile(pl, inp, layers):+.2f}"
        curves["localization"][m] = {"layers": layers,
                                     "contrast_mean": [float(v) for v in profs[m]],
                                     "ci95_half": [float(v) for v in cis[m]]}
    ymax = max(max(np.array(profs[m]) + np.array(cis[m])) for m in profs) * 1.28

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)
    for ax, m in zip(axes, ("dit", "dit95", "unet")):
        n_l = len(keys[m])
        n_in = 6 if m != "unet" else len(UNET_IN)
        x = np.arange(n_l)
        for i in range(n_l):
            ax.bar(x[i], profs[m][i], width=0.72, color=C[m],
                   hatch="///" if i < n_in else None,
                   edgecolor="white", linewidth=0.8)
        ax.errorbar(x, profs[m], yerr=cis[m], fmt="none", ecolor="#4d4d4d",
                    elinewidth=0.9, capsize=1.8)
        ax.axvline(n_in - 0.5, color="#999999", lw=0.9, ls=(0, (4, 3)))
        if m == "unet":
            ax.set_xticks(x)
            ax.set_xticklabels(["enc.4", "enc.5", "dec.6", "dec.7", "dec.8"],
                               fontsize=7)
            side_lbl = ("encoder", "decoder")
        else:
            ax.set_xticks(x[::2])
            ax.set_xticklabels([str(i) for i in range(0, n_l, 2)], fontsize=8)
            ax.set_xlabel("block")
            side_lbl = ("input (0–5)", "deep (6–11)")
        ax.text((n_in - 0.5) / 2 / n_l, 0.97, side_lbl[0], transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#4d4d4d")
        ax.text((n_in - 0.5 + n_l) / 2 / n_l, 0.97, side_lbl[1],
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color="#4d4d4d")
        ax.set_title(LBL[m], color=C[m], fontsize=10.5, fontweight="bold")
        ax.text(0.985, 0.83, s_note[m], transform=ax.transAxes, ha="right",
                fontsize=8.5, color="#1a1a1a")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("per-layer collapse contrast $c$\n(late window)")
    save(fig, "fig_localization")


def detector_fig(curves):
    """Print version of F4 (ROC at primary eps + AUC vs eps)."""
    det = json.load(open(BASE / "detector.json"))
    C3 = dict(C, dit95="#56B4E9")
    L3 = dict(LBL, dit95="DiT-B/2 @ FID95")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax1.plot([0, 1], [0, 1], color="#BBBBBB", lw=1, ls=(0, (4, 3)))
    for m in ("dit", "unet", "dit95"):
        r = det["roc"][m]
        auc = det["auc"][f"{m}:pgd@{PRIMARY_EPS}"]["auc"]
        ax1.plot(r["fpr"], r["tpr"], color=C3[m], lw=2,
                 label=f"{L3[m]}  AUC {auc:.2f}")
    ax1.set_xlabel("false positive rate (benign flagged)")
    ax1.set_ylabel("true positive rate (attacks flagged)")
    ax1.set_title(f"ROC, benign vs attacked, $\\varepsilon$ = {PRIMARY_EPS}",
                  fontsize=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)

    for m in ("dit", "unet"):
        mu = [det["auc"][f"{m}:pgd@{e}"]["auc"] for e in EPS_GRID]
        lo = [mu[i] - det["auc"][f"{m}:pgd@{e}"]["ci95"][0]
              for i, e in enumerate(EPS_GRID)]
        hi = [det["auc"][f"{m}:pgd@{e}"]["ci95"][1] - mu[i]
              for i, e in enumerate(EPS_GRID)]
        ax2.errorbar(EPS_GRID, mu, yerr=[lo, hi], color=C[m], lw=2, marker="o",
                     ms=5, capsize=2.5)
        ax2.annotate(LBL[m], (EPS_GRID[-1], mu[-1]),
                     xytext=(6, -2 if m == "unet" else 6),
                     textcoords="offset points", color=C[m], fontsize=9,
                     fontweight="bold")
        rnd = [det["auc"][f"{m}:rand@{e}"]["auc"] for e in EPS_GRID]
        ax2.plot(EPS_GRID, rnd, color=C[m], lw=1.2, ls=(0, (4, 3)), alpha=0.55)
    ax2.axhline(0.5, color="#BBBBBB", lw=1)
    ax2.text(0.011, 0.515, "chance", fontsize=8, color="#4d4d4d")
    ax2.text(0.011, 0.44, "random control", fontsize=8,
             color="#4d4d4d")
    ax2.set_xscale("log")
    ax2.set_xticks(EPS_GRID); ax2.set_xticklabels([str(e) for e in EPS_GRID])
    ax2.set_xlim(0.0085, 0.16); ax2.set_ylim(0.4, 1.02)
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("AUC (flags low rank)")
    ax2.set_title("AUC as a function of the budget", fontsize=10)
    ax2.grid(axis="x", visible=False)
    save(fig, "fig_detector")


def energy_fig(height=6.2):
    """Print version of the energy-probe figure (presentation-only:
    reads the persisted energy substrate and energy_analysis.json)."""
    import torch
    from analyze_softmax_energy import EPS, SLUGS, per_step_curve
    EBASE = REPO_ROOT / "experiments/energy_probe"
    ea = json.load(open(EBASE / "energy_analysis.json"))
    blobs = {m: torch.load(EBASE / s / f"energy_eps{EPS}.pt",
                           map_location="cpu", weights_only=False)
             for m, s in SLUGS.items()}
    n_steps = blobs["dit"]["n_steps_ode"]

    fig = plt.figure(figsize=(8.6, height))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, :])
    t_axis = (np.arange(n_steps) + 1) / n_steps
    for m in ("dit", "unet", "dit95"):
        mu, err = per_step_curve(blobs[m], n_steps)
        ax.plot(t_axis, mu, color=C[m], lw=2)
        ax.fill_between(t_axis, mu - err, mu + err, color=C[m], alpha=0.15,
                        lw=0)
        ax.annotate(LBL[m], (t_axis[-1], mu[-1]), xytext=(5, 0),
                    textcoords="offset points", color=C[m], fontsize=9,
                    va="center", fontweight="bold")
    ax.axvspan(0.72, 1.0, color="#000000", alpha=0.05, lw=0)
    ax.axhline(0, color="#BBBBBB", lw=1)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("sampler time $t$")
    ax.set_ylabel("row log-sum-exp contrast, nats\n"
                  "(attacked $-$ random, locus $N=256$)")
    ax.grid(axis="x", visible=False)

    ax_d = fig.add_subplot(gs[1, :3])
    ax_u = fig.add_subplot(gs[1, 3:])
    for ax, m in ((ax_d, "dit"), (ax_u, "unet")):
        prof = ea["models"][m]["per_layer_late_shift"]
        layers = list(prof.keys())
        x = np.arange(len(layers))
        ax.bar(x, [prof[n] for n in layers], width=0.72, color=C[m],
               edgecolor="white", linewidth=0.8)
        if m == "dit":
            prof95 = ea["models"]["dit95"]["per_layer_late_shift"]
            ax.bar(x, [prof95[n] for n in layers], width=0.72, fill=False,
                   edgecolor=C["dit95"], linewidth=1.4)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in range(len(layers))],
                               fontsize=8)
            ax.set_xlabel("block")
            ax.set_title(f"{LBL['dit']} (outline: @ FID95)", fontsize=10,
                         color=C[m])
            ax.set_ylabel("late shift, nats")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(["enc.4", "enc.5", "dec.6", "dec.7", "dec.8"],
                               fontsize=8)
            ax.set_title(LBL[m], fontsize=10, color=C[m])
        ax.axhline(0, color="#BBBBBB", lw=1)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(h_pad=2.0)
    save(fig, "fig_energy")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    steps = late_window(25)
    legs = {"dit": load_leg(BASE / "SiT-B-2", EPS_GRID, with_nfe=False),
            "unet": load_leg(BASE / "UNet-B", EPS_GRID, with_nfe=False),
            "dit95": load_leg(BASE / "SiT-B-2-FID95", [PRIMARY_EPS],
                              with_nfe=False)}
    curves = {"dose_response": {}, "temporal": {}, "localization": {}}
    dose_response(legs, steps, curves)
    temporal(legs, curves)
    dose_temporal(legs, steps, curves)
    localization(legs, steps, curves)
    detector_fig(curves)
    energy_fig(height=float(sys.argv[1]) if len(sys.argv) > 1 else 6.2)
    with open(OUT / "curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
