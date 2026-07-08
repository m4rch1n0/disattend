#!/usr/bin/env python3
"""Phase 3 thesis figures and tables from the persisted substrate.

Reads experiments/phase3_main/{analysis.json,detector.json} plus the raw
substrate (via the frozen analyzer's loaders; nothing recomputed differs from
analysis.json). Writes:
  figures/f1_dose_response.(png|pdf)   contrast vs eps, both models, CI95
  figures/f2_localization.(png|pdf)    per-layer profiles DiT / DiT@FID95 / UNet
  figures/f3_temporal.(png|pdf)        per-step contrast, late window shaded
  figures/f4_detector.(png|pdf)        ROC @ primary eps + AUC vs eps
  figures/f6_convergence.(png|pdf)     PGD loss curves + contrast vs iters
  tables/t1_confirmatory.md            co-primaries + family, TOST, verdicts
  tables/t2_metric_vs_eps.md           three metrics x eps x model, CI95
  tables/t3_controls.md                fid95 / nfe / attack gate / saturation / iters
  tables/t5_detector.md                AUC + operating points + transfer

Palette (CVD-checked): DiT #0072B2, UNet #D55E00, DiT@FID95 #56B4E9 (same hue
family as DiT, lightness = convergence), reference gray #7F7F7F.
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
from scipy import stats

from analyze_phase3 import (DIT_ALL, DIT_IN, EPS_GRID, PRIMARY_EPS, UNET_ALL,
                            UNET_IN, agg_scalar, entropy_echo_contrast,
                            erank_contrast, late_window, load_leg,
                            per_layer_contrast, share_profile)

BASE = REPO_ROOT / "experiments/phase3_main"
FIG = BASE / "figures"
TAB = BASE / "tables"
C = {"dit": "#0072B2", "unet": "#D55E00", "dit95": "#56B4E9", "ref": "#7F7F7F"}
LBL = {"dit": "DiT-B/2", "unet": "UNet-B", "dit95": "DiT-B/2 @ FID95"}

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
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig {name}")


# ---------- F1 dose-response ----------


def f1(legs, steps):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for m in ("dit", "unet"):
        mu = [erank_contrast(legs[m], e, steps).mean() for e in EPS_GRID]
        err = [ci95(erank_contrast(legs[m], e, steps)) for e in EPS_GRID]
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
    ax.set_ylabel("erank contrast (rand − PGD)")
    ax.grid(axis="x", visible=False)
    save(fig, "f1_dose_response")


# ---------- F2 localization profiles ----------


def f2(legs, steps, analysis):
    profs, cis, keys, s_note = {}, {}, {}, {}
    for m, layers, inp in (("dit", DIT_ALL, DIT_IN), ("dit95", DIT_ALL, DIT_IN),
                           ("unet", UNET_ALL, UNET_IN)):
        d = legs[m]["eps"][PRIMARY_EPS]
        pl = per_layer_contrast(d["attn_rand"], d["attn_pgd"], steps, layers)
        profs[m] = [pl[n].mean() for n in layers]
        cis[m] = [ci95(pl[n]) for n in layers]
        keys[m] = layers
        s_note[m] = f"S = {share_profile(pl, inp, layers):+.2f}"
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
            ax.set_xticklabels(["in.4", "in.5", "out.6", "out.7", "out.8"],
                               fontsize=8)
            side_lbl = ("encoder", "decoder")
        else:
            ax.set_xticks(x[::2])
            ax.set_xticklabels([str(i) for i in range(0, n_l, 2)], fontsize=8)
            ax.set_xlabel("block")
            side_lbl = ("input side (0–5)", "deep side (6–11)")
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
    axes[0].set_ylabel("erank contrast per layer\n(rand − PGD, late window)")
    fig.suptitle("Where the fingerprint lives: hatched = input side  "
                 "(S = excess input-side share of clamped contrast mass)",
                 fontsize=9, y=1.04, color="#4d4d4d")
    save(fig, "f2_localization")


# ---------- F3 temporal localization ----------


def f3(legs):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
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
        ax.plot(t_axis, mu, color=C[m], lw=2)
        ax.fill_between(t_axis, mu - err, mu + err, color=C[m], alpha=0.18, lw=0)
        ax.annotate(LBL[m], (t_axis[-1], mu[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C[m], fontsize=9,
                    va="center", fontweight="bold")
    ax.axvspan(0.72, 1.0, color="#000000", alpha=0.05, lw=0)
    ax.text(0.86, ax.get_ylim()[1] * 0.06, "late window\n(t ≥ 0.72)",
            ha="center", fontsize=8, color="#4d4d4d")
    ax.set_xlabel("ODE time t (post-step, noise → data)")
    ax.set_ylabel("erank contrast (rand − PGD)")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", visible=False)
    save(fig, "f3_temporal")


# ---------- F4 detector ----------


def f4(det):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax1.plot([0, 1], [0, 1], color="#BBBBBB", lw=1, ls=(0, (4, 3)))
    for m in ("dit", "unet", "dit95"):
        r = det["roc"][m]
        auc = det["auc"][f"{m}:pgd@{PRIMARY_EPS}"]["auc"]
        ax1.plot(r["fpr"], r["tpr"], color=C[m], lw=2,
                 label=f"{LBL[m]}  AUC {auc:.2f}")
    ax1.set_xlabel("false positive rate (benign flagged)")
    ax1.set_ylabel("true positive rate (PGD flagged)")
    ax1.set_title(f"ROC, benign vs PGD @ ε = {PRIMARY_EPS}", fontsize=10)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)

    for m in ("dit", "unet"):
        mu = [det["auc"][f"{m}:pgd@{e}"]["auc"] for e in EPS_GRID]
        lo = [mu[i] - det["auc"][f"{m}:pgd@{e}"]["ci95"][0] for i, e in enumerate(EPS_GRID)]
        hi = [det["auc"][f"{m}:pgd@{e}"]["ci95"][1] - mu[i] for i, e in enumerate(EPS_GRID)]
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
    ax2.text(0.011, 0.44, "dashed: Rademacher control", fontsize=8, color="#4d4d4d")
    ax2.set_xscale("log")
    ax2.set_xticks(EPS_GRID); ax2.set_xticklabels([str(e) for e in EPS_GRID])
    ax2.set_xlim(0.0085, 0.16); ax2.set_ylim(0.4, 1.02)
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel("AUC (flag when erank low)")
    ax2.set_title("Detector skill vs attack budget", fontsize=10)
    ax2.grid(axis="x", visible=False)
    save(fig, "f4_detector")


# ---------- F6 attack convergence ----------


def f6(legs, steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for m in ("dit", "unet"):
        curves = np.array(legs[m]["eps"][PRIMARY_EPS]["pgd_loss_curves"])
        mu, sd = curves.mean(0), curves.std(0)
        it = np.arange(curves.shape[1])
        ax1.plot(it, mu, color=C[m], lw=2)
        ax1.fill_between(it, mu - sd, mu + sd, color=C[m], alpha=0.15, lw=0)
        ax1.annotate(LBL[m], (it[-1], mu[-1]), xytext=(6, 7 if m == "dit" else -9),
                     textcoords="offset points", color=C[m], fontsize=9,
                     va="center", fontweight="bold")
    ax1.set_xlabel("PGD iteration")
    ax1.set_ylabel("attack objective (batch mean ± SD)")
    ax1.set_title(f"Attack convergence @ ε = {PRIMARY_EPS} (40 iters)", fontsize=10)
    ax1.set_xlim(0, 52)

    iters = [20, 40, 80]
    for m, slug in (("dit", "SiT-B-2"), ("unet", "UNet-B")):
        cs = []
        for it in iters:
            leg = legs[m] if it == 40 else load_leg(BASE / f"{slug}.it{it}",
                                                    [PRIMARY_EPS], with_nfe=False)
            cs.append(erank_contrast(leg, PRIMARY_EPS, steps).mean())
        ax2.plot(iters, cs, color=C[m], lw=2, marker="o", ms=5)
        ax2.fill_between([15, 90], cs[1] * 0.9, cs[1] * 1.1, color=C[m],
                         alpha=0.08, lw=0)
        ax2.annotate(LBL[m], (iters[-1], cs[-1]), xytext=(6, 0),
                     textcoords="offset points", color=C[m], fontsize=9,
                     va="center", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(iters); ax2.set_xticklabels([str(i) for i in iters])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlim(17, 108)
    ax2.set_xlabel("PGD iterations")
    ax2.set_ylabel("erank contrast (late window)")
    ax2.set_title("Plateau check (band = ±10% of the 40-iter value)", fontsize=10)
    ax2.grid(axis="x", visible=False)
    save(fig, "f6_convergence")


# ---------- tables ----------


def fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def t1(analysis):
    a, b, fam = analysis["coprimary_A"], analysis["coprimary_B"], analysis["family"]
    v = analysis["verdicts"]
    rows = [
        "# T1 - confirmatory family (Holm alpha=0.05) and verdicts",
        "",
        "| member | estimate | CI95 (boot) | p (max t,W) | p Holm | SESOI | TOST 90% CI | verdict |",
        "|---|---|---|---|---|---|---|---|",
        f"| **A: magnitude** (erank contrast Δ, DiT−UNet) | {a['delta']['mean']:+.4f} | "
        f"{fmt_ci(a['delta']['ci95_boot'])} | {a['delta']['p']:.2e} | "
        f"{fam['A_magnitude']['holm']['p_holm']:.2e} | 0.010 | "
        f"{fmt_ci(a['tost']['ci90_t'])} (in) | **{v['A']}** |",
        f"| **B: structure** (excess input-share D) | {b['D']['mean']:+.4f} "
        f"(profile {b['D_profile']:+.4f}) | {fmt_ci(b['D']['ci95_boot'])} | "
        f"{b['D']['p']:.2e} | {fam['B_structure']['holm']['p_holm']:.2e} | 0.10 (both) | "
        f"{fmt_ci(b['tost']['ci90_t'])} (out) | **{v['B']}** |",
        f"| dose-response (Δ slope on log ε) | {fam['dose_response']['mean']:+.4f} | "
        f"{fmt_ci(fam['dose_response']['ci95_boot'])} | {fam['dose_response']['p']:.2e} | "
        f"{fam['dose_response']['holm']['p_holm']:.2e} | - | - | class ≤ 2 by rule |",
        f"| entropy echo of A | {fam['entropy_echo']['mean']:+.4f} | "
        f"{fmt_ci(fam['entropy_echo']['ci95_boot'])} | {fam['entropy_echo']['p']:.2e} | "
        f"{fam['entropy_echo']['holm']['p_holm']:.2e} | - | - | echo |",
        f"| flatness echo of A | {fam['flatness_echo']['mean']:+.4f} | "
        f"{fmt_ci(fam['flatness_echo']['ci95_boot'])} | {fam['flatness_echo']['p']:.2e} | "
        f"{fam['flatness_echo']['holm']['p_holm']:.2e} | - | - | echo |",
        "",
        f"Overall (PREREG rule): architectures distinguishable = "
        f"**{analysis['verdicts']['overall_distinguishable']}**. "
        f"n = {analysis['n_seeds']} paired seeds; within-model contrasts at "
        f"ε = {PRIMARY_EPS}: DiT {a['contrast_dit']['mean']:+.4f} "
        f"(d = {a['contrast_dit']['d_within']:.2f}), "
        f"UNet {a['contrast_unet']['mean']:+.4f} "
        f"(d = {a['contrast_unet']['d_within']:.2f}); pair corr "
        f"{a['pair_corr']:+.2f}. B exclusions: {b['exclusions']}.",
    ]
    (TAB / "t1_confirmatory.md").write_text("\n".join(rows) + "\n")
    print("  tab t1_confirmatory")


def t2(legs, steps):
    rows = ["# T2 - contrast vs eps, three metrics (mean [CI95], n=500)",
            "",
            "| model | ε | erank_rv | entropy (echo form) | flatness_ratio |",
            "|---|---|---|---|---|"]
    for m in ("dit", "unet"):
        for e in EPS_GRID:
            er = erank_contrast(legs[m], e, steps)
            en = entropy_echo_contrast(legs[m], e, steps)
            fl = erank_contrast(legs[m], e, steps, metric="flatness_ratio")
            cells = []
            for x in (er, en, fl):
                cells.append(f"{x.mean():+.4f} ± {ci95(x):.4f}")
            rows.append(f"| {LBL[m]} | {e} | " + " | ".join(cells) + " |")
    (TAB / "t2_metric_vs_eps.md").write_text("\n".join(rows) + "\n")
    print("  tab t2_metric_vs_eps")


def t3(analysis):
    c = analysis["controls"]
    f95, nfe, atk = c["fid95"], c["nfe_transfer"], c["attack"]
    sat, conv = c["saturation"], c["convergence"]
    rows = ["# T3 - registered controls", "",
            "## Matched-quality (DiT@FID95, batch-64 regime caveat)",
            "| co-primary | Δ(95 − 68) | TOST 90% CI | equivalent | sign vs UNet matches | survives |",
            "|---|---|---|---|---|---|"]
    for k in ("A", "B"):
        key = "delta_95_vs_68" if k == "A" else "D_95_vs_68"
        rows.append(f"| {k} | {f95[k][key]['mean']:+.4f} | "
                    f"{fmt_ci(f95[k]['tost']['ci90_t'])} | {f95[k]['tost']['equivalent']} | "
                    f"{f95[k]['sign_matches_primary']} | **{f95[k]['survives']}** |")
    rows += ["", "## NFE transfer (late window, ε = 0.05)",
             "| NFE | member | estimate | ratio to primary | Holm reject | survives |",
             "|---|---|---|---|---|---|"]
    prim = {"A": analysis["coprimary_A"]["delta"]["mean"],
            "B": analysis["coprimary_B"]["D"]["mean"]}
    for n in ("50", "100"):
        for k in ("A", "B"):
            r = nfe[n][k]
            rows.append(f"| {n} | {k} | {r['mean']:+.4f} | "
                        f"{r['mean'] / prim[k]:.2f}x | {r['holm']['reject']} | "
                        f"**{r['survives']}** |")
    rows += ["", "## Attack comparability + saturation + iteration plateau",
             "| check | value | criterion | pass |", "|---|---|---|---|"]
    for e, v in atk["per_eps"].items():
        rows.append(f"| l2_out ratio DiT/UNet @ ε={e} | {v['l2_ratio_dit_over_unet']:.3f} | "
                    f"[0.8, 1.25] | {v['in_band']} |")
    for m in ("dit", "unet"):
        s = sat[m]
        rows.append(f"| saturation {LBL[m]} | c@0.1 / c@0.05 = "
                    f"{s['contrast@0.1'] / s['contrast@0.05']:.2f} | < 2 | {s['saturating']} |")
        cv = conv[m]
        rows.append(f"| iters plateau {LBL[m]} | c20/c40/c80 = {cv['c20']:.4f}/"
                    f"{cv['c40']:.4f}/{cv['c80']:.4f} | \\|c80−c40\\| ≤ 10% c40 | "
                    f"{cv['plateau']} |")
    (TAB / "t3_controls.md").write_text("\n".join(rows) + "\n")
    print("  tab t3_controls")


def t5(det):
    rows = ["# T5 - exploratory detector (post-hoc, outside the PREREG)",
            "",
            f"Score: late-window N=256 erank_rv; flag when below τ. "
            f"Operating points at {int(det['fpr_target'] * 100)}% FPR on own benign.",
            "",
            "| model | ε | AUC benign-vs-PGD [CI95] | AUC benign-vs-rand | TPR @ 5% FPR |",
            "|---|---|---|---|---|"]
    for m in ("dit", "unet", "dit95"):
        eps_list = EPS_GRID if m != "dit95" else [PRIMARY_EPS]
        for e in eps_list:
            a = det["auc"][f"{m}:pgd@{e}"]
            r = det["auc"][f"{m}:rand@{e}"]["auc"]
            tpr = det["operating_points"][m][f"tpr:pgd@{e}"]
            rows.append(f"| {LBL[m]} | {e} | {a['auc']:.3f} "
                        f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {r:.3f} | {tpr:.2f} |")
    rows += ["", "## Cross-model transfer of the benign-z threshold (fit at 5% FPR)",
             "| fit → deploy | FPR on deploy | TPR @ ε=0.05 | TPR @ ε=0.1 |",
             "|---|---|---|---|"]
    for k, v in det["transfer"].items():
        t10 = f"{v.get('tpr:pgd@0.1', float('nan')):.2f}" if "tpr:pgd@0.1" in v else "-"
        rows.append(f"| {k} | {v['fpr_on_dst']:.3f} | {v['tpr:pgd@0.05']:.2f} | {t10} |")
    (TAB / "t5_detector.md").write_text("\n".join(rows) + "\n")
    print("  tab t5_detector")


def main() -> int:
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    steps = late_window(25)
    analysis = json.load(open(BASE / "analysis.json"))
    det = json.load(open(BASE / "detector.json"))
    legs = {"dit": load_leg(BASE / "SiT-B-2", EPS_GRID, with_nfe=False),
            "unet": load_leg(BASE / "UNet-B", EPS_GRID, with_nfe=False),
            "dit95": load_leg(BASE / "SiT-B-2-FID95", [PRIMARY_EPS], with_nfe=False)}
    f1(legs, steps)
    f2(legs, steps, analysis)
    f3(legs)
    f4(det)
    f6(legs, steps)
    t1(analysis)
    t2(legs, steps)
    t3(analysis)
    t5(det)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
