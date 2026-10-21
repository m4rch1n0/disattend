#!/usr/bin/env python3
"""Ch. 4 thesis figure: attack + measurement pipeline diagram.

Pure diagram, no artifact input: the content restates the frozen design of
experiments/phase2_pilot/PREREG.md and experiments/phase3_main/PREREG.md
(triplet branches, PGD loop, hooks, immediate spectral reduction, late
window). Writes paper/figures/fig_pipeline.pdf (and .png).

Deliberately model-agnostic (no DiT/UNet colors): the pipeline is identical
for both backbones, and the caption says so. Neutral grays, one dark-red
accent on the PGD branch and its feedback loop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "paper/figures/fig_pipeline.pdf"
OUT_PNG = OUT.with_suffix(".png")

GRAY_EDGE = "#666666"
GRAY_FILL = "#f5f5f5"
TEXT = "#1a1a1a"
ACCENT = "#A02020"          # PGD branch + feedback loop
MUTED = "#4d4d4d"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 8.0,
    "text.color": TEXT,
})


def box(ax, x, y, w, h, text, ec=GRAY_EDGE, fc=GRAY_FILL, lw=1.0, ls="-",
        fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        fc=fc, ec=ec, lw=lw, linestyle=ls, mutation_aspect=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=TEXT, linespacing=1.35)


def arrow(ax, p0, p1, color=MUTED, lw=1.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 68)
    ax.axis("off")

    # ---- top lane: generation ------------------------------------------
    box(ax, 2, 45, 24, 9.5,
        "attacked\n$z_T + \\delta_{\\mathrm{PGD}}$", ec=ACCENT, lw=1.4)
    box(ax, 2, 33, 24, 9.5,
        "random (Rademacher)\n$z_T + \\delta_{\\mathrm{rand}}$", fs=7.6)
    box(ax, 2, 21, 24, 9.5, "benign\n$z_T$")
    ax.text(12, 18.2, "same seed, same class:\nthree branches per seed $i$",
            ha="center", va="top", fontsize=6.8, color=MUTED, style="italic")

    box(ax, 34, 30.5, 24, 14,
        "Euler sampler\n25 steps, deterministic\nfp32", fs=7.8)
    box(ax, 64, 33.5, 12, 9, "output\n$z_0^{(b)}$")
    box(ax, 82, 33.5, 19, 9,
        "VAE decoding:\nLPIPS, $L_2$", ls=(0, (4, 3)), fs=7.6)
    ax.text(91.5, 31.2, "effectiveness check", ha="center", va="top",
            fontsize=6.8, color=MUTED, style="italic")

    arrow(ax, (26.6, 49.5), (33.4, 39.5), rad=-0.15)
    arrow(ax, (26.6, 37.8), (33.4, 37.8))
    arrow(ax, (26.6, 25.5), (33.4, 35.5), rad=0.15)
    arrow(ax, (58.6, 38), (63.4, 38))
    arrow(ax, (76.6, 38), (81.4, 38))

    # PGD feedback loop over the top: dashed = backward pass, and the arc's
    # apex sits above the sampler on purpose (the gradient re-traverses it)
    arrow(ax, (70, 43.2), (27, 52.5), color=ACCENT, lw=1.4, rad=0.25,
          ls=(0, (4, 3)))
    ax.text(49, 59.4,
            "the gradient re-traverses the 25 steps backward\n"
            "and updates $\\delta_{\\mathrm{PGD}}$",
            ha="center", va="center", fontsize=6.8, color=ACCENT,
            style="italic")
    ax.text(51, 64.2,
            "PGD loop: maximize $\\|z_0^{(\\mathrm{PGD})} - z_0^{(\\mathrm{ben})}\\|_2$"
            "  (20/40 iterations, step $\\varepsilon/4$, $L_\\infty$ projection, random start)",
            ha="center", va="center", fontsize=7.4, color=ACCENT)

    # ---- bottom lane: measurement --------------------------------------
    arrow(ax, (44, 29.7), (40, 19.6))
    ax.text(45.2, 25.0, "hook on every layer $\\times$ step\n(fidelity verified)",
            ha="left", va="center", fontsize=7.0, color=MUTED)

    box(ax, 25, 5, 24, 13.5,
        "attention maps\nper head, $N \\times N$\nlocus $N = 256$", fs=7.6)
    box(ax, 55, 5, 24.5, 13.5,
        "immediate spectral\nreduction, on CPU:\nentropy, flatness, erank", fs=7.6)
    box(ax, 86, 5, 17.5, 13.5,
        "$M_i^{(b)}$\nlate window\n$t \\geq 0.72$", fs=7.4)

    arrow(ax, (49.8, 11.7), (54.2, 11.7))
    arrow(ax, (80.3, 11.7), (85.2, 11.7))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
