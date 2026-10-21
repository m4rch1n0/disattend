#!/usr/bin/env python3
"""Ch. 3 thesis figure: UNet-B weight-norm growth, constant vs cosine LR.

Reads experiments/weight_norm_comparison.json (read-only; norms measured from
the checkpoints of 20260603-UNet-B-bf16-clip, constant LR, stopped at step
5 541 200, and 20260611-UNet-B-cosine-6p4M, the final run). Writes
paper/figures/fig_weight_norm.pdf. Nothing is recomputed.

Palette matches make_phase3_figures.py: UNet #D55E00 for the final run,
reference gray #7F7F7F (dashed) for the abandoned constant-LR run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "experiments/weight_norm_comparison.json"
OUT = REPO_ROOT / "paper/figures/fig_weight_norm.pdf"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.edgecolor": "#666666",
    "axes.labelcolor": "#1a1a1a", "text.color": "#1a1a1a",
    "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
})


def main() -> None:
    rows = json.loads(SRC.read_text())
    steps_m = [r["step"] / 1e6 for r in rows]
    cosine = [(s, r["cosine_wnorm"]) for s, r in zip(steps_m, rows)
              if r["cosine_wnorm"] is not None]
    const = [(s, r["constant_wnorm"]) for s, r in zip(steps_m, rows)
             if r["constant_wnorm"] is not None]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(*zip(*const), color="#7F7F7F", lw=2, ls="--", marker="s", ms=4.5,
            label="constant LR (run stopped)")
    ax.plot(*zip(*cosine), color="#D55E00", lw=2, marker="o", ms=4.5,
            label="cosine LR (final run)")
    ax.annotate(f"{const[-1][1]:.0f}", const[-1], xytext=(0, 8),
                textcoords="offset points", ha="center", color="#5c5c5c")
    ax.annotate(f"{cosine[-1][1]:.0f}", cosine[-1], xytext=(4, 8),
                textcoords="offset points", ha="center", color="#D55E00")
    ax.set_xlabel("training step (millions)")
    ax.set_ylabel("L2 norm of the weights")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
