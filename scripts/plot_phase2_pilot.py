#!/usr/bin/env python3
"""Render the Phase 2 pilot figures to notebooks/out/ (matplotlib Agg).

Standalone version of the notebook plots, so the figures exist as files
without a Jupyter kernel. Reads analysis.json + the ab_results.pt substrates.

Usage: uv run python scripts/plot_phase2_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
for _p in (str(REPO_ROOT), str(SIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SLUGS = {"SiT-B/2": "SiT-B-2", "UNet-B": "UNet-B"}
METRICS = ["entropy", "flatness_ratio", "erank_rv"]
PILOT = REPO_ROOT / "experiments/phase2_pilot"
OUT = REPO_ROOT / "notebooks/out"


def fig_effect_sizes(analysis: dict) -> None:
    models = list(analysis)
    x = np.arange(len(METRICS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(models):
        d = [analysis[m]["ab_n256"][k]["gate"]["cohens_d"] for k in METRICS]
        bars = ax.bar(x + (i - 0.5) * w, d, w, label=m)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(0.8, color="gray", ls=":", lw=0.8)
    ax.text(len(METRICS) - 0.5, 0.82, "d=0.8 (large)", fontsize=7,
            color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["entropy", "flatness ratio", "eff. rank (RV)"])
    ax.set_ylabel("Cohen's d  (PGD vs random contrast)")
    ax.set_title("Attention-metric effect size on the N=256 locus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase2_effect_sizes.png", dpi=140)
    plt.close(fig)


def per_step(attn: dict, branch: str, metric: str, token_filter: int = 256):
    st = attn[branch]
    names = [n for n in st if int(st[n]["n_tokens"]) == token_filter]
    return torch.stack([st[n][metric].mean(dim=(1, 2)) for n in names]).mean(0).numpy()


def fig_timestep(models: list[str]) -> None:
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.4 * len(models)),
                             squeeze=False)
    for i, m in enumerate(models):
        attn = torch.load(PILOT / SLUGS[m] / "ab_results.pt",
                          weights_only=False)["attn"]
        for j, metric in enumerate(METRICS):
            ax = axes[i][j]
            ax.plot(per_step(attn, "ben", metric), label="benign", lw=2)
            ax.plot(per_step(attn, "pgd", metric), label="PGD", lw=2, ls="--")
            ax.plot(per_step(attn, "rand0", metric), label="random", lw=1,
                    alpha=0.6)
            ax.set_title(f"{m} -- {metric}")
            ax.set_xlabel("Euler step (t=0 -> 1)")
            if j == 0:
                ax.set_ylabel("metric (N=256, mean over L,H)")
            if i == 0 and j == 2:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_timestep_profiles.png", dpi=140)
    plt.close(fig)


def fig_qualitative(models: list[str]) -> None:
    from diffusers import AutoencoderKL
    from src.evaluation.fid import VAE_REPO, SCALING_FACTOR
    device = torch.device("cuda")
    vae = AutoencoderKL.from_pretrained(
        VAE_REPO, torch_dtype=torch.float16).to(device).eval()

    def dec(z):
        with torch.inference_mode():
            img = vae.decode(z.to(device).to(torch.float16) / SCALING_FACTOR).sample
        return ((img.float() + 1) / 2).clamp(0, 1).cpu()

    for m in models:
        d = torch.load(PILOT / SLUGS[m] / "ab_results.pt", weights_only=False)
        l2 = d["l2_out"]["pgd"].numpy()
        order = np.argsort(l2)
        picks = {"worst": order[-1], "median": order[len(order) // 2],
                 "best": order[0]}
        fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
        for c, (label, idx) in enumerate(picks.items()):
            axes[0][c].imshow(dec(d["z0"]["ben"][idx:idx + 1])[0].permute(1, 2, 0))
            axes[0][c].set_title(f"benign ({label})")
            axes[1][c].imshow(dec(d["z0"]["pgd"][idx:idx + 1])[0].permute(1, 2, 0))
            axes[1][c].set_title(f"PGD  L2={l2[idx]:.1f}")
        for a in axes.flat:
            a.axis("off")
        fig.suptitle(f"{m}: benign (top) vs PGD (bottom), by output L2")
        fig.tight_layout()
        fig.savefig(OUT / f"phase2_qualitative_{SLUGS[m]}.png", dpi=130)
        plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = json.load(open(PILOT / "analysis.json"))
    models = list(analysis)
    fig_effect_sizes(analysis)
    fig_timestep(models)
    fig_qualitative(models)
    print(f"wrote figures to {OUT}:")
    for p in sorted(OUT.glob("phase2_*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
