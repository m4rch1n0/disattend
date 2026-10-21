#!/usr/bin/env python3
"""Chapter 5 qualitative figure with all three branches (benign/random/PGD).

Re-render of make_phase3_qualitative.py for the paper (column-width layout, English labels)
with the random branch added, under binding guards:

  G1  For the shown seed, the regenerated z0 of EVERY branch (benign; random
      from the bit-exact reconstructed z_T_rand; PGD from the persisted delta)
      must reproduce the persisted z0 within max|diff| <= 1e-4 (declared
      before running; expected ~0 on the same fp32 GPU). Observed gaps are
      reported regardless.
  G2  Cross-check on the shown cell: the head-mean erank_rv computed on the
      recaptured last-step map must match the persisted reduced metric at
      (layer, step 24) within the same tolerance.
  If any guard fails: nothing is written and the failure is reported.

Seed and layer selection are deterministic from persisted data only: same
frozen median-rank-band rule as the original script; shown layer = largest
late-contrast layer per model from the persisted per-layer metrics.
Output: paper/figures/fig_qualitative.(png|pdf) + qualitative_guards.json.
experiments/ is never written to.
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
import torch
from scipy import stats

from analyze_phase3 import (PRIMARY_EPS, erank_contrast, late_window, load_leg,
                            per_layer_contrast)
from src.attacks.pgd_latent import euler_step, load_model
from src.evaluation.attention_metrics import effective_rank_per_layer
from src.evaluation.fid import VAE_REPO
from src.utils.attention_hooks import AttentionCollector
from scripts.run_phase2_pilot import decode_images
from scripts.run_phase3_grid import CKPTS, MODEL_KEY, make_rademacher

BASE = REPO_ROOT / "experiments/phase3_main"
OUT = REPO_ROOT / "paper/figures"
TOL = 1e-4  # declared before execution (same threshold as hook fidelity)

SHOW = {"dit": ("SiT-B/2", "SiT-B-2"), "unet": ("UNet-B", "UNet-B")}
C = {"dit": "#0072B2", "unet": "#D55E00"}
LBL = {"dit": "DiT-B/2", "unet": "UNet-B"}
BRANCHES = ("ben", "rand", "pgd")
COL_TITLES = {"ben": "benign sample", "rand": "random sample",
              "pgd": "attacked sample"}
MAP_TITLES = {"ben": "map, benign", "rand": "map, random",
              "pgd": "map, attacked"}
# display names for the shown layers (paper vocabulary instead of module names)
LAYER_NAME = {"blocks.9.attn": "block 9", "output_blocks.7.1": "decoder block 7"}


@torch.inference_mode()
def sample_with_attn(model, z, y, layer, device, n_steps=25):
    """One trajectory: returns (z0, head-mean last-step map, head-mean erank)."""
    col = AttentionCollector(model, store_dtype=torch.float32,
                             store_device="cpu")
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    x = z
    with col:
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y)
        snap = {layer: col.snapshot()[layer]}
    er = effective_rank_per_layer(snap)[layer]["erank_rv"].mean().item()
    return x, snap[layer][0].mean(0).numpy(), er


def main() -> int:
    device = torch.device("cuda")
    steps = late_window(25)
    legs = {m: load_leg(BASE / slug, [PRIMARY_EPS], with_nfe=False)
            for m, (_, slug) in SHOW.items()}

    # shown layer: largest late-contrast layer per model, from persisted
    # metrics only (deterministic)
    layer_of = {}
    for m in SHOW:
        d = legs[m]["eps"][PRIMARY_EPS]
        layers = list(d["attn_pgd"].keys())
        pl = per_layer_contrast(d["attn_rand"], d["attn_pgd"], steps, layers)
        layer_of[m] = max(layers, key=lambda n: float(pl[n].mean()))
    print("shown layers:", layer_of)

    # frozen seed rule, identical to make_phase3_qualitative.py
    ranks, cell_drop, cell_mean = [], {}, {}
    for m in SHOW:
        c = erank_contrast(legs[m], PRIMARY_EPS, steps)
        ranks.append(stats.rankdata(c))
        lay = layer_of[m]
        ben = legs[m]["ben"]["attn"][lay]["erank_rv"][24].mean(dim=1).numpy()
        pgd = legs[m]["eps"][PRIMARY_EPS]["attn_pgd"][lay]["erank_rv"][24] \
            .mean(dim=1).numpy()
        cell_drop[m] = ben - pgd
        cell_mean[m] = float((ben - pgd).mean())
    mean_rank = np.mean(ranks, axis=0)
    n = len(mean_rank)
    order = np.argsort(mean_rank)
    band = set(order[n // 3: 2 * n // 3].tolist())
    ok = [i for i in band
          if all(cell_drop[m][i] >= 0.6 * cell_mean[m] for m in SHOW)]
    med = n // 2
    seed = int(min(ok, key=lambda i: abs(mean_rank[i] - med))) if ok \
        else int(order[med])
    print(f"seed {seed} (mean paired rank {mean_rank[seed]:.1f}/{n})")

    guards = {"tolerance": TOL, "seed": seed, "layers": layer_of,
              "z0": {}, "cell": {}}
    vae = None
    panels = {}
    for m, (name, slug) in SHOW.items():
        d = torch.load(BASE / slug / f"eps_{PRIMARY_EPS}.pt",
                       map_location="cpu", weights_only=False)
        ben = torch.load(BASE / slug / "benign.pt", map_location="cpu",
                         weights_only=False)
        model = load_model(MODEL_KEY[name], REPO_ROOT / CKPTS[name], device)
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                VAE_REPO, torch_dtype=torch.float16).to(device).eval()
        y = ben["y"][seed:seed + 1].to(device)
        z_T = ben["z_T"][seed:seed + 1].to(device)
        # bit-exact reconstruction of the random branch input
        delta = make_rademacher(z_T, PRIMARY_EPS, [seed], device)
        z_in = {"ben": z_T, "rand": z_T + delta,
                "pgd": d["z_T_adv"][seed:seed + 1].to(device)}
        z0_ref = {"ben": ben["z0"][seed], "rand": d["z0_rand"][seed],
                  "pgd": d["z0_pgd"][seed]}
        attn_ref = {"ben": ben["attn"], "rand": d["attn_rand"],
                    "pgd": d["attn_pgd"]}
        lay = layer_of[m]
        panels[m] = {"layer": lay, "lpips": float(d["lpips"]["pgd"][seed])}
        for b in BRANCHES:
            z0, amap, er = sample_with_attn(model, z_in[b], y, lay, device)
            g1 = float((z0[0].cpu() - z0_ref[b]).abs().max())
            er_ref = float(attn_ref[b][lay]["erank_rv"][24][seed].mean())
            g2 = abs(er - er_ref)
            guards["z0"][f"{m}:{b}"] = g1
            guards["cell"][f"{m}:{b}"] = {"recaptured": er,
                                          "persisted": er_ref, "gap": g2}
            print(f"  {m}:{b}  G1 max|dz0| = {g1:.3e}   "
                  f"G2 |d erank| = {g2:.3e}")
            panels[m][f"img_{b}"] = decode_images(
                vae, z0_ref[b][None].to(device))[0]
            panels[m][f"map_{b}"] = amap
            panels[m][f"er_{b}"] = er
        del model
        torch.cuda.empty_cache()

    failed = [k for k, v in guards["z0"].items() if v > TOL] + \
             [k for k, v in guards["cell"].items() if v["gap"] > TOL]
    guards["passed"] = not failed
    with open(OUT / "qualitative_guards.json", "w") as f:
        json.dump(guards, f, indent=2)
    if failed:
        print(f"GUARD FAILURE on {failed}: figure NOT written "
              f"(tolerance {TOL}); the figure was not written.")
        return 1

    fig, axes = plt.subplots(4, 3, figsize=(3.4, 4.9))
    ROWS = (("dit", "img"), ("dit", "map"), ("unet", "img"), ("unet", "map"))
    for r, (m, kind) in enumerate(ROWS):
        p = panels[m]
        vmax = np.percentile(np.stack([p[f"map_{b}"] for b in BRANCHES]), 99.5)
        for c_i, b in enumerate(BRANCHES):
            ax = axes[r, c_i]
            if kind == "img":
                img = p[f"img_{b}"]
                ax.imshow(((img.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1))
                ax.set_title(COL_TITLES[b] if r == 0 else "", fontsize=7)
                ax.axis("off")
            else:
                ax.imshow(p[f"map_{b}"], cmap="Blues", vmin=0, vmax=vmax,
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"erank/N = {p[f'er_{b}']:.2f}", fontsize=6.5, labelpad=1.5)
                if c_i == 0:
                    ax.text(0.03, 0.03, LAYER_NAME.get(p["layer"], p["layer"]), transform=ax.transAxes,
                            fontsize=5.5, color="#4d4d4d", va="bottom")
        lbl = LBL[m] + ("\n(samples)" if kind == "img" else "\n(maps)")
        axes[r, 0].text(-0.2, 0.5, lbl, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=7,
                        fontweight="bold", color=C[m])
    fig.tight_layout(h_pad=0.9, w_pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_qualitative.{ext}", bbox_inches="tight")
    print("wrote paper/figures/fig_qualitative.png|pdf and guards json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
