#!/usr/bin/env python3
"""F5 qualitative: decoded images + raw attention maps, benign vs PGD.

Representative seed = median of the mean paired rank of the per-seed late
erank contrast across the two models (a middle-of-the-road seed for both) at
the primary eps. Images are decoded from the persisted z0; attention maps are
re-captured with a single hooked forward per branch per model (the substrate
stores reduced metrics only). Shown map: head-mean attention at the last ODE
step (t = 1), the layer with the largest late contrast per model
(DiT blocks.9, UNet output_blocks.7.1); benign and PGD share the color scale.

Writes figures/f5_qualitative.(png|pdf).
"""

from __future__ import annotations

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

from analyze_phase3 import PRIMARY_EPS, erank_contrast, late_window, load_leg
from src.attacks.pgd_latent import euler_step, load_model
from src.evaluation.attention_metrics import effective_rank_per_layer
from src.evaluation.fid import VAE_REPO
from src.utils.attention_hooks import AttentionCollector
from scripts.run_phase2_pilot import decode_images
from scripts.run_phase3_grid import CKPTS, MODEL_KEY

BASE = REPO_ROOT / "experiments/phase3_main"
SHOW = {"dit": ("SiT-B/2", "SiT-B-2", "blocks.9.attn"),
        "unet": ("UNet-B", "UNet-B", "output_blocks.7.1")}
C = {"dit": "#0072B2", "unet": "#D55E00"}
LBL = {"dit": "DiT-B/2", "unet": "UNet-B"}


@torch.inference_mode()
def attn_at_last_step(model, z, y, layer, device, n_steps=25):
    col = AttentionCollector(model, store_dtype=torch.float32, store_device="cpu")
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    x = z
    with col:
        for i in range(n_steps):
            x = euler_step(model, x, ts[i], ts[i + 1] - ts[i], y)
        snap = {layer: col.snapshot()[layer]}
    er = effective_rank_per_layer(snap)[layer]["erank_rv"].mean().item()
    return snap[layer][0].mean(0).numpy(), er  # head-mean (N, N), erank


def main() -> int:
    device = torch.device("cuda")
    steps = late_window(25)
    legs = {m: load_leg(BASE / slug, [PRIMARY_EPS], with_nfe=False)
            for m, (_, slug, _) in SHOW.items()}

    # representative seed: middle third of the mean paired rank of the late
    # contrast, constrained to show a near-typical drop (>= 60% of the cell
    # mean) on BOTH displayed cells (layer, last step) -- a single cell of a
    # median seed can be silent by chance while the aggregate is not.
    ranks, cell_drop, cell_mean = [], {}, {}
    for m, (_, _, layer) in SHOW.items():
        c = erank_contrast(legs[m], PRIMARY_EPS, steps)
        ranks.append(stats.rankdata(c))
        ben = legs[m]["ben"]["attn"][layer]["erank_rv"][24].mean(dim=1).numpy()
        pgd = legs[m]["eps"][PRIMARY_EPS]["attn_pgd"][layer]["erank_rv"][24] \
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

    vae = None
    panels = {}
    for m, (name, slug, layer) in SHOW.items():
        d = torch.load(BASE / slug / f"eps_{PRIMARY_EPS}.pt", map_location="cpu",
                       weights_only=False)
        ben = torch.load(BASE / slug / "benign.pt", map_location="cpu",
                         weights_only=False)
        model = load_model(MODEL_KEY[name], REPO_ROOT / CKPTS[name], device)
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(
                VAE_REPO, torch_dtype=torch.float16).to(device).eval()
        z_T = ben["z_T"][seed:seed + 1].to(device)
        z_adv = d["z_T_adv"][seed:seed + 1].to(device)
        y = ben["y"][seed:seed + 1].to(device)
        img_b = decode_images(vae, ben["z0"][seed:seed + 1].to(device))[0]
        img_a = decode_images(vae, d["z0_pgd"][seed:seed + 1].to(device))[0]
        map_b, er_b = attn_at_last_step(model, z_T, y, layer, device)
        map_a, er_a = attn_at_last_step(model, z_adv, y, layer, device)
        panels[m] = {"img_b": img_b, "img_a": img_a, "map_b": map_b,
                     "map_a": map_a, "er_b": er_b, "er_a": er_a,
                     "lpips": float(d["lpips"]["pgd"][seed]), "layer": layer}
        del model
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.4))
    for r, m in enumerate(("dit", "unet")):
        p = panels[m]
        for c_i, (key, title) in enumerate((("img_b", "benign sample"),
                                            ("img_a", "PGD sample"))):
            ax = axes[r, c_i]
            ax.imshow(((p[key].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1))
            ax.set_title(title if r == 0 else "", fontsize=9.5)
            ax.axis("off")
        vmax = np.percentile(np.stack([p["map_b"], p["map_a"]]), 99.5)
        for c_i, (key, er, title) in enumerate((
                ("map_b", p["er_b"], "attention, benign"),
                ("map_a", p["er_a"], "attention, PGD")), start=2):
            ax = axes[r, c_i]
            ax.imshow(p[key], cmap="Blues", vmin=0, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(title if r == 0 else "", fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"erank/N = {er:.3f}", fontsize=8.5)
        axes[r, 0].text(-0.12, 0.5, LBL[m], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=11,
                        fontweight="bold", color=C[m])
        axes[r, 2].text(0.02, 0.02, p["layer"], transform=axes[r, 2].transAxes,
                        fontsize=7.5, color="#4d4d4d", va="bottom")
    fig.suptitle(f"Paired seed {seed} (median-rank band), ε = {PRIMARY_EPS}, last ODE step; "
                 f"attention = head-mean, shared color scale per model  "
                 f"(LPIPS: DiT {panels['dit']['lpips']:.2f}, "
                 f"UNet {panels['unet']['lpips']:.2f})", fontsize=9, y=0.99)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(BASE / "figures" / f"f5_qualitative.{ext}",
                    bbox_inches="tight")
    print(f"seed {seed}: DiT erank/N {panels['dit']['er_b']:.3f}->{panels['dit']['er_a']:.3f}, "
          f"UNet {panels['unet']['er_b']:.3f}->{panels['unet']['er_a']:.3f}")
    print("wrote figures/f5_qualitative.png|pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
