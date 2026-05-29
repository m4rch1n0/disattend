#!/usr/bin/env python3
"""Measure how well a latent-space horizontal flip approximates an
image-space horizontal flip through the SD VAE.

We trained with flip applied to the latent (torch.flip(z, dims=[-1])),
not the source image. The VAE is only approximately flip-equivariant
(its convolutional kernels are learned, not symmetric), so half of our
training data is "latent-flipped" tensors that don't correspond to
encoding any real flipped image. This script measures the discrepancy.

For each of N val images x:
  imgA = decode(flip(encode(x)))                  # what latent-flip training sees
  imgB = decode(encode(flip(x)))                  # what image-flip training would see
  target = flip(x)                                # the actual flipped image
  recon_noflip = decode(encode(x))                # VAE round-trip baseline

We report PSNR for the comparisons that matter:
  - PSNR(x, recon_noflip)     -- baseline VAE quality (should match ~24.8 dB)
  - PSNR(target, imgB)        -- baseline VAE quality on flipped input
  - PSNR(target, imgA)        -- how well latent-flip approximates flip(image)
  - PSNR(imgA, imgB)          -- how close the two training distributions are

If imgA and imgB are close (PSNR > the VAE baseline), latent-flip is
training-equivalent to image-flip modulo VAE noise. If meaningfully
worse, our flip is introducing real error.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HOME_LOCAL = REPO_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_HOME_LOCAL))
if "HF_TOKEN" not in os.environ:
    for _c in (HF_HOME_LOCAL / "token",
               Path.home() / ".cache" / "huggingface" / "token"):
        if _c.exists():
            os.environ["HF_TOKEN"] = _c.read_text().strip()
            break

import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import make_grid, save_image


DATASET_REPO = "ILSVRC/imagenet-1k"
VAE_REPO = "stabilityai/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
IMG_SIZE = 256
ENCODE_BATCH = 16
N_SAMPLES = 500
N_GRID = 8

OUT_PNG = REPO_ROOT / "notebooks" / "out" / "latent_flip_grid.png"
OUT_JSON = REPO_ROOT / "notebooks" / "out" / "latent_flip_metrics.json"


def get_image_bytes(field):
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return None


def psnr_minus1to1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR for tensors in [-1, 1] (peak-to-peak = 2)."""
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))


@torch.inference_mode()
def main() -> int:
    device = torch.device("cuda")
    dtype = torch.float16
    print(f"device={device} dtype={dtype}")

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    local = hf_hub_download(DATASET_REPO, "data/validation-00000-of-00014.parquet",
                            repo_type="dataset")
    table = pq.read_table(local, columns=["image"])
    images_col = table.column("image")
    n_avail = len(table)
    n = min(N_SAMPLES, n_avail)
    print(f"using {n} val images")

    psnr_recon_noflip: list[float] = []  # baseline
    psnr_target_vs_B: list[float] = []   # baseline on flipped
    psnr_target_vs_A: list[float] = []   # how good latent-flip approximates
    psnr_A_vs_B: list[float] = []        # closeness of training inputs

    grid_orig: list[torch.Tensor] = []
    grid_target: list[torch.Tensor] = []
    grid_B: list[torch.Tensor] = []
    grid_A: list[torch.Tensor] = []

    t0 = time.perf_counter()
    for i in range(0, n, ENCODE_BATCH):
        batch = []
        for j in range(i, min(i + ENCODE_BATCH, n)):
            ib = get_image_bytes(images_col[j].as_py())
            if ib is None:
                continue
            img = Image.open(io.BytesIO(ib)).convert("RGB")
            batch.append(transform(img))
        if not batch:
            continue
        x = torch.stack(batch).to(device, dtype=dtype)
        x_flip = torch.flip(x, dims=[-1])

        # Encode both
        z_x = vae.encode(x).latent_dist.sample() * SCALING_FACTOR
        z_xflip = vae.encode(x_flip).latent_dist.sample() * SCALING_FACTOR
        # Latent-flip of z_x
        z_x_flipped = torch.flip(z_x, dims=[-1])

        # Decode
        recon_noflip = vae.decode(z_x / SCALING_FACTOR).sample
        imgA = vae.decode(z_x_flipped / SCALING_FACTOR).sample  # latent-flip decoded
        imgB = vae.decode(z_xflip / SCALING_FACTOR).sample      # image-flip decoded

        # PSNR (all in [-1, 1])
        psnr_recon_noflip.extend(psnr_minus1to1(x.float(), recon_noflip.float()).cpu().tolist())
        psnr_target_vs_B.extend(psnr_minus1to1(x_flip.float(), imgB.float()).cpu().tolist())
        psnr_target_vs_A.extend(psnr_minus1to1(x_flip.float(), imgA.float()).cpu().tolist())
        psnr_A_vs_B.extend(psnr_minus1to1(imgA.float(), imgB.float()).cpu().tolist())

        if len(grid_orig) * ENCODE_BATCH < N_GRID:
            need = N_GRID - len(grid_orig) * ENCODE_BATCH
            take = min(need, x.shape[0])
            grid_orig.append(x[:take].float().cpu())
            grid_target.append(x_flip[:take].float().cpu())
            grid_B.append(imgB[:take].float().cpu())
            grid_A.append(imgA[:take].float().cpu())

    elapsed = time.perf_counter() - t0
    n_actual = len(psnr_recon_noflip)

    def stats(name, values):
        t = torch.tensor(values)
        return {"mean": t.mean().item(), "median": t.median().item(),
                "min": t.min().item(), "max": t.max().item()}

    metrics = {
        "n_samples": n_actual,
        "elapsed_s": round(elapsed, 1),
        "psnr_x_vs_recon_noflip": stats("baseline VAE quality", psnr_recon_noflip),
        "psnr_target_vs_B": stats("baseline on flipped (image-flip path)", psnr_target_vs_B),
        "psnr_target_vs_A": stats("latent-flip approximates flipped image", psnr_target_vs_A),
        "psnr_A_vs_B": stats("latent-flip vs image-flip (training input gap)", psnr_A_vs_B),
        "vae_repo": VAE_REPO,
    }

    print(f"\n=== latent-flip equivariance, n={n_actual}, elapsed={elapsed:.1f}s ===")
    for key in ("psnr_x_vs_recon_noflip", "psnr_target_vs_B",
                "psnr_target_vs_A", "psnr_A_vs_B"):
        s = metrics[key]
        print(f"  {key:30s}: mean {s['mean']:6.2f} dB | median {s['median']:6.2f} dB "
              f"| range [{s['min']:.2f}, {s['max']:.2f}]")

    # Interpretation hint
    base = metrics["psnr_x_vs_recon_noflip"]["mean"]
    AvsB = metrics["psnr_A_vs_B"]["mean"]
    print(f"\n  baseline (VAE round-trip)           : {base:.2f} dB")
    print(f"  latent-flip vs image-flip outputs   : {AvsB:.2f} dB")
    delta = base - AvsB
    print(f"  delta                                : {delta:+.2f} dB  "
          f"({'similar to round-trip noise' if delta < 3 else 'meaningful extra error'})")

    # Grid: [orig | target=flip(img) | imgB=img-flip path | imgA=latent-flip path]
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    o = torch.cat(grid_orig, dim=0)[:N_GRID]
    t = torch.cat(grid_target, dim=0)[:N_GRID]
    B = torch.cat(grid_B, dim=0)[:N_GRID]
    A = torch.cat(grid_A, dim=0)[:N_GRID]
    rows = []
    for tensor in (o, t, B, A):
        rows.append(((tensor + 1) / 2).clamp(0, 1))
    full = torch.cat(rows, dim=0)
    grid = make_grid(full, nrow=N_GRID, padding=2)
    save_image(grid, str(OUT_PNG))
    print(f"\nsaved grid (rows: orig | target=flip | imgB image-flip | imgA latent-flip) "
          f"-> {OUT_PNG}")

    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {OUT_JSON}")

    # cleanup
    blob = os.path.realpath(local)
    for p in (local, blob):
        try:
            if os.path.exists(p) or os.path.islink(p):
                os.remove(p)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
