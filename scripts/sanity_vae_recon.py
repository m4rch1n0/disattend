#!/usr/bin/env python3
"""VAE reconstruction sanity check (Phase 1 Task 1.1, Rischio H mitigation).

Two checks:
  (A) Round-trip PSNR on N=1000 val images: orig -> VAE encode -> VAE decode
      -> compare to orig. Pass criterion: mean PSNR >= 24 dB.
  (B) Visual integrity of precomputed latents: decode 16 random rows from
      data/imagenet_latents/train_0000.pt -> save as PNG grid for eyeball.

Outputs:
  notebooks/out/vae_recon_sanity.png    # orig (top row) vs recon (bottom)
  notebooks/out/vae_decode_train0000.png # decoded precomputed latents
  notebooks/out/vae_recon_psnr.json     # numeric metrics
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_HOME_LOCAL = REPO_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_HOME_LOCAL))

if "HF_TOKEN" not in os.environ:
    for _cand in (HF_HOME_LOCAL / "token",
                  Path.home() / ".cache" / "huggingface" / "token"):
        if _cand.exists():
            os.environ["HF_TOKEN"] = _cand.read_text().strip()
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
ROUND_TRIP_BATCH = 16
N_SAMPLES = 1000
N_GRID = 16
PASS_THRESHOLD_DB = 24.0

OUT_PNG_RECON = REPO_ROOT / "notebooks" / "out" / "vae_recon_sanity.png"
OUT_PNG_LATENTS = REPO_ROOT / "notebooks" / "out" / "vae_decode_train0000.png"
OUT_JSON = REPO_ROOT / "notebooks" / "out" / "vae_recon_psnr.json"
PRECOMPUTED_SHARD = REPO_ROOT / "data" / "imagenet_latents" / "train_0000.pt"


def get_image_bytes(field) -> bytes | None:
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return None


@torch.inference_mode()
def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"device={device} dtype={dtype}")

    print("loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    # (A) Round-trip PSNR on val shard 0
    print("downloading val shard 0 (will be cleaned up after)...")
    val_remote = "data/validation-00000-of-00014.parquet"
    local = hf_hub_download(DATASET_REPO, val_remote, repo_type="dataset")
    table = pq.read_table(local, columns=["image", "label"])
    n_avail = len(table)
    n = min(N_SAMPLES, n_avail)
    print(f"available rows={n_avail}, taking first {n}")

    images_col = table.column("image")
    psnr_vals: list[float] = []
    grid_orig: list[torch.Tensor] = []
    grid_recon: list[torch.Tensor] = []

    t_start = time.perf_counter()
    for i in range(0, n, ROUND_TRIP_BATCH):
        batch = []
        for j in range(i, min(i + ROUND_TRIP_BATCH, n)):
            ib = get_image_bytes(images_col[j].as_py())
            if ib is None:
                continue
            img = Image.open(io.BytesIO(ib)).convert("RGB")
            batch.append(transform(img))
        if not batch:
            continue
        x = torch.stack(batch).to(device, dtype=dtype)
        z = vae.encode(x).latent_dist.sample() * SCALING_FACTOR
        x_recon = vae.decode(z / SCALING_FACTOR).sample
        # MSE in [-1,1] space, peak-to-peak = 2
        mse = ((x.float() - x_recon.float()) ** 2).mean(dim=(1, 2, 3))
        peak = 2.0
        psnr = 10.0 * torch.log10(peak ** 2 / mse.clamp_min(1e-12))
        psnr_vals.extend(psnr.detach().cpu().tolist())

        if len(grid_orig) * ROUND_TRIP_BATCH < N_GRID:
            grid_orig.append(x.float().cpu())
            grid_recon.append(x_recon.float().cpu())

    elapsed = time.perf_counter() - t_start
    psnr_t = torch.tensor(psnr_vals)
    mean_psnr = psnr_t.mean().item()
    median_psnr = psnr_t.median().item()
    min_psnr = psnr_t.min().item()
    max_psnr = psnr_t.max().item()
    p10 = psnr_t.quantile(0.10).item()
    passed = mean_psnr >= PASS_THRESHOLD_DB

    print(f"\n=== VAE round-trip PSNR (n={len(psnr_vals)}, elapsed={elapsed:.1f}s) ===")
    print(f"  mean   = {mean_psnr:.2f} dB")
    print(f"  median = {median_psnr:.2f} dB")
    print(f"  p10    = {p10:.2f} dB")
    print(f"  min    = {min_psnr:.2f} dB")
    print(f"  max    = {max_psnr:.2f} dB")
    print(f"  pass (>= {PASS_THRESHOLD_DB} dB)? {'YES' if passed else 'NO'}")

    # Save side-by-side grid
    orig_cat = torch.cat(grid_orig, dim=0)[:N_GRID]
    recon_cat = torch.cat(grid_recon, dim=0)[:N_GRID]
    orig_cat = ((orig_cat + 1) / 2).clamp(0, 1)
    recon_cat = ((recon_cat + 1) / 2).clamp(0, 1)
    full = torch.cat([orig_cat, recon_cat], dim=0)
    grid = make_grid(full, nrow=N_GRID, padding=2)
    OUT_PNG_RECON.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(OUT_PNG_RECON))
    print(f"saved orig (top) vs recon (bottom) -> {OUT_PNG_RECON}")

    # (B) Decode-only of 16 precomputed latents
    print("\ndecoding 16 random latents from train_0000.pt...")
    pre = torch.load(PRECOMPUTED_SHARD, weights_only=True)
    lat_all = pre["latents"]
    lab_all = pre["labels"]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(lat_all.shape[0], generator=g)[:N_GRID]
    z = lat_all[idx].to(device, dtype=dtype)
    x_dec = vae.decode(z / SCALING_FACTOR).sample
    x_dec = ((x_dec.float().cpu() + 1) / 2).clamp(0, 1)
    grid2 = make_grid(x_dec, nrow=8, padding=2)
    save_image(grid2, str(OUT_PNG_LATENTS))
    sample_labels = lab_all[idx].tolist()
    print(f"saved decoded precomputed latents -> {OUT_PNG_LATENTS}")
    print(f"  classes shown: {sample_labels}")

    # Cleanup val parquet (we don't keep it: FID will re-download separately)
    blob = os.path.realpath(local)
    for p in (local, blob):
        try:
            if os.path.exists(p) or os.path.islink(p):
                os.remove(p)
        except OSError:
            pass
    print("cleaned up val parquet from cache")

    OUT_JSON.write_text(json.dumps({
        "n_samples": len(psnr_vals),
        "mean_psnr_db": round(mean_psnr, 3),
        "median_psnr_db": round(median_psnr, 3),
        "p10_psnr_db": round(p10, 3),
        "min_psnr_db": round(min_psnr, 3),
        "max_psnr_db": round(max_psnr, 3),
        "pass_threshold_db": PASS_THRESHOLD_DB,
        "passed": passed,
        "elapsed_s": round(elapsed, 1),
        "vae_repo": VAE_REPO,
        "scaling_factor": SCALING_FACTOR,
        "img_size": IMG_SIZE,
        "decode_sample_labels_from_train_0000": sample_labels,
    }, indent=2))
    print(f"saved metrics -> {OUT_JSON}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
