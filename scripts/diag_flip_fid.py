#!/usr/bin/env python3
"""Quantify the FID cost of latent-space horizontal-flip augmentation.

We augment training by flipping the *latent* (decision D), not the image.
sanity_latent_flip.py showed this is ~5.6 dB worse in pixel PSNR; this
script converts that into FID, the metric we actually report.

Decodes real ImageNet val images under 4 conditions and computes FID of
each against the precomputed 50k real-val reference
(data/imagenet_latents/fid_ref_stats.pt -- real images, no VAE):

  1. real        -- x straight to Inception (no VAE).  Sampling-bias floor.
  2. clean       -- decode(encode(x)).                 VAE round-trip ceiling.
  3. imageflip   -- decode(encode(flip(x))).           Correct flip aug.
  4. latentflip  -- decode(flip(encode(x))).           Our aug (decision D).

Encoding uses .latent_dist.sample() * 0.18215, exactly as the training
precompute (scripts/precompute_latents.py). latentflip reuses the SAME
sampled z as clean (flip applied after), so the only difference between
clean and latentflip is the flip operation, not VAE posterior noise.

Key readouts:
  clean - real           = pure VAE reconstruction cost
  imageflip - clean       ~ 0 expected (flip preserves the distribution)
  latentflip - imageflip = cost of flipping in latent vs image space  <-- ANSWER
  latentflip - clean     = total extra FID our flip injects over no-flip
"""

from __future__ import annotations

import argparse
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
    for _c in (HF_HOME_LOCAL / "token",
               Path.home() / ".cache" / "huggingface" / "token"):
        if _c.exists():
            os.environ["HF_TOKEN"] = _c.read_text().strip()
            break

for _p in (str(REPO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download, HfApi
from diffusers import AutoencoderKL
from torchvision import transforms

from src.evaluation.fid import (
    load_inception, inception_features, compute_stats, fid_from_stats,
)

DATASET_REPO = "ILSVRC/imagenet-1k"
VAE_REPO = "stabilityai/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
IMG_SIZE = 256
REF_PATH = REPO_ROOT / "data" / "imagenet_latents" / "fid_ref_stats.pt"
OUT_JSON = REPO_ROOT / "notebooks" / "out" / "flip_fid_diag.json"


def get_image_bytes(field):
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return None


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of val images")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"device={device} dtype={dtype} n={args.n}")

    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device).eval()
    inc = load_inception(device)

    ref = torch.load(REF_PATH, weights_only=True)
    mu_r = ref["mu"].numpy(); sigma_r = ref["sigma"].numpy()
    print(f"reference: n={ref.get('n')} split={ref.get('split')}")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),  # -> [-1, 1] for VAE
    ])

    conditions = ["real", "clean", "imageflip", "latentflip"]
    feats = {c: [] for c in conditions}

    def decode01(z):
        imgs = vae.decode(z.to(dtype) / SCALING_FACTOR).sample
        return ((imgs.float() + 1) / 2).clamp(0, 1)

    files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
    shard_files = sorted(f for f in files if f.startswith("data/validation-"))

    collected = 0
    t0 = time.perf_counter()
    for shard in shard_files:
        if collected >= args.n:
            break
        local = hf_hub_download(DATASET_REPO, shard, repo_type="dataset")
        table = pq.read_table(local, columns=["image"])
        col = table.column("image")

        buf = []
        for j in range(len(table)):
            if collected + len(buf) >= args.n:
                break
            ib = get_image_bytes(col[j].as_py())
            if ib is None:
                continue
            try:
                img = Image.open(io.BytesIO(ib)).convert("RGB")
            except Exception:
                continue
            buf.append(transform(img))
            if len(buf) == args.batch:
                _process(buf, device, dtype, vae, inc, feats, decode01)
                collected += len(buf); buf = []
                if collected % 800 == 0:
                    print(f"  {collected}/{args.n}  ({time.perf_counter()-t0:.0f}s)")
        if buf:
            _process(buf, device, dtype, vae, inc, feats, decode01)
            collected += len(buf); buf = []

        # cleanup downloaded parquet (snapshot symlink + blob)
        blob = os.path.realpath(local)
        for p in (local, blob):
            try:
                if os.path.exists(p) or os.path.islink(p):
                    os.remove(p)
            except OSError:
                pass

    elapsed = time.perf_counter() - t0
    print(f"\ndecoded {collected} images in {elapsed:.0f}s; computing FID...")

    results = {}
    for c in conditions:
        f = np.concatenate(feats[c], axis=0)
        mu_g, sigma_g = compute_stats(f)
        results[c] = round(fid_from_stats(mu_r, sigma_r, mu_g, sigma_g), 3)

    out = {
        "n_samples": collected,
        "reference_n": int(ref.get("n", 0)),
        "elapsed_s": round(elapsed, 1),
        "fid": results,
        "derived": {
            "vae_recon_cost (clean - real)": round(results["clean"] - results["real"], 3),
            "imageflip - clean": round(results["imageflip"] - results["clean"], 3),
            "latentflip_cost (latentflip - imageflip)": round(results["latentflip"] - results["imageflip"], 3),
            "latentflip_total (latentflip - clean)": round(results["latentflip"] - results["clean"], 3),
        },
    }

    print("\n=== FID vs real 50k reference ===")
    for c in conditions:
        print(f"  {c:12s}: {results[c]:8.3f}")
    print("\n=== derived ===")
    for k, v in out["derived"].items():
        print(f"  {k:42s}: {v:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


@torch.inference_mode()
def _process(buf, device, dtype, vae, inc, feats, decode01):
    x = torch.stack(buf).to(device, dtype=dtype)          # [-1, 1]
    x_flip = torch.flip(x, dims=[-1])

    z = vae.encode(x).latent_dist.sample() * SCALING_FACTOR
    z_imgflip = vae.encode(x_flip).latent_dist.sample() * SCALING_FACTOR
    z_latflip = torch.flip(z, dims=[-1])

    imgs = {
        "real": ((x.float() + 1) / 2).clamp(0, 1),
        "clean": decode01(z),
        "imageflip": decode01(z_imgflip),
        "latentflip": decode01(z_latflip),
    }
    for c, im in imgs.items():
        feats[c].append(inception_features(inc, im).cpu().numpy())


if __name__ == "__main__":
    sys.exit(main())
