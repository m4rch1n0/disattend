#!/usr/bin/env python3
"""Precompute FID reference statistics from the ImageNet validation set.

For each of the 14 val parquet shards:
  - download from HF (rolling delete: at most one shard on disk at a time);
  - decode each JPEG, resize+center-crop 256, scale to [0, 1];
  - feed to pytorch-fid InceptionV3 (which resizes to 299 internally) to
    obtain 2048-d features;
  - accumulate features in RAM.

At the end compute (mu, sigma) and save to
data/imagenet_latents/fid_ref_stats.pt. Saving (mu, sigma) only (~16 MB)
not raw features (~400 MB) — features can always be regenerated.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HF_HOME_LOCAL = REPO_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_HOME_LOCAL))

if "HF_TOKEN" not in os.environ:
    for _cand in (HF_HOME_LOCAL / "token",
                  Path.home() / ".cache" / "huggingface" / "token"):
        if _cand.exists():
            os.environ["HF_TOKEN"] = _cand.read_text().strip()
            break

import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from torchvision import transforms

from src.evaluation.fid import (
    INCEPTION_DIM, load_inception, inception_features, compute_stats,
)


DATASET_REPO = "ILSVRC/imagenet-1k"
IMG_SIZE = 256
ENCODE_BATCH = 32

OUT_PATH = REPO_ROOT / "data" / "imagenet_latents" / "fid_ref_stats.pt"
LOG_PATH = REPO_ROOT / "notebooks" / "out" / "precompute_fid_ref.log"


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.fh.write(line + "\n")
        print(line, flush=True)


def get_image_bytes(field):
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return None


def main() -> int:
    log = Logger(LOG_PATH)
    log("=== precompute_fid_ref start ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    log("loading InceptionV3 (pytorch-fid)...")
    inc = load_inception(device)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),  # [0, 1]
    ])

    files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
    shard_files = sorted(f for f in files
                         if f.startswith("data/validation-"))
    log(f"found {len(shard_files)} validation shards on HF")

    all_feats: list[np.ndarray] = []
    grand_start = time.perf_counter()

    for shard_idx, shard_remote in enumerate(shard_files):
        log(f"--- shard {shard_idx} ({shard_remote})")
        t_dl = time.perf_counter()
        local = hf_hub_download(DATASET_REPO, shard_remote, repo_type="dataset")
        size_mb = os.path.getsize(local) / 1e6
        log(f"    downloaded {size_mb:.1f} MB in "
            f"{time.perf_counter()-t_dl:.1f}s")

        table = pq.read_table(local, columns=["image"])
        n = len(table)
        images_col = table.column("image")
        log(f"    rows={n}, encoding...")

        t_enc = time.perf_counter()
        buf: list[torch.Tensor] = []
        skipped = 0
        for i in range(n):
            ib = get_image_bytes(images_col[i].as_py())
            if ib is None:
                skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(ib)).convert("RGB")
            except Exception as e:
                log(f"    WARN row {i}: PIL decode failed ({e!r})")
                skipped += 1
                continue
            buf.append(transform(img))
            if len(buf) == ENCODE_BATCH:
                x = torch.stack(buf).to(device, non_blocking=True)
                feats = inception_features(inc, x).cpu().numpy()
                all_feats.append(feats)
                buf.clear()
        if buf:
            x = torch.stack(buf).to(device, non_blocking=True)
            feats = inception_features(inc, x).cpu().numpy()
            all_feats.append(feats)
            buf.clear()
        if skipped:
            log(f"    skipped {skipped} rows")

        elapsed = time.perf_counter() - t_enc
        log(f"    encoded {n - skipped} rows in {elapsed:.1f}s "
            f"({(n - skipped)/elapsed:.1f} img/s)")

        # Rolling delete (HF Hub 1.x: snapshot symlink + blob)
        blob = os.path.realpath(local)
        for p in (local, blob):
            try:
                if os.path.exists(p) or os.path.islink(p):
                    os.remove(p)
            except OSError as e:
                log(f"    WARN failed to remove {p}: {e}")

        cum = sum(f.shape[0] for f in all_feats)
        log(f"    total features so far: {cum}")

    feats = np.concatenate(all_feats, axis=0)
    log(f"\nstacked features: shape={feats.shape}")

    mu, sigma = compute_stats(feats)
    log(f"stats: mu shape={mu.shape}, sigma shape={sigma.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mu": torch.from_numpy(mu),
        "sigma": torch.from_numpy(sigma),
        "n": int(feats.shape[0]),
        "feature_dim": INCEPTION_DIM,
        "img_size": IMG_SIZE,
        "split": "validation",
        "dataset": DATASET_REPO,
    }, OUT_PATH)
    log(f"saved -> {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.perf_counter() - grand_start
    log(f"=== done. {feats.shape[0]} ref images in {elapsed/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
