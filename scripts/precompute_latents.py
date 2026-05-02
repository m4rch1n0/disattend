#!/usr/bin/env python3
"""Precompute SD VAE latents for ImageNet-1K parquet shards.

Pipeline per shard:
  parquet (HF) -> JPEG bytes -> PIL -> resize 256 + center crop
  -> [-1,1] tensor -> VAE encode (fp16, .sample() * 0.18215)
  -> save .pt shard (latents + labels) -> delete parquet from cache

Output: data/imagenet_latents/{split}_{idx:04d}.pt + index.json
Log:    notebooks/out/precompute_latents.log  (line-buffered, tail -f)

Usage:
  Sanity (1000 val imgs):
    uv run python scripts/precompute_latents.py \
        --split validation --limit-rows 1000 --out-subdir sanity

  Full train (background):
    nohup uv run python scripts/precompute_latents.py --split train \
        > /dev/null 2>&1 & disown
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
    for _cand in (HF_HOME_LOCAL / "token",
                  Path.home() / ".cache" / "huggingface" / "token"):
        if _cand.exists():
            os.environ["HF_TOKEN"] = _cand.read_text().strip()
            break

import torch
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from diffusers import AutoencoderKL
from torchvision import transforms


DATASET_REPO = "ILSVRC/imagenet-1k"
VAE_REPO = "stabilityai/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
IMG_SIZE = 256
ENCODE_BATCH = 32

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "imagenet_latents"
LOG_PATH = REPO_ROOT / "notebooks" / "out" / "precompute_latents.log"


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.fh.write(line + "\n")
        print(line, flush=True)

    def close(self) -> None:
        self.fh.close()


def load_vae(device: torch.device, dtype: torch.dtype) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])


def get_image_bytes(field) -> bytes | None:
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    return None


@torch.inference_mode()
def encode_batch(imgs: torch.Tensor, vae: AutoencoderKL,
                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = imgs.to(device, dtype=dtype, non_blocking=True)
    z = vae.encode(x).latent_dist.sample() * SCALING_FACTOR
    return z.to(torch.float16).cpu()


def process_shard(parquet_path: str, transform, vae: AutoencoderKL,
                  device: torch.device, dtype: torch.dtype,
                  log: Logger) -> tuple[torch.Tensor, torch.Tensor]:
    table = pq.read_table(parquet_path, columns=["image", "label"])
    n = len(table)
    images_col = table.column("image")
    labels_col = table.column("label")

    latents_chunks: list[torch.Tensor] = []
    labels_list: list[int] = []
    buf: list[torch.Tensor] = []
    skipped = 0

    for i in range(n):
        img_bytes = get_image_bytes(images_col[i].as_py())
        if img_bytes is None:
            skipped += 1
            continue
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            log(f"    WARN row {i}: PIL decode failed ({e!r}), skipping")
            skipped += 1
            continue
        buf.append(transform(img))
        labels_list.append(labels_col[i].as_py())
        if len(buf) == ENCODE_BATCH:
            latents_chunks.append(encode_batch(torch.stack(buf), vae, device, dtype))
            buf.clear()

    if buf:
        latents_chunks.append(encode_batch(torch.stack(buf), vae, device, dtype))
        buf.clear()

    if skipped:
        log(f"    skipped={skipped} rows (decode/missing)")

    latents = torch.cat(latents_chunks, dim=0).contiguous()
    labels = torch.tensor(labels_list, dtype=torch.int32)
    return latents, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], default="train")
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--end-shard", type=int, default=None,
                    help="exclusive; default = all shards in split")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="stop after this many rows total (sanity)")
    ap.add_argument("--out-subdir", type=str, default=None,
                    help="write to data/imagenet_latents/<subdir>/ "
                         "(use 'sanity' for sanity runs)")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="do not delete parquet shard from .hf_cache after processing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    log = Logger(LOG_PATH)
    log("=== precompute_latents start ===")
    log(f"split={args.split} start={args.start_shard} end={args.end_shard} "
        f"limit_rows={args.limit_rows} out_subdir={args.out_subdir} seed={args.seed}")

    out_dir = DEFAULT_OUT_DIR / args.out_subdir if args.out_subdir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    log(f"device={device} dtype={dtype}")

    t_vae = time.perf_counter()
    vae = load_vae(device, dtype)
    n_params = sum(p.numel() for p in vae.parameters()) / 1e6
    log(f"VAE {VAE_REPO} loaded in {time.perf_counter()-t_vae:.1f}s, params={n_params:.1f}M")

    transform = make_transform()

    files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
    shard_files = sorted(f for f in files if f.startswith(f"data/{args.split}-"))
    n_shards = len(shard_files)
    log(f"found {n_shards} {args.split} shards on HF")

    end = args.end_shard if args.end_shard is not None else n_shards
    if not (0 <= args.start_shard < end <= n_shards):
        log(f"FATAL bad shard range: start={args.start_shard} end={end} n={n_shards}")
        return 2

    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    grand_start = time.perf_counter()
    total_rows = 0

    for shard_idx in range(args.start_shard, end):
        shard_remote = shard_files[shard_idx]
        log(f"--- shard {shard_idx} ({shard_remote})")

        t_dl = time.perf_counter()
        local = hf_hub_download(DATASET_REPO, shard_remote, repo_type="dataset")
        size_mb = os.path.getsize(local) / 1e6
        dl_s = time.perf_counter() - t_dl
        log(f"    downloaded {size_mb:.1f} MB in {dl_s:.1f}s "
            f"({size_mb/max(dl_s, 1e-6):.1f} MB/s)")

        t_enc = time.perf_counter()
        latents, labels = process_shard(local, transform, vae, device, dtype, log)
        enc_elapsed = time.perf_counter() - t_enc
        n_rows = latents.shape[0]
        rate = n_rows / enc_elapsed if enc_elapsed > 0 else 0
        log(f"    encoded {n_rows} rows in {enc_elapsed:.1f}s ({rate:.1f} img/s)")

        out_path = out_dir / f"{args.split}_{shard_idx:04d}.pt"
        torch.save({"latents": latents, "labels": labels}, out_path)
        out_mb = out_path.stat().st_size / 1e6
        log(f"    saved -> {out_path.name} ({out_mb:.1f} MB)")

        index[out_path.name] = {"n": int(n_rows), "shard_remote": shard_remote}
        index_path.write_text(json.dumps(index, indent=2))

        if not args.keep_parquet:
            # hf_hub_download returns a snapshot path (symlink or regular file)
            # pointing to a blob under .hf_cache/hub/<repo>/blobs/<hash>.
            # We must remove both to actually free disk.
            blob_path = os.path.realpath(local)
            for p in (local, blob_path):
                if p == local or p != local:  # always try both
                    try:
                        if os.path.exists(p) or os.path.islink(p):
                            os.remove(p)
                    except OSError as e:
                        log(f"    WARN failed to remove {p}: {e}")
            log(f"    removed parquet from cache (snapshot+blob, freed {size_mb:.1f} MB)")

        total_rows += n_rows
        elapsed = time.perf_counter() - grand_start
        done = shard_idx - args.start_shard + 1
        todo = end - shard_idx - 1
        eta_h = (elapsed / done * todo) / 3600 if todo > 0 else 0
        log(f"    [progress] {done}/{end-args.start_shard} shards, "
            f"{total_rows} rows, elapsed={elapsed/3600:.2f}h, ETA={eta_h:.2f}h")

        if args.limit_rows is not None and total_rows >= args.limit_rows:
            log(f"    hit --limit-rows={args.limit_rows}, stopping")
            break

    total_elapsed = time.perf_counter() - grand_start
    log(f"=== done. {total_rows} rows in {total_elapsed/3600:.2f}h ===")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
