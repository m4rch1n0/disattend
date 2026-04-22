"""Sanity load + single-sample inference for DiT-XL/2 (Phase 0 Task 0.3).

Loads `facebook/DiT-XL-2-256` from HuggingFace into local `.hf_cache/`,
measures VRAM in fp32 and fp16, runs one DDPM sampling at default step count
for ImageNet class 207 (golden retriever), and writes:

  notebooks/out/golden_retriever_fp16.png
  notebooks/out/sanity_load_summary.json

Run from repo root:

    uv run python scripts/sanity_load_dit.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

import torch
from diffusers import DiTPipeline

MODEL_ID = "facebook/DiT-XL-2-256"
OUT_DIR = REPO_ROOT / "notebooks" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABEL = 207  # ImageNet class 207 = golden retriever
NUM_INFERENCE_STEPS = 250
SEED = 42


def _gb(bytes_: int) -> float:
    return bytes_ / (1024**3)


def vram_alloc_gb() -> float:
    return _gb(torch.cuda.memory_allocated())


def vram_peak_gb() -> float:
    return _gb(torch.cuda.max_memory_allocated())


def main() -> None:
    assert torch.cuda.is_available(), "CUDA/ROCm not available"
    device_name = torch.cuda.get_device_name(0)
    print(f"[env] torch {torch.__version__} | device: {device_name}")
    print(f"[env] HF_HOME = {os.environ['HF_HOME']}")
    print(f"[env] model: {MODEL_ID}")

    # ---- fp32 load ----
    print("\n[fp32] loading DiT-XL/2 in fp32 ...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe32 = DiTPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to("cuda")
    torch.cuda.synchronize()
    load_fp32_sec = time.perf_counter() - t0
    vram_fp32 = vram_alloc_gb()
    print(f"[fp32] load wallclock: {load_fp32_sec:.2f} s | VRAM allocated: {vram_fp32:.3f} GB")

    del pipe32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- fp16 load ----
    print("\n[fp16] loading DiT-XL/2 in fp16 ...")
    t0 = time.perf_counter()
    pipe = DiTPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    torch.cuda.synchronize()
    load_fp16_sec = time.perf_counter() - t0
    vram_fp16 = vram_alloc_gb()
    scheduler_name = pipe.scheduler.__class__.__name__
    print(f"[fp16] load wallclock: {load_fp16_sec:.2f} s | VRAM allocated: {vram_fp16:.3f} GB")
    print(f"[fp16] default scheduler: {scheduler_name}")

    # ---- Sample ----
    print(f"\n[sample] class {CLASS_LABEL} (golden retriever), {NUM_INFERENCE_STEPS} {scheduler_name} steps, seed {SEED}")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        result = pipe(
            class_labels=[CLASS_LABEL],
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        )
    torch.cuda.synchronize()
    sample_sec = time.perf_counter() - t0
    vram_peak_sample = vram_peak_gb()
    print(f"[sample] wallclock: {sample_sec:.2f} s | VRAM peak during sampling: {vram_peak_sample:.3f} GB")

    img = result.images[0]
    png_path = OUT_DIR / "golden_retriever_fp16.png"
    img.save(png_path)
    print(f"[sample] saved: {png_path.relative_to(REPO_ROOT)}")

    summary = {
        "device": device_name,
        "torch": torch.__version__,
        "model": MODEL_ID,
        "scheduler": scheduler_name,
        "seed": SEED,
        "class_label": CLASS_LABEL,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "load_fp32_sec": round(load_fp32_sec, 3),
        "load_fp16_sec": round(load_fp16_sec, 3),
        "vram_fp32_gb": round(vram_fp32, 3),
        "vram_fp16_gb": round(vram_fp16, 3),
        "sample_wallclock_sec": round(sample_sec, 3),
        "vram_peak_sample_fp16_gb": round(vram_peak_sample, 3),
        "png_path": str(png_path.relative_to(REPO_ROOT)),
    }
    summary_path = OUT_DIR / "sanity_load_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n[summary]")
    print(json.dumps(summary, indent=2))
    print(f"\n[summary] saved: {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
