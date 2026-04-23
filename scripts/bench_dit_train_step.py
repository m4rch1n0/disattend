"""Micro-benchmark: forward+backward cost of DiT-S/B/XL on the 6900 XT.

NOT a deliverable of Phase 0. One-shot diagnostic to inform the scope
decision (train-from-scratch? which size? which resolution?). Run once,
read numbers, delete if noisy.

For each (DiT variant, latent-shape, dtype, batch) config:
  - instantiate DiTTransformer2DModel from scratch (random init) with
    Peebles & Xie 2022 Table 1 hyperparameters,
  - run N warmup + M timed training-like steps (MSE noise-prediction
    loss on random latents),
  - report per-step wall-clock + peak VRAM,
  - extrapolate to full-training budgets for CIFAR-10 (~400k steps)
    and ImageNet-256 (~400k steps).

Run from repo root:

    uv run python scripts/bench_dit_train_step.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DiTTransformer2DModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "notebooks" / "out" / "bench_dit_train_step.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Peebles & Xie 2022, Table 1. patch_size=2 fixed.
DIT_CONFIGS = {
    "DiT-S/2":  {"num_layers": 12, "num_attention_heads":  6, "attention_head_dim": 64, "patch_size": 2},
    "DiT-B/2":  {"num_layers": 12, "num_attention_heads": 12, "attention_head_dim": 64, "patch_size": 2},
    "DiT-L/2":  {"num_layers": 24, "num_attention_heads": 16, "attention_head_dim": 64, "patch_size": 2},
    "DiT-XL/2": {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 2},
}

# (sample_size_latent, in_channels, label). 32x32x4 is the SD-VAE latent of a
# 256x256 image; 4x4x4 is what you'd get by encoding a 32x32 CIFAR image.
LATENT_SHAPES = [
    {"name": "ImageNet-256 VAE latent", "sample_size": 32, "in_channels": 4},
    {"name": "CIFAR-32 pixel-space",    "sample_size": 32, "in_channels": 3},
]

BATCHES = [8, 16, 32]
WARMUP_STEPS = 3
TIMED_STEPS = 10
NUM_CLASSES = 1000  # ImageNet-scale class embedding; doesn't affect compute meaningfully


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def bench_one(name: str, cfg: dict, shape_cfg: dict, batch: int, dtype: torch.dtype) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = DiTTransformer2DModel(
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
        in_channels=shape_cfg["in_channels"],
        num_layers=cfg["num_layers"],
        sample_size=shape_cfg["sample_size"],
        patch_size=cfg["patch_size"],
        num_embeds_ada_norm=NUM_CLASSES,
    ).to("cuda", dtype=dtype)
    model.train()

    params_M = param_count(model) / 1e6
    latent_shape = (batch, shape_cfg["in_channels"], shape_cfg["sample_size"], shape_cfg["sample_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def make_batch():
        x = torch.randn(*latent_shape, device="cuda", dtype=dtype)
        t = torch.randint(0, 1000, (batch,), device="cuda")
        y = torch.randint(0, NUM_CLASSES, (batch,), device="cuda")
        return x, t, y

    try:
        for _ in range(WARMUP_STEPS):
            x, t, y = make_batch()
            opt.zero_grad(set_to_none=True)
            pred = model(x, timestep=t, class_labels=y).sample
            loss = F.mse_loss(pred, x)  # dummy MSE target — we only care about compute cost
            loss.backward()
            opt.step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            x, t, y = make_batch()
            opt.zero_grad(set_to_none=True)
            pred = model(x, timestep=t, class_labels=y).sample
            loss = F.mse_loss(pred, x)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_ms = (elapsed / TIMED_STEPS) * 1000
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        status = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        step_ms = None
        peak_gb = None
        status = f"OOM: {str(exc)[:80]}"
    except Exception as exc:  # noqa: BLE001
        step_ms = None
        peak_gb = None
        status = f"ERR: {type(exc).__name__}: {str(exc)[:120]}"
    finally:
        del model, opt
        torch.cuda.empty_cache()

    return {
        "model": name,
        "shape": shape_cfg["name"],
        "batch": batch,
        "dtype": str(dtype).replace("torch.", ""),
        "params_M": round(params_M, 2),
        "step_ms": None if step_ms is None else round(step_ms, 2),
        "peak_vram_gb": None if peak_gb is None else round(peak_gb, 3),
        "status": status,
    }


def main() -> None:
    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    print(f"[env] torch {torch.__version__} | device: {device_name}")

    rows: list[dict] = []
    for model_name, cfg in DIT_CONFIGS.items():
        for shape_cfg in LATENT_SHAPES:
            for batch in BATCHES:
                # fp16 is the realistic training dtype on RDNA 2 per PROJECT_PLAN §1 (no bf16).
                row = bench_one(model_name, cfg, shape_cfg, batch, torch.float16)
                print(
                    f"[{row['model']:9s}] {row['shape']:24s} batch={batch:3d} fp16 "
                    f"params={row['params_M']:7.1f}M  step={row['step_ms']}  peak={row['peak_vram_gb']}GB  {row['status']}"
                )
                rows.append(row)

    # Extrapolations — assume 400k training steps (standard DiT scale).
    # For each ok row, compute projected wall-clock hours / days.
    STEPS_IMAGENET = 400_000
    STEPS_CIFAR = 200_000

    projections = []
    for r in rows:
        if r["step_ms"] is None:
            continue
        if "CIFAR" in r["shape"]:
            total_steps = STEPS_CIFAR
            scale_note = f"{STEPS_CIFAR//1000}k steps (CIFAR-10 scale)"
        else:
            total_steps = STEPS_IMAGENET
            scale_note = f"{STEPS_IMAGENET//1000}k steps (ImageNet-256 scale)"
        hours = (r["step_ms"] / 1000) * total_steps / 3600
        projections.append({
            **r,
            "total_steps_assumed": total_steps,
            "projected_hours": round(hours, 1),
            "projected_days": round(hours / 24, 2),
            "scale": scale_note,
        })

    summary = {
        "device": device_name,
        "torch": torch.__version__,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "rows": rows,
        "projections": projections,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] saved: {OUT_PATH.relative_to(REPO_ROOT)}")

    # Compact human-readable table of projections.
    print("\n=== projected full-training wall-clock (single 6900 XT fp16) ===")
    print(f"{'model':10s}  {'shape':24s}  {'batch':>5s}  {'step_ms':>8s}  {'VRAM_GB':>7s}  {'scale':30s}  {'hours':>6s}  {'days':>6s}")
    for p in projections:
        print(f"{p['model']:10s}  {p['shape']:24s}  {p['batch']:5d}  {p['step_ms']:8.2f}  {p['peak_vram_gb']:7.3f}  {p['scale']:30s}  {p['projected_hours']:6.1f}  {p['projected_days']:6.2f}")


if __name__ == "__main__":
    main()
