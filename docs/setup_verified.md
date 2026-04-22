# Setup Verification

## System info

| Component | Value |
|-----------|-------|
| OS | Ubuntu 24.04 LTS |
| Kernel | 6.8.0-110-generic |
| CPU | Intel Core i9-10900K |
| GPU | AMD Radeon RX 6900 XT (16 GB VRAM) |
| ROCm | 7.2.2 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+rocm7.2 |
| torchvision | 0.26.0+rocm7.2 |

## Setup instructions

```bash
# Prerequisites: Linux with ROCm 7.2+ installed system-wide.
# Install uv: https://docs.astral.sh/uv/getting-started/installation/

cd disattend
uv sync
uv run python scripts/verify_setup.py
```

## Verification output

Output of `uv run python scripts/verify_setup.py`, collected 2026-04-22
during repository bootstrap.

```
============================================================
GPU CHECK
============================================================
Python:        3.12.3
PyTorch:       2.11.0+rocm7.2
CUDA available:True
Device:        AMD Radeon RX 6900 XT
VRAM:          17.2 GB
Tensor op:     OK (4x4 matmul on GPU -> CPU)

============================================================
MATMUL BENCHMARK
============================================================
1024x1024 fp32:  0.64 ms  (3377 GFLOPS)
2048x2048 fp32:  0.80 ms  (21553 GFLOPS)
4096x4096 fp32:  6.01 ms  (22879 GFLOPS)

All checks passed.
```
