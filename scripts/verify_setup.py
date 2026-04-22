"""Verify that the core software stack works: GPU and matmul benchmark.

This script is intentionally narrow. Model-specific libraries (diffusers,
transformers) are not installed at bootstrap time and will be added in
Phase 1; extend this script when they are.
"""

import sys
import time

import torch


def check_gpu():
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA available:{torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: No GPU detected. Check ROCm installation.")
        sys.exit(1)
    print(f"Device:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Quick tensor round-trip
    x = torch.randn(4, 4, device="cuda")
    _ = (x @ x.T).cpu()
    print(f"Tensor op:     OK (4x4 matmul on GPU -> CPU)")
    print()


def benchmark_matmul():
    print("=" * 60)
    print("MATMUL BENCHMARK")
    print("=" * 60)

    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()

        # Timed runs
        n_runs = 10
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = a @ b
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs

        gflops = (2 * size**3) / elapsed / 1e9
        print(f"{size}x{size} fp32:  {elapsed * 1000:.2f} ms  ({gflops:.0f} GFLOPS)")

    print()


if __name__ == "__main__":
    check_gpu()
    benchmark_matmul()
    print("All checks passed.")
