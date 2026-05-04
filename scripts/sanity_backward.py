#!/usr/bin/env python3
"""Verify a single backward pass through the Euler ODE sampler fits in VRAM.

This is the Phase 1 Task 1.6 sanity: when Phase 2 implements PGD on z_T,
the attack inner loop needs a *differentiable* sampler. We use Euler ODE
on the velocity model (deterministic and naturally differentiable, no
DDIM tricks needed). The risk is VRAM: without help, 25-50 forward
activations of SiT-B/2 (130M) all stack on the autograd graph and OOM
the 16 GB 6900 XT.

Mitigation: torch.utils.checkpoint.checkpoint per Euler step. Each step
discards its activations and recomputes them during backward; ~2x compute
for ~25x memory reduction. The script measures peak VRAM and confirms a
finite, non-zero gradient on z_T to prove the path actually carries
signal back to the perturbation site.

Usage (each takes ~minutes on the 6900 XT, depending on model size):

    # SiT-B/2 from a training checkpoint (preferred once available):
    uv run python scripts/sanity_backward.py \
        --model SiT-B/2 \
        --ckpt experiments/.../checkpoints/step_00500000.pt \
        --n-steps 25 --batch 1

    # Random-init smoke (no ckpt; just confirms the mechanics):
    uv run python scripts/sanity_backward.py \
        --model SiT-B/2 --n-steps 25 --batch 1

    # CPU dry-run with a tiny model (fast, validates code path only):
    uv run python scripts/sanity_backward.py \
        --model SiT-S/2 --n-steps 5 --batch 1 --cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SIT_DIR = REPO_ROOT / "third_party" / "sit"
if str(SIT_DIR) not in sys.path:
    sys.path.insert(0, str(SIT_DIR))

import torch
from torch.utils.checkpoint import checkpoint

from models import SiT_models  # third_party/sit
from src.models.unet_b import UNet_models


MODEL_REGISTRY = {**SiT_models, **UNet_models}


def euler_step(model, x, t_cur, dt, y):
    """One Euler integration step. Wrapped so checkpoint() can re-run it."""
    t_batch = t_cur.expand(x.shape[0])
    v = model(x, t_batch, y)
    return x + dt * v


def sample_with_checkpoint(model, z_T, y, n_steps: int,
                           use_checkpoint: bool = True):
    """Euler ODE from t=0 (noise) to t=1 (data) with optional gradient checkpointing.

    Convention matches the SiT codebase ICPlan: integrate forward in t
    from noise (t=0) to data (t=1) with positive dt. Differentiable wrt
    z_T so gradients flow back through every step.
    """
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=z_T.device, dtype=z_T.dtype)
    x = z_T
    for i in range(n_steps):
        dt = (ts[i + 1] - ts[i]).detach()
        t_cur = ts[i].detach()
        if use_checkpoint:
            x = checkpoint(euler_step, model, x, t_cur, dt, y, use_reentrant=False)
        else:
            x = euler_step(model, x, t_cur, dt, y)
    return x


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SiT-B/2",
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="optional path to a training ckpt (.pt). "
                         "Without it the model is random-init.")
    ap.add_argument("--use-ema", action="store_true",
                    help="load the EMA weights from --ckpt instead of model")
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--latent-size", type=int, default=32)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable gradient checkpointing (likely OOM "
                         "for SiT-B/2, useful only as A/B comparison)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (no VRAM measurement; smoke only)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"device={device}")

    model = MODEL_REGISTRY[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=0.0,
    ).to(device)

    if args.ckpt:
        print(f"loading {args.ckpt} (use_ema={args.use_ema})")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        key = "ema" if args.use_ema else "model"
        model.load_state_dict(ckpt[key])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params/1e6:.1f}M")

    # Eval mode: disable dropout (CFG class_dropout). Keep grads enabled
    # only on z_T (the perturbation site for Phase 2 PGD).
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    z_T = torch.randn(args.batch, 4, args.latent_size, args.latent_size,
                      device=device, requires_grad=True)
    y = torch.randint(0, args.num_classes, (args.batch,), device=device)
    target = torch.randn_like(z_T)  # dummy target; structure of loss does not matter for the test

    use_checkpoint = not args.no_checkpoint
    print(f"sampling {args.n_steps} Euler steps (checkpoint={use_checkpoint})...")

    t0 = time.perf_counter()
    z_0 = sample_with_checkpoint(
        model, z_T, y, n_steps=args.n_steps, use_checkpoint=use_checkpoint,
    )
    t_fwd = time.perf_counter() - t0

    if device.type == "cuda":
        peak_fwd = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"forward done in {t_fwd:.1f}s, peak VRAM after fwd: {peak_fwd:.2f} GB")
    else:
        print(f"forward done in {t_fwd:.1f}s (CPU, no VRAM stats)")

    loss = ((z_0 - target.detach()) ** 2).mean()
    print(f"loss = {loss.item():.4f}")

    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0

    if device.type == "cuda":
        peak_total = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"backward done in {t_bwd:.1f}s, peak VRAM total: {peak_total:.2f} GB")
        within_budget = peak_total < 16.0
        print(f"within 16 GB budget? {'YES' if within_budget else 'NO'}")
    else:
        print(f"backward done in {t_bwd:.1f}s (CPU, no VRAM stats)")
        within_budget = True

    grad = z_T.grad
    if grad is None:
        print("FAIL: z_T.grad is None")
        return 1
    finite = torch.isfinite(grad).all().item()
    nz = (grad.abs() > 0).any().item()
    gnorm = grad.norm().item()
    print(f"z_T.grad: shape={tuple(grad.shape)} dtype={grad.dtype}")
    print(f"  finite={finite}  has_nonzero={nz}  ||grad||_2={gnorm:.4g}")

    ok = finite and nz and within_budget
    print(f"=== result: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
