"""Untargeted L_inf PGD on the initial latent z_T, backprop through the sampler.

Maximizes ||sample(z_T + delta) - sample(z_T)||_2. Forward-Euler sampler
(t=0 noise -> t=1 data, as in sanity_backward.py / fid.py), deterministic, so
the benign branch is computed once and reused as the target.

Don't attach an AttentionCollector inside the attack loop: the checkpointed
backward re-runs every Euler step, so a hook fires twice per step. Measure
attention in a separate inference_mode forward on the returned z_T_adv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
if str(SIT_DIR) not in sys.path:
    sys.path.insert(0, str(SIT_DIR))

from models import SiT_models  # noqa: E402  (after sys.path inject)
from src.models.unet_b import UNet_models  # noqa: E402

MODEL_REGISTRY = {**SiT_models, **UNet_models}
NUM_CLASSES = 1000  # null CFG token = index NUM_CLASSES in the label embedding


def load_model(name: str, ckpt_path: str | Path, device: torch.device) -> nn.Module:
    """Build a MODEL_REGISTRY model, load EMA weights, freeze, eval, fp32.

    class_dropout_prob must be > 0 so the label embedding has the +1 null row
    the checkpoints were trained with; dropout behavior itself is disabled by
    .eval().
    """
    model = MODEL_REGISTRY[name](
        input_size=32,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.1,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def euler_step(model, x, t_cur, dt, y, cfg_scale: float = 1.0):
    """One Euler integration step. Wrapped so checkpoint() can re-run it.

    cfg_scale == 1.0 -> single conditional forward. Otherwise manual
    classifier-free guidance (model-agnostic, differentiable): the guard
    matters because the cat-duplicated batch doubles compute and VRAM on
    every step, and under checkpointing every step runs twice.
    """
    t_batch = t_cur.expand(x.shape[0])
    if cfg_scale != 1.0:
        y_null = torch.full_like(y, NUM_CLASSES)
        v = model(
            torch.cat([x, x], dim=0),
            torch.cat([t_batch, t_batch], dim=0),
            torch.cat([y, y_null], dim=0),
        )
        v_cond, v_uncond = v.chunk(2, dim=0)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
    else:
        v = model(x, t_batch, y)
    return x + dt * v


def sample_with_checkpoint(model, z_T, y, n_steps: int,
                           cfg_scale: float = 1.0,
                           use_checkpoint: bool = True):
    """Euler ODE t=0 (noise) -> t=1 (data), differentiable wrt z_T.

    With use_checkpoint each step discards activations and recomputes them in
    backward (~2x compute, ~n_steps x memory saving). Disable it for
    grad-free legs (benign target, measurement): under no_grad checkpointing
    buys nothing and only warns.
    """
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=z_T.device, dtype=z_T.dtype)
    x = z_T
    for i in range(n_steps):
        dt = (ts[i + 1] - ts[i]).detach()
        t_cur = ts[i].detach()
        if use_checkpoint:
            x = checkpoint(euler_step, model, x, t_cur, dt, y, cfg_scale,
                           use_reentrant=False)
        else:
            x = euler_step(model, x, t_cur, dt, y, cfg_scale)
    return x


def rademacher_delta(z_T: torch.Tensor, eps: float, *,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Equal-budget random control: +-eps on every coordinate.

    Matches both the L_inf and the L2 norm of a sign-saturated PGD
    perturbation; only the direction is random. The caller averages metrics
    over K >= 3 independent draws per seed for a stable control branch.
    """
    r = torch.empty_like(z_T).normal_(generator=generator)
    return torch.where(r >= 0, eps, -eps).to(z_T.dtype)


def pgd_latent(
    model: nn.Module,
    z_T: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.05,
    n_steps_attack: int = 20,
    step_size: float | None = None,
    n_steps_ode: int = 25,
    cfg_scale: float = 1.0,
    loss_fn: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "l2_output",
    generator: torch.Generator | None = None,
    use_checkpoint: bool = True,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Untargeted L_inf PGD on z_T: maximize the output displacement.

    delta_0 ~ Uniform(-eps, eps)  (random start; mandatory, see module doc)
    delta_{k+1} = clamp(delta_k + alpha * sign(grad_delta loss), -eps, eps)

    No data-range clamp is applied to z_T + delta: latents are unbounded
    Gaussian coordinates, there is no [0,1] box as in image-space PGD.

    Args:
        model: frozen velocity model from load_model (fp32, eval).
        z_T: benign noise seeds (B, 4, 32, 32), no grad required.
        y: class labels (B,).
        eps: L_inf budget in latent space.
        n_steps_attack: PGD iterations.
        step_size: PGD step alpha; default eps/4.
        n_steps_ode: Euler steps of the differentiable sampler.
        cfg_scale: 1.0 = no guidance (default). >1 = manual CFG.
        loss_fn: "l2_output" = MSE between perturbed and benign generated
            latents (maximized). Or a callable (z0_adv, z0_benign_detached)
            -> scalar to maximize, e.g. an image-space objective that decodes
            in-graph (the fallback if the latent-L2 attack turns out not to be
            perceptual).
        generator: optional torch.Generator (device-matched) for the random
            start, for reproducibility.
        use_checkpoint: gradient checkpointing in the attack loop.
        verbose: print one line per iteration.

    Returns:
        (z_T_adv, info): z_T_adv detached, ||z_T_adv - z_T||_inf <= eps.
        info: losses (n_steps_attack+1 floats: loss at each iterate, then at
        the final one), grad_norms (n_steps_attack), l2_out_final (B,) L2
        output displacement per sample, linf_final, plus the config echo.
    """
    if step_size is None:
        step_size = eps / 4.0
    if isinstance(loss_fn, str):
        if loss_fn != "l2_output":
            raise ValueError(f"unknown loss_fn {loss_fn!r}")
        _loss = lambda z0_adv, z0_ben: F.mse_loss(z0_adv, z0_ben)  # noqa: E731
        loss_name = "l2_output"
    else:
        _loss = loss_fn
        loss_name = getattr(loss_fn, "__name__", "custom")

    z_T = z_T.detach()

    # Benign target branch: computed once, fixed. no_grad (not inference_mode:
    # inference tensors cannot participate in a later autograd graph).
    with torch.no_grad():
        z0_ben = sample_with_checkpoint(model, z_T, y, n_steps=n_steps_ode,
                                        cfg_scale=cfg_scale, use_checkpoint=False)

    # random start (Madry). needed here, not just standard: the sampler is
    # deterministic, so at delta=0 the grad is exactly 0 and sign(0)=0 -> stuck.
    delta = torch.empty_like(z_T).uniform_(-eps, eps, generator=generator)

    losses: list[float] = []
    grad_norms: list[float] = []
    for it in range(n_steps_attack):
        delta.requires_grad_(True)
        with torch.enable_grad():
            z0_adv = sample_with_checkpoint(model, z_T + delta, y,
                                            n_steps=n_steps_ode,
                                            cfg_scale=cfg_scale,
                                            use_checkpoint=use_checkpoint)
            loss = _loss(z0_adv, z0_ben)
        (grad,) = torch.autograd.grad(loss, delta)
        losses.append(loss.item())
        grad_norms.append(grad.norm().item())
        with torch.no_grad():
            delta = (delta + step_size * grad.sign()).clamp_(-eps, eps)
        delta = delta.detach()
        if verbose:
            print(f"  pgd it {it + 1:3d}/{n_steps_attack}  "
                  f"loss={losses[-1]:.6f}  ||grad||={grad_norms[-1]:.3e}")

    # Loss/displacement at the final iterate (the loop above measures each
    # delta_k before its update, so delta_final would otherwise go unmeasured).
    with torch.no_grad():
        z0_adv = sample_with_checkpoint(model, z_T + delta, y,
                                        n_steps=n_steps_ode,
                                        cfg_scale=cfg_scale, use_checkpoint=False)
        losses.append(_loss(z0_adv, z0_ben).item())
        l2_out_final = (z0_adv - z0_ben).flatten(1).norm(dim=1)

    z_T_adv = z_T + delta
    info = {
        "losses": losses,
        "grad_norms": grad_norms,
        "l2_out_final": l2_out_final.cpu(),
        "linf_final": delta.abs().max().item(),
        "eps": eps,
        "step_size": step_size,
        "n_steps_attack": n_steps_attack,
        "n_steps_ode": n_steps_ode,
        "cfg_scale": cfg_scale,
        "loss_fn": loss_name,
    }
    return z_T_adv, info
