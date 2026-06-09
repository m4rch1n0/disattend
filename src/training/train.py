"""Single-GPU training loop for flow-matching DiT-B/2 (and later UNet-B).

Derived from third_party/sit/train.py. Differences:
  - DDP stripped (single GPU only).
  - ImageFolder + online VAE encode replaced with LatentShardDataset
    (latents precomputed offline by scripts/precompute_latents.py).
  - fp16 mixed precision via torch.amp + GradScaler.
  - JSON line-delimited logging instead of wandb.
  - Step-based loop with resume support (model + EMA + opt + scaler + RNG).
  - No periodic image sampling during training (FID is a separate script
    that loads a checkpoint and runs offline).

Usage:
  Smoke test (100 steps, 1 ckpt):
    uv run python -m src.training.train --smoke

  Full DiT-B/2 run (100M sample budget, ~10.7 days):
    nohup uv run python -m src.training.train \\
        --model SiT-B/2 \\
        --total-steps 6400000 \\
        > /dev/null 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
if str(SIT_DIR) not in sys.path:
    sys.path.insert(0, str(SIT_DIR))

from models import SiT_models  # noqa: E402  (after sys.path inject)
from transport import create_transport  # noqa: E402

from src.evaluation.fid import FIDEvaluator  # noqa: E402
from src.models.unet_b import UNet_models  # noqa: E402
from src.training.dataset import LatentShardDataset  # noqa: E402

# Unified registry: SiT-{S,B,L,XL}/{2,4,8} from third_party + UNet-{S,B} from src/models
MODEL_REGISTRY: dict[str, callable] = {**SiT_models, **UNet_models}


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module,
               model: torch.nn.Module,
               decay: float = 0.9999) -> None:
    """ema <- decay * ema + (1 - decay) * model, parameter-wise."""
    ema_p = dict(ema_model.named_parameters())
    cur_p = dict(model.named_parameters())
    for name, p in cur_p.items():
        ema_p[name].mul_(decay).add_(p.data, alpha=1.0 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def save_ckpt(path: Path, *, step: int,
              model: torch.nn.Module,
              ema: torch.nn.Module,
              opt: torch.optim.Optimizer,
              scaler: torch.amp.GradScaler | None,
              args: argparse.Namespace) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        "args": vars(args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def prune_ckpts(ckpt_dir: Path, *, keep_last: int,
                permanent: set[int]) -> list[Path]:
    """Keep the last `keep_last` step ckpts plus those in `permanent`. Delete the rest."""
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return []
    keep_last_set = set(ckpts[-keep_last:])
    deleted: list[Path] = []
    for c in ckpts:
        try:
            cs = int(c.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if c not in keep_last_set and cs not in permanent:
            c.unlink()
            deleted.append(c)
    return deleted


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SiT-B/2",
                    choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--data-dir", type=str,
                    default=str(REPO_ROOT / "data" / "imagenet_latents"))
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--class-dropout-prob", type=float, default=0.1,
                    help="CFG class-dropout in training")
    ap.add_argument("--latent-size", type=int, default=32,
                    help="latent spatial size (256/8 for SD VAE)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=6_400_000,
                    help="default = 100M sample budget at batch 16")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"],
                    default="constant",
                    help="LR schedule. 'constant' = DiT-B (LayerNorm is "
                         "scale-invariant, tolerates a constant LR). 'cosine' = "
                         "warmup + cosine decay to --lr-min, needed for UNet-B: "
                         "GroupNorm is NOT scale-invariant, so a constant LR "
                         "drives unbounded weight-norm growth -> FID degrades "
                         "monotonically (observed in 20260603-UNet-B-bf16-clip).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup steps (0 = off)")
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="floor LR reached at --lr-decay-steps (cosine only)")
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help="step at which cosine reaches --lr-min "
                         "(0 = use --total-steps)")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="max grad-norm (0 = off, = DiT-B). NOT the fix for the "
                         "UNet-B fp16 NaN (that was attention QK^T overflow, "
                         "fixed in unet_b.py); kept only as an optional knob.")
    ap.add_argument("--nan-abort", type=int, default=500,
                    help="abort after this many CONSECUTIVE non-finite steps "
                         "(watchdog vs the silent multi-hour NaN spin). Isolated "
                         "NaNs that recover reset the streak.")
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16", "float32"],
                    default="float16",
                    help="autocast dtype. DiT-B used float16 (its activations fit "
                         "fp16's range). UNet-B needs bfloat16 (fp32 range -> no "
                         "65504 overflow; NVIDIA only) and then weight_decay can "
                         "go back to 0 = DiT recipe. float32 disables autocast. "
                         "GradScaler is used only with float16.")
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--hflip-prob", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=100_000)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--results-dir", type=str,
                    default=str(REPO_ROOT / "experiments"))
    ap.add_argument("--run-name", type=str, default=None,
                    help="default: timestamp + model name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke override: 100 steps, log every 10, ckpt every 50")
    ap.add_argument("--compile", action="store_true",
                    help="wrap the model forward with torch.compile "
                         "(~1.4x speedup on RDNA 2; optimizer/EMA/ckpt stay "
                         "on the original module)")
    ap.add_argument("--fid-every", type=int, default=0,
                    help="evaluate FID every N steps on the EMA model (0 = off)")
    ap.add_argument("--fid-samples", type=int, default=5000,
                    help="number of generated samples per FID eval")
    ap.add_argument("--fid-steps-ode", type=int, default=25,
                    help="Euler ODE steps for FID sampling")
    ap.add_argument("--fid-ref-stats", type=str,
                    default=str(REPO_ROOT / "data" / "imagenet_latents"
                                / "fid_ref_stats.pt"))
    ap.add_argument("--fid-sample-batch", type=int, default=16)
    return ap.parse_args()


def lr_at_step(step: int, *, base_lr: float, warmup: int, decay_steps: int,
               lr_min: float, schedule: str) -> float:
    """LR as a pure function of step (resume-safe: no scheduler state to save).

    Linear warmup 0->base_lr over `warmup` steps, then either constant or a
    cosine decay base_lr->lr_min reached at `decay_steps` and held after.
    """
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    if step >= decay_steps:
        return lr_min
    progress = (step - warmup) / max(1, decay_steps - warmup)
    return lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.total_steps = 100
        args.log_every = 10
        args.ckpt_every = 50
        args.num_workers = 0  # avoid worker startup cost in smoke

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Run dir
    if args.run_name is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        model_safe = args.model.replace("/", "-")
        args.run_name = f"{ts}-{model_safe}"
    run_dir = Path(args.results_dir) / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "results.jsonl"
    print(f"run_dir={run_dir}")
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # Model
    model = MODEL_REGISTRY[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params/1e6:.1f}M")

    # EMA: deep copy, frozen, init from current model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0.0)  # ema := model
    ema.eval()

    # Optional torch.compile: compile only the forward path. The optimizer,
    # EMA, and checkpointing keep operating on the original `model` (the
    # compiled wrapper shares the same parameter tensors), so state dicts
    # stay free of the `_orig_mod.` prefix and resume cleanly.
    fwd_model = torch.compile(model) if args.compile else model
    if args.compile:
        print("torch.compile enabled (forward path)")

    # Transport: Linear interpolant + Velocity prediction = OT linear FM.
    # train_eps=sample_eps=0 stable for VELOCITY+LINEAR (transport.create_transport).
    transport = create_transport(path_type="Linear", prediction="velocity")

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # AMP. float16 needs a GradScaler; bfloat16 (fp32 range) does not and is the
    # cloud/NVIDIA path for UNet-B; float32 disables autocast entirely.
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.amp_dtype]
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    scaler = (torch.amp.GradScaler("cuda")
              if use_amp and amp_dtype == torch.float16 else None)
    print(f"amp: dtype={args.amp_dtype} autocast={use_amp} grad_scaler={scaler is not None}")

    # Data
    print("loading dataset...")
    t_load = time.perf_counter()
    ds = LatentShardDataset(
        data_dir=args.data_dir,
        split="train",
        hflip_prob=args.hflip_prob,
    )
    print(f"  {repr(ds)} (load {time.perf_counter()-t_load:.1f}s)")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # Resume
    start_step = 0
    if args.resume_from:
        print(f"resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng_torch"])
        if device.type == "cuda" and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        start_step = int(ckpt["step"])
        print(f"  resumed at step {start_step}")

    model.train()

    # FID evaluator (lazy: only created if --fid-every > 0)
    fid_eval: FIDEvaluator | None = None
    if args.fid_every > 0:
        ref_path = Path(args.fid_ref_stats)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"--fid-every set but ref stats missing: {ref_path}\n"
                f"run scripts/precompute_fid_ref.py first")
        print(f"setting up FID evaluator (n_samples={args.fid_samples}, "
              f"n_steps_ode={args.fid_steps_ode})...")
        fid_eval = FIDEvaluator(
            device=device,
            ref_stats_path=ref_path,
            n_samples=args.fid_samples,
            n_steps=args.fid_steps_ode,
            sample_batch=args.fid_sample_batch,
            amp_dtype=amp_dtype,
        )
        print(f"  ref n={fid_eval.ref_n}")

    permanent_ckpts = {500_000, 1_000_000, 1_500_000, 2_000_000,
                       3_000_000, 4_000_000, 5_000_000, 6_400_000}

    lr_decay_steps = (args.lr_decay_steps if args.lr_decay_steps > 0
                      else args.total_steps)
    if args.lr_schedule != "constant" or args.warmup_steps > 0:
        print(f"lr schedule: {args.lr_schedule} base={args.lr} "
              f"warmup={args.warmup_steps} decay_steps={lr_decay_steps} "
              f"min={args.lr_min}")

    print(f"training to step {args.total_steps}")
    step = start_step
    log_window_loss = 0.0
    log_window_steps = 0
    t_window = time.perf_counter()
    t_global = time.perf_counter()
    nan_streak = 0  # consecutive non-finite steps; watchdog aborts past --nan-abort

    loader_iter = iter(loader)
    while step < args.total_steps:
        try:
            z, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            z, y = next(loader_iter)

        z = z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        cur_lr = lr_at_step(step, base_lr=args.lr, warmup=args.warmup_steps,
                            decay_steps=lr_decay_steps, lr_min=args.lr_min,
                            schedule=args.lr_schedule)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            losses = transport.training_losses(fwd_model, z, dict(y=y))
            loss = losses["loss"].mean()

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                # Unscale before clipping; scaler still records inf/nan found
                # during unscale_ and skips the step in scaler.step().
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        loss_val = loss.item()
        if math.isfinite(loss_val):
            # Only fold a finite step into EMA; the scaler already skipped the
            # opt update on any inf/nan grad, so model+EMA stay clean.
            update_ema(ema, model, decay=args.ema_decay)
            nan_streak = 0
        else:
            nan_streak += 1
            jsonl_append(log_path, {
                "step": step + 1,
                "event": "non_finite_loss",
                "loss": loss_val,
                "nan_streak": nan_streak,
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"WARN step={step+1}: non-finite loss {loss_val} "
                  f"(skipping; streak={nan_streak})")
            if nan_streak >= args.nan_abort:
                # Watchdog: a long unbroken NaN streak = irrecoverable divergence
                # (the scaler keeps skipping but cannot un-poison). Abort instead
                # of spinning uselessly for hours, as the pre-fix runs did.
                jsonl_append(log_path, {
                    "step": step + 1, "event": "nan_abort",
                    "nan_streak": nan_streak,
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"FATAL step={step+1}: {nan_streak} consecutive non-finite "
                      f"steps -> aborting (diverged).")
                return 1
            loss_val = 0.0
        log_window_loss += loss_val
        log_window_steps += 1
        step += 1

        if step % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            window_elapsed = time.perf_counter() - t_window
            avg_loss = log_window_loss / log_window_steps
            sps = log_window_steps / max(window_elapsed, 1e-6)
            global_elapsed = time.perf_counter() - t_global
            record = {
                "step": step,
                "loss": round(avg_loss, 6),
                "steps_per_sec": round(sps, 3),
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": round(global_elapsed, 1),
                "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"step={step:>8d} loss={avg_loss:.4f} "
                  f"steps/s={sps:.2f} elapsed={global_elapsed/3600:.2f}h")
            jsonl_append(log_path, record)
            log_window_loss = 0.0
            log_window_steps = 0
            t_window = time.perf_counter()

        if fid_eval is not None and step % args.fid_every == 0 and step > 0:
            print(f"step={step}: running FID eval on EMA "
                  f"(n={args.fid_samples}, n_steps={args.fid_steps_ode})...")
            t_fid = time.perf_counter()
            try:
                fid_metrics = fid_eval.evaluate(ema)
                fid_elapsed = time.perf_counter() - t_fid
                print(f"  FID = {fid_metrics['fid']:.3f} "
                      f"(eval took {fid_elapsed/60:.1f} min)")
                jsonl_append(log_path, {
                    "step": step,
                    "event": "fid_eval",
                    "fid": round(fid_metrics["fid"], 4),
                    "fid_n_samples": fid_metrics["n_samples"],
                    "fid_n_steps_ode": fid_metrics["n_steps_ode"],
                    "fid_elapsed_s": round(fid_elapsed, 1),
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception as e:
                fid_elapsed = time.perf_counter() - t_fid
                err = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(f"  WARN FID eval failed: {err} (continuing training)")
                print(tb)
                jsonl_append(log_path, {
                    "step": step,
                    "event": "fid_eval_error",
                    "error": err,
                    "fid_elapsed_s": round(fid_elapsed, 1),
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                model.train()  # in case evaluate() left it in eval mode after exception
            t_window = time.perf_counter()  # reset to not skew next steps/sec

        if step % args.ckpt_every == 0 and step > 0:
            ckpt_path = run_dir / "checkpoints" / f"step_{step:08d}.pt"
            save_ckpt(ckpt_path, step=step, model=model, ema=ema,
                      opt=opt, scaler=scaler, args=args)
            deleted = prune_ckpts(
                run_dir / "checkpoints",
                keep_last=args.keep_last,
                permanent=permanent_ckpts,
            )
            kept = len(list((run_dir / "checkpoints").glob("step_*.pt")))
            print(f"saved {ckpt_path.name} (pruned {len(deleted)}, kept {kept})")

    final_path = run_dir / "checkpoints" / f"step_{step:08d}_final.pt"
    save_ckpt(final_path, step=step, model=model, ema=ema,
              opt=opt, scaler=scaler, args=args)
    print(f"=== done. final ckpt: {final_path.name} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
