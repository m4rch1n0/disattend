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
import os
import sys
import time
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

    # AMP for fp16 (RDNA 2: bf16 banned, fp32 fallback if no GPU)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

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
        )
        print(f"  ref n={fid_eval.ref_n}")

    permanent_ckpts = {500_000, 1_000_000, 1_500_000, 2_000_000,
                       3_000_000, 4_000_000, 5_000_000, 6_400_000}

    print(f"training to step {args.total_steps}")
    step = start_step
    log_window_loss = 0.0
    log_window_steps = 0
    t_window = time.perf_counter()
    t_global = time.perf_counter()

    loader_iter = iter(loader)
    while step < args.total_steps:
        try:
            z, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            z, y = next(loader_iter)

        z = z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            losses = transport.training_losses(model, z, dict(y=y))
            loss = losses["loss"].mean()

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        update_ema(ema, model, decay=args.ema_decay)

        log_window_loss += loss.item()
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
