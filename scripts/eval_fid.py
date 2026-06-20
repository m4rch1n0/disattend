#!/usr/bin/env python3
"""Standalone FID evaluation on a saved checkpoint (EMA weights).

Usage:
    python scripts/eval_fid.py \
        --checkpoint experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt \
        --model SiT-B/2 \
        --n-steps 50 \
        --n-samples 5000 \
        --out experiments/20260520-SiT-B-2-recovery/fid_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
for _p in (str(REPO_ROOT), str(SIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from models import SiT_models
from src.models.unet_b import UNet_models
from src.evaluation.fid import FIDEvaluator

MODEL_REGISTRY = {**SiT_models, **UNet_models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="SiT-B/2")
    ap.add_argument("--latent-size", type=int, default=32)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--class-dropout-prob", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--sample-batch", type=int, default=16)
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16", "float32"],
                    default="float16",
                    help="autocast dtype for the velocity-model sampling. "
                         "DiT-B used float16 (fits fp16 range). UNet-B must NOT "
                         "use float16 (activation overflow -> NaN/garbage FID); "
                         "use float32 (canonical, overflow-safe on any GPU) or "
                         "bfloat16 (NVIDIA). For a matched DiT-vs-UNet comparison "
                         "run both in float32.")
    ap.add_argument("--ref-stats", default="data/imagenet_latents/fid_ref_stats.pt")
    ap.add_argument("--out", default=None,
                    help="jsonl file to append result to (default: print only)")
    args = ap.parse_args()

    device = torch.device("cuda")

    model = MODEL_REGISTRY[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    ema.load_state_dict(ckpt["ema"])
    step = ckpt.get("step", -1)
    print(f"loaded checkpoint step={step} from {args.checkpoint}")

    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.amp_dtype]
    evaluator = FIDEvaluator(
        device=device,
        ref_stats_path=Path(args.ref_stats),
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        sample_batch=args.sample_batch,
        amp_dtype=amp_dtype,
    )
    print(f"running FID: n_samples={args.n_samples} n_steps={args.n_steps} ...")
    t0 = time.time()
    result = evaluator.evaluate(ema)
    elapsed = time.time() - t0

    record = {
        "step": step,
        "checkpoint": args.checkpoint,
        "fid": result["fid"],
        "n_samples": result["n_samples"],
        "n_steps_ode": result["n_steps_ode"],
        "elapsed_s": round(elapsed, 1),
        "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(record))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
