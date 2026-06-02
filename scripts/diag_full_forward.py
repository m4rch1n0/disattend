#!/usr/bin/env python3
"""Comprehensive overflow hunt: hook EVERY module output in a UNet-B forward at
the most-grown available checkpoint and find the single hottest fp16 tensor.

Five NaN runs, onset pushed 78k -> 186k -> 234k by successive fp16 patches.
Attention is now fp32 (autocast-off) yet it still NaN'd at 234k, so the
overflow has moved to a NON-attention op. This probe loads the 200k checkpoint
(closest to the 234k onset), runs many real batches under the exact training
fp16 autocast, and ranks every module by max|activation| / 65504. It also
re-derives the attention scores BOTH ways (fp16 vs fp32) to confirm the fp32
fix is what keeps attention alive while something else grows. Read-only.
"""
from __future__ import annotations

import math
import os
import sys

REPO = "/home/marco/disattend"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "sit"))

import torch
import torch.nn as nn
from src.models.unet_b import UNet_models, AttentionBlock

DEVICE = torch.device("cuda")
FP16_MAX = 65504.0
CKPT = os.path.join(REPO, "experiments/20260601-UNet-B-wd/checkpoints/step_00200000.pt")
N_BATCHES, BS = 64, 16


def main() -> int:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    print(f"ckpt step={ck.get('step')}  "
          f"finite={all(torch.isfinite(v).all().item() for v in sd.values() if v.is_floating_point())}")

    model = UNet_models["UNet-B"](input_size=32, num_classes=1000,
                                  class_dropout_prob=0.1).to(DEVICE).eval()
    model.load_state_dict(sd)

    rec: dict[str, dict] = {}

    def hook(name, kind):
        def h(mod, inp, out):
            outs = out if isinstance(out, (tuple, list)) else [out]
            for o in outs:
                if not torch.is_tensor(o):
                    continue
                f = torch.isfinite(o)
                mx = o[f].abs().max().item() if f.any() else float("inf")
                r = rec.setdefault(name, {"kind": kind, "max": 0.0, "bad": False})
                r["bad"] = r["bad"] or (not f.all().item())
                r["max"] = max(r["max"], mx)
        return h

    # attention fp16-vs-fp32 score check
    attn_chk: dict[str, dict] = {}
    nh_of: dict[str, int] = {}

    def attn_hook(name):
        def h(mod, inp, out):
            B, C3, N = out.shape
            ch = (C3 // 3) // nh_of[name]
            q, k, _ = out.reshape(B, 3, nh_of[name], ch, N).unbind(dim=1)
            s = 1.0 / math.sqrt(math.sqrt(ch))
            fp16 = torch.einsum("bhci,bhcj->bhij", q * s, k * s)            # what autocast would do
            fp32 = torch.einsum("bhci,bhcj->bhij", q.float() * s, k.float() * s)  # current model path
            r = attn_chk.setdefault(name, {"ch": ch, "fp16_bad": False, "fp16_max": 0.0, "fp32_max": 0.0})
            cf = torch.isfinite(fp16)
            r["fp16_bad"] = r["fp16_bad"] or (not cf.all().item())
            if cf.any():
                r["fp16_max"] = max(r["fp16_max"], fp16[cf].abs().max().item())
            r["fp32_max"] = max(r["fp32_max"], fp32.abs().max().item())
        return h

    for n, m in model.named_modules():
        if len(list(m.children())) == 0 and n:        # leaf modules
            m.register_forward_hook(hook(n, type(m).__name__))
        if isinstance(m, AttentionBlock):
            nh_of[n] = m.num_heads
            m.qkv.register_forward_hook(attn_hook(n))

    shard = torch.load(os.path.join(REPO, "data/imagenet_latents/train_0000.pt"),
                       map_location="cpu", weights_only=False)
    lat = shard["latents"] if isinstance(shard, dict) else shard[0]

    torch.manual_seed(1)
    out_max = 0.0
    with torch.inference_mode():
        for _ in range(N_BATCHES):
            idx = torch.randint(0, lat.shape[0], (BS,))
            x = lat[idx].to(DEVICE, dtype=torch.float16)
            t = torch.rand(BS, device=DEVICE)
            y = torch.randint(0, 1000, (BS,), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                o = model(x, t, y)
            out_max = max(out_max, o.float().abs().max().item())

    print(f"\nmodel velocity-output max|.| = {out_max:.1f}  ({out_max/FP16_MAX*100:.0f}% of fp16 ceil)")

    print(f"\n=== TOP-25 hottest module outputs (fp16 ceil={FP16_MAX:.0f}) ===")
    print(f"{'module':44s} {'type':>10s} {'max|act|':>12s} {'%ceil':>7s} {'bad':>5s}")
    for n in sorted(rec, key=lambda k: -rec[k]["max"])[:25]:
        r = rec[n]
        print(f"{n:44s} {r['kind']:>10s} {r['max']:>12.1f} {r['max']/FP16_MAX*100:>6.0f}% {str(r['bad']):>5s}")

    print(f"\n=== attention scores fp16(autocast) vs fp32(current model) ===")
    print(f"{'block':24s} {'ch':>4s} {'fp16 max':>12s} {'fp16 bad':>9s} {'fp32 max':>12s}")
    for n in sorted(attn_chk, key=lambda k: -attn_chk[k]["fp32_max"]):
        r = attn_chk[n]
        print(f"{n:24s} {r['ch']:>4d} {r['fp16_max']:>12.1f} {str(r['fp16_bad']):>9s} {r['fp32_max']:>12.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
