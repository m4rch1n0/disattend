#!/usr/bin/env python3
"""Find ALL fp16-overflow-prone sites in UNet-B at the most sensitive
available checkpoint, to pick a robust fix instead of another patch.

The attention scale-before fix pushed the NaN onset from ~85k to ~186k but
did not eliminate it: the QK^T einsum is still computed in fp16, so larger
q,k eventually overflow again. This probe, on the latest finite checkpoint
(closest to the 186k onset), measures under fp16 autocast:
  - per AttentionBlock: max logit with the CURRENT fix (fp16 einsum) vs an
    fp32-upcast einsum -> how close is fp16 to 65504, does fp32 stay finite;
  - GLOBAL per-module max-abs activation -> the single hottest tensor in the
    forward, whatever its type (attention / GroupNorm / conv / residual).
If attention is the ONLY thing near 65504 and everything else is far below,
fp32 attention scores suffice. If many ops climb toward 65504, the root is
unbounded fp16 growth (wd=0) and only full fp32 / weight decay fixes it.
Read-only: no training, no writes.
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
CKPT = os.path.join(REPO, "experiments/20260531-UNet-B-fix/checkpoints/step_00150000.pt")


def main() -> int:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    finite = all(torch.isfinite(v).all().item() for v in sd.values() if v.is_floating_point())
    qkv_max = max((v.abs().max().item() for kk, v in sd.items() if "qkv.weight" in kk), default=0.0)
    print(f"ckpt step={ck.get('step')}  model_finite={finite}  max|qkv.w|={qkv_max:.3f}")

    model = UNet_models["UNet-B"](input_size=32, num_classes=1000,
                                  class_dropout_prob=0.1).to(DEVICE).eval()
    model.load_state_dict(sd)

    attn_rec: dict[str, dict] = {}
    nh_of: dict[str, int] = {}

    def attn_hook(name):
        def hook(mod, inp, out):
            B, C3, N = out.shape
            C = C3 // 3
            ch = C // nh_of[name]
            qkv = out.reshape(B, 3, nh_of[name], ch, N)
            q, k, _ = qkv.unbind(dim=1)
            s = 1.0 / math.sqrt(math.sqrt(ch))
            cur = torch.einsum("bhci,bhcj->bhij", q * s, k * s)            # current fix (fp16)
            f32 = torch.einsum("bhci,bhcj->bhij", q.float() * s, k.float() * s)  # fp32 upcast
            r = attn_rec.setdefault(name, {"ch": ch, "cur_max": 0.0, "cur_bad": False, "f32_max": 0.0})
            cf = torch.isfinite(cur)
            r["cur_bad"] = r["cur_bad"] or (not cf.all().item())
            if cf.any():
                r["cur_max"] = max(r["cur_max"], cur[cf].abs().max().item())
            r["f32_max"] = max(r["f32_max"], f32.abs().max().item())
        return hook

    glob: dict[str, dict] = {}

    def glob_hook(name, kind):
        def hook(mod, inp, out):
            if not torch.is_tensor(out):
                return
            f = torch.isfinite(out)
            mx = out[f].abs().max().item() if f.any() else float("inf")
            r = glob.setdefault(name, {"kind": kind, "max": 0.0, "bad": False})
            r["bad"] = r["bad"] or (not f.all().item())
            r["max"] = max(r["max"], mx)
        return hook

    for n, m in model.named_modules():
        if isinstance(m, AttentionBlock):
            nh_of[n] = m.num_heads
            m.qkv.register_forward_hook(attn_hook(n))
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.GroupNorm)):
            m.register_forward_hook(glob_hook(n, type(m).__name__))

    shard = torch.load(os.path.join(REPO, "data/imagenet_latents/train_0000.pt"),
                       map_location="cpu", weights_only=False)
    lat = shard["latents"] if isinstance(shard, dict) else shard[0]

    torch.manual_seed(0)
    with torch.inference_mode():
        for _ in range(12):
            idx = torch.randint(0, lat.shape[0], (16,))
            x = lat[idx].to(DEVICE, dtype=torch.float16)
            t = torch.rand(16, device=DEVICE)
            y = torch.randint(0, 1000, (16,), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(x, t, y)

    print(f"\n=== attention logits @150k (fp16 max={FP16_MAX:.0f}) ===")
    print(f"{'block':24s} {'ch':>4s} {'CUR(fp16) max':>14s} {'CUR bad':>8s} {'fp32 max':>12s}")
    for n in sorted(attn_rec, key=lambda k: -attn_rec[k]["cur_max"]):
        r = attn_rec[n]
        frac = r["cur_max"] / FP16_MAX
        print(f"{n:24s} {r['ch']:>4d} {r['cur_max']:>14.1f} {str(r['cur_bad']):>8s} "
              f"{r['f32_max']:>12.1f}   ({frac*100:.0f}% of fp16 ceil)")

    print(f"\n=== TOP-12 hottest activations across ALL conv/GN modules ===")
    print(f"{'module':40s} {'type':>10s} {'max|act|':>12s} {'%ceil':>7s} {'bad':>5s}")
    for n in sorted(glob, key=lambda k: -glob[k]["max"])[:12]:
        r = glob[n]
        print(f"{n:40s} {r['kind']:>10s} {r['max']:>12.1f} {r['max']/FP16_MAX*100:>6.0f}% {str(r['bad']):>5s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
