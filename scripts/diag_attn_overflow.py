#!/usr/bin/env python3
"""A/B confirmation of the UNet-B fp16 attention overflow root cause.

Loads a UNet-B checkpoint from the sensitive (post-onset) weight regime,
runs real latent batches through the model under fp16 autocast, and for
every AttentionBlock captures the real q,k and computes the pre-softmax
logits TWO ways on the SAME activations:

  OLD: einsum(q, k) * scale            (scale AFTER matmul -> can overflow fp16)
  NEW: einsum(q*sqrt(scale), k*sqrt(scale))   (scale BEFORE matmul, fp16-safe)

If OLD hits inf/nan while NEW stays finite, the attention scale-ordering is
confirmed as the overflow site. Also reports GroupNorm output magnitudes
(secondary suspect) and whether each checkpoint's weights are finite (tests
the 'finite-but-sensitive weights' vs 'NaN weights' hypothesis). Read-only:
no training, no writes.
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

CANDIDATES = [
    ("0529/50k", "experiments/20260529-UNet-B/checkpoints/step_00050000.pt"),
    ("0529/100k", "experiments/20260529-UNet-B/checkpoints/step_00100000.pt"),
    ("0529/150k", "experiments/20260529-UNet-B/checkpoints/step_00150000.pt"),
    ("0531/100k", "experiments/20260531-UNet-B/checkpoints/step_00100000.pt"),
    ("0531/150k", "experiments/20260531-UNet-B/checkpoints/step_00150000.pt"),
]


def main() -> int:
    print("=== checkpoint finiteness + max|qkv.weight| ===")
    finite_ckpts = []
    for name, rel in CANDIDATES:
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            print(f"  {name:10s}: (missing)")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        finite = all(torch.isfinite(v).all().item()
                     for v in sd.values() if v.is_floating_point())
        qkv_max = max((v.abs().max().item() for kk, v in sd.items()
                       if "qkv.weight" in kk), default=0.0)
        print(f"  {name:10s}: step={ck.get('step')}  model_finite={finite}  "
              f"max|qkv.w|={qkv_max:.3f}")
        if finite:
            finite_ckpts.append((name, p, qkv_max, ck))

    if not finite_ckpts:
        print("No finite checkpoint available for the A/B."); return 1
    # pick the finite checkpoint with the LARGEST qkv weights (most sensitive)
    name, path, qkv_max, ck = max(finite_ckpts, key=lambda t: t[2])
    print(f"\n=== A/B on most-sensitive finite ckpt: {name} (max|qkv.w|={qkv_max:.3f}) ===")

    model = UNet_models["UNet-B"](input_size=32, num_classes=1000,
                                  class_dropout_prob=0.1).to(DEVICE).eval()
    model.load_state_dict(ck["model"])

    rec: dict[str, dict] = {}

    def attn_hook(mod_name):
        def hook(mod, inp, out):
            qkv = out  # (B, 3C, N)
            B, C3, N = qkv.shape
            C = C3 // 3
            nh = mod_name_nh[mod_name]
            ch = C // nh
            qkv = qkv.reshape(B, 3, nh, ch, N)
            q, k, _ = qkv.unbind(dim=1)
            old = torch.einsum("bhci,bhcj->bhij", q, k) * (1.0 / math.sqrt(ch))
            s = 1.0 / math.sqrt(math.sqrt(ch))
            new = torch.einsum("bhci,bhcj->bhij", q * s, k * s)
            r = rec.setdefault(mod_name, {"ch": ch, "old_max": 0.0, "old_bad": False,
                                          "new_max": 0.0, "new_bad": False})
            old_finite = torch.isfinite(old)
            new_finite = torch.isfinite(new)
            r["old_bad"] = r["old_bad"] or (not old_finite.all().item())
            r["new_bad"] = r["new_bad"] or (not new_finite.all().item())
            if old_finite.any():
                r["old_max"] = max(r["old_max"], old[old_finite].abs().max().item())
            if new_finite.any():
                r["new_max"] = max(r["new_max"], new[new_finite].abs().max().item())
        return hook

    gn_rec: dict[str, float] = {}

    def gn_hook(mod_name):
        def hook(mod, inp, out):
            f = torch.isfinite(out)
            v = out[f].abs().max().item() if f.any() else float("inf")
            gn_rec[mod_name] = max(gn_rec.get(mod_name, 0.0), v)
        return hook

    mod_name_nh = {}
    for n, m in model.named_modules():
        if isinstance(m, AttentionBlock):
            mod_name_nh[n] = m.num_heads
            m.qkv.register_forward_hook(attn_hook(n))
        if isinstance(m, nn.GroupNorm):
            m.register_forward_hook(gn_hook(n))

    # real latents from one shard
    shard = torch.load(os.path.join(REPO, "data/imagenet_latents/train_0000.pt"),
                       map_location="cpu", weights_only=False)
    lat = shard["latents"] if isinstance(shard, dict) else shard[0]
    print(f"shard latents shape={tuple(lat.shape)} dtype={lat.dtype}")

    torch.manual_seed(0)
    n_batches, bs = 8, 16
    with torch.inference_mode():
        for b in range(n_batches):
            idx = torch.randint(0, lat.shape[0], (bs,))
            x = lat[idx].to(DEVICE, dtype=torch.float16)
            t = torch.rand(bs, device=DEVICE)           # spread over [0,1]
            y = torch.randint(0, 1000, (bs,), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(x, t, y)

    print(f"\n{'attention block':40s} {'ch':>4s} {'OLD max':>12s} {'OLD bad':>8s} "
          f"{'NEW max':>10s} {'NEW bad':>8s}")
    any_old_bad = False
    for n in sorted(rec):
        r = rec[n]
        any_old_bad |= r["old_bad"]
        print(f"{n:40s} {r['ch']:>4d} {r['old_max']:>12.1f} {str(r['old_bad']):>8s} "
              f"{r['new_max']:>10.2f} {str(r['new_bad']):>8s}  (fp16 max={FP16_MAX:.0f})")

    print(f"\nGroupNorm output max|.| (secondary suspect; fp16 max={FP16_MAX:.0f}):")
    for n in sorted(gn_rec, key=lambda k: -gn_rec[k])[:6]:
        print(f"  {n:42s} {gn_rec[n]:.1f}")

    print("\n=== verdict ===")
    print(f"OLD attention produced inf/nan: {any_old_bad}")
    print(f"NEW attention finite everywhere: {not any(rec[n]['new_bad'] for n in rec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
