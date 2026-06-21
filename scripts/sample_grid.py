#!/usr/bin/env python3
"""Generate a sample-image grid from a trained checkpoint (EMA weights).

Inference only -> runs on the RX 6900 XT (ROCm, device 'cuda'). Use
--amp-dtype float32 for UNet-B (fp16 overflows, the reason it was trained in
bf16). Optional classifier-free guidance via --cfg-scale: both models were
trained with class-dropout, so the null token is the index `num_classes`
(=1000) of the label embedding, applied manually here (works for SiT and the
ADM UNet alike, no model-specific forward_with_cfg needed).

Examples:
    # UNet-B, CFG 2.0, 50 ODE steps, fp32
    python scripts/sample_grid.py \
        --checkpoint experiments/20260611-UNet-B-cosine-6p4M/checkpoints/step_06400000_final.pt \
        --model UNet-B --amp-dtype float32 --cfg-scale 2.0 --n-steps 50 \
        --out notebooks/out/grid_unet_cfg2.png

    # DiT-B, no guidance (matches the FID protocol)
    python scripts/sample_grid.py \
        --checkpoint experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt \
        --model SiT-B/2 --amp-dtype float32 --cfg-scale 1.0 \
        --out notebooks/out/grid_dit.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
for _p in (str(REPO_ROOT), str(SIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

from models import SiT_models
from src.models.unet_b import UNet_models
from src.evaluation.fid import VAE_REPO, SCALING_FACTOR

MODEL_REGISTRY = {**SiT_models, **UNet_models}
NUM_CLASSES = 1000


@torch.inference_mode()
def sample(model, z, y, n_steps, cfg_scale, amp_dtype, device):
    """Deterministic Euler ODE t=0 (noise) -> t=1 (data), matching fid.py.

    cfg_scale==1.0 -> plain conditional. Otherwise classifier-free guidance:
    v = v_uncond + cfg_scale * (v_cond - v_uncond), with the uncond pass using
    the null label (index NUM_CLASSES).
    """
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    x = z
    use_cfg = bool(cfg_scale) and cfg_scale != 1.0
    y_null = torch.full_like(y, NUM_CLASSES)
    autocast_on = amp_dtype != torch.float32
    for i in range(n_steps):
        dt = ts[i + 1] - ts[i]
        t = ts[i].expand(x.shape[0])
        with torch.autocast(device.type, dtype=amp_dtype, enabled=autocast_on):
            if use_cfg:
                v = model(torch.cat([x, x], 0),
                          torch.cat([t, t], 0),
                          torch.cat([y, y_null], 0))
                v_cond, v_uncond = v.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, y)
        x = x + dt * v.float()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="UNet-B",
                    help="MODEL_REGISTRY key, e.g. 'UNet-B' or 'SiT-B/2'")
    ap.add_argument("--latent-size", type=int, default=32)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--class-dropout-prob", type=float, default=0.1,
                    help="must be >0 to load a ckpt whose label embedding has "
                         "the +1 null row")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n", type=int, default=16, help="number of samples")
    ap.add_argument("--nrow", type=int, default=4)
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="1.0 = no guidance; 1.5-4.0 sharpens object semantics")
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16", "float32"],
                    default="float32",
                    help="UNet-B MUST use float32 (fp16 overflow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", type=str, default=None,
                    help="comma-separated ImageNet class ids (cycled to fill "
                         "the grid); default = random")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.amp_dtype]

    model = MODEL_REGISTRY[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device).eval()
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["ema"])
    print(f"loaded {args.model} EMA @ step {ckpt.get('step')} from {args.checkpoint}")

    vae = AutoencoderKL.from_pretrained(
        VAE_REPO, torch_dtype=torch.float16).to(device).eval()

    gen = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(args.n, 4, args.latent_size, args.latent_size,
                    device=device, generator=gen)
    if args.classes:
        ids = [int(c) for c in args.classes.split(",")]
        y = torch.tensor((ids * args.n)[:args.n], device=device)
    else:
        y = torch.randint(0, NUM_CLASSES, (args.n,), device=device, generator=gen)

    z0 = sample(model, z, y, args.n_steps, args.cfg_scale, amp_dtype, device)
    imgs = vae.decode(z0.to(torch.float16) / SCALING_FACTOR).sample
    imgs = ((imgs.float() + 1) / 2).clamp(0, 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(imgs, str(out), nrow=args.nrow)
    print(f"saved {out}  (n={args.n} cfg={args.cfg_scale} "
          f"nfe={args.n_steps} {args.amp_dtype})")


if __name__ == "__main__":
    main()
