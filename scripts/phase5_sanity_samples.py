#!/usr/bin/env python3
"""Phase 5 test 2: decode a few class-conditional samples from the official SiT
checkpoints with OUR sampler (25 Euler steps, Linear-velocity convention) to
prove that the state dict loads and the sampling convention is right.
Recognizable images of the requested classes = pass. Writes one PNG grid per
model under experiments/phase5_pretrained/sanity/.

  .venv/bin/python scripts/phase5_sanity_samples.py --models SiT-B/2-official SiT-XL/2-official
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

from src.attacks.pgd_latent import sample_with_checkpoint
from src.evaluation.fid import VAE_REPO
from scripts.run_phase2_pilot import decode_images
from scripts.run_phase5 import CKPTS, MODEL_KEY, SLUG, load_model_any

# golden retriever, macaw, red panda, arctic fox, otter, hot-air balloon, geyser, valley
CLASSES = [207, 88, 387, 279, 360, 417, 974, 979]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["SiT-B/2-official", "SiT-XL/2-official"],
                    choices=list(CKPTS.keys()))
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--n-steps-ode", type=int, default=25)
    ap.add_argument("--out-dir", default="experiments/phase5_pretrained/sanity")
    args = ap.parse_args()
    device = torch.device("cuda")
    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float16).to(device).eval()
    y = torch.tensor(CLASSES, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(len(CLASSES), 4, 32, 32, device=device, generator=g)
    for name in args.models:
        model = load_model_any(MODEL_KEY[name], REPO_ROOT / CKPTS[name], device)
        with torch.no_grad():
            z0 = sample_with_checkpoint(model, z, y, n_steps=args.n_steps_ode,
                                        cfg_scale=args.cfg_scale, use_checkpoint=False)
        imgs = (decode_images(vae, z0) + 1) / 2
        path = out / f"{SLUG[name]}_cfg{args.cfg_scale}.png"
        save_image(imgs, path, nrow=4)
        print(f"{name}: z0 std {z0.std().item():.3f}, wrote {path}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
