"""FID evaluation against precomputed ImageNet val reference statistics.

Pipeline:
  1. Sample N latents z_T ~ N(0, I), labels y ~ Uniform(1000).
  2. Integrate z_T -> z_0 with Euler ODE on the velocity model (deterministic
     and differentiable, matching the SiT convention t=0 data, t=1 noise).
  3. VAE.decode(z_0 / 0.18215) -> images in [-1, 1].
  4. Map to [0, 1], pass through pytorch-fid InceptionV3 (which normalizes
     and resizes internally) to get 2048-dim features.
  5. Compute (mu_g, sigma_g) and return FID against (mu_r, sigma_r).

Reference (mu_r, sigma_r) are precomputed once by
scripts/precompute_fid_ref.py on the full ImageNet validation set
(~50k images) and saved to data/imagenet_latents/fid_ref_stats.pt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


VAE_REPO = "stabilityai/sd-vae-ft-ema"
SCALING_FACTOR = 0.18215
INCEPTION_DIM = 2048
NUM_CLASSES = 1000


@torch.inference_mode()
def euler_ode_sample(model, z_T, y, n_steps: int = 25, t_min: float = 1e-3):
    """Deterministic Euler ODE from t=1 (noise) to t=t_min (data).

    Convention: SiT velocity model with t in [0, 1], v = dx/dt of the linear
    interpolant. `t_min > 0` avoids the slight instability at exactly t=0.
    """
    ts = torch.linspace(1.0, t_min, n_steps + 1, device=z_T.device)
    x = z_T
    for i in range(n_steps):
        dt = ts[i + 1] - ts[i]  # negative, integrating downward in t
        t_batch = ts[i].expand(x.shape[0])
        v = model(x, t_batch, y)
        x = x + dt * v
    return x


def load_inception(device: torch.device) -> InceptionV3:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_DIM]
    net = InceptionV3([block_idx]).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def inception_features(net: InceptionV3,
                       images_01: torch.Tensor) -> torch.Tensor:
    """images_01: (B, 3, H, W) in [0, 1]. Returns (B, 2048) features.

    pytorch-fid InceptionV3 normalizes and resizes (to 299) internally
    when constructed with default normalize_input=True, resize_input=True.
    """
    feats = net(images_01)[0]
    if feats.ndim == 4:
        feats = feats.squeeze(-1).squeeze(-1)
    return feats


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def fid_from_stats(mu_r: np.ndarray, sigma_r: np.ndarray,
                   mu_g: np.ndarray, sigma_g: np.ndarray) -> float:
    return float(calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g))


class FIDEvaluator:
    """Holds VAE + Inception + reference statistics; evaluates a model."""

    def __init__(
        self,
        device: torch.device,
        ref_stats_path: Path,
        vae: AutoencoderKL | None = None,
        n_samples: int = 5000,
        n_steps: int = 25,
        sample_batch: int = 16,
        seed: int = 42,
    ):
        self.device = device
        self.n_samples = int(n_samples)
        self.n_steps = int(n_steps)
        self.sample_batch = int(sample_batch)
        self.seed = int(seed)

        if vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                VAE_REPO, torch_dtype=torch.float16
            ).to(device).eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)
        else:
            self.vae = vae

        self.inception = load_inception(device)

        ref = torch.load(ref_stats_path, weights_only=True)
        self.mu_r = ref["mu"].numpy() if torch.is_tensor(ref["mu"]) else ref["mu"]
        self.sigma_r = (ref["sigma"].numpy()
                        if torch.is_tensor(ref["sigma"]) else ref["sigma"])
        self.ref_n = int(ref.get("n", 0))

    @torch.inference_mode()
    def _decode_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """latents (B, 4, 32, 32) fp32 or fp16 on device -> images (B, 3, H, W)
        in [0, 1] fp32."""
        z = latents.to(dtype=torch.float16) / SCALING_FACTOR
        imgs = self.vae.decode(z).sample
        imgs = ((imgs.float() + 1) / 2).clamp(0, 1)
        return imgs

    @torch.inference_mode()
    def evaluate(self, model: torch.nn.Module) -> dict:
        """Run sampling + FID on `model`. Caller usually passes the EMA model."""
        was_training = model.training
        model.eval()
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        feats_chunks: list[torch.Tensor] = []
        n_done = 0
        while n_done < self.n_samples:
            B = min(self.sample_batch, self.n_samples - n_done)
            z = torch.randn(B, 4, 32, 32, device=self.device, generator=gen)
            y = torch.randint(0, NUM_CLASSES, (B,), device=self.device,
                              generator=gen)
            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                z0 = euler_ode_sample(model, z, y, n_steps=self.n_steps)
            imgs = self._decode_to_images(z0)
            feats = inception_features(self.inception, imgs)
            feats_chunks.append(feats.cpu())
            n_done += B

        if was_training:
            model.train()

        feats = torch.cat(feats_chunks, dim=0).numpy()
        mu_g, sigma_g = compute_stats(feats)
        fid = fid_from_stats(self.mu_r, self.sigma_r, mu_g, sigma_g)
        return {
            "fid": fid,
            "n_samples": int(self.n_samples),
            "n_steps_ode": int(self.n_steps),
            "ref_n": self.ref_n,
        }
