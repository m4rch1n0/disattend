"""ADM-style class-conditional UNet for latent flow matching.

Reference: Dhariwal & Nichol 2021, "Diffusion Models Beat GANs", and the
guided-diffusion implementation at github.com/openai/guided-diffusion.

Project-specific adaptations:
  - Operates in latent space: 4 channels, 32x32 (post-VAE).
  - Outputs velocity (4 channels), not noise+sigma. The original
    learn_sigma path is dropped because flow matching predicts v directly.
  - forward(x, t, y) signature matches SiT so the same transport and
    training loop work without branches.
  - Continuous t in [0, 1] (SiT convention) handled by the sinusoidal
    embedding without scaling.
  - Classifier-free guidance dropout matches SiT: a class is replaced
    with the null index (== num_classes) with probability class_dropout_prob.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-init a module's parameters in place. Used on the last conv of
    each ResBlock and the final output conv (paper init recipe)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels: int) -> nn.Module:
    """GroupNorm with 32 groups when possible, else as many groups as channels."""
    return nn.GroupNorm(min(32, channels), channels)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP. Identical to SiT's."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args_ = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Embedding with CFG dropout. Index `num_classes` is the null class."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor,
                   force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop = force_drop_ids == 1
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: torch.Tensor, train: bool,
                force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if (train and self.dropout_prob > 0.0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class ResBlock(nn.Module):
    """ADM residual block with time/class conditioning via additive embedding."""

    def __init__(self, in_channels: int, out_channels: int,
                 emb_channels: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over the spatial axis of a 4D tensor."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)
        qkv = self.qkv(h)
        ch = C // self.num_heads
        qkv = qkv.reshape(B, 3, self.num_heads, ch, H * W)
        q, k, v = qkv.unbind(dim=1)
        # fp16-safe attention. Compute the scores (QK^T + softmax) in fp32 with
        # autocast DISABLED so they cannot overflow fp16's 65504 ceiling no
        # matter how large q,k grow. The earlier scale-before-matmul + fp32
        # softmax only *delayed* the overflow (onset 85k -> 186k) because the
        # einsum still ran in fp16 under autocast; a plain .float() is not enough
        # either, since autocast re-casts einsum inputs back to fp16. The AV
        # matmul stays fp16. (SiT/DiT never hit this: timm Attention is already
        # fp32-safe. A small weight_decay additionally bounds the conv weights so
        # no fp16 activation drifts toward the ceiling over a long run.)
        scale = 1.0 / math.sqrt(math.sqrt(ch))
        with torch.autocast(device_type=q.device.type, enabled=False):
            attn = torch.einsum("bhci,bhcj->bhij", q.float() * scale, k.float() * scale)
            attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn.to(v.dtype), v).reshape(B, C, H * W)
        out = self.proj_out(out).reshape(B, C, H, W)
        return out + x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TimestepEmbedSequential(nn.Sequential):
    """Sequential where ResBlocks receive (x, emb) and others just (x)."""

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """ADM-style UNet, class-conditional, velocity output."""

    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 4,
        model_channels: int = 192,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        attention_resolutions: Iterable[int] = (16, 8),
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # velocity has same shape as latent
        self.model_channels = model_channels
        self.input_size = input_size
        attention_resolutions = set(attention_resolutions)

        emb_channels = model_channels * 4
        self.t_embedder = TimestepEmbedder(emb_channels)
        self.y_embedder = LabelEmbedder(num_classes, emb_channels, class_dropout_prob)

        # Encoder (input blocks)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))]
        )
        ch = model_channels
        ds = 1
        input_chans: list[int] = [model_channels]
        for level, mult in enumerate(channel_mult):
            out_ch = mult * model_channels
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResBlock(ch, out_ch, emb_channels, dropout)]
                ch = out_ch
                if input_size // ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_chans.append(ch)
                ds *= 2

        # Bottleneck
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, ch, emb_channels, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, emb_channels, dropout),
        )

        # Decoder (output blocks)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = mult * model_channels
            for i in range(num_res_blocks + 1):
                skip_ch = input_chans.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, emb_channels, dropout)]
                ch = out_ch
                if input_size // ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (N, 4, 32, 32), t: (N,), y: (N,) class labels."""
        emb = self.t_embedder(t) + self.y_embedder(y, self.training)
        h = x
        skips: list[torch.Tensor] = []
        for module in self.input_blocks:
            h = module(h, emb)
            skips.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)


# ----- Config registry, mirroring SiT_models in willisma/sit -----

def UNet_B(**kwargs) -> UNet:
    """ADM-B target ~130M params (matches SiT-B/2 within tolerance).
    160 base channels, 4 levels (1,2,3,4)."""
    return UNet(
        model_channels=160,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        num_heads=4,
        **kwargs,
    )


def UNet_S(**kwargs) -> UNet:
    """ADM-S smaller fallback ~50M params."""
    return UNet(
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        num_heads=4,
        **kwargs,
    )


UNet_models = {
    "UNet-B": UNet_B,
    "UNet-S": UNet_S,
}
