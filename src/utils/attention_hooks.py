"""Capture self-attention softmax maps from DiT (SiT) and UNet during inference.

Both backbones contain self-attention layers we will inspect in Phase 2-3:
  * SiT/DiT: 12 timm Attention modules at `model.blocks[i].attn`. They
    compute `softmax(Q @ K^T / sqrt(d_h))` inline and discard it (only the
    output `attn @ v` survives the forward).
  * UNet:    AttentionBlock instances at fixed resolutions (16, 8 in our
    UNet-B config plus the middle block at the bottleneck). Same story:
    the softmax is intermediate.

Strategy: register a forward hook on the `qkv` projection. From that we
have Q and K (post any q_norm/k_norm, which are Identity in our SiT
config because qk_norm=False is the default in timm and SiT does not
opt in). We recompute the softmax matrix; the math is identical to what
the layer did internally, just at the cost of a single extra matmul +
softmax per layer. Output is moved to fp16 CPU to free VRAM eagerly.

Usage (Phase 2 sampling will call this in a loop over timesteps):

    collector = AttentionCollector(model)
    collector.attach()
    with torch.inference_mode():
        out = model(x, t, y)
        maps_t = collector.snapshot()    # dict[str -> Tensor (B, H, N, N) fp16]
    collector.detach()

The keys of `maps_t` are dotted module paths, e.g. `blocks.0.attn` for
SiT or `output_blocks.5.1` for UNet (path through TimestepEmbedSequential).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


SIT_ATTENTION_CLASS = "Attention"          # timm.models.vision_transformer.Attention
UNET_ATTENTION_CLASS = "AttentionBlock"    # src.models.unet_b.AttentionBlock


@dataclass(frozen=True)
class _LayerSpec:
    name: str
    kind: str          # "linear_qkv" (SiT) or "conv1d_qkv" (UNet)
    num_heads: int
    head_dim: int


def discover_attention_layers(model: nn.Module) -> list[_LayerSpec]:
    """Walk the model and return one _LayerSpec per self-attention layer."""
    specs: list[_LayerSpec] = []
    for name, m in model.named_modules():
        cls = type(m).__name__
        qkv = getattr(m, "qkv", None)
        nh = getattr(m, "num_heads", None)
        if qkv is None or nh is None:
            continue
        if cls == SIT_ATTENTION_CLASS and isinstance(qkv, nn.Linear):
            head_dim = qkv.out_features // (3 * nh)
            specs.append(_LayerSpec(name, "linear_qkv", nh, head_dim))
        elif cls == UNET_ATTENTION_CLASS and isinstance(qkv, nn.Conv1d):
            head_dim = qkv.out_channels // (3 * nh)
            specs.append(_LayerSpec(name, "conv1d_qkv", nh, head_dim))
    return specs


class AttentionCollector:
    """Capture per-layer softmax(QK^T / sqrt(d_h)) on each forward."""

    def __init__(
        self,
        model: nn.Module,
        layer_filter: Iterable[str] | None = None,
        store_dtype: torch.dtype = torch.float16,
        store_device: str | torch.device = "cpu",
    ):
        """
        Args:
            model: SiT or UNet (or any nn.Module containing matching attention).
            layer_filter: optional iterable of substrings; only layers whose
                name contains any one substring are hooked. None = hook all.
            store_dtype: dtype to cast captured maps to (fp16 by default).
            store_device: where captured maps live (CPU by default; pin to
                a CUDA device only if you need on-GPU access immediately).
        """
        self.model = model
        self.store_dtype = store_dtype
        self.store_device = torch.device(store_device)
        all_specs = discover_attention_layers(model)
        if layer_filter is None:
            self._specs = all_specs
        else:
            patterns = tuple(layer_filter)
            self._specs = [s for s in all_specs if any(p in s.name for p in patterns)]
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._maps: dict[str, torch.Tensor] = {}

    # ---------- lifecycle ----------

    @property
    def specs(self) -> list[_LayerSpec]:
        return list(self._specs)

    def attach(self) -> None:
        if self._hooks:
            return
        for spec in self._specs:
            qkv = self.model.get_submodule(spec.name).qkv
            handle = qkv.register_forward_hook(self._make_hook(spec))
            self._hooks.append(handle)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "AttentionCollector":
        self.attach()
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    # ---------- capture ----------

    def clear(self) -> None:
        self._maps.clear()

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return current maps (dict copy) and clear internal state.

        Each value is a tensor of shape (B, num_heads, N, N) where N depends
        on the layer's spatial size: 256 for SiT-B/2 patch 2, and 256 / 64
        / 16 for UNet attention at resolution 16 / 8 / 4.
        """
        out = dict(self._maps)
        self._maps.clear()
        return out

    def _make_hook(self, spec: _LayerSpec):
        nh, hd = spec.num_heads, spec.head_dim
        scale = 1.0 / math.sqrt(hd)
        store_dtype = self.store_dtype
        store_device = self.store_device

        if spec.kind == "linear_qkv":
            def hook(module, args, output):
                # output: (B, N, 3*nh*hd)
                B, N, _ = output.shape
                qkv = output.reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)            # (B, nh, N, hd)
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                self._maps[spec.name] = attn.detach().to(
                    dtype=store_dtype, device=store_device
                )
        elif spec.kind == "conv1d_qkv":
            def hook(module, args, output):
                # output: (B, 3*nh*hd, N)  with N = H*W
                B = output.shape[0]
                N = output.shape[2]
                qkv = output.reshape(B, 3, nh, hd, N)
                q = qkv[:, 0]          # (B, nh, hd, N)
                k = qkv[:, 1]
                # attn: (B, nh, N, N)  -- queries over keys
                attn = torch.einsum("bhci,bhcj->bhij", q, k) * scale
                attn = attn.softmax(dim=-1)
                self._maps[spec.name] = attn.detach().to(
                    dtype=store_dtype, device=store_device
                )
        else:
            raise ValueError(f"unknown kind {spec.kind!r}")

        return hook

    # ---------- introspection ----------

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return (
            f"AttentionCollector(model={type(self.model).__name__}, "
            f"n_layers={len(self._specs)}, "
            f"store_dtype={self.store_dtype}, store_device={self.store_device})"
        )
