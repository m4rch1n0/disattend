"""Row-wise softmax energy of attention logits (exploratory probe).

The attention map softmax(L) with L = Q K^T / sqrt(d_h) is invariant to a
per-row constant shift of L: the normalizing denominator sum_j exp(L_ij) is
the degree of freedom the post-softmax map cannot see. Following Masi et al.
(arXiv 2407.06315), where untargeted PGD makes classifier inputs LOWER-energy
in the EBM sense (energy = -logsumexp of the logits, low energy = more
in-distribution) -- i.e. it RAISES the softmax denominator -- this module
measures the same quantity on attention rows:

    energy_i = logsumexp_j L_ij          (one value per query row)

per layer, per head, per sampling step, reduced to row means. Also kept:
row_max = max_j L_ij (the peak logit; energy = peak + a log tail mass, so the
pair says whether an energy move comes from the peak or from the tail).

Q and K are extracted exactly as src.utils.attention_hooks.AttentionCollector
does (same qkv hook, same reshape and 1/sqrt(d_h) scaling; that recomputation
was verified to match the models' internal attention). The UNet's fp16-safe
scale-before-matmul reordering is multiplicatively identical to L, so the
logits here are the models' effective logits in both backbones. Everything is
reduced inside the hook on GPU; only (B, H) means are stored, never N x N.

This is a post-hoc exploratory measurement, outside the pre-registered
Phase-3 family; it is analyzed descriptively.

Self-test: python -m src.evaluation.softmax_energy
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from src.utils.attention_hooks import discover_attention_layers

ENERGY_KEYS = ("row_lse", "row_max")

Stacked = dict[str, dict[str, torch.Tensor]]   # layer -> key -> (T, B, H)


def _reduce_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """(B, H, N, N) pre-softmax logits -> {'row_lse','row_max': (B, H)} fp32."""
    l32 = logits.float()
    return {
        "row_lse": torch.logsumexp(l32, dim=-1).mean(dim=-1),
        "row_max": l32.amax(dim=-1).mean(dim=-1),
    }


class EnergyCollector:
    """Capture per-layer row-energy statistics of attention logits.

    Same lifecycle as AttentionCollector: attach()/detach() or context
    manager, snapshot() after each forward returns
    {layer: {'row_lse': (B, H), 'row_max': (B, H), 'n_tokens': N}} on CPU
    and clears the internal state.

    keep_softmax=True additionally stores the recomputed softmax map per
    layer (fp32 CPU) in .softmax_maps; only for fidelity checks against
    AttentionCollector, never in production runs.
    """

    def __init__(self, model: nn.Module,
                 layer_filter: Iterable[str] | None = None,
                 keep_softmax: bool = False):
        self.model = model
        self.keep_softmax = keep_softmax
        all_specs = discover_attention_layers(model)
        if layer_filter is None:
            self._specs = all_specs
        else:
            patterns = tuple(layer_filter)
            self._specs = [s for s in all_specs if any(p in s.name for p in patterns)]
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._stats: dict[str, dict[str, torch.Tensor]] = {}
        self.softmax_maps: dict[str, torch.Tensor] = {}

    def attach(self) -> None:
        if self._hooks:
            return
        for spec in self._specs:
            qkv = self.model.get_submodule(spec.name).qkv
            self._hooks.append(qkv.register_forward_hook(self._make_hook(spec)))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "EnergyCollector":
        self.attach()
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    def snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        out = dict(self._stats)
        self._stats = {}
        return out

    def _store(self, name: str, logits: torch.Tensor, n_tokens: int) -> None:
        stats = {k: v.detach().cpu() for k, v in _reduce_logits(logits).items()}
        stats["n_tokens"] = torch.tensor(n_tokens)
        self._stats[name] = stats
        if self.keep_softmax:
            self.softmax_maps[name] = logits.float().softmax(dim=-1).detach().cpu()

    def _make_hook(self, spec):
        nh, hd = spec.num_heads, spec.head_dim
        scale = 1.0 / math.sqrt(hd)

        if spec.kind == "linear_qkv":
            def hook(module, args, output):
                B, N, _ = output.shape
                qkv = output.reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                self._store(spec.name, (q @ k.transpose(-2, -1)) * scale, N)
        elif spec.kind == "conv1d_qkv":
            def hook(module, args, output):
                B, N = output.shape[0], output.shape[2]
                qkv = output.reshape(B, 3, nh, hd, N)
                q, k = qkv[:, 0], qkv[:, 1]
                self._store(spec.name,
                            torch.einsum("bhci,bhcj->bhij", q, k) * scale, N)
        else:
            raise ValueError(f"unknown kind {spec.kind!r}")
        return hook

    def __len__(self) -> int:
        return len(self._specs)


def stack_energy_steps(steps: list[dict]) -> Stacked:
    """Stack per-step snapshots along a leading T axis: (T, B, H) per key."""
    if not steps:
        raise ValueError("no steps to stack")
    out: Stacked = {}
    for name in steps[0]:
        out[name] = {k: torch.stack([s[name][k] for s in steps], dim=0)
                     for k in ENERGY_KEYS}
        out[name]["n_tokens"] = steps[0][name]["n_tokens"]
    return out


def aggregate_energy(stacked: Stacked, key: str, steps: list[int] | None,
                     token_filter: int | None = 256) -> torch.Tensor:
    """Flat mean over (locus layer, step window, head) -> (B,) per sample."""
    per = []
    for name, d in stacked.items():
        if token_filter is not None and int(d["n_tokens"]) != token_filter:
            continue
        x = d[key] if steps is None else d[key][steps]
        per.append(x.permute(1, 0, 2).flatten(1))
    if not per:
        raise ValueError(f"no layers match token_filter={token_filter}")
    return torch.cat(per, dim=1).mean(dim=1).double()


if __name__ == "__main__":
    # Closed forms. Uniform logits (all c): row_lse = c + log N, row_max = c.
    B, H, N, c = 2, 3, 64, 1.7
    r = _reduce_logits(torch.full((B, H, N, N), c))
    assert torch.allclose(r["row_lse"], torch.full((B, H), c + math.log(N)),
                          atol=1e-5), r["row_lse"]
    assert torch.allclose(r["row_max"], torch.full((B, H), c), atol=1e-6)

    # Diagonal a, off-diagonal b: row_lse = log(exp(a) + (N-1) exp(b)).
    a, b = 3.0, -1.0
    L = torch.full((1, 1, N, N), b) + (a - b) * torch.eye(N)
    want = math.log(math.exp(a) + (N - 1) * math.exp(b))
    r = _reduce_logits(L)
    assert torch.allclose(r["row_lse"], torch.tensor(want), atol=1e-5)
    assert torch.allclose(r["row_max"], torch.tensor(a), atol=1e-6)

    # The captured degree of freedom: a per-row shift moves the energy by
    # exactly that shift and the softmax not at all.
    torch.manual_seed(0)
    L = torch.randn(B, H, N, N)
    shift = 0.83
    r0, r1 = _reduce_logits(L), _reduce_logits(L + shift)
    assert torch.allclose(r1["row_lse"] - r0["row_lse"],
                          torch.full((B, H), shift), atol=1e-5)
    assert torch.allclose(L.softmax(-1), (L + shift).softmax(-1), atol=1e-6)

    # stack + aggregate shapes and window slicing
    snap = {"lay256": {**_reduce_logits(L), "n_tokens": torch.tensor(256)},
            "lay16": {**_reduce_logits(L), "n_tokens": torch.tensor(16)}}
    st = stack_energy_steps([snap, snap, snap])
    assert st["lay256"]["row_lse"].shape == (3, B, H)
    agg = aggregate_energy(st, "row_lse", steps=[1, 2], token_filter=256)
    assert agg.shape == (B,)
    print("softmax_energy self-test: OK")
