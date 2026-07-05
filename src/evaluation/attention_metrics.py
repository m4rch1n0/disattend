"""Attention metrics A (entropy) and B (effective rank) for Phase 2.

Consumes the measurement dicts produced by
src.utils.attention_hooks.AttentionCollector.snapshot(): {layer_name:
Tensor(B, H, N, N)}, softmax over the last dim (each row a distribution over
N keys). Use AttentionCollector(store_dtype=torch.float32): torch has no CPU
fp16 SVD, and Phase 2 is fp32 everywhere anyway. All functions cast to fp32
and are device-agnostic, but reduce on CPU -- svdvals batched on ROCm is ~49x
slower than on CPU (measured 2026-07-05), so store the snapshots on "cpu".

Conventions:
- entropy (A): row-wise Shannon entropy in nats, mean over rows, /log(N) so
  it's comparable across the UNet's N in {256,64,16} and the DiT's N=256.
- effective rank (B): per-head SVD (not on head-averaged maps, which inflate
  the rank), two defs from one svdvals, both /N: flatness_ratio =
  sum(sigma)/sigma_max (the spectral flatness ratio, not the literature's
  effective rank) and erank_rv = exp(H(sigma_normalized)) (Roy & Vetterli).
- signs: entropy_shift = perturbed - benign; rank drop = benign - perturbed.
- aggregation: flat mean over (layer, timestep, head) in a token locus, per
  sample. Primary cross-model locus is N=256 (12 DiT blocks vs the UNet's 5
  res-16 blocks). It matches in N but not depth (the UNet's N=256 blocks sit
  at encoder-start/decoder-end), so compare per-model aggregates, not layers.

snapshot() after each Euler step and reduce right away (reduce_snapshot ->
stack_steps -> aggregate_per_sample); the raw maps are dropped as reduced.

Measurement-only (post-hoc, on detached snapshots). An attention-based loss
must not go through the collector -- compute it from live q, k in a
non-detaching hook instead.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

Snapshot = Mapping[str, torch.Tensor]          # layer -> (B, H, N, N)
Reduced = dict[str, dict[str, torch.Tensor]]   # layer -> metric -> (B, H)

METRIC_KEYS = ("entropy", "flatness_ratio", "erank_rv")


def _rows_entropy_mean(attn: torch.Tensor) -> torch.Tensor:
    """(B, H, N, N) softmax maps -> (B, H) mean row entropy in nats."""
    row_h = torch.special.entr(attn).sum(dim=-1)  # (B, H, N); entr(0) = 0
    return row_h.mean(dim=-1)


def entropy_per_layer(attn: Snapshot, *, normalize: bool = True
                      ) -> dict[str, torch.Tensor]:
    """Metric A per layer: (B, H) mean row entropy, /log(N) if normalize."""
    out: dict[str, torch.Tensor] = {}
    for name, a in attn.items():
        a = a.float()
        h = _rows_entropy_mean(a)
        if normalize:
            n = a.shape[-1]
            h = h / torch.log(torch.tensor(float(n), device=h.device))
        out[name] = h
    return out


def svdvals_per_layer(attn: Snapshot) -> dict[str, torch.Tensor]:
    """Per-head singular values, descending: layer -> (B, H, N)."""
    return {name: torch.linalg.svdvals(a.float()) for name, a in attn.items()}


def effective_rank_per_layer(attn: Snapshot, *, normalize: bool = True
                             ) -> dict[str, dict[str, torch.Tensor]]:
    """Metric B per layer, both definitions from one SVD.

    Returns layer -> {"flatness_ratio": (B, H), "erank_rv": (B, H)},
    each divided by N when normalize (fraction of maximal rank).
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, sv in svdvals_per_layer(attn).items():
        n = sv.shape[-1]
        flatness = sv.sum(dim=-1) / sv[..., 0]
        p = sv / sv.sum(dim=-1, keepdim=True)
        erank = torch.exp(torch.special.entr(p).sum(dim=-1))
        if normalize:
            flatness = flatness / n
            erank = erank / n
        out[name] = {"flatness_ratio": flatness, "erank_rv": erank}
    return out


def entropy_shift(benign: dict[str, torch.Tensor],
                  perturbed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Signed per-layer shift, PERTURBED - BENIGN (inputs from entropy_per_layer)."""
    return {name: perturbed[name] - benign[name] for name in benign}


def effective_rank_drop(benign: dict[str, dict[str, torch.Tensor]],
                        perturbed: dict[str, dict[str, torch.Tensor]]
                        ) -> dict[str, dict[str, torch.Tensor]]:
    """Signed per-layer drop, BENIGN - PERTURBED (positive = rank collapsed)."""
    return {
        name: {k: benign[name][k] - perturbed[name][k] for k in benign[name]}
        for name in benign
    }


# ---------- per-step reduction pipeline: snapshot -> reduce -> drop ----------


def reduce_snapshot(attn: Snapshot, *, normalize: bool = True) -> Reduced:
    """All metrics of one timestep's snapshot; raw maps can be dropped after.

    Returns layer -> {"entropy" | "flatness_ratio" | "erank_rv": (B, H),
    "n_tokens": 0-d long tensor N} (N kept for later locus filtering).
    """
    ent = entropy_per_layer(attn, normalize=normalize)
    rank = effective_rank_per_layer(attn, normalize=normalize)
    out: Reduced = {}
    for name, a in attn.items():
        out[name] = {
            "entropy": ent[name],
            **rank[name],
            "n_tokens": torch.tensor(a.shape[-1]),
        }
    return out


def stack_steps(steps: list[Reduced]) -> Reduced:
    """Stack per-step reductions along a new leading T axis: (T, B, H)."""
    if not steps:
        raise ValueError("no steps to stack")
    out: Reduced = {}
    for name in steps[0]:
        out[name] = {
            k: torch.stack([s[name][k] for s in steps], dim=0)
            for k in METRIC_KEYS
        }
        out[name]["n_tokens"] = steps[0][name]["n_tokens"]
    return out


def aggregate_per_sample(stacked: Reduced, *, token_filter: int | None = None
                         ) -> dict[str, torch.Tensor]:
    """Pre-registered aggregation: flat mean over (L, T, H) -> (B,) per metric.

    token_filter: keep only layers with that N (256 = the primary shared
    locus); None = all layers (descriptive full profile -- only meaningful
    with normalized metrics, never average raw values across mixed N).
    """
    names = [
        n for n in stacked
        if token_filter is None or int(stacked[n]["n_tokens"]) == token_filter
    ]
    if not names:
        raise ValueError(f"no layers match token_filter={token_filter}")
    out: dict[str, torch.Tensor] = {}
    for key in METRIC_KEYS:
        per_layer = [stacked[n][key] for n in names]        # each (T, B, H)
        cells = torch.cat([x.movedim(1, 0).flatten(1) for x in per_layer],
                          dim=1)                            # (B, L*T*H)
        out[key] = cells.mean(dim=1)
    out["n_layers"] = torch.tensor(len(names))
    return out


def summarize(benign_attn: Snapshot, perturbed_attn: Snapshot, *,
              token_filter: int | None = None, normalize: bool = True) -> dict:
    """One comparison row from a single-step snapshot pair.

    Positive entropy_shift = entropy grew under perturbation; positive
    *_drop = rank collapsed. Returns per-sample (B,) tensors plus their
    means as floats.
    """
    ben = stack_steps([reduce_snapshot(benign_attn, normalize=normalize)])
    per = stack_steps([reduce_snapshot(perturbed_attn, normalize=normalize)])
    b = aggregate_per_sample(ben, token_filter=token_filter)
    p = aggregate_per_sample(per, token_filter=token_filter)
    per_sample = {
        "entropy_shift": p["entropy"] - b["entropy"],
        "flatness_ratio_drop": b["flatness_ratio"] - p["flatness_ratio"],
        "erank_rv_drop": b["erank_rv"] - p["erank_rv"],
    }
    row = {k: v.mean().item() for k, v in per_sample.items()}
    row["per_sample"] = per_sample
    row["n_layers"] = int(b["n_layers"])
    return row


if __name__ == "__main__":
    # Closed-form self-test (python -m src.evaluation.attention_metrics):
    # uniform map (1/N) 11^T -> max entropy (1.0 normalized), rank 1
    # (1/N normalized); identity map -> zero entropy, full rank (1.0).
    torch.manual_seed(0)
    B, H, N = 2, 3, 64
    uniform = torch.full((B, H, N, N), 1.0 / N)
    identity = torch.eye(N).expand(B, H, N, N).contiguous()
    snap_u = {"layer": uniform}
    snap_i = {"layer": identity}

    e_u = entropy_per_layer(snap_u)["layer"]
    e_i = entropy_per_layer(snap_i)["layer"]
    r_u = effective_rank_per_layer(snap_u)["layer"]
    r_i = effective_rank_per_layer(snap_i)["layer"]

    assert torch.allclose(e_u, torch.ones_like(e_u), atol=1e-5), e_u
    assert torch.allclose(e_i, torch.zeros_like(e_i), atol=1e-5), e_i
    assert torch.allclose(r_u["flatness_ratio"], torch.full_like(e_u, 1.0 / N),
                          atol=1e-4), r_u
    assert torch.allclose(r_u["erank_rv"], torch.full_like(e_u, 1.0 / N),
                          atol=1e-4), r_u
    assert torch.allclose(r_i["flatness_ratio"], torch.ones_like(e_u),
                          atol=1e-4), r_i
    assert torch.allclose(r_i["erank_rv"], torch.ones_like(e_u),
                          atol=1e-4), r_i

    row = summarize(snap_i, snap_u)  # identity -> uniform: entropy up, rank down
    assert abs(row["entropy_shift"] - 1.0) < 1e-4
    assert abs(row["flatness_ratio_drop"] - (1.0 - 1.0 / N)) < 1e-3
    assert abs(row["erank_rv_drop"] - (1.0 - 1.0 / N)) < 1e-3
    assert row["n_layers"] == 1
    assert row["per_sample"]["entropy_shift"].shape == (B,)

    # pipeline shape check: 4 fake steps stacked then aggregated
    steps = [reduce_snapshot({"a": torch.softmax(torch.randn(B, H, N, N), -1),
                              "b": torch.softmax(torch.randn(B, H, 16, 16), -1)})
             for _ in range(4)]
    agg_all = aggregate_per_sample(stack_steps(steps))
    agg_64 = aggregate_per_sample(stack_steps(steps), token_filter=64)
    assert agg_all["entropy"].shape == (B,) and int(agg_all["n_layers"]) == 2
    assert int(agg_64["n_layers"]) == 1
    print("attention_metrics self-test: OK")
