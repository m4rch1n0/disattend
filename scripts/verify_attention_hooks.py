#!/usr/bin/env python3
"""Hook-fidelity artifact: the collector's attention reproduces each layer.

The whole attention measurement rests on AttentionCollector recomputing
softmax(QK^T/sqrt(d)) from a hook on the qkv projection. This script proves
it end to end and persists the evidence: for every attention layer of both
backbones, it reconstructs the module's ACTUAL output using the collector's
map plus the module's own v and output projection, and compares with the
real output from the same forward. Any error in q/k extraction, head
reshaping, scaling or softmax would show up here; matching the output is
strictly stronger than matching an internal map.

Writes experiments/phase3_main/hook_fidelity.json (per-layer max abs diff,
fp32; outputs are O(1) activations).

Usage: uv run python scripts/verify_attention_hooks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIT_DIR = REPO_ROOT / "third_party" / "sit"
for _p in (str(REPO_ROOT), str(SIT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from src.attacks.pgd_latent import euler_step, load_model
from src.utils.attention_hooks import AttentionCollector, discover_attention_layers
from scripts.run_phase3_grid import CKPTS, MODEL_KEY

TOL = 1e-4  # fp32 activations O(1). The UNet's explicit attention path
# matches the reconstruction to ~1e-6; the SiT module runs fused SDPA
# internally, so the reconstruction differs by summation order (~1e-4 over
# N=256 accumulations). Both are numerical noise, not extraction errors.


@torch.inference_mode()
def check_model(name: str, device: torch.device) -> dict[str, float]:
    model = load_model(MODEL_KEY[name], REPO_ROOT / CKPTS[name], device)
    specs = discover_attention_layers(model)
    io: dict[str, tuple] = {}
    handles = []
    for s in specs:
        def mk(nm):
            def hook(module, args, output):
                io[nm] = (args[0].detach(), output.detach())
            return hook
        handles.append(model.get_submodule(s.name).register_forward_hook(mk(s.name)))

    col = AttentionCollector(model, store_dtype=torch.float32, store_device="cpu")
    g = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(2, 4, 32, 32, device=device, generator=g)
    y = torch.tensor([1, 2], device=device)
    with col:
        euler_step(model, z, torch.zeros((), device=device),
                   torch.tensor(1.0 / 25, device=device), y)
        maps = col.snapshot()
    for h in handles:
        h.remove()

    diffs: dict[str, float] = {}
    for s in specs:
        x_in, out = io[s.name]
        attn = maps[s.name].to(device)
        m = model.get_submodule(s.name)
        if s.kind == "linear_qkv":          # timm Attention (SiT)
            B, N, _ = x_in.shape
            qkv = m.qkv(x_in).reshape(B, N, 3, s.num_heads, s.head_dim
                                      ).permute(2, 0, 3, 1, 4)
            v = qkv[2]
            rec = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            rec = m.proj(m.norm(rec))
        else:                                # AttentionBlock (UNet)
            B, C, H, W = x_in.shape
            h_ = m.norm(x_in).reshape(B, C, H * W)
            qkv = m.qkv(h_).reshape(B, 3, s.num_heads, s.head_dim, H * W)
            v = qkv[:, 2]
            rec = torch.einsum("bhij,bhcj->bhci", attn, v).reshape(B, C, H * W)
            rec = m.proj_out(rec).reshape(B, C, H, W) + x_in
        diffs[s.name] = float((rec - out).abs().max().item())
    del model
    torch.cuda.empty_cache()
    return diffs


def main() -> int:
    device = torch.device("cuda")
    results = {}
    for name in ("SiT-B/2", "UNet-B"):
        diffs = check_model(name, device)
        worst = max(diffs.values())
        print(f"{name}: worst output reconstruction diff {worst:.2e} "
              f"over {len(diffs)} layers -> {'OK' if worst < TOL else 'FAIL'}")
        assert worst < TOL, f"{name} hook fidelity broken: {worst:.2e}"
        results[name] = {"per_layer_max_abs_diff": diffs, "worst": worst}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    out = REPO_ROOT / "experiments/phase3_main/hook_fidelity.json"
    with open(out, "w") as f:
        json.dump({"tolerance": TOL, "method": "output reconstruction from "
                   "collector map + module v/proj", "models": results,
                   "git_commit": commit, "torch": torch.__version__,
                   "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime())}, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
