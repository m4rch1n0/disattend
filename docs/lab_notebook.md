# Lab Notebook

Chronological log of experimental work, decisions, and observations.
Update this every working session, even if just a few lines.

---

## 2026-04-22 — Repository bootstrap, pivot from slowflow

New repository `~/disattend` created as sibling of `~/slowflow`. The two
projects share hardware, stack conventions, and style, but have disjoint
scientific scope: `slowflow` studies latency-surging attacks on Flow
Matching UNet (Phase 2 complete, FID 10.40 on CIFAR-10 with a 35.75M
UNet), `disattend` studies how standard gradient-based attacks manifest
in the self-attention layers of Diffusion Transformers versus UNet
backbones. The `slowflow` UNet may later serve as the UNet baseline for
the comparison, but that decision is deferred to Phase 3.

**Decisions fixed this session (see `PROJECT_PLAN.md` §1, §3).**

- Research question framed descriptively/comparatively: characterize the
  attention-level fingerprint of a standard PGD attack on DiT vs UNet. A
  null result is acceptable.
- Focal DiT model: DiT-XL/2 (675M params), class-conditional ImageNet
  256×256, from the pre-trained release of Peebles & Xie 2022. Smaller
  variants (DiT-S/2, DiT-B/2) kept as VRAM fallback.
- Attack family: PGD on the initial latent `z_T`, `L_inf`-bounded, white-box,
  gradients through the sampling loop with gradient checkpointing.
- Three candidate metrics to route into Phase 2 (one will be elected
  primary afterwards): attention entropy shift, effective-rank drop,
  differential FID.
- Engineering budget: gradient checkpointing + fp16 inference + subset of
  sampling steps in backward are permitted; final FID measurements in fp32.

**Environment setup.**

- `uv init` equivalent via hand-written `pyproject.toml` (Python 3.12,
  PyTorch 2.11 on ROCm 7.2, numpy, matplotlib, tqdm, torchvision,
  triton-rocm). First `uv sync` failed with the same `triton-rocm` index
  resolution error seen on slowflow 2026-04-20; fixed by adding
  `triton-rocm` to the dependency list with the `pytorch-rocm` index
  hint. Second `uv sync` completed using the existing uv cache — no
  significant download.
- `diffusers`, `transformers`, `accelerate`, `huggingface_hub` are
  intentionally NOT installed. They will be added in Phase 1 when the
  first script needs them, so that the dependency surface grows with
  actual need rather than speculation.

**Hardware constraints inherited from slowflow.**

- RDNA 2 on ROCm 7.2 has no native bf16: bf16 autocast is ~5× slower
  than fp32. Use fp16 or fp32, never bf16.
- `tmux` is not usable in the VSCode integrated terminal (xterm.js);
  long-running jobs must use `nohup ... & disown`.
- `torch.compile()` incompatibility with torchdiffeq is not expected to
  bite here (no torchdiffeq in the dependency set), but the compatibility
  of `torch.compile()` with `diffusers` pipelines will need to be checked
  in Phase 1 before relying on it.

**Next.** Phase 0 Task 1 in a fresh session: run `verify_setup.py`, paste
the output into `docs/setup_verified.md`, then perform a read-only sanity
load of DiT-XL/2 in a notebook (inference only, no attack). Handoff in
`docs/phase0_plan.md`.
