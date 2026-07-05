# Lab Notebook

Chronological log of experimental work, decisions, and observations.
Update this every working session, even if just a few lines.

---

## 2026-04-22 - Repository bootstrap, pivot from slowflow

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
  hint. Second `uv sync` completed using the existing uv cache - no
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

---

## 2026-07-05 - Phase 2 kickoff: PGD module + attention metrics land

Phase 1 closed 2026-07-01 with two symmetric checkpoints (6.4M steps, fp32
FID-50k: SiT-B/2 = 68.67, UNet-B = 99.14). Phase 2 (metrics + pilot PGD)
starts today; the two net-new modules shipped and smoke-tested on both
checkpoints.

**Shipped.**

- `src/attacks/pgd_latent.py`: untargeted L_inf PGD on the noise seed
  `z_T` through the 25-step differentiable Euler-ODE sampler (gradient
  checkpointing, fp32), plus `rademacher_delta` as the equal-budget
  random control (+-eps per coordinate: matches both L_inf and L2 of
  sign-saturated PGD). Optional CFG (off by default), optional callable
  objective for the image-space fallback.
- `src/evaluation/attention_metrics.py`: Metric A (row entropy, nats,
  normalized by log N) and Metric B in both definitions from one SVD
  (nuclear/spectral "flatness ratio" + Roy-Vetterli effective rank,
  per-head, normalized by N). The pipeline snapshots after each Euler step,
  reduces to scalars right away, stacks the timestep axis, then aggregates
  on the shared N=256 locus. Closed-form self-test in `__main__`.
- Deps: `lpips` added (perceptual efficacy check on the attack), `scipy`
  pinned (Wilcoxon).

**Random start is not optional.** With a deterministic sampler and the benign
output as the loss target, delta=0 is an exact stationary point: the first PGD
gradient is identically zero, sign(0)=0, and the attack would never leave the
seed. So delta is initialized ~ U(-eps, eps). This bit me in the first smoke
run before I worked out why the loss stayed at 0.

**Smoke results (B=2, 3 PGD iters, 25 ODE steps, eps=0.05, fp32).**

- Sampler determinism bit-exact on both models (max|a-b| = 0) -- good, the
  paired benign/random/PGD design relies on it.
- PGD: loss strictly increasing, grads finite/nonzero, ||delta||_inf = eps
  exactly. Peak VRAM 0.87 GB (SiT) / 0.76 GB (UNet) at B=2, so the pilot can
  batch wide. 3 iters: 4.5 s (SiT) / 7.0 s (UNet).
- Attention layer census confirmed: SiT-B/2 12 layers all N=256; UNet-B 5 @
  N=256, 5 @ N=64, 1 @ N=16 (the N=256 ones sit at input_blocks.4-5 and
  output_blocks.6-8, i.e. encoder-start / decoder-end, not spread through the
  depth like the DiT).
- Measuring attention with the maps kept on GPU took 241.8 s for a 25-step
  SiT pass vs 16.2 s on CPU (~15x). svdvals batched on ROCm is the culprit --
  a separate micro-bench put the isolated call at ~49x the CPU. So the runner
  stores snapshots on CPU and reduces there.

**Next.** Pilot runner (100 seeds x 3 branches x 2 models) + the perceptual
check, and write `experiments/phase2_pilot/PREREG.md` (aggregation, gate,
election rule) and tag it before the comparison runs, so the analysis choices
are fixed before I see any results.
