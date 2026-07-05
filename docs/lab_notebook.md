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

---

## 2026-07-05 (later) - Phase 2 pilot ran, GO, primary metric = effective rank

Ran the full pilot the same day the modules landed (both models, AB 100 seeds +
FID 1000 seeds, fp32, eps=0.05, 20 PGD iters). Timings on the 6900 XT: SiT AB
91 min (12 N=256 layers x 12 heads of CPU SVD is the cost), UNet AB 15 min,
SiT FID 124 min, UNet FID 46 min. Pre-registration tagged `phase2-prereg`
before the comparison. Figures in `notebooks/out/phase2_*.png`.

**GO.** Metrics A (entropy) and B (effective rank, both definitions) pass the
triplet gate on BOTH models, Wilcoxon p<0.001, Cohen's d 0.77-1.37. LPIPS
efficacy passes on both (PGD/random ratio 14.9 and 19.9) so the attention
result is interpretable -- no image-space fallback needed.

Effect sizes (d, PGD-vs-random on the N=256 locus): entropy 1.24 (DiT) / 1.37
(UNet); flatness 0.93 / 0.77; effective rank (RV) 1.09 / 1.09.

**Primary metric = effective rank (Roy-Vetterli).** Identical d on both models,
clean direction (PGD collapses the rank), and it's the Phase 4 target. Entropy
has a marginally higher d but its signed shift nearly cancels across seeds
(mixed direction: the attack perturbs sharpness strongly but not consistently
up or down), so it's a weaker dose-response axis -- kept as secondary.

**Differential FID eliminated, and it's an interesting reason.** ddFID is
strongly non-zero but the WRONG way: PGD *lowers* FID (SiT 107.8->91.7, UNet
136.1->119.0), so the perturbed batch is slightly closer to real ImageNet. The
untargeted displacement attack changes each image a lot (LPIPS 0.54, L2 ~45)
but doesn't hurt the batch's distributional realism -- see the qualitative
grids, the PGD samples are clearly different but still plausible images. So
Diff-FID is the wrong detector for this attack; per the pre-registration a
wrong-direction result fails the gate. Absolute FID at n=1000 is inflated vs
the 50k headline (small-n bias, expected).

**Cross-architecture, preliminary.** At eps=0.05 the fingerprint is comparable
in size across the two backbones (erank d identical). Benign levels differ a
lot though -- the UNet's attention is intrinsically sparser/lower-rank at
N=256 (erank 0.196 vs 0.411) -- so keep the "attention is less central in the
UNet" caveat and frame everything as an architecture fingerprint, not
robustness.

Phase 3 hand-off: primary erank_rv, secondary entropy, eps grid unchanged
({0.01,0.02,0.05,0.1}, 0.05 is responsive and not saturated), disjoint seeds,
add the DiT@FID95 matched-quality control and the NFE-transfer check.
`docs/phase2_plan.md` closed (section 9).
