# Phase 2 pilot -- pre-registration

Committed and git-tagged (`phase2-prereg`) before the pilot comparison is run.
The point is to fix, in writing and before seeing any data, the three things
that would otherwise be easy to tune after the fact: how each metric is
reduced to one number per sample, the go/no-go test, and how the primary
metric is chosen. The pilot is exploratory. Phase 3 is the confirmatory
experiment (disjoint seeds, a single pre-declared primary metric, Holm
correction on any secondary metrics reported inferentially).

Full spec and background: `docs/phase2_plan.md`.

Convention: the flow-matching sampler runs t=0 (noise) -> t=1 (data), forward
Euler, 25 steps; `z_T` is the initial noise seed that PGD perturbs.

---

## 1. Design

Two frozen checkpoints, fp32 everywhere (fp16 overflows the UNet-B attention;
bf16 is ~5x slower on this GPU):

- DiT-B/2 (SiT-B/2): `experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt`, key `"ema"`. 12 attention layers, all N=256, 12 heads.
- UNet-B: `experiments/20260611-UNet-B-cosine-6p4M/checkpoints/step_06400000_final.pt`, key `"ema"`. 11 attention layers: N=256 at `input_blocks.{4,5}.1` and `output_blocks.{6,7,8}.1` (5 layers), N=64 at `input_blocks.{7,8}.1` and `output_blocks.{3,4,5}.1` (5 layers), N=16 at `middle_block.1` (1 layer), 4 heads.

Attack, identical for both models (this symmetry is the experiment): untargeted
L_inf PGD on `z_T`, eps=0.05, 20 iterations, step size eps/4, 25 ODE steps,
no CFG, objective = maximize ||sample(z_T+delta) - sample(z_T)||_2. Random
start delta_0 ~ U(-eps, eps) -- mandatory, because the deterministic sampler
makes delta=0 a stationary point where the gradient is zero.

Three branches per shared seed: benign (`z_T`), random (`z_T + delta_rand`),
PGD (`z_T + delta_PGD`). The random control is Rademacher, delta_rand =
eps * (+-1) per coordinate, so it has the same L_inf and the same L2 as a
sign-saturated PGD perturbation and only the direction differs; averaged over
K=3 independent draws per seed.

Sample sets (seed ledger in `scripts/run_phase2_pilot.py`; disjoint from
Phase 3 by construction):
- Attention (parts A/B): n=100 shared seeds, labels `y_i = (10*i) mod 1000`.
- Diff-FID (part C): n=1000, labels `y_j = j mod 1000` (class-balanced,
  1000 classes x 1). May drop to n=500 only if wall-clock forces it, and only
  if said so in the results.

---

## 2. How each metric becomes one number per sample

Everything is normalized so the two models are comparable despite different N
(DiT N=256; UNet N in {256, 64, 16}):

- Metric A, entropy: row-wise Shannon entropy in nats, mean over the N rows,
  divided by log N so it lands in [0, 1]. Computed per (layer, timestep, head).
- Metric B, effective rank: SVD of each (N, N) map, per head (do not average
  maps over heads first). Two definitions come out of the same SVD, both
  divided by N so they land in (0, 1]:
  - `flatness_ratio` = sum(sigma) / sigma_max, the nuclear/spectral ratio.
    Named plainly as spectral flatness, not "effective rank".
  - `erank_rv` = exp(H(sigma_normalized)), the Roy & Vetterli effective rank.

Per-sample scalar for A and B: the flat mean over all (layer, timestep, head)
cells inside the N=256 locus (`aggregate_per_sample(..., token_filter=256)` in
`src/evaluation/attention_metrics.py`). N=256 is the primary cross-model locus
-- all 12 DiT blocks against the UNet's 5 resolution-16 blocks. Note it is
comparable in N but not in depth (the DiT's N=256 blocks span the whole
network, the UNet's sit at encoder-start / decoder-end), so I compare per-model
aggregates there and do not pair layers or read depth into it. The full-profile
aggregate over all layers (normalized) is reported as context only, never as
the gate. The random branch's scalar is the mean over its K=3 draws.

Signs: `entropy_shift = M_perturbed - M_benign` (positive = more diffuse);
`drop = M_benign - M_perturbed` (positive = rank collapsed).

Metric C, Diff-FID: for each branch, decode its 1000 latents, get InceptionV3
features, compute FID against the existing 50k reference
(`data/imagenet_latents/fid_ref_stats.pt`). Report FID for each branch, the
deltas `dFID_pgd = FID_pgd - FID_ben` and `dFID_rand`, and the contrast
`ddFID = dFID_pgd - dFID_rand`. Absolute FID at n<=1000 is upward-biased and is
not comparable to the 68.67 / 99.14 headline (n=50k); only the deltas are the
metric.

---

## 3. Go/no-go test

The sampler is deterministic given `(z_T, y)`, so a benign re-seed gives an
identical output and a "2 sigma of a re-seed" baseline is degenerate (its
spread is zero). Instead I use the contrast between the two perturbations.
For A and B (paired over the 100 seeds), per seed i:

    D_i = (M_PGD,i - M_ben,i) - (M_rand,i - M_ben,i) = M_PGD,i - M_rand,i

(the benign anchor cancels; it is written out because it is what each shift
means). A metric PASSES iff BOTH:

1. `mean(D) >= 2*SE(D)`, with `SE(D) = std(D)/sqrt(n)`, one-sided in the
   direction fixed below; and
2. a one-sided Wilcoxon signed-rank test on {D_i} agrees (p < 0.05).
   Adversarial effects can be bimodal, so I do not trust the mean alone.

Pre-registered direction of D (fixed now, before the data):
- A (entropy): two-sided -- an attack could sharpen or diffuse attention and I
  have no prior, so gate on the magnitude of the shift:
  `D_i = |M_PGD,i - M_ben,i| - |M_rand,i - M_ben,i|`, one-sided
  `mean(D) >= 2*SE(D)` (PGD moves entropy farther than random, either way).
- B (both rank definitions): rank should drop more under PGD than under random,
  so `D_i = drop_PGD,i - drop_rand,i = M_rand,i - M_PGD,i`, one-sided. If the
  data instead show rank rising, that is a failed-direction result, not an
  excuse to flip the sign afterwards.

For Metric C (no per-seed pairing): PASS iff the bootstrap 95% CI of ddFID
(>=200 paired resamples of the per-sample Inception features, recomputing FID
each time) excludes 0 on the positive side.

Overall: GO iff at least one of the three metrics passes. No multiple-comparison
correction on this gate -- it is a screening test (a false pass costs Phase-3
compute, not a wrong conclusion), and correcting at n=100 would only add false
negatives. The real safeguard is that Phase 3 is confirmatory with one
pre-declared metric and disjoint seeds. If nothing passes -> write a diagnostic
note, do not jump to the Phase 3 grid.

---

## 4. Perceptual efficacy check (gates the interpretation)

The attack maximizes a latent-space L2, but this VAE's latent geometry is not
faithfully perceptual (the latent-flip test gave 19.2 vs 24.8 dB), so a
latent-L2 gain can decode to almost no visible change. So I report
`LPIPS(decode(benign), decode(perturbed))` for the PGD and random branches
(AlexNet LPIPS), and the latent-L2 output displacement alongside.

Fallback rule, fixed now: if `mean LPIPS(PGD)` is not clearly above
`mean LPIPS(rand)` -- specifically if the ratio is below 1.5 -- then a null
attention result is uninterpretable ("robust attention" vs "the attack did
nothing visible"), and I re-run the pilot with an image-space objective
(decode inside the graph, maximize pixel-space L2; VRAM is fine, the backward
peaked ~2.86 GB) before reporting anything. A ratio >= 1.5 means the attack is
perceptually effective and the attention metrics can be read as they are.

---

## 5. Which metric becomes primary

Among the metrics that pass the gate, pick the primary one for Phase 3 by:

1. Largest standardized effect size on the N=256 locus: `mean(D)/std(D)`
   (Cohen's d of the contrast) for A/B; for C, ddFID over its bootstrap SD.
2. Tie-break (within 15% on effect size): prefer Metric B (effective rank),
   because the optional Phase 4 rank-collapse attack targets it and it carries
   the most downstream value. If B is elected, record which definition
   (flatness_ratio or erank_rv) had the larger effect; Phase 3 uses that one.
3. If only one metric passes, it is primary by default (report the others'
   effect sizes as context).

If nothing passes, there is no election; the deliverable is the diagnostic
note and a calibrated-eps recommendation.

eps calibration: report whether eps=0.05 produced a measurable above-random
shift. If the effect is saturated or absent, recommend re-centering the Phase 3
grid {0.01, 0.02, 0.05, 0.1}. The recommendation is descriptive -- the grid
values do not change unless 0.05 is clearly degenerate.

---

## 6. Not pre-registered (flagged honestly)

- The qualitative best/worst/median attention-map figures (post-hoc, for the
  notebook, not inferential).
- The full-UNet-profile aggregate (context, not a gate).
- Any per-layer or per-timestep breakdown (exploratory; would be confirmatory
  in Phase 3 if pursued).
