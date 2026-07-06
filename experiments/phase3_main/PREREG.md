# Phase 3 -- pre-registration

Confirmatory experiment: does the PGD attention fingerprint DIFFER between
DiT-B/2 and UNet-B? The pilot (exploratory, tag `phase2-prereg`) showed the
fingerprint exists and is large on both models but could not resolve a
difference at n=100. This file fixes the analysis before any Phase-3 data is
generated; committed and git-tagged `phase3-prereg` before the grid runs.

Revised twice: after an internal 4-lens review (seed base, absolute contrast,
TOST framing) and after an independent external review that re-analyzed the
pilot's late-step substrate (`experiments/phase3_main/DESIGN_REVIEW_20260706.md`).
The external review changed the confirmatory core -- co-primary B is now a
positional share statistic, not an entropy index (the entropy permutation-null
was mathematically degenerate) -- and fixed the SESOI as numbers. Rationale is
inline.

All substrate numbers below are pilot late-step values (ODE steps 17-24 of 25),
i.e. the Phase-3 primary scope; the pilot analysis.json aggregates are
full-trajectory and ~2.5x smaller.

## 0. Provenance / self-contained spec

- Attack (identical to the pilot): untargeted PGD on the initial latent z_T,
  L_inf ball of radius eps, 40 iterations, step size eps/4, random start
  delta_0 ~ U(-eps,eps); objective maximizes ||sample(z_T+delta)-sample(z_T)||_2
  through a 25-step Euler-ODE sampler (t=0 noise -> t=1 data). Rademacher
  control: delta = eps * (+-1) per coordinate, same eps.
- Seeds DISJOINT from the pilot: `SEED_BASE = 21260705` (pilot used 20260705;
  +1.0e6 clears the largest offset band, ~500,500, at n=500). The runner
  asserts zero overlap with the pilot's consumed offsets.
- fp32 throughout; attention SVD reduced on CPU; reduction order per section 2.
- Metric code frozen at the commit that tags this file.

## 1. Design (data generation)

Grid: eps in {0.01, 0.02, 0.05, 0.1}, n=500 class-distinct samples/config
(labels 0-499 once each, shared across models and eps), 40 PGD iters, 25 ODE
steps. Three branches per shared seed: benign, Rademacher random (K=1;
across-draw noise was ~5% of across-seed spread in the pilot), PGD. Benign is
measured ONCE per (model, seed) and reused across all eps (it is
eps-independent). Shared z_T (1x4x32x32, seed BASE+i), shared PGD-init, shared
labels across models -> the per-seed cross-model difference Delta_i is valid at
any cross-model correlation (pairing affects efficiency, not validity; the
pilot's measured correlation is +0.56, ~32% SE reduction). The runner persists
the full per-layer x per-step x per-head substrate per branch (co-primary B and
all robustness scopes depend on it); post-step time convention t = (i+1)/25.

Models: DiT-B/2, UNet-B (full grid), plus a matched-quality control (section 6).

## 2. Metrics, aggregation, reduction order

Three metrics (entropy, flatness_ratio, erank_rv), normalized as in the pilot.
Reduction order, fixed: per-head SVD -> mean over heads -> mean over the locus
layers -> mean over the late-step window. (The flat-mean implementation equals
this nested mean because head count is constant within each model's N=256
locus.) Late-step window = normalized t >= 0.72 = 0-indexed steps 17-24; it maps
across NFE for the transfer check.

**Primary metric: erank_rv, N=256 locus, late-step scope. Primary eps = 0.05**
(pilot-validated, mid-grid, and approximately output-damage-matched across
models: pilot LPIPS 0.546 vs 0.536, l2_out 45.2 vs 44.7). The whole
confirmatory family is scoped to this eps except the dose-response member
(section 5), which by definition uses all four.

## 3. Confirmatory family and co-primary tests

Two co-primary members, both two-sided (the cross-model difference has no a
priori sign). The benign-free contrast is contrast_i(model) = drop_PGD,i -
drop_rand,i = rand_i - pgd_i (benign cancels).

**Co-primary A -- magnitude.** ABSOLUTE contrast (erank is already /N-normalized
-> dimensionless, cross-model comparable). Delta_i = contrast_i(DiT) -
contrast_i(UNet); paired two-sided t + Wilcoxon, bootstrap 95% CI (percentile,
10k resamples), per-seed Delta distribution reported. Substrate forecast:
DiT 0.0329, UNet 0.0457, Delta = -0.0128, SE at n=500 ~0.0014.
- *Why absolute, not relative:* the contrast is benign-free by construction, so
  dividing by benign only re-encodes the ~2x benign-rank gap (DiT 0.41 vs
  UNet 0.20). Absolute is the scale the Phase-2 metric was elected on and the
  most conservative framing. There is no framing-free "same size" on a bounded
  nonlinear scale, so relative-to-mean-benign and standardized d are reported as
  DESCRIPTIVE framings (no veto). All A claims are worded "in absolute erank
  units on the shared N=256 locus".
- *Locus-contingency disclosure:* on the all-layer locus the sign flips
  (pilot: DiT 0.0178 > UNet 0.0104). A's answer is locus-specific; the N=256
  scoping is a choice with consequences, stated as such.

**Co-primary B -- structure (excess input-side share).** The pilot difference is
positional (UNet decoder-concentrated, DiT block-0-plus-deep), not entropic. Per
seed, per model, on the model's own N=256 layers in depth order:

    S_i = sum_{l in input-side} max(c_l,i, 0) / sum_l max(c_l,i, 0)  -  L_in/L

c_l,i = per-layer late contrast (rand - pgd). Input-side = {input_blocks.4,
input_blocks.5} for UNet (structural encoder, L_in/L = 2/5) and blocks 0-5 for
DiT (6/12). Under a within-seed layer shuffle E[S] = 0 exactly and analytically
(the permutation-null instinct becomes a closed form because the statistic is
positional). D_i = S_i(DiT) - S_i(UNet); paired two-sided t + Wilcoxon, same
machinery as A. Substrate: per-seed D = +0.156 (SD 0.233, SE at n=500 ~0.0104,
z ~15); seed-mean-profile D = +0.117.
- *Riders:* (a) clamp max(c,0); exclude seeds below a minimum-total-mass floor
  fixed from the pilot substrate, reporting exclusion counts; (b) report D under
  the DiT boundary shifted +-1 block (0-4/5-11 and 0-6/7-11) as a pre-registered
  sensitivity; (c) within-model corroborations: UNet decoder-vs-encoder paired
  contrast, DiT block-0 share > 0.
- *Why not the entropy index (dropped):* Shannon entropy is permutation-
  invariant, so the previously-registered "excess over a within-seed
  permutation null" is identically zero -- a 0/0-degenerate test. Even repaired,
  entropy is location-blind while the finding is positional, and its L-dependent
  bias (0.11 DiT vs 0.01 UNet on the substrate) exceeds the 0.06 signal.

**Family (5 members, Holm-corrected, alpha = 0.05):** {co-primary A, co-primary
B, dose-response interaction, entropy-echo of A, flatness-echo of A}. Each member
contributes ONE p-value = max(p_t, p_Wilcoxon) (conservative conjunction). Other
eps and the full-trajectory scope are robustness, reported with CIs, outside the
family.

## 4. Verdict rule (total function)

The study is over-powered (|Delta|/SE ~9 at the primary scope), so the verdict
rests on effect size + SESOI, not p-values. Per co-primary, three-way:

1. rejects after Holm AND |point estimate| >= SESOI AND survives the
   matched-quality control -> DISTINGUISHABLE on that axis;
2. rejects but |estimate| < SESOI -> statistically DETECTABLE, below the
   architecturally meaningful threshold;
3. no rejection and 90% CI within +-SESOI -> EQUIVALENT within the SESOI;
   no rejection and CI exceeding +-SESOI -> INCONCLUSIVE.

TOST = two one-sided tests at alpha = 0.05 (equivalently the 90% CI within
+-SESOI), primary eps only, both co-primaries. The dose-response member's
rejection alone yields at most class 2 (it cannot make "distinguishable" by
itself). Overall "architectures distinguishable" iff at least one co-primary
reaches class 1.

Mandatory wording: a raw-Delta TOST equivalence pass must NOT be phrased "the
architectures respond equivalently" -- relative susceptibility differs ~3x
(0.033/0.41 vs 0.046/0.20); report contrast-over-mean-benign descriptively.

## 5. SESOI (frozen numbers)

**SESOI-A = 0.010** (absolute erank contrast, late-step, N=256, eps=0.05).
Anchor: 25% of the pooled late fingerprint C = 0.039 (0.25 x 0.039 ~= 0.010).
Below this the Phase-4 targeted-attack headroom is under ~0.3 within-model SD
(not certifiable on a single-GPU budget); above it a difference is worth a
targeted-attack phase. Calibration: 10% is the MDE (would collapse meaningful
into detectable), 50% would bless a third-of-the-effect difference. Frozen as a
number, not the C-dependent rule, so it can't drift with Phase-3's own level
estimate. Feasibility: SE(Delta) ~0.0014 (<=0.0024 at rho=0), MDE ~0.004-0.008;
forecast CI ~[0.010, 0.016] -> the lower bound sits at the SESOI, so neither
verdict is pre-ordained.

**SESOI-B = 0.10** (excess input-share difference D). Anchor: robustness to the
one analyst choice -- the DiT half-depth cut, whose one-block move shifts S by
1/12 ~= 0.083; round up to 0.10, paired with the +-1-block sensitivity. Pass
requires BOTH the per-seed D and the seed-mean-profile D >= 0.10. Feasibility:
SE ~0.0104, MDE ~0.03; pilot clears it (0.156 / 0.117). (An alternative anchor
of 0.20 was considered and rejected: it treats the UNet's structural boundary as
movable.)

Rejected SESOI anchors: Cohen's-d conventions (noise-dependent; cell-count
asymmetry 1152 vs 160 makes cross-model d unreliable), benign-gap fraction
(resting level != response), observed pilot Delta (circular), MDE/variance floor
(defines detectable).

## 6. Confound controls

- **Matched-quality control.** The FID pre-check (2026-07-06) measured the
  same-regime batch-256 checkpoints step_04M/05M at FID-5k = 77.7 / 75.8 -- both
  too close to the DiT final (~75 at 5k) and far from the UNet's ~104, so NO
  same-regime UNet-matched checkpoint exists (batch-256 converges early). Use the
  batch-64 `experiments/20260518-SiT-B-2-b64/checkpoints/step_00150000.pt`
  (FID-50k 95.4), with the regime caveat PROMINENT (batch-64 / ~9.6M samples
  varies convergence AND regime). Selection rule frozen: the checkpoint with
  FID-50k closest to the UNet's 99.1 within [80,110]; the batch-64 point is the
  only one available. Control leg = primary eps only (n=500), ~half a GPU-day
  saved, zero confirmatory loss. Decision rule: DiT@FID95-vs-DiT@FID68 must be
  TOST-equivalent within SESOI-A to count as "patterns with the DiT" (survives);
  DiT@FID95-vs-UNet must have the same sign as DiT@FID68-vs-UNet. If DiT@FID95
  patterns with the UNet instead -> the effect tracks convergence; state the
  ambiguous-middle verdict explicitly (inconclusive on architecture-vs-
  convergence). Applies to both co-primaries.
- **NFE-transfer.** Re-run co-primary A and B at NFE 50 and 100 (forward-only)
  at the primary eps on both models, re-sampling benign/rand/PGD all at that NFE
  (never perturbed@50 vs benign@25), normalized-t window. "Survives" = same sign,
  Holm-significant, magnitude within x/div 2 of the primary.
- **Attack success.** Report l2_out per eps per model; interpret conditional on
  comparable success (per-eps l2_out ratio in [0.8, 1.25]); flag any eps outside.
- **Attack convergence.** {20,40,80}-iter ablation on both models at eps=0.05;
  plateau = |contrast@80 - contrast@40| <= 10% of contrast@40; justify 40.
- **Saturation** (from the grid): contrast@0.1 < 2 x contrast@0.05 for both.

## 7. Disclosures / limitations

Head geometry differs on N=256 (DiT 12 heads x dim 64; UNet 4 heads x dim 96);
disclosed, not corrected (softmax breaks the QK^T rank ceiling -- DiT benign
erank ~105 > head_dim 64 -- and the ceiling runs opposite to the benign gap, so
head_dim is not the driver). Cheap probe: does benign erank fall with head_dim
across the UNet's own resolution blocks? K=1 random (5% justification); single
UNet convergence control (asymmetry disclosed); the N=256 aggregate mixes the
UNet's active decoder and quiet encoder blocks, so co-primary B (structure)
carries the "differs" claim and magnitude (A) is secondary.

## 8. Blinding / deliverables

The analyzer (co-primary A/B, TOST, dose-response, control rule) may be
finalized DURING the runs, but NO cross-model comparison is computed until the
analyzer is frozen (that is the blinding rule). It operates on the persisted
substrate, so no analysis change alters generated data. Only the runner
(section 1) and the control-checkpoint choice (section 6, now resolved) gate the
launch. Deliverables: `experiments/phase3_main/` per-config results +
analysis.json; tables and figures per docs/phase3_plan.md section 5.

Dose-response member: per-seed OLS slope of the absolute contrast on log(eps)
over the four grid points, paired two-sided cross-model test on the slopes
(uses all four eps -- the scoped exception to the family's primary-eps rule).

## 9. Pre-launch amendments (2026-07-06; no Phase-3 data existed)

The first launch attempt showed the CPU spectral reduction dominating wall
time (~57 h forecast for the three legs, two thirds of it SVD on NFE-transfer
steps no registered test consumes). Two implementation changes were adopted
and verified BEFORE any Phase-3 data generation; forecast ~28 h. Component
benches and real-map equivalence measurements dated 2026-07-06, on the run
hardware (i9-10900K / RX 6900 XT), smoke-tested end to end.

**9a. Singular values via the Gram identity (amends the section-0 "metric
code frozen" line).** `svdvals_per_layer` computes per-head spectra as
sigma_i(A) = sqrt(eigvalsh(A^T A)) instead of direct `torch.linalg.svdvals`
(1.9x faster at phase-3 shapes; the run is SVD-bound). The identity is exact;
in fp32 the squaring floors singular values below ~sigma_max*sqrt(eps), which
perturbs erank_rv on near-degenerate heads (DiT block-0 attention sinks).
Measured against direct svdvals on real DiT maps (batch 25, benign /
Rademacher / PGD branches, steps spanning the trajectory): per-cell
(layer, step, head) worst 7.6e-4 (degenerate pgd@late cells; typical ~1e-5);
worst per-layer head-mean late-window contrast -- the co-primary-B input --
4.4e-5 on block 0 (share-statistic impact ~3e-4 vs SESOI-B = 0.10), all other
layers <= 4e-6; N=256-locus late contrast -- the co-primary-A input --
<= 4.4e-6 (SESOI-A / 2300). Every registered statistic therefore moves >=
300x less than its decision threshold. Flatness agrees to <= 2e-6 everywhere;
entropy involves no spectra and is untouched. Runtime guard: the first
reduced snapshot of every run recomputes direct svdvals on the same real maps
and hard-fails above a 2e-3 per-cell tolerance (2.6x the measured worst;
genuine breakage -- transpose/ordering bugs -- manifests at >= 1e-2).

**9b. NFE-transfer substrate restricted to its registered scope (section 6).**
At NFE 50 and 100 the runner attaches capture hooks and reduces attention
only on the normalized-t window t >= 0.72 -- post-step 0-indexed Euler steps
35-49 of 50 and 71-99 of 100 -- exactly the scope the section-6 transfer
tests (co-primaries A and B) are defined on; steps before the window run with
no hooks. The values entering every registered transfer statistic are
bit-identical to a full-trajectory measure; what is no longer produced is
early-step substrate at NFE 50/100, which no registered analysis consumes.
The 25-step grid substrate (section 1) remains full-trajectory (all layers x
25 steps x heads, all three metrics). `nfe*_eps*.pt` files persist
`step_indices` and `t_min`.

Considered and NOT adopted, keeping registered scopes intact: subsampling
early steps of the 25-step substrate (would downgrade the full-trajectory
robustness scope to a stratified estimate; measured bias would be <= 2-3%,
but the scope stays exact instead); reducing the transfer n from 500 to 100
(the x/div-2 transfer magnitude window would acquire a ~4% noise-driven
false-flag probability); fp64 eigvalsh (per-cell <= 1.9e-6, but only 1.19x --
the fp32 route's aggregate-level equivalence above is the operative
guarantee).
