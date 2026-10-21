# Phase 5a — guidance ablation (post-thesis, exploratory)

Written before launch, 2026-10-21. Not pre-registered; the statistics below
are the Phase-3 ones re-applied, and every number produced here is reported
as "run after the thesis, not pre-registered".

## Question

Every result of the thesis and of the paper was measured at classifier-free
guidance (CFG) scale 1.0, i.e. pure conditional sampling. Does the attention
fingerprint of the latent PGD attack survive guidance at a practical scale,
and does the cross-model comparison (DiT vs UNet) change?

## Models and settings

| leg | checkpoint | cfg | seeds | batch | eps | PGD | sampler |
|-----|-----------|-----|-------|-------|-----|-----|---------|
| SiT-B-2 | ours, `experiments/20260520-SiT-B-2-recovery/checkpoints/step_06400000.pt` (6.4M steps, batch 16) | 1.5 | 500 (ids 0..499, labels i mod 1000) | 25 | 0.05 | 40 iters, step eps/4, random start | 25 Euler steps, fp32 |
| UNet-B | ours, `experiments/20260611-UNet-B-cosine-6p4M/checkpoints/step_06400000_final.pt` | 1.5 | same | 25 | 0.05 | same | same |

Guidance scale 1.5 is the reference setting of the SiT paper (Ma et al.
2024, FID 2.06 for SiT-XL/2); the thesis used 2.0 only for qualitative grids.
The null label is index 1000 in both models (trained with class dropout
0.1). Guidance is applied manually and identically to both models in
`src/attacks/pgd_latent.euler_step`: `v = v_uncond + s (v_cond - v_uncond)`.
The attack maximizes the displacement of the GUIDED trajectory and the
measurement runs on the same guided trajectory (`scripts/run_phase5.py`,
`measure_attention_cfg`). The hooks see the duplicated batch [cond; uncond]:
the conditional half is the primary substrate (same file layout as Phase 3),
the unconditional half is saved as `*_uncond.pt` for a secondary reading.

Seeds, Rademacher directions and PGD starts are the Phase-3 ones
(`SEED_BASE 21260705`, eps-value bands), so each (model, seed) pairs with the
Phase-3 leg at cfg 1.0 (`experiments/phase3_main/<slug>/`). The eps grid is
the primary budget only; no NFE-transfer legs.

## Primary and secondary readings

Primary (per model): the late-window (t >= 0.72, steps 17–24) effective-rank
contrast c = M_rand − M_pgd on the N=256 locus, mean, SD, Cohen's d,
bootstrap CI95, paired t / Wilcoxon p; and the paired per-seed difference
c(cfg 1.5) − c(cfg 1.0) against Phase 3.

Secondary: cross-model Delta_A = c_DiT − c_UNet at cfg 1.5 with its CI90,
read against the registered SESOI-A = 0.010 as a sensitivity of the
confirmatory verdict (not a new verdict); the per-layer profile and the
input-side share (co-primary B statistic D) at cfg 1.5; benign levels and
relative susceptibility; LPIPS and L2 displacement ratios; the unconditional
half; the energy probe (`energy_eps0.05.pt`, LSE shift).

What would change the paper: a fingerprint that vanishes under guidance
(d near 0) would confine the result to unguided sampling; a sign change of
Delta_A or a profile reversal would be reported as a guidance dependence.
A fingerprint of the same sign and comparable d removes limitation (xv)'s
"class conditioning at cfg 1 only" clause.

## Cost (from the dry run, filled in before launch)

See `../phase5_pretrained/DESIGN.md`, section "Dry runs", for the measured
rates; the queue is `scripts/run_phase5_all.sh`.

## Provenance

Runner `scripts/run_phase5.py` (imports the seed helpers from the frozen
`scripts/run_phase3_grid.py`, which is not modified); regression test
`scripts/test_phase5_regression.py` (the new runner at cfg 1.0 replays the
Phase-3 substrate for the first 50 seeds); each leg writes `meta.json` with
the checkpoint md5, the git commit, the torch version, the peak VRAM and the
wall time. Launched with nohup from the Rome machine; the owner works from
Munich and monitors the logs read-only.
