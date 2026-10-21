# Phase 5b — official SiT checkpoints (post-thesis, exploratory)

Written before launch, 2026-10-21. Not pre-registered; every number produced
here is reported as "run after the thesis, not pre-registered".

## Question

The thesis measured one pair of models trained by us at size B. Does the
attention fingerprint of the latent PGD attack appear on models trained by
someone else, at state-of-the-art quality, and how does it change with size?

## Checkpoints

Source: Hugging Face `nyu-visionx/SiT-collections`, revision
`9a43ddf08f80ac7f3525208ce72943e98fe98343`, MIT license, released by the
SiT authors (willisma/SiT issue #42; the Dropbox link of the README is dead).
Downloaded 2026-10-21 into `checkpoints/pretrained/` (gitignored); md5 in
each leg's `meta.json`. Plain state dicts (no "ema" key), loaded with
`strict=True` into the vendored `third_party/sit/models.py` definitions.

| name | blocks | hidden | heads | params | training | file |
|------|--------|--------|-------|--------|----------|------|
| SiT-S/2 | 12 | 384 | 6 | 33 M | 800k steps, global batch 256 | SiT-S-2-256.pt (0.13 GB) |
| SiT-B/2 | 12 | 768 | 12 | 130 M | 800k steps, global batch 256 | SiT-B-2-256.pt (0.52 GB) |
| SiT-L/2 | 24 | 1024 | 16 | 458 M | 800k steps, global batch 256 | SiT-L-2-256.pt (1.83 GB) |
| SiT-XL/2 | 28 | 1152 | 16 | 675 M | 7M steps (paper model, FID 2.06 at cfg 1.5) | SiT-XL-2-256.pt (2.70 GB) |

All four: Linear interpolant + velocity prediction (the SiT training
default, `third_party/sit/train_utils.py`), the same convention as our
training and our Euler sampler (t=0 noise, t=1 data, x + dt v);
`sd-vae-ft-ema` latents 32x32x4, patch 2, 256 tokens; class dropout 0.1 so
the null label row (index 1000) exists. Sanity check passed 2026-10-21:
`scripts/phase5_sanity_samples.py` decodes recognizable images of the eight
requested classes from B/2 and XL/2 with our 25-step Euler sampler at cfg
1.5 (`sanity/*.png`).

The official SiT-B/2 is architecturally identical to our DiT (same
`SiT_B_2` definition); it differs only in who trained it and how (800k steps
at batch 256 = 205 M samples, against our 6.4M steps at batch 16 = 102 M
samples). It is the leg that separates "who trained it" from "how big it is".

## Legs and settings

| id | model | cfg | seeds | batch | eps | PGD | sampler |
|----|-------|-----|-------|-------|-----|-----|---------|
| E2 | SiT-B/2-official | 1.0 | 500 (ids 0..499, labels i mod 1000) | 25 | 0.05 | 40 iters, eps/4, random start | 25 Euler, fp32 |
| E3 | SiT-XL/2-official | 1.0 | 500 | see dry run | 0.05 | same | same |
| E4 | SiT-XL/2-official | 1.5 | 500 | see dry run | 0.05 | same | same |
| E5 | SiT-S/2-official, SiT-L/2-official | 1.0 | 500 | 25 / 15 | 0.05 | same | same |

Seeds, Rademacher directions and PGD starts are the Phase-3 ones
(`SEED_BASE 21260705`), so every leg pairs seed by seed with every other
leg and with the Phase-3 legs. Locus: N = 256 = every block of every
transformer; late window t >= 0.72 (steps 17–24). Guidance as in
`../phase5_cfg/DESIGN.md`.

## Readings

Within-model only (there is no public UNet counterpart): per leg the
late-window effective-rank contrast c = M_rand − M_pgd (mean, SD, d,
bootstrap CI95, paired p), the benign level and the relative
susceptibility, late vs all-trajectory, the per-layer profile and the
input-side share, the presence of the block-0 attention sink in the benign
maps (entropy and erank per block), the within-model detector AUC (z-units
from the leg's own benign branch) and the transfer of the Phase-3
thresholds, the energy LSE shift, LPIPS and L2 displacement ratios.

Comparisons that the design supports: official B/2 vs our DiT (same
network, different training: replication and a third maturity point next
to DiT@FID95 and DiT@FID68); S/B/L/XL at cfg 1 (size curve, same recipe for
S/B/L); XL at cfg 1 vs cfg 1.5 (guidance at scale). No cross-architecture
Delta is computed here.

## Dry runs (filled in before launch)

Measured 2026-10-21 on the RX 6900 XT (torch 2.11.0+rocm7.2), one batch each,
`experiments/phase5_tests/`:

| run | batch | s/seed | peak VRAM | 500 seeds |
|-----|-------|--------|-----------|-----------|
| our SiT-B/2, cfg 1.0 (regression replay, 50 seeds) | 25 | 27.5 | 4.11 GB | 3.8 h |
| SiT-XL/2 official, cfg 1.0 | 10 | 118.0 | 7.33 GB | 16.4 h |
| SiT-XL/2 official, cfg 1.5 | 5 | see `SiT-XL-2-official.dry15/meta.json` | | |
| our SiT-B/2, cfg 1.5 | 25 | see `SiT-B-2.dry15/meta.json` | | |

Regression replay: the new runner at cfg 1.0 reproduces the Phase-3
substrate for seeds 0–49 bit for bit (z_T, z0, z_T_adv, every per-cell
metric, per-seed late-window contrast: max |diff| = 0). Batch sizes in the
queue: 25 for S/B and the UNet, 15 for L, 10 for XL at cfg 1, 5 for XL at
cfg 1.5.

## Provenance

Runner `scripts/run_phase5.py`; energy probe `scripts/run_phase5_energy.py`;
queue `scripts/run_phase5_all.sh` (idempotent, resumes from `partial.pt`,
skips legs with `meta.json`); each leg's `meta.json` carries checkpoint md5,
HF revision, git commit, torch version, peak VRAM and wall time.
