# FID numbers: one number = one artifact + one protocol

Every FID in this repo was computed against OUR reference statistics
(`data/imagenet_latents/fid_ref_stats.pt`, precomputed from the ImageNet-256
set with `scripts/precompute_fid_ref.py`), so no value here is comparable
with published paper numbers (different reference suite and preprocessing).
FID depends strongly on n_samples, NFE and sampling dtype: numbers are
comparable ONLY within the same row-protocol. `scripts/eval_fid.py` defaults
to `--amp-dtype float16`; the July fp32 standardization exists because the
UNet cannot sample in fp16 (attention overflow), so the only protocol shared
fairly by both backbones is fp32. The result jsonl schema does not record
the dtype; the file NAME and the log carry it (fixed going forward by this
table).

## Canonical (thesis-facing) numbers - 50k samples, NFE 25, fp32

| model | FID-50k | artifact | date |
|---|---|---|---|
| SiT-B/2 @ 6.4M steps | 68.67 | `experiments/20260520-SiT-B-2-recovery/fid_50k_fp32.jsonl` | 2026-07-01 |
| UNet-B @ 6.4M steps | 99.14 | `experiments/20260611-UNet-B-cosine-6p4M/fid_50k.jsonl` | 2026-06-20 |

The UNet jsonl and log do not echo the dtype (schema gap): fp32 is
documented in `docs/lab_notebook.md` ("fp32 FID-50k: SiT-B/2 = 68.67,
UNet-B = 99.14"), and fp16 is excluded a fortiori because the UNet's fp16
attention overflow yields NaN/garbage FID, not a sane value.
| SiT-B/2 @ 150k steps, batch 64 (matched-quality control) | 95.40 | `experiments/20260518-SiT-B-2-b64/fid_50k_fp32_step150k.jsonl` | 2026-07-04 |

## Registered pre-checks and diagnostics (different n, still fp32-era)

| purpose | value | protocol | artifact |
|---|---|---|---|
| matched-quality pre-check @ 4M steps | 77.67 | 5k, NFE 25 | `.../fid_precheck_step04000000.json` (2026-07-06) |
| matched-quality pre-check @ 5M steps | 75.79 | 5k, NFE 25 | `.../fid_precheck_step05000000.json` (2026-07-06) |
| pilot branch levels (benign/rand/PGD) | 107.8 / 107.5 / 91.7 (SiT), 136.1 / 136.6 / 119.0 (UNet) | 1k per branch, small-n inflated: use only for the within-pilot contrast | `experiments/phase2_pilot/analysis.json` |
| latent-flip diagnostic | flip-FID +71 | see file | `notebooks/out/flip_fid_diag.json` |

## Superseded (fp16-era, pre-standardization) - do not cite in the thesis

| value | protocol | artifact | note |
|---|---|---|---|
| 68.18 | 50k, NFE 25, fp16 | `notebooks/out/fid_50k.json` | early notebook eval; superseded by 68.67 fp32 |
| 68.63 | 50k, NFE 25, fp16 | `.../fid_sweep.jsonl` last row (2026-05-29) | same, sweep-era |
| 71.36 / 69.69 / 68.73 | 5k, NFE 50/100/250, fp16 | `.../fid_sweep.jsonl` (2026-05-29) | NFE sweep, small n |
| 74.46 / 70.54 / 68.83 / 67.89 | 5k, NFE 25/50/100/250, fp16 | `notebooks/out/fid_sweep_final.json` | notebook-era NFE sweep |

The three 50k@NFE25 SiT values (68.18 fp16 notebook, 68.63 fp16 sweep, 68.67
fp32 standard) agree within ~0.5 FID: the discrepancy the numbers show is
dtype plus sampling-seed noise, not a real re-evaluation drift. The thesis
cites 68.67 and 99.14 (and 95.40 for the control) and nothing else at 50k.
