# Disattend

Adversarial characterization of attention patterns in Diffusion Transformers
(DiT) versus UNet backbones under gradient-based perturbations.

Bachelor's thesis in Mathematical Sciences for Artificial Intelligence,
Sapienza University of Rome.

## Overview

The project asks a descriptive question: when a standard gradient-based
adversarial attack (PGD on the initial latent `z_T`, `L_inf`-bounded) is
applied to a pre-trained class-conditional Diffusion Transformer
([DiT-XL/2, Peebles & Xie 2022](https://arxiv.org/abs/2212.09748)), how does
the perturbation manifest in the model's self-attention maps, compared to
the same attack applied to a UNet diffusion backbone?

The goal is empirical characterization, not a robustness claim. A null
result ("no distinctive pattern") is a valid outcome.

### Planned contributions

1. Infrastructure to extract per-layer, per-timestep attention maps from
   DiT-XL/2 during class-conditional sampling.
2. A PGD pipeline on `z_T` that propagates gradients through the sampling
   loop, fitting in 16 GB of VRAM via gradient checkpointing.
3. Side-by-side measurement of attention-based metrics (entropy shift,
   effective-rank drop) and image-quality metrics (differential FID) on
   benign versus adversarial samples, across DiT and UNet backbones.

## Setup

Requires Linux with AMD ROCm 7.2+ and Python 3.12.

```bash
# Install uv if not present (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install the current dependency set
uv sync

# Verify GPU and core stack
uv run python scripts/verify_setup.py
```

`uv sync` populates `.venv/` with Python 3.12, PyTorch 2.11 on ROCm 7.2,
and the minimal scientific stack. Model-specific libraries (`diffusers`,
`transformers`, `accelerate`, `huggingface_hub`) will be added when Phase 1
actually needs them.

## Repository layout

```
disattend/
├── src/           # models, attacks, evaluation, utils
├── scripts/       # entry points (verify_setup, future sampling/attack runners)
├── configs/       # experiment configs
├── notebooks/     # exploration only, not final results
├── docs/          # setup, lab notebook, per-phase planning
├── experiments/   # timestamped run outputs (gitignored)
├── checkpoints/   # model weights (gitignored)
└── data/          # datasets (gitignored)
```

## Key references

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023.
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018.
- Salman et al., *PhotoGuard: Raising the Cost of Malicious AI-Powered Image Editing*, ICML 2023.
- Dong et al., *Attention is not all you need: Pure attention loses rank doubly exponentially with depth*, ICML 2021.

## Status

- **Phase 0** — Repository bootstrap and environment setup (April 2026) — in progress.

This is an early-stage repository. A sibling project
([`slowflow`](https://github.com/m4rch1n0/slowflow), private) provides the
UNet flow-matching baseline that may be referenced in later phases.
