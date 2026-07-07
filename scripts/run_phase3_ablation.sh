#!/usr/bin/env bash
# Attack-convergence ablation (PREREG section 6): {20, 80}-iter legs at the
# primary eps, both models; the 40-iter point is the main grid itself. Same
# runner and same per-seed anchoring (z_T, Rademacher, PGD init keyed by eps
# value and seed id), so the three iteration counts share their start points.
# No NFE-transfer here (empty --nfe-extra). Launch via nohup:
#   nohup bash scripts/run_phase3_ablation.sh > experiments/phase3_main/run_ablation.log 2>&1 &
# Resumable: each leg resumes from its own partial.pt -- relaunch the same command.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/third_party/sit"
PY=".venv/bin/python"

run() {
    echo "=================================================================="
    echo ">>> START $* :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "=================================================================="
    "$PY" -u scripts/run_phase3_grid.py "$@"
    echo "<<< DONE $* :: exit=$? :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run --model UNet-B --eps-grid 0.05 --nfe-extra --n-steps-attack 20 --tag it20
run --model UNet-B --eps-grid 0.05 --nfe-extra --n-steps-attack 80 --tag it80
run --model SiT-B/2 --eps-grid 0.05 --nfe-extra --n-steps-attack 20 --tag it20
run --model SiT-B/2 --eps-grid 0.05 --nfe-extra --n-steps-attack 80 --tag it80

echo "ABLATION RUNS COMPLETE :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
