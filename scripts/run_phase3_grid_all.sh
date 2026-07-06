#!/usr/bin/env bash
# Drive the three Phase 3 runs sequentially (single GPU). SiT and UNet on the
# full eps grid; the DiT@FID95 convergence control at the primary eps only
# (no NFE-transfer on the control). Launch via nohup:
#   nohup bash scripts/run_phase3_grid_all.sh > experiments/phase3_main/run_all.log 2>&1 &
# Resumable: each run resumes from its own partial.pt if interrupted -- just
# relaunch the same command.
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

run --model SiT-B/2
run --model UNet-B
run --model SiT-B/2-FID95 --eps-grid 0.05 --nfe-extra

echo "ALL PHASE 3 RUNS COMPLETE :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
