#!/usr/bin/env bash
# Drive all four Phase 2 pilot runs sequentially (single GPU, compute-bound).
# AB parts first (core paired attention comparison, cheaper), then the FID
# parts (1000 PGD attacks/model, the expensive tail). Launch via nohup:
#   nohup bash scripts/run_phase2_pilot_all.sh > experiments/phase2_pilot/run_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH="$PWD:$PWD/third_party/sit"

run() {
    echo "=================================================================="
    echo ">>> START $* :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "=================================================================="
    "$PY" -u scripts/run_phase2_pilot.py "$@"
    echo "<<< DONE $* :: exit=$? :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run --model SiT-B/2 --part ab  --batch-size 25
run --model UNet-B  --part ab  --batch-size 25
run --model SiT-B/2 --part fid --batch-size 25
run --model UNet-B  --part fid --batch-size 25

echo "ALL PHASE 2 PILOT RUNS COMPLETE :: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
