#!/bin/bash
# Phase 5 queue: guidance ablation (E1) and official SiT checkpoints (E2-E5),
# then the energy probe on every leg (E6). Sequential on the one GPU.
# Idempotent: a leg whose meta.json exists is skipped by the runner itself,
# a half-done leg resumes from its partial.pt. Relaunch after any crash with
#   nohup bash scripts/run_phase5_all.sh > experiments/phase5_pretrained/run_all.log 2>&1 &
# Design notes: experiments/phase5_cfg/DESIGN.md, experiments/phase5_pretrained/DESIGN.md
set -u
cd /home/marco/disattend
PY=.venv/bin/python
CFG=experiments/phase5_cfg
PRE=experiments/phase5_pretrained
N=${N_SEEDS:-500}
BS_XL1=${BS_XL1:-10}     # XL/2 at cfg 1   (dry run: see DESIGN.md)
BS_XL15=${BS_XL15:-5}    # XL/2 at cfg 1.5
BS_L=${BS_L:-15}

leg () {  # leg <model> <cfg> <batch> <out-dir> <log-name> [tag]
  local model=$1 cfg=$2 bs=$3 out=$4 log=$5 tag=${6:-}
  echo "== $(date '+%F %T') leg $model cfg$cfg bs$bs -> $out/$log"
  $PY -u scripts/run_phase5.py --model "$model" --cfg-scale "$cfg" --n-seeds "$N" \
      --batch-size "$bs" --out-dir "$out" ${tag:+--tag "$tag"} 2>&1 | grep --line-buffered -v "^(null)" > "$out/$log.log"
  tail -n 2 "$out/$log.log"
}
energy () {  # energy <model> <cfg> <leg-dir> <batch>
  echo "== $(date '+%F %T') energy $1 cfg$2 on $3"
  $PY -u scripts/run_phase5_energy.py --model "$1" --cfg-scale "$2" --leg "$3" \
      --batch-size "$4" 2>&1 | grep --line-buffered -v "^(null)" >> "$3/energy.log"
  tail -n 1 "$3/energy.log"
}

# E1  guidance ablation on our two models (paired with Phase 3 at cfg 1)
leg SiT-B/2 1.5 25 $CFG SiT-B-2
leg UNet-B  1.5 25 $CFG UNet-B
# E2  official SiT-B/2 (800k steps, batch 256), cfg 1
leg SiT-B/2-official 1.0 25 $PRE SiT-B-2-official
# E3  official SiT-XL/2, cfg 1
leg SiT-XL/2-official 1.0 $BS_XL1 $PRE SiT-XL-2-official
# E4  official SiT-XL/2, cfg 1.5
leg SiT-XL/2-official 1.5 $BS_XL15 $PRE SiT-XL-2-official.cfg15 cfg15
# E5  size curve, cfg 1
leg SiT-S/2-official 1.0 25 $PRE SiT-S-2-official
leg SiT-L/2-official 1.0 $BS_L $PRE SiT-L-2-official
# E6  energy probe on every leg
energy SiT-B/2 1.5 $CFG/SiT-B-2 25
energy UNet-B  1.5 $CFG/UNet-B 25
energy SiT-B/2-official 1.0 $PRE/SiT-B-2-official 25
energy SiT-XL/2-official 1.0 $PRE/SiT-XL-2-official $BS_XL1
energy SiT-XL/2-official 1.5 $PRE/SiT-XL-2-official.cfg15 $BS_XL15
energy SiT-S/2-official 1.0 $PRE/SiT-S-2-official 25
energy SiT-L/2-official 1.0 $PRE/SiT-L-2-official $BS_L
echo "== $(date '+%F %T') PHASE 5 QUEUE COMPLETE"
