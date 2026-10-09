#!/bin/bash
# retrain_dagger1.sh -- Path 4 round 1: re-train BC from the oracle tapes PLUS
# the newly recorded DAgger tape of the o3 hybrid's own trajectories
# (tape_o3_hyb_1.pt), with the SAME config that produced the strongest clone
# (results/bc_v2/run_o3, select=turn_bias_wall_following).
#
# All paths absolute; run from anywhere (cd's into python/ internally).
set -o errexit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(dirname "$SCRIPT_DIR")"
REPO="$(dirname "$PY_DIR")"
cd "$PY_DIR"

DAGGER="$REPO/results/plan_execution/dagger"
OUT="$DAGGER/run_dagger1"
mkdir -p "$OUT"

python bc_offline.py "$OUT" \
    --train-tape "$REPO/results/bc_v2/tape_train.pt" \
    "$REPO/results/bc_v2/fixed_dagger_1a.pt" \
    "$REPO/results/bc_v2/fixed_dagger_1b.pt" \
    "$REPO/results/bc_v2/fixed_dagger_2a.pt" \
    "$REPO/results/bc_v2/fixed_dagger_2b.pt" \
    "$DAGGER/tape_o3_hyb_1.pt" \
    --val-tape "$REPO/results/bc_v2/tape_val.pt" \
    --epochs 100 --batch-robots 128 --chunk 96 --lr 0.001 --warmup 200 \
    --max-grad-norm 1.0 --activation elu --state-balance --state-weight-cap 8.0 \
    --skip-arrived --arrived-motor-weight 0.1 --arrived-motor-floor 0.02 \
    --arrived-loss-weight 1.0 --arrived-balance balanced --arrived-threshold 0.95 \
    --state-head-weight 1.0 --wall-head-weight 1.0 --oracle-head --oracle-residual 0.05 \
    --select turn_bias_wall_following --wall-label latched --obs-noise 0.03 \
    --use-arrived-head --use-turn-anchor --device cuda --eval-every 1 \
    > "$OUT/train.log" 2>&1 < /dev/null &
echo "launched dagger1 retrain pid $! -> $OUT/train.log"