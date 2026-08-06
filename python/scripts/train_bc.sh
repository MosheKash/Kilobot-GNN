#!/bin/bash
# Behaviour cloning against simple_oracle.py, the supported teacher.
#
# Replaces two earlier scripts that drove the removed lock-based oracle through
# launch.py's KILOBOT_MODE=bc path. run_bc_monitored.py is the maintained
# driver: it holds out formations for validation, records a val tape, and
# reports per-oracle-state imitation error rather than coverage alone.
#
# Writes to <out-dir>/: actor_latest.pt, actor_best.pt, history.jsonl, the val
# tape and the progress plots.
#
# Safe to interrupt and resume: rerun with the SAME out-dir and it picks up
# actor_latest.pt, history.jsonl and the persisted reservoir automatically.
# There is no resume flag.
#
# usage: ./scripts/train_bc.sh <out-dir> [iterations] [instances] [arenas]
#   ./scripts/train_bc.sh ../results/bc_run
#   ./scripts/train_bc.sh ../results/bc_run 300 4 4

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:?usage: ./scripts/train_bc.sh <out-dir> [iterations] [instances] [arenas]}"
ITERATIONS="${2:-300}"
INSTANCES="${3:-4}"
ARENAS="${4:-4}"

echo "BC: teacher=simple_oracle, $INSTANCES player(s) x $ARENAS arena(s), $ITERATIONS iterations -> $OUT_DIR"
echo ""

python run_bc_monitored.py "$OUT_DIR" \
    --iterations "$ITERATIONS" \
    --instances "$INSTANCES" \
    --arenas "$ARENAS" \
    --min-bots 40 --max-bots 60 \
    --heartbeat 48 \
    --use-arrived-head --use-turn-anchor \
    --bc-replay-capacity 2000000 --bc-replay-persist \
    --swarm-rng 0 \
    "${@:5}"
