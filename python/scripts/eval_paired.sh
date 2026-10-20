#!/bin/bash
# eval_paired.sh -- the canonical paired evaluation harness on the BC clone.
#
# One line per driver: same --swarm-rng, same --seed, same arenas, so two
# runs differ only in the driver. Phase 169's method; this script exists
# because an earlier one-off invocation omitted a backslash and ran
# "--swarm-rng: command not found", wasting an eval round.
#
# Base is the A7 recipe (clone + protection stack, no fusion). Pass extra
# flags after the out-path to change it. IMPORTANT: quote everything in the
# extra args -- they are passed through as-is (that is the point of the
# helper, and the exact bug it exists to prevent).
#
# Usage: eval_paired.sh RESULTS_FILE.json [EXTRA ARGS...]
#   ./scripts/eval_paired.sh out.json --peer-fix --peer-conf-min 0.70

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(dirname "$SCRIPT_DIR")"
REPO="$(dirname "$PY_DIR")"
cd "$PY_DIR"

OUT="${1:?usage: eval_paired.sh RESULTS_FILE.json [extra.args...]}"
shift
EXTRA=("$@")

python -u tools/eval_closed_loop.py "$OUT" \
  --mode actor \
  --weights "$REPO/results/bc_v2/run_r0/actor_best.pt" \
  --ticks 8000 --sample-every 200 --instances 1 --arenas 4 \
  --arrived-release-threshold 0.5 \
  --closed-form-arrived --closed-form-dist 0.08 --closed-form-hybrid \
  --belief-comms --beacon-arrived-only --tc-peer-zero \
  --peer-innovation-min 1.1 --arrival-min-anchored --arrival-persist 2 \
  --filter-seed 7 --swarm-rng 500 --seed 7 --device cuda \
  "${EXTRA[@]}"