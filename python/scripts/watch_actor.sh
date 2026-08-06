#!/bin/bash
# Drive one visible Unity arena from a trained checkpoint, with no training, so
# you can watch what the actor actually does.
#
# Needs a HEADED build; see scripts/watch_oracle.sh for why, and point
# KILOBOT_BUILD_PATH at it.
#
# usage: ./scripts/watch_actor.sh <checkpoint> [seed_layout] [num_arenas] [time_scale] [formations_dir]
#   ./scripts/watch_actor.sh ../results/tb/run_.../actor_final.pt

set -euo pipefail
cd "$(dirname "$0")/.."


if [ -z "${1:-}" ]; then
    echo "usage: $0 <path-to-ckpt.pt-or-actor_final.pt> [episodes] [seed_layout] [max_formations] [formations_dir] [use_arrived_head] [use_turn_anchor]"
    echo "  e.g.: $0 ../results/tb/run_20260714_182306/ckpt.pt"
    exit 1
fi

WEIGHTS="$1"
EPISODES="${2:-999999}"
SEED_LAYOUT="${3:-corners}"
MAX_FORMATIONS="${4:-1}"
FORMATIONS_DIR="${5:-../data/formations}"
# New, optional, trailing -- default false preserves every existing
# invocation unchanged. Set to true only for a checkpoint actually trained
# with the matching run_bc_monitored.py flag; passing true against a
# checkpoint trained without it produces the same load_state_dict error
# in reverse (missing keys / a narrower size expected).
USE_ARRIVED_HEAD="${6:-false}"
USE_TURN_ANCHOR="${7:-false}"
# Colors a robot green the moment its own arrived head switches it off, using
# the phase-48 visual-state channel that already exists end to end (python ->
# CriticChannel -> SwarmManager.SetRobotStates -> KilobotAgent.SetVisualState,
# where state 4 is already defined as "arrived, stopped -> green"). No Unity
# change is involved. Defaults ON for this script specifically: with
# bc_motor_skip_arrived the motor head is deliberately never trained to output
# a stop, so the arrived head is the ONLY thing that halts a robot, and being
# able to see it fire live is the point of watching at all. Set to false to
# restore the previous, uncolored behaviour.
SHOW_ARRIVED="${8:-true}"

if [ ! -f "$WEIGHTS" ]; then
    echo "error: $WEIGHTS does not exist"
    exit 1
fi

echo "watching $WEIGHTS, seed_layout=$SEED_LAYOUT, max_formations=$MAX_FORMATIONS, formations=$FORMATIONS_DIR, use_arrived_head=$USE_ARRIVED_HEAD, use_turn_anchor=$USE_TURN_ANCHOR (Ctrl+C to stop; pass an explicit episode count as \$2 for a bounded run instead)"
echo "a Unity window should open shortly -- if it doesn't, check that KILOBOT_BUILD_PATH points at a headed (non-server) build; a headless build has no window to show regardless of KILOBOT_NO_GRAPHICS"
echo ""

env KILOBOT_EVAL=1 KILOBOT_EVAL_WEIGHTS="$WEIGHTS" KILOBOT_EVAL_ITERS="$EPISODES" \
    KILOBOT_MAX_FORMATIONS="$MAX_FORMATIONS" \
    KILOBOT_MAX_EPISODE_STEPS=1000000000 KILOBOT_SUCCESS_THRESHOLD=1.1 \
    KILOBOT_NUM_ARENAS=1 KILOBOT_NO_GRAPHICS=false KILOBOT_TIME_SCALE=1 \
    KILOBOT_SHOW_RADIUS=true KILOBOT_LOG_FORMATIONS=true \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_SEED_LAYOUT="$SEED_LAYOUT" KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_USE_ARRIVED_HEAD="$USE_ARRIVED_HEAD" KILOBOT_USE_TURN_ANCHOR="$USE_TURN_ANCHOR" \
    KILOBOT_ORACLE_SEND_VISUAL_STATE="$SHOW_ARRIVED" \
    KILOBOT_FORMATIONS="$FORMATIONS_DIR" python launch.py


