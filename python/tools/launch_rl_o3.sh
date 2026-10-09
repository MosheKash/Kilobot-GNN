#!/bin/bash
# launch_rl_o3.sh -- Path 1: MAPPO fine-tune from the o3 actor.
#
# All paths are ABSOLUTE (derived from this script's own location), so nothing
# can resolve to an external directory. CWD is set to python/ as the plan
# requires and every ../results path is spelled out to the repo's real path.
#
# CORRECTED vs results/immediate_plan.md's command block:
#   * KILOBOT_SMOKE=false      (launch.py's SMOKE default is TRUE: without this
#                               the run only does 3 tiny iterations)
#   * KILOBOT_ACTOR_TYPE=gru_split_observation, KILOBOT_SPLIT_ACTIVATION=elu,
#     the four head flags and heartbeat -- all must match what
#     results/bc_v2/run_o3/actor_best.pt was trained with or load_for_eval
#     refuses the checkpoint (shape mismatch).
#   * KILOBOT_ITERATIONS=300   (the proposal's curriculum length)
#   * KILOBOT_NUM_ARENAS=9, rollout 4096, max_episode 2048 = the config.json
#     phase-2 settings (also the Config defaults).
#   * The reward-shaping curriculum (5.0/2.0/0.0 at iters 0/50/150) is now
#     applied by trainer.py's shaping_schedule + parallel.py's payload pass.
#
# MEMORY DEVIATION: the plan's config.json phase-2 says num_arenas=9, which
# OOM-killed the main process twice on this 30GB box (the merged PPO buffer of
# 4 workers x 9 arenas x 4096 steps x ~50 robots spikes ~11GB; only ~18GB was
# available). num_arenas lowered 9->4 AND rollout_steps 4096->2048 here so the
# run fits; everything else from the plan/config.json is kept.

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(dirname "$SCRIPT_DIR")"                       # <repo>/python
REPO="$(dirname "$PY_DIR")"                             # <repo>
cd "$PY_DIR"

RESULTS="$REPO/results"
LOGDIR="$RESULTS/plan_execution/rl_o3"
mkdir -p "$LOGDIR"

KILOBOT_INIT_ACTOR="$RESULTS/bc_v2/run_o3/actor_best.pt" \
KILOBOT_MODE=rl \
KILOBOT_NUM_WORKERS=4 \
KILOBOT_DEVICE=cuda \
KILOBOT_SMOKE=false \
KILOBOT_ITERATIONS=300 \
KILOBOT_ACTOR_TYPE=gru_split_observation \
KILOBOT_SPLIT_ACTIVATION=elu \
KILOBOT_USE_ARRIVED_HEAD=1 \
KILOBOT_USE_TURN_ANCHOR=1 \
KILOBOT_USE_STATE_HEAD=1 \
KILOBOT_USE_WALL_HEAD=1 \
KILOBOT_USE_ORACLE_HEAD=1 \
KILOBOT_USE_STEER_FEATURE=0 \
KILOBOT_HEARTBEAT_TICKS=48 \
KILOBOT_NUM_ARENAS=4 \
KILOBOT_ROLLOUT_STEPS=2048 \
KILOBOT_REWARD_SHAPING=5.0 \
KILOBOT_ENTROPY_COEF=0.001 \
KILOBOT_LOG_STD_INIT=-0.5 \
KILOBOT_BELIEF_CONF_BONUS=0.5 \
KILOBOT_BELIEF_CONF_BONUS_ITERS=200 \
KILOBOT_SEED_FIND_BONUS=0.01 \
KILOBOT_WALL_FIND_PENALTY=0.01 \
KILOBOT_CRITIC_BELIEF_FEATURES=1 \
KILOBOT_SUMMARY="$LOGDIR/live_summary.json" \
KILOBOT_LOGDIR="$RESULTS/tb" \
nohup python -u launch.py > "$LOGDIR/train.log" 2>&1 < /dev/null &
echo "launched rl_o3 pid $! -> $LOGDIR/train.log"