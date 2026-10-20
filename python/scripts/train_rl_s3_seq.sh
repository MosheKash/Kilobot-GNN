#!/bin/bash
# Sequential Stage 3 pair: comms arm first, then the no-comms control.
#
# Why sequential: running both arms concurrently means 16 Unity players +
# 2 CUDA learners inside one cgroup. Measured (systemd journal,
# app-org.kde.konsole-98395.scope): 21.8G memory peak + 21.5G swap peak from
# that one scope -- the terminal froze, systemd tore the scope down, and the
# whole pair died with it. zram swap is RAM, so it relieves nothing. One arm
# at a time (8 players ~ 9-11G) fits alongside the desktop.
#
# Run via systemd --user (see train_rl_s3_launch_note.sh or the systemd-run
# line in the tuning log) so the training lives in its own cgroup and
# survives terminal crashes and logouts.
#
# Usage: train_rl_s3_seq.sh <iterations> <out_root>
set -o errexit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(dirname "$SCRIPT_DIR")"
REPO="$(dirname "$PY_DIR")"
cd "$PY_DIR"

ITERATIONS="${1:-100}"
OUT="${2:-$REPO/results/rl_s3}"
mkdir -p "$OUT/comms" "$OUT/control"

# The systemd --user environment has no conda activation: name the env's
# interpreter explicitly (a bare `python` here resolves to the base env and
# dies on `import numpy` before any training starts).
PY=/home/moshe/miniconda3/envs/kilobot-gnn/bin/python
# Allocator hygiene for the learner (host RAM only; cannot change learning results):
# glibc arena ratchet was measured as the main consumer of the learner's 7-8G RSS.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=134217728

env_common() {
  echo "KILOBOT_INIT_ACTOR=$REPO/results/bc_v2/run_r0/actor_best.pt"
  echo "KILOBOT_MODE=rl"
  echo "KILOBOT_SMOKE=0"
  echo "KILOBOT_ITERATIONS=$ITERATIONS"
  echo "KILOBOT_NUM_WORKERS=4"
  echo "KILOBOT_DEVICE=cuda"
  echo "KILOBOT_ACTOR_TYPE=gru_split_observation"
  echo "KILOBOT_SPLIT_ACTIVATION=elu"
  echo "KILOBOT_USE_ARRIVED_HEAD=1"
  echo "KILOBOT_USE_TURN_ANCHOR=1"
  echo "KILOBOT_USE_STATE_HEAD=1"
  echo "KILOBOT_USE_WALL_HEAD=1"
  echo "KILOBOT_USE_ORACLE_HEAD=1"
  echo "KILOBOT_HEARTBEAT_TICKS=48"
  echo "KILOBOT_MIN_BOTS=40"
  echo "KILOBOT_MAX_BOTS=60"
  echo "KILOBOT_NUM_ARENAS=9"
  echo "KILOBOT_ROLLOUT_STEPS=1024"
  echo "KILOBOT_MAX_EPISODE_STEPS=8000"
  echo "KILOBOT_REWARD_MODE=target"
  echo "KILOBOT_TARGET_ACC_WEIGHT=1.0"
  echo "KILOBOT_TARGET_ACC_SIGMA=0.05"
  echo "KILOBOT_TARGET_TIME_COST=0.1"
  echo "KILOBOT_TARGET_SHAPING=5.0"
  echo "KILOBOT_STOP_BONUS=1.0"
  echo "KILOBOT_STOP_RADIUS=0.05"
  echo "KILOBOT_GLOBAL_SHAPING=2.0"
  echo "KILOBOT_BELIEF_SHAPING=2.0"
  echo "KILOBOT_SUCCESS_METRIC=settled"
  echo "KILOBOT_SUCCESS_THRESHOLD=0.75"
  echo "KILOBOT_USE_CLOSED_FORM_ARRIVED=1"
  echo "KILOBOT_CLOSED_FORM_HYBRID=1"
  echo "KILOBOT_CLOSED_FORM_ARRIVAL_DIST=0.08"
  echo "KILOBOT_ARRIVED_RELEASE_THRESHOLD=0.5"
  echo "KILOBOT_ARRIVED_FREEZE_HIDDEN=0"
  echo "KILOBOT_ARRIVAL_MIN_ANCHORED=1"
  echo "KILOBOT_ARRIVAL_PERSIST=2"
  echo "KILOBOT_TC_PEER_ZERO=1"
  echo "KILOBOT_CRITIC_BELIEF_FEATURES=1"
  echo "KILOBOT_QUIET_MSG_HEAD=1"
  echo "KILOBOT_MSG_LOG_STD=-2.5"
  echo "KILOBOT_MOTOR_LOG_STD=-2.5"
  echo "KILOBOT_LOG_STD_MIN=-4.0"
  echo "KILOBOT_MSG_LOG_STD_MAX=-2.5"
  echo "KILOBOT_ACTOR_LR=2e-5"
  echo "KILOBOT_PPO_EPOCHS=2"
  echo "KILOBOT_ENTROPY_COEF=0.001"
  echo "KILOBOT_ORACLE_KNOWN_START_HEADING=1"
  echo "KILOBOT_SWARM_RNG=500"
  echo "KILOBOT_FORMATIONS=$REPO/data/formations"
}

run_arm() { # name port extra...
  local name=$1 port=$2
  shift 2
  echo "[$(date +%H:%M:%S)] starting arm $name (port $port)"
  set +e
  env $(env_common) \
    KILOBOT_LOGDIR="$OUT/$name/tb" \
    KILOBOT_UNITY_LOG_DIR="$OUT/$name/unity-logs" \
    KILOBOT_SUMMARY="$OUT/$name/live_summary.json" \
    KILOBOT_BASE_PORT=$port \
    "$@" \
    "$PY" -u launch.py > "$OUT/$name/train.log" 2>&1
  local rc=$?
  set -e
  echo "[$(date +%H:%M:%S)] arm $name finished (exit $rc)"
  return $rc
}

# Arm A: the learned transmission channel
run_arm comms 7100 \
  KILOBOT_BELIEF_COMMS=1 \
  KILOBOT_BELIEF_COMMS_LEARNED=1 \
  KILOBOT_MSG_SUPERVISION=0.5 \
  KILOBOT_MSG_HEAD_PROP_INPUT=1 \
  KILOBOT_BEACON_HONESTY=4.0 \
  || echo "[$(date +%H:%M:%S)] arm comms FAILED -- continuing to the control"

# Arm B: the no-comms control
run_arm control 7200 \
  KILOBOT_BELIEF_COMMS=0 \
  KILOBOT_BELIEF_COMMS_LEARNED=0 \
  KILOBOT_MSG_SUPERVISION=0 \
  KILOBOT_MSG_HEAD_PROP_INPUT=0 \
  KILOBOT_BEACON_HONESTY=0 \
  || echo "[$(date +%H:%M:%S)] arm control FAILED"

echo "[$(date +%H:%M:%S)] pair complete"
