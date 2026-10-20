#!/bin/bash
# train_rl_transmission.sh -- RL fine-tune that trains a transmission policy
# on top of the BC clone (results/bc_v2/run_r0).
#
# The corrected recipe. Three things were wrong with the phase-164..167
# attempts, all fixed here and each unit-tested:
#
#   1. THE WARM START DIED BEFORE PPO EVER RAN. A bc_offline checkpoint's
#      saved log_std is an untrained placeholder (std 0.61): BC fits the
#      mean, deterministic eval executes the mean, but RL's stochastic
#      rollout executes 0.5*(tanh(N(mean, 0.61))+1) per wheel -- random wheel
#      speeds from tick one (measured: settled 0.002 at iteration 0,
#      conf_pos 0.004, std_m/i 0.05/0.61, vs 0.63 settled<5u for the same
#      weights under deterministic eval). KILOBOT_MOTOR_LOG_STD=-2.5 starts
#      motor exploration quiet instead.
#
#   2. THE SENDER CREDIT WAS PAYING FOR THE SWARM GETTING WORSE. The shared
#      arena potential is a higher-is-better kernel, and phase 164/165
#      computed k*(prev_pot - gamma*pot) -- the exact negative of the PBRS
#      form -- so a helpful broadcast that raised the potential PENALIZED
#      its own sender. Corrected to k*(gamma*pot - prev_pot), and the
#      potential now has a second band over the belief error
#      (|filter mean - truth|), which is the quantity that actually moves
#      during the localization phase a broadcast speeds up: the settled
#      kernel alone is exp(-(d/0.05)^2), structurally zero until a robot
#      is nearly at its pixel, while the settle curve spends its first
#      ~3000 ticks localizing (0.016 settled<5u at tick 2000).
#
#   3. THE LEARNED BEACON COULD NEVER SWITCH ON. The receiver-side fusion
#      gate (PEER_CONF_MIN=0.85) needs the squashed conf slot past 0.85,
#      i.e. a pre-tanh mean above ~1.73 -- unreachable from a quiet start
#      (entropy explores amplitude, not offset). KILOBOT_MSG_HEAD_PROP_INPUT
#      widens head_msg's input to cat([g, prop]) so the beacon is a linear
#      readout of the robot's own filter state -- installed as the head's
#      INITIALIZATION (the honest protocol), refined in action space by
#      KILOBOT_MSG_SUPERVISION (trunk-detached: a trunk-connected
#      supervision was measured to wreck the motor policy within five
#      iterations), and free to be modulated by PPO's own gradient.
#
#   4. THE 1/sigma^2 AMPLIFICATION. Quiet stds (motor 0.082, message 0.05)
#      preserve the warm start's behavior but multiply every policy
#      gradient into the shared trunk by 1/sigma^2 ~ 150-400 -- measured as
#      kl 1 -> 10 per iteration and a ~0.09 motor-mean drift that quietly
#      destroyed the clone's steering. ACTOR_LR=2e-5 and PPO_EPOCHS=2 keep
#      the per-update drift inside the clone's steering precision (the
#      oracle's steering differential is 0.0093 rms), and the message std
#      starts at the cap (0.082) to halve its own amplification.
#
#   5. THE FILTER THE CLONE NEVER SAW. eval_closed_loop.py and
#      run_bc_monitored.py both set oracle_known_start_heading=True (the
#      spawn heading is a real setup convention: SwarmManager spawns robots
#      at cardinal headings and tells the filter which one). launch.py's RL
#      path defaults it to False and no previous recipe ever set it -- so
#      every RL run's particle filter started with uniformly random
#      per-particle headings, never localized (conf_pos 0.004,
#      frac_localized < 1%, belief error CLIMBING through every episode --
#      the same signature in the phase-167 run and both of today's), and
#      fed the clone belief features it was never trained on. Same
#      divergence family as phase 167's arrived_freeze_hidden.
#
# Also fixed in this phase: KILOBOT_CRITIC_BELIEF_FEATURES existed in a
# launch script but was never read by launch.py, and both launch paths built
# Critic() directly instead of make_critic -- wired now.
#
# Usage: train_rl_transmission.sh <iterations> <out_dir> [extra env as ENV=...]
# Run from anywhere; CWD is set to python/ as launch.py requires.

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$(dirname "$SCRIPT_DIR")"
REPO="$(dirname "$PY_DIR")"
cd "$PY_DIR"

ITERATIONS="${1:-100}"
OUT="${2:-$REPO/results/rl_transmission}"
mkdir -p "$OUT"

KILOBOT_INIT_ACTOR="$REPO/results/bc_v2/run_r0/actor_best.pt" \
KILOBOT_MODE=rl \
KILOBOT_SMOKE=0 \
KILOBOT_ITERATIONS="$ITERATIONS" \
KILOBOT_NUM_WORKERS=4 \
KILOBOT_DEVICE=cuda \
KILOBOT_ACTOR_TYPE=gru_split_observation \
KILOBOT_SPLIT_ACTIVATION=elu \
KILOBOT_USE_ARRIVED_HEAD=1 \
KILOBOT_USE_TURN_ANCHOR=1 \
KILOBOT_USE_STATE_HEAD=1 \
KILOBOT_USE_WALL_HEAD=1 \
KILOBOT_USE_ORACLE_HEAD=1 \
KILOBOT_HEARTBEAT_TICKS=48 \
KILOBOT_MIN_BOTS=40 \
KILOBOT_MAX_BOTS=60 \
KILOBOT_NUM_ARENAS=9 \
KILOBOT_ROLLOUT_STEPS=1024 \
KILOBOT_MAX_EPISODE_STEPS=8000 \
KILOBOT_LOGDIR="$OUT/tb" \
KILOBOT_UNITY_LOG_DIR="$OUT/unity-logs" \
KILOBOT_SUMMARY="$OUT/live_summary.json" \
\
KILOBOT_REWARD_MODE=target \
KILOBOT_TARGET_ACC_WEIGHT=1.0 \
KILOBOT_TARGET_ACC_SIGMA=0.05 \
KILOBOT_TARGET_TIME_COST=0.1 \
KILOBOT_TARGET_SHAPING=5.0 \
KILOBOT_STOP_BONUS=1.0 \
KILOBOT_STOP_RADIUS=0.05 \
KILOBOT_GLOBAL_SHAPING=2.0 \
KILOBOT_BELIEF_SHAPING=2.0 \
KILOBOT_SUCCESS_METRIC=settled \
KILOBOT_SUCCESS_THRESHOLD=0.75 \
\
KILOBOT_BELIEF_COMMS=1 \
KILOBOT_BELIEF_COMMS_LEARNED=1 \
KILOBOT_MSG_SUPERVISION=0.5 \
KILOBOT_MSG_HEAD_PROP_INPUT=1 \
KILOBOT_BEACON_HONESTY=4.0 \
KILOBOT_CRITIC_BELIEF_FEATURES=1 \
\
KILOBOT_QUIET_MSG_HEAD=1 \
KILOBOT_MSG_LOG_STD=-2.5 \
KILOBOT_MOTOR_LOG_STD=-2.5 \
KILOBOT_LOG_STD_MIN=-4.0 \
KILOBOT_MSG_LOG_STD_MAX=-2.5 \
\
KILOBOT_USE_CLOSED_FORM_ARRIVED=1 \
KILOBOT_CLOSED_FORM_HYBRID=1 \
KILOBOT_CLOSED_FORM_ARRIVAL_DIST=0.08 \
KILOBOT_ARRIVED_RELEASE_THRESHOLD=0.5 \
KILOBOT_ARRIVED_FREEZE_HIDDEN=0 \
\
KILOBOT_ACTOR_LR=2e-5 \
KILOBOT_PPO_EPOCHS=2 \
KILOBOT_ENTROPY_COEF=0.001 \
KILOBOT_ORACLE_KNOWN_START_HEADING=1 \
KILOBOT_SWARM_RNG=500 \
KILOBOT_FORMATIONS="$REPO/data/formations" \
nohup python -u launch.py > "$OUT/train.log" 2>&1 < /dev/null &
echo "launched rl_transmission pid $! -> $OUT/train.log"
