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
    echo "usage: $0 <path-to-ckpt.pt-or-actor_final.pt> [episodes] [seed_layout] [max_formations] [formations_dir] [use_arrived_head] [use_turn_anchor] [show_arrived] [use_closed_form_arrived] [closed_form_arrival_dist] [closed_form_hybrid] [use_state_head] [use_wall_head] [use_oracle_head] [use_steer_feature] [split_activation] [split_prop_time_scale]"
    echo "  e.g.: $0 ../results/tb/run_20260714_182306/ckpt.pt"
    echo "  watch the hybrid gate: $0 ../results/bc_v2/run_o3/actor_best.pt 999999 corners 1 ../results/bc_v2/val_formations true true true 1 0.08 1"
    exit 1
fi

WEIGHTS="$1"
EPISODES="${2:-999999}"
SEED_LAYOUT="${3:-corners}"
MAX_FORMATIONS="${4:-1}"
FORMATIONS_DIR="${5:-../data/formations}"
# KILOBOT_FORMATIONS is read by BOTH python (resolved against this script's
# working directory) and the Unity player's ImageLibrary, which resolves it
# against its OWN Application.dataPath -- NOT against python/. A relative path
# like ../results/... therefore resolves on the python side but, for a player
# built at Builds/Kilobot.x86_64, becomes <build>/results/... on the Unity side,
# which does not exist. ImageLibrary then finds an empty formations folder: the
# baked distance field (node[:,4], the reward) reads 1.0 everywhere AND the
# target-formation background image never appears on the floor, even though
# KILOBOT_SHOW_TARGET_FLOOR and floorRenderer are both set. This is the exact
# foot-gun unity_env.set_player_env() warns about and why it abspaths. Canonical
# length here so the player and python agree on which shapes are showing.
if command -v realpath >/dev/null 2>&1; then
    FORMATIONS_DIR="$(realpath -m "$FORMATIONS_DIR")"
elif [ -d "$FORMATIONS_DIR" ]; then
    FORMATIONS_DIR="$(cd "$FORMATIONS_DIR" && pwd)"
fi
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
# oracle_known_start_heading is set true (KILOBOT_ORACLE_KNOWN_START_HEADING in
# the env block below): this is what run_bc_monitored's build_train_cfg and
# eval_closed_loop BOTH use (the closed-loop numbers in ../results/hybrid_cloning
# were measured under it). At Config's own default (false) the belief filter
# starts spread across all four cardinal headings with heading jitter, so the
# trained network sees an off-distribution observation for the whole episode --
# which shows up as wandering-random behaviour that matches no number in the
# report. The player's SwarmManager spawns at a known heading (knownStartHeading
# on the prefab), so this is the physically-true prior, not a privileged readout.
# The closed-form arrival gate (config.py's use_closed_form_arrived /
# closed_form_arrival_dist / closed_form_hybrid; see the report in
# ../results/hybrid_cloning/). Defaults off, so every existing invocation
# behaves exactly as before. Set $9 to 1 and $11 to 1 to watch the HYBRID gate
# -- the oracle's own arrival rule computed from the actor's filter, ORed with
# the learned head. $10 is the arrival radius in normalized units; 0 means
# cfg.tau_v, which for this actor's conservative filter parks almost nobody,
# so the hybrid's measured 0.08 is the sane value.
USE_CLOSED_FORM_ARRIVED="${9:-0}"
CLOSED_FORM_ARRIVAL_DIST="${10:-0}"
CLOSED_FORM_HYBRID="${11:-0}"
# Same class of bug as use_arrived_head above, one notch worse: bad_bridge's
# bc_offline runs ship all four of these on (state/wall/oracle heads and the
# steer feature), and launch.py's build_actor(cfg) cannot know that, so without
# passing them here a watch of such a checkpoint dies in load_for_eval with
# "head_state.weight unexpected" / up1.width 40 vs 42. Defaults false, so
# nothing that worked before changes.
USE_STATE_HEAD="${12:-false}"
USE_WALL_HEAD="${13:-false}"
USE_ORACLE_HEAD="${14:-false}"
USE_STEER_FEATURE="${15:-false}"
# The split network's hidden activation. launch.py reads KILOBOT_SPLIT_ACTIVATION
# straight into cfg.split_activation, and nothing in a state_dict records which
# activation a checkpoint was trained under -- evaluating under a different one
# is a silently different function (the o3 actor is elu). Default empty means
# launch.py config default (relu), preserving existing invocations.
SPLIT_ACTIVATION="${16:-}"
# The time-scale on the actor's relative-track observations (observation.py
# reads cfg.split_prop_time_scale). Training and eval_closed_loop BOTH use
# 0.058 (run_bc_monitored.build_train_cfg); config.py's default is 0.02, so a
# watch that forgets this runs the trained weights against inputs scaled 2.9x
# differently -- the failure looks like a bad policy, and was exactly the
# "deliberates a long time / wanders randomly / no coherent shape" behaviour
# seen here. Defaulted to the trained value, override with $17 if ever needed.
SPLIT_PROP_TIME_SCALE="${17:-0.058}"

if [ ! -f "$WEIGHTS" ]; then
    echo "error: $WEIGHTS does not exist"
    exit 1
fi

echo "watching $WEIGHTS, seed_layout=$SEED_LAYOUT, max_formations=$MAX_FORMATIONS, formations=$FORMATIONS_DIR, use_arrived_head=$USE_ARRIVED_HEAD, use_turn_anchor=$USE_TURN_ANCHOR, use_closed_form_arrived=$USE_CLOSED_FORM_ARRIVED, closed_form_arrival_dist=$CLOSED_FORM_ARRIVAL_DIST, closed_form_hybrid=$CLOSED_FORM_HYBRID (Ctrl+C to stop; pass an explicit episode count as \$2 for a bounded run instead)"
echo "a Unity window should open shortly -- if it doesn't, check that KILOBOT_BUILD_PATH points at a headed (non-server) build; a headless build has no window to show regardless of KILOBOT_NO_GRAPHICS"
echo ""

env KILOBOT_EVAL=1 KILOBOT_EVAL_WEIGHTS="$WEIGHTS" KILOBOT_EVAL_ITERS="$EPISODES" \
    KILOBOT_MAX_FORMATIONS="$MAX_FORMATIONS" \
    KILOBOT_MAX_EPISODE_STEPS=1000000000 KILOBOT_SUCCESS_THRESHOLD=1.1 \
    KILOBOT_NUM_ARENAS=1 KILOBOT_NO_GRAPHICS=false KILOBOT_TIME_SCALE=1 \
    KILOBOT_SHOW_RADIUS=true KILOBOT_LOG_FORMATIONS=true \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_SEED_LAYOUT="$SEED_LAYOUT" KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_USE_ARRIVED_HEAD="$USE_ARRIVED_HEAD" KILOBOT_USE_TURN_ANCHOR="$USE_TURN_ANCHOR" \
    KILOBOT_USE_CLOSED_FORM_ARRIVED="$USE_CLOSED_FORM_ARRIVED" \
    KILOBOT_CLOSED_FORM_ARRIVAL_DIST="$CLOSED_FORM_ARRIVAL_DIST" \
    KILOBOT_CLOSED_FORM_HYBRID="$CLOSED_FORM_HYBRID" \
    KILOBOT_USE_STATE_HEAD="$USE_STATE_HEAD" KILOBOT_USE_WALL_HEAD="$USE_WALL_HEAD" \
    KILOBOT_USE_ORACLE_HEAD="$USE_ORACLE_HEAD" KILOBOT_USE_STEER_FEATURE="$USE_STEER_FEATURE" \
    ${SPLIT_ACTIVATION:+KILOBOT_SPLIT_ACTIVATION="$SPLIT_ACTIVATION"} \
    KILOBOT_SPLIT_PROP_TIME_SCALE="$SPLIT_PROP_TIME_SCALE" \
    KILOBOT_SHOW_TARGET_FLOOR=true \
    KILOBOT_ORACLE_SEND_VISUAL_STATE="$SHOW_ARRIVED" \
    KILOBOT_ORACLE_KNOWN_START_HEADING=true \
    KILOBOT_FORMATIONS="$FORMATIONS_DIR" python launch.py


