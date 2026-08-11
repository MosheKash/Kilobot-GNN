#!/bin/bash
# Drive visible Unity arenas from simple_oracle.py, with no training, so you
# can watch how the swarm organises the shape on its own.
#
# Needs a HEADED build. The normal training binary (Builds/Kilobot.x86_64) is a
# server build with no rendering pipeline compiled in, so
# KILOBOT_NO_GRAPHICS=false does nothing against it. Build a second,
# non-"Server Build" player to a different path and point KILOBOT_BUILD_PATH at
# it.
#
# The episode is deliberately unbounded: the step limit is set beyond anything a
# watching session reaches and the success threshold above 1.0, which coverage
# can never reach, so neither can fire and respawn the swarm mid-watch.
#
# Body colour tracks oracle state: ivory=go_north, amber=turning,
# gold=wall_following, deep red=navigating, green=arrived.
#
# The oracle assumes every robot spawns at one of belief.CARDINAL_HEADINGS,
# which holds only if the build was compiled with SwarmManager's
# knownStartHeading on. KILOBOT_ORACLE_DEBUG_WALL_LOG=1 adds a per-robot
# SIMPLE_ORACLE_SPAWN_CHECK against Unity's real heading (flagged MISMATCH past
# 5 degrees) -- faster than guessing from the window.
#
# More than two arenas with graphics on is outside what this mode was built for;
# launch.py warns, and Unity can go unresponsive rather than fail cleanly. Don't
# run this alongside a training run on the same machine.
#
# usage: ./scripts/watch_oracle.sh [seed_layout] [num_arenas] [time_scale]
#   ./scripts/watch_oracle.sh
#   ./scripts/watch_oracle.sh corners 1 5
#   KILOBOT_ORACLE_DEBUG_WALL_LOG=1 ./scripts/watch_oracle.sh corners 1 1

set -euo pipefail
cd "$(dirname "$0")/.."

SEED_LAYOUT="${1:-corners}"
NUM_ARENAS="${2:-1}"
TIME_SCALE="${3:-1}"
# Same foot-gun watch_actor.sh documents in detail: the player's ImageLibrary
# resolves KILOBOT_FORMATIONS against its OWN Application.dataPath, not this
# script's cwd, so a relative ../data/formations leaves the player's image list
# empty -- the target-formation background image on the floor never appears.
# Canonicalise so the player and python agree on the shapes being shown.
FORMATION_DIR=../data/formations
if command -v realpath >/dev/null 2>&1; then
    FORMATION_DIR="$(realpath -m "$FORMATION_DIR")"
elif [ -d "$FORMATION_DIR" ]; then
    FORMATION_DIR="$(cd "$FORMATION_DIR" && pwd)"
fi

echo "watching simple_oracle.py on $NUM_ARENAS arena(s), seed_layout=$SEED_LAYOUT, time_scale=$TIME_SCALE, no training"
if [ "$NUM_ARENAS" -gt 2 ]; then
    echo "note: >2 arenas with graphics on is untested territory -- if the window never appears, this is why"
fi
echo "a Unity window should open shortly; if not, check KILOBOT_BUILD_PATH points at a HEADED build"
echo "Ctrl+C to stop"
echo ""

env KILOBOT_MODE=watch_oracle KILOBOT_MOTOR_OVERRIDE=simple_oracle \
    KILOBOT_MAX_FORMATIONS=1 \
    KILOBOT_MAX_EPISODE_STEPS=1000000000 KILOBOT_SUCCESS_THRESHOLD=1.1 \
    KILOBOT_NUM_ARENAS="$NUM_ARENAS" KILOBOT_NO_GRAPHICS=false KILOBOT_TIME_SCALE="$TIME_SCALE" \
    KILOBOT_SHOW_RADIUS=true KILOBOT_LOG_FORMATIONS=true KILOBOT_ORACLE_SEND_VISUAL_STATE=true \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_SEED_LAYOUT="$SEED_LAYOUT" KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_ORACLE_KNOWN_START_HEADING=true \
    KILOBOT_FORMATIONS="$FORMATION_DIR" python launch.py
