#!/usr/bin/env bash
# Self-checking entrypoint for the Kilobot-GNN container.
# Validates that the artifacts the run needs are actually present, reports GPU
# visibility, then dispatches on KILOBOT_MODE. train/smoke/eval are container
# aliases; any other value is passed through to launch.py unchanged (rl, bc,
# probe, reward_probe, audit, control, watch_oracle).
set -euo pipefail

fail=0
need() {
    if [ ! -e "$1" ]; then
        echo "PREFLIGHT ERROR: missing $2" >&2
        echo "                expected at: $1" >&2
        fail=1
    fi
}

echo "== Kilobot-GNN preflight =="
need "${KILOBOT_BUILD_PATH}"   "Unity headless player (KILOBOT_BUILD_PATH)"
need "${KILOBOT_ENCODER_PATH}" "image encoder (KILOBOT_ENCODER_PATH)"
need "${KILOBOT_FORMATIONS}"   "formations directory (KILOBOT_FORMATIONS)"

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "Preflight failed. The build context must contain Builds/, data/image_encoder.pt," >&2
    echo "and data/formations/ (see docs/DOCKER.md)." >&2
    exit 1
fi

if [ -x "${KILOBOT_BUILD_PATH}" ]; then
    echo "ok: player is executable"
else
    echo "note: ${KILOBOT_BUILD_PATH} is not marked executable; attempting chmod +x" >&2
    chmod +x "${KILOBOT_BUILD_PATH}" || true
fi

# GPU visibility is informational - CPU runs still work.
if [ "${KILOBOT_DEVICE:-cpu}" = "cuda" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    else
        echo "WARNING: KILOBOT_DEVICE=cuda but nvidia-smi is not available." >&2
        echo "         Run with '--gpus all' on a host with a CUDA driver, or set KILOBOT_DEVICE=cpu." >&2
    fi
fi

# Unity's player attaches its windowing/input subsystem to an X display at scene
# load even under -nographics (rendering still uses the NullGfxDevice). Without a
# display it picks the "(null)" backend and segfaults. Provide a virtual
# framebuffer; every Unity worker process inherits DISPLAY and attaches to it.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
mkdir -p "${XDG_RUNTIME_DIR}" && chmod 700 "${XDG_RUNTIME_DIR}"
export DISPLAY=:99
Xvfb :99 -screen 0 640x480x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
sleep 1
if kill -0 "${XVFB_PID}" 2>/dev/null; then
    echo "started Xvfb on :99"
else
    echo "WARNING: Xvfb failed to start; the player will likely segfault. Log:" >&2
    cat /tmp/xvfb.log >&2
fi

MODE="${KILOBOT_MODE:-train}"
echo "== mode: ${MODE} (device=${KILOBOT_DEVICE:-cpu} workers=${KILOBOT_NUM_WORKERS:-1} arenas=${KILOBOT_NUM_ARENAS:-?}) =="

# train/smoke/eval are container-level aliases that set the flag launch.py
# actually reads. Everything else is passed straight through as KILOBOT_MODE,
# so the container supports every mode launch.py does -- rl, bc, and the probe
# modes -- without this list needing to track that one.
case "${MODE}" in
    train)
        exec env KILOBOT_MODE=rl KILOBOT_SMOKE=0 python3 launch.py
        ;;
    smoke)
        exec env KILOBOT_MODE=rl KILOBOT_SMOKE=1 python3 launch.py
        ;;
    eval)
        exec env KILOBOT_EVAL=1 python3 launch.py
        ;;
    *)
        exec env KILOBOT_MODE="${MODE}" python3 launch.py
        ;;
esac
