# Kilobot-GNN training container.
#
# Architecture: CPU env-worker processes step Unity and run the small actor;
# one GPU learner runs PPO (the privileged GATv2 critic over all arena graphs).
# That split is why this image wants a single-GPU + many-CPU-core box.
#
# The build context MUST be the project root and MUST contain:
#   python/                       the training code (in this repo)
#   Builds/Kilobot.x86_64         the compiled HEADLESS Linux Unity player + Kilobot_Data/
#   data/image_encoder.pt         the trained image encoder
#   data/formations/              the target-image PNGs (>= KILOBOT_MAX_FORMATIONS of them)
# Builds/ and data/ are not in version control - place them before building.
#
# Build (defaults target CUDA 12.1):
#   docker build -t kilobot-gnn .
# Retarget a different CUDA / torch build without editing this file:
#   docker build -t kilobot-gnn \
#     --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#     --build-arg TORCH_VERSION=2.5.1 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
# Run (real training, all GPUs, persist logs + checkpoints):
#   docker run --gpus all --rm -v "$(pwd)/results:/app/results" kilobot-gnn
# Other modes via KILOBOT_MODE:
#   docker run --gpus all --rm -e KILOBOT_MODE=smoke kilobot-gnn
#   docker run --gpus all --rm -e KILOBOT_MODE=eval \
#     -e KILOBOT_EVAL_WEIGHTS=/app/results/tb/run_XXXX/actor_final.pt \
#     -v "$(pwd)/results:/app/results" kilobot-gnn

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG MLAGENTS_REF=release_23

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Ubuntu 22.04 ships Python 3.10 (matches the dev environment). The X / GL
# libraries below are RUNTIME dependencies of the Unity Linux player. libgtk-3-0
# and libdbus-1-3 satisfy the bundled libAppUINativePlugin.so (Unity "App UI"
# package); without libgtk-3 that plugin fails to load and the headless player
# segfaults at scene init. libgtk-3-0 pulls in the glib/gobject/gdk-pixbuf/
# pango/cairo chain it also needs. If the player still fails to start, this apt
# list is the first thing to extend (xvfb is the usual fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        libgl1 \
        libglu1-mesa \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxi6 \
        libxss1 \
        libgtk-3-0 \
        libdbus-1-3 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ----------------------------------------------------
# torch first (from the chosen CUDA wheel index) so the rest resolve against
# it. torch_geometric's GATv2Conv + global pooling run in pure PyTorch on
# modern PyG, so the compiled torch-scatter / torch-sparse companions are
# intentionally NOT installed.
RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
        torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}

RUN pip3 install --no-cache-dir \
        "numpy<2" \
        pillow \
        tensorboard \
        torch_geometric

# mlagents-envs MUST match the Unity com.unity.ml-agents package (4.0.3),
# which is the release_23 line. Do NOT swap this for the plain PyPI release -
# the gRPC comms protocol version has to line up with the Unity build.
RUN pip3 install --no-cache-dir \
        "git+https://github.com/Unity-Technologies/ml-agents.git@${MLAGENTS_REF}#subdirectory=ml-agents-envs"

# --- Project ----------------------------------------------------------------
WORKDIR /app
COPY python/ /app/python/
COPY Builds/ /app/Builds/
COPY data/ /app/data/
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh && (chmod +x /app/Builds/Kilobot.x86_64 || true)

WORKDIR /app/python

# Defaults for a single-GPU cloud box. Override any with `docker run -e`.
# The learner uses the GPU; workers are forced onto CPU in code regardless of
# this value, so KILOBOT_DEVICE only controls the PPO/critic device.
ENV KILOBOT_MODE=train
ENV KILOBOT_DEVICE=cuda
ENV KILOBOT_NO_GRAPHICS=1
ENV KILOBOT_SMOKE=0
ENV KILOBOT_NUM_WORKERS=4
ENV KILOBOT_NUM_ARENAS=9
ENV KILOBOT_ITERATIONS=100
ENV KILOBOT_CKPT_EVERY=10
ENV KILOBOT_BUILD_PATH=/app/Builds/Kilobot.x86_64
ENV KILOBOT_ENCODER_PATH=/app/data/image_encoder.pt
ENV KILOBOT_FORMATIONS=/app/data/formations
ENV KILOBOT_LOGDIR=/app/results/tb

ENTRYPOINT ["/app/entrypoint.sh"]
