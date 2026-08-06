# Development

This covers setting up to work on the project the way it is currently developed: editing the Unity simulator and the Python training code, building the headless player, running the tests, and building the container.

## Prerequisites

- Unity 6000.4.8f1, with the ML-Agents package (the project is pinned to release_23).
- Python 3.10.
- A C compiler toolchain is not needed; the Python side is pure PyTorch and friends.
- A GPU is optional. CPU works for development and smoke tests. The learner can use an NVIDIA GPU (CUDA) or, with the right Torch build, an AMD GPU (ROCm). Evaluation always runs on CPU.

## Get the repository

```
git clone <your-fork-or-repo-url>
cd Kilobot-GNN
```

The clone includes the full Unity project, the Python code, the trained encoder (`data/image_encoder.pt`), and the Docker files.

Two things are **not** in the clone, because both are reproducible: a 158 MB player and 679 MB of PNGs. Produce them before anything will run:

**The headless player.** Build it without opening the editor:

```
<unity-editor> -batchmode -quit -nographics -projectPath . \
    -executeMethod BuildPlayer.BuildLinux -logFile -
```

That writes `Builds/Kilobot.x86_64` and exits non-zero on failure. It is the same entry point CI would use.

**The target images.** Generate the formation pool from QuickDraw:

```
python data-prep/quickdraw_to_png.py --categories cat dog bicycle airplane tree \
    --max-total 50000 --output-dir data/formations
```

Then verify the whole pipeline end to end:

```
cd python && python tools/verify_unity_pipeline.py
```

## Python side

Create an environment and install the dependencies. From the repository root:

```
cd python
python -m venv .venv && source .venv/bin/activate    # or conda, your choice
pip install -r requirements.txt
pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents-envs"
```

The `mlagents-envs` version has to match the Unity ML-Agents package, which is why it is installed from the release_23 branch rather than from PyPI. If the communication handshake fails at startup, a version mismatch here is the first thing to check.

Run the tests:

```
python -m pytest tests -q
```

Most are pure Python, but 26 launch a real Unity player -- there is no simulator
stub. With a build present that is 256 passed; without one, 230 pass and 26 skip
with the build command in the skip reason, so a fresh clone still gets a
meaningful run before doing the Unity setup.

Run a smoke training loop against the player:

```
env KILOBOT_NUM_WORKERS=2 KILOBOT_NUM_ARENAS=9 KILOBOT_SMOKE=1 \
    KILOBOT_DEVICE=cpu KILOBOT_FORMATIONS=../data/formations python launch.py
```

A healthy start prints `all N workers ready`, then iteration lines with a coverage value around 0.2. Coverage near zero at the first iteration means the encoder or the images did not load. The player's own output -- including its per-arena `SwarmManager arena 0: ... kilobots` line -- goes to `results/unity-logs/Player-<worker>.log`, not the console; set `KILOBOT_UNITY_LOG_DIR=""` to put it back on the console while debugging a player that will not start.

`docs/configuration.md` has the full set of run recipes.

To search for hyperparameters that train, install Optuna (`pip install optuna`) and use `sweep.py`. The workflow is in `docs/sweep.md`.

## Unity side

Open the project folder in Unity 6000.4.8f1. The training scene is `Assets/Scenes/KilobotShapeTraining.unity`. The simulation scripts are in `Assets/Scripts/` and are described in `docs/code-overview.md`.

If you change anything in the simulator, rebuild the headless player that Python and Docker use — the one-line `-executeMethod BuildPlayer.BuildLinux` command above, which writes to `Builds/Kilobot.x86_64` with its data folder at `Builds/Kilobot_Data/`. The Python side launches that binary by path; it does not talk to the editor. After rebuilding, rerun `tools/verify_unity_pipeline.py` to confirm the new build connects and the observation layout is unchanged.

`scripts/watch_actor.sh` / `scripts/watch_oracle.sh` (`python/scripts/`) need a second, separate build: a normal, non-"Server Build" player, so it has a rendering pipeline at all. Build it to a different path than `Builds/Kilobot.x86_64` so it doesn't overwrite the headless training binary, and point `KILOBOT_BUILD_PATH` at it when running either script.

The arena count is read from `KILOBOT_NUM_ARENAS` at scene start, so you do not need to rebuild to change how many arenas run.

## The development loop

For Python-only changes, edit the files in `python/`, run the tests, and run a smoke loop. Nothing else is needed.

For simulator changes, edit the scripts in `Assets/Scripts/`, rebuild the headless player to `Builds/`, and then run a smoke loop to confirm Python still connects and the arenas spawn.

Watch training in TensorBoard:

```
tensorboard --logdir results/tb
```

## Building the container

The Docker image bundles the Python code, the player and the data, so it runs without Unity. Do the first-time setup above before building it -- the image copies `Builds/` and `data/` in, and neither is in the clone. Build from the repository root:

```
docker build -t kilobot-gnn .
```

The image starts from an NVIDIA CUDA base and installs Torch, torch-geometric, and the matching `mlagents-envs`. The build arguments in `docs/configuration.md` let you retarget a different CUDA or Torch version.

`entrypoint.sh` runs first inside the container. It checks that the player, encoder, and images are present, starts a virtual display (the headless Unity player needs an X display to attach to even with rendering disabled), reports the GPU if one is requested, and then launches the chosen mode. The README quickstart shows how to run the container for smoke, training, and evaluation.

## Where outputs go

Each run creates a folder under `KILOBOT_LOGDIR` (by default `results/tb/run_<timestamp>/`). That folder holds the TensorBoard logs, the periodic `ckpt.pt`, and the final `actor_final.pt`. `results/` is gitignored. When running in Docker, mount a host folder at `/app/results` if you want those to survive the container.

## A note on the headless player in Docker

The Unity Linux player links several system libraries and needs an X display at scene load, even when run with graphics disabled. The Dockerfile installs those libraries (including the GTK ones a bundled plugin pulls in) and the entrypoint provides a virtual display with Xvfb. If you replace the player with a build that uses different Unity packages and it fails to start in the container, the missing piece is usually another shared library; find it by running `ldd` against the player and its plugins inside the image, and add the matching package to the Dockerfile.
