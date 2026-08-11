# Kilobot-GNN

Training simulated Kilobot swarms to self-assemble into target shapes with multi-agent reinforcement learning. Hundreds of identical robots, each able to see only its nearby neighbours over short-range infrared, learn one shared policy that moves the whole group into a shape supplied as an image.

Unity runs the physics and the infrared messaging. All the learning happens in Python through the ML-Agents low-level API. Training is centralised — a privileged graph critic sees the whole arena — while execution is decentralised: every robot runs the same small policy on local information only, so a trained policy could in principle run on a real swarm.

## Status

A research work in progress. The pipeline runs end to end — collect rollouts, train an actor and critic with MAPPO, checkpoint, resume, export, evaluate — but it does **not** yet converge to reliable shape assembly, and that is the current focus.

The current approach is a two-stage one: behaviour-clone a scripted teacher (`python/simple_oracle.py`, a five-state machine that uses no privileged information), then fine-tune with RL from that warm start.

The investigation is long and is documented rather than summarised here:

- **[`docs/tuning.md`](docs/tuning.md)** — the chronological research log, phase by phase. The last entries are where things currently stand.
- **[`docs/code-history.md`](docs/code-history.md)** — the same material indexed by symbol: why a given constant, flag or sign is what it is.

Treat this as a working research codebase, not a finished product.

## First-time setup

Two things the repository does **not** track, because both are reproducible: a 158 MB player and 679 MB of PNGs (837 MB of disk, and 99.4% of what used to be tracked).

**1. The Unity player** (`Builds/`). Build it headlessly from the Unity project:

```
<unity-editor> -batchmode -quit -nographics -projectPath . \
    -executeMethod BuildPlayer.BuildLinux -logFile -
```

Produces `Builds/Kilobot.x86_64`. Rebuild whenever you change anything under `Assets/`.

**2. The target images** (`data/formations/`). Generate them from QuickDraw:

```
python data-prep/quickdraw_to_png.py --categories cat dog bicycle airplane tree \
    --max-total 50000 --output-dir data/formations
```

`data/image_encoder.pt`, the trained image encoder, **is** tracked — it is not reproducible on demand. `data-prep/autoencoder_latent_search.ipynb` is the notebook that produced it.

Then install the Python side. You need Python 3.10 and the same `mlagents-envs` the Unity project is built against (release_23):

```
cd python
pip install -r requirements.txt
pip install "git+https://github.com/Unity-Technologies/ml-agents.git@release_23#subdirectory=ml-agents-envs"
```

Confirm the whole pipeline is wired up — 12 checks, from "no Python simulator survives" to a full closed loop:

```
python tools/verify_unity_pipeline.py
```

## Repository layout

```
.
├── Assets/                 Unity project: simulation logic, prefabs, scene
│   ├── Scripts/            C# -- robot, movement, arena spawning, side channel
│   ├── Editor/             BuildPlayer.cs, the headless build entry point
│   ├── Prefabs/            Arena, Kilobot, Seed, WallSeed, Heartbeat
│   └── Scenes/             KilobotShapeTraining.unity
├── python/                 all training code -- see docs/code-overview.md
│   ├── scripts/            shell entry points (train_bc, watch_oracle, watch_actor)
│   ├── tests/              pytest suite, runs against a real Unity player
│   ├── tools/              diagnostics that run against the live pipeline
│   └── temp_test_material/ scratch; not part of the pipeline, not tracked
├── Packages/               Unity package manifest and the vendored ML-Agents package
├── ProjectSettings/        Unity project settings (Fixed Timestep, layers, tags)
├── data-prep/              offline: QuickDraw -> PNGs, and the encoder notebook
├── data/
│   └── image_encoder.pt    the trained target-image encoder (tracked)
├── docs/                   architecture, configuration, tuning log, code history
├── results/                all run output: checkpoints, logs, TensorBoard (not tracked)
├── Dockerfile
└── entrypoint.sh           container startup: preflight, virtual display, dispatch
```

Generated and therefore absent from a fresh clone: `Builds/`, `data/formations/`, `results/`, and Unity's `Library/`.

## Quickstart

Every recipe below runs from `python/`, uses the split-observation actor, and uses the `corners` seed layout — the only supported one.

**Smoke test**, headless, a few short iterations to confirm everything connects:

```
env KILOBOT_SMOKE=1 KILOBOT_DEVICE=cpu KILOBOT_NUM_ARENAS=9 \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_FORMATIONS=../data/formations python launch.py
```

**Behaviour cloning** against the oracle — the recommended starting point, since RL from scratch does not currently converge. Two paths exist; the offline one is the one under active work:

```
./scripts/bc_offline_pipeline.sh ../results/bc_v2      # record tapes, fit, DAgger, evaluate, report
./scripts/train_bc.sh ../results/bc_run 300            # the older online loop
```

The offline path records the oracle's rollouts to disk **once** (`tools/record_tape.py`) and then fits the actor to them as per-robot *sequences* with truncated BPTT, so training costs no simulation and a run is reproducible from a tape file and a seed. It also runs DAgger rounds, which are not optional here.

**The task metric is not coverage.** It is: per robot, did it *stop* and did it stop *within X units of the point it was assigned* — reported as a distribution over arenas, counting only arenas the driver finished (>= 95% stopped). `tools/settle_report.py` measures it; `reward.coverage` asks the weaker question "near any on-pixel" and has a 0.286 chance floor.

Where it stands (`docs/tuning.md` phases 156-159, which end with a handover section): the clone reproduces the oracle's **decisions** — 97% of held-out decisions within 0.05 of the teacher's command — but not its **outcome**. On the task metric the oracle settles a median 40% of robots within 5 units and finishes 21 of 24 arenas; the best clone settles 2% and finishes none, because its robots never get near their targets (median closest approach 55 units against the oracle's 19). Held-out imitation error and task outcome are anti-correlated across runs, which is why the recommended next step is reward-based fine-tuning from the BC warm start rather than more cloning.

**RL**, warm-started from that clone:

```
env KILOBOT_MODE=rl KILOBOT_SMOKE=0 KILOBOT_ITERATIONS=200 KILOBOT_DEVICE=cuda \
    KILOBOT_INIT_ACTOR=../results/bc_run/actor_best.pt \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_REWARD_SHAPING=5.0 KILOBOT_ENTROPY_COEF=0.001 \
    KILOBOT_FORMATIONS=../data/formations python launch.py
```

`KILOBOT_REWARD_SHAPING` and `KILOBOT_ENTROPY_COEF` are set explicitly because both defaults were *measured* to prevent progress (`docs/tuning.md`): shaping is off by default and is the only gradient a robot far from the target gets, and the default entropy coefficient drives action noise to its ceiling within the first third of a run.

**Watch it run** (needs a separate headed build — see `docs/development.md`):

```
./scripts/watch_oracle.sh             # the scripted teacher
./scripts/watch_actor.sh <checkpoint> # a trained policy
```

**Tests**:

```
python -m pytest tests -q
```

The suite launches real Unity players; there is no simulator stub. Tests skip if no build is present.

## Quickstart with Docker

From the repository root. The image copies in `Builds/` and `data/`, so do the first-time setup above before building it.

```
docker build -t kilobot-gnn .

# smoke test on CPU
docker run --rm -e KILOBOT_MODE=smoke -e KILOBOT_DEVICE=cpu -e KILOBOT_NUM_WORKERS=2 kilobot-gnn

# full run, checkpoints to ./results on the host
docker run --rm -v "$(pwd)/results:/app/results" -e KILOBOT_NUM_WORKERS=4 kilobot-gnn

# with an NVIDIA GPU (learner on GPU, workers on CPU)
docker run --rm --gpus all -v "$(pwd)/results:/app/results" kilobot-gnn

# evaluate a saved policy
docker run --rm -v "$(pwd)/results:/app/results" \
    -e KILOBOT_MODE=eval \
    -e KILOBOT_EVAL_WEIGHTS=/app/results/tb/run_YYYYMMDD_HHMMSS/actor_final.pt \
    kilobot-gnn
```

The image defaults to `KILOBOT_DEVICE=cuda`; set `cpu` if you have no GPU.

## Parameters you will probably touch

Everything is configured through `KILOBOT_*` environment variables.

| Variable | Default | What it does |
|---|---|---|
| `KILOBOT_MODE` | `rl` | `rl`, `bc`, `eval`, `smoke`, or a probe mode |
| `KILOBOT_DEVICE` | `cpu` (`cuda` in the image) | device for the learner's PPO and critic |
| `KILOBOT_NUM_WORKERS` | `1` | parallel Unity worker processes; `>= 2` uses the learner/worker split |
| `KILOBOT_NUM_ARENAS` | `9` | arenas per Unity instance |
| `KILOBOT_ACTOR` | `deepset` | `deepset`, `gru`, or `gru_split_observation` (the one under active work) |
| `KILOBOT_SEED_LAYOUT` | `corners` | the only supported layout; `cluster` exists for backward compatibility and is deprecated |
| `KILOBOT_HEARTBEAT_TICKS` | `0` | force a decision after this many idle ticks so a robot cannot coast forever. Set `48` |
| `KILOBOT_REWARD_SHAPING` | `0.0` | potential-based shaping; the only gradient for a robot beyond `l_scale`. `5.0` recommended |
| `KILOBOT_ENTROPY_COEF` | `0.01` | PPO entropy weight; the default pins action noise at its ceiling. `0.001` fixes that |
| `KILOBOT_SWARM_RNG` | unset | seed the player's spawn RNG, making a run's sequence of arenas replayable |
| `KILOBOT_ITERATIONS` | `100` | training iterations |
| `KILOBOT_SMOKE` | `1` (the image sets `0`) | short validation loop instead of full training — a bare `python launch.py` smoke-tests |
| `KILOBOT_RESUME` | unset | path to a `ckpt.pt` to continue from |
| `KILOBOT_NO_GRAPHICS` | `1` | `false` opens a Unity window instead of running headless |
| `KILOBOT_UNITY_LOG_DIR` | `results/unity-logs` | where the player writes its own log; `""` puts it back on the console |
| `KILOBOT_FORMATIONS` | `../data/formations` | folder of target images, relative to `python/` |

The complete list, and the `config.py` hyperparameters, are in `docs/configuration.md`.

## Hyperparameter search

A search harness that looks for settings which train without the entropy climbing. Install Optuna, then run from `python/`:

```
pip install optuna

# explore on a small proxy task
python sweep.py --trials 40 --iters 40 --formations 8 --workers 2 --device cuda

# refine the best region, reusing the same study
python sweep.py --trials 30 --iters 120 --formations 64 --device cuda

# confirm the winner across seeds
python sweep.py --trials 0 --confirm-seeds 3 --iters 120 --formations 64 --device cuda
```

Before a large search, prove one easy shape can be learned at all (`--formations 1`, watch coverage rise and entropy fall); a search cannot fix a signal that cannot learn. Full workflow in `docs/sweep.md`.

## Documentation

| doc | what it covers |
|---|---|
| [`docs/code-overview.md`](docs/code-overview.md) | what every module owns, and how the Unity and Python sides meet |
| [`docs/architecture.md`](docs/architecture.md) | the environment, the observation model, the reward |
| [`docs/configuration.md`](docs/configuration.md) | every `KILOBOT_*` variable and `Config` field |
| [`docs/development.md`](docs/development.md) | building the player, running the tests, headed builds |
| [`docs/tuning.md`](docs/tuning.md) | the chronological research log |
| [`docs/code-history.md`](docs/code-history.md) | why a given constant or flag is what it is, by symbol |
| [`docs/sweep.md`](docs/sweep.md) | the hyperparameter search |

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Moshe Kashlinsky — kashlinskymoshe@gmail.com
