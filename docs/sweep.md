# Hyperparameter search (the sweep)

This explains how to search for hyperparameters that train and keep entropy bounded, using the sweep harness in `python/sweep.py`. It assumes you have read the configuration doc. This is the tool to reach for once the policy trains at all; for the record of getting it to train, see `tuning.md`.

## The goal

Two things, in order. First, find a configuration that actually trains: coverage rises over a run and the policy entropy does not climb. Second, push that configuration to train as well as possible. The sweep does both with one score, so you do not have to switch tools between the phases. Early trials explore widely, and the sampler then concentrates on the region that trains.

## Before you sweep: confirm the task is learnable

A search can only find good settings if some setting can work. Spend a little time first proving that the signal is not broken:

- Run a single short training run on a deliberately easy task: one or a few target shapes (`KILOBOT_MAX_FORMATIONS=1`), a small swarm, short episodes.
- Watch the critic's explained variance and the policy entropy in TensorBoard. If explained variance climbs above zero, the critic is learning and the advantages are meaningful. If it stays near zero or negative, the actor has nothing useful to follow, which is the usual cause of entropy drifting up. Fix that first (critic learning rate, more critic epochs, or reward scaling) before running a wide search.
- If `log_std` grows without bound, the policy is just widening. A lower `KILOBOT_LOG_STD_INIT` is the first thing to try.

If you cannot get one easy shape to improve at all, the problem is the reward or the critic, not the hyperparameters, and no sweep will fix it.

**Set `KILOBOT_REWARD_SHAPING` before you sweep, in the shell you launch `sweep.py` from -- it is not searched and not set anywhere in `sweep.py`.** `env = dict(os.environ)` in `sweep.py` means every trial inherits whatever the launching shell had set and nothing more; if `KILOBOT_REWARD_SHAPING` is unset there, every trial silently runs at `Config`'s default of `0.0`. Phase 8 (`docs/tuning.md`) found this specific field, not `k_pos` (see the table below), is what gives a robot far from the target any reward gradient to move toward it at all -- without it, `off_penalty` saturates at a fixed value beyond `l_scale` and stays exactly flat no matter how much farther away the robot gets, so a whole sweep can run to completion tuning learning rates and entropy around a task that structurally cannot be learned. `export KILOBOT_REWARD_SHAPING=5.0` (the project's standing recommendation) before running any of the commands below, the same way you would set `KILOBOT_HEARTBEAT_TICKS` or `KILOBOT_ACTOR` -- a structural prerequisite, not a hyperparameter to search.

## The knobs the sweep varies

Each is an environment variable read by `launch.py`, so you can also set any of them by hand for a single run. The sweep samples the ones below; the rest of the `KILOBOT_*` variables are in the configuration doc.

| Variable | Searched range | Notes |
|---|---|---|
| `KILOBOT_ACTOR_LR` | 3e-5 to 1e-3, log | actor (policy) learning rate |
| `KILOBOT_CRITIC_LR` | 1e-4 to 3e-3, log | critic learning rate, often higher than the actor |
| `KILOBOT_ENTROPY_COEF` | 1e-5 to 3e-2, log | prime suspect when entropy climbs -- confirmed directly in phase 8/9 (`docs/tuning.md`): the default `0.01` reliably pinned `policy/std_mean` at its hard ceiling by ~iteration 55-60 in two separate real-Unity runs; `0.001` fixed that specific symptom (the largest, least ambiguous effect size found in that investigation) but did not, on its own, accelerate coverage -- a real, necessary fix, not sufficient by itself |
| `KILOBOT_CLIP` | 0.1 to 0.3 | PPO clip range |
| `KILOBOT_PPO_EPOCHS` | 3 to 12 | passes over each rollout |
| `KILOBOT_GAE_LAMBDA` | 0.90 to 0.99 | advantage smoothing |
| `KILOBOT_LOG_STD_INIT` | -1.5 to 0.0 | initial policy spread |
| `KILOBOT_K_POS` | 0.25 to 4.0, log | strength of the off-shape reward gradient |
| `KILOBOT_R_PACK` | 0.25 to 4.0, log | peak packing bonus for a well-spaced on-shape neighbor -- see "The packing reward" below |
| `KILOBOT_PACK_RANGE` | 0.08 to 0.5 | how far out the packing bonus decays |

Other reward and training variables you can set by hand but the sweep leaves alone by default: `KILOBOT_GAMMA`, `KILOBOT_MAX_GRAD_NORM`, `KILOBOT_MINIBATCH`, `KILOBOT_R_ON`, `KILOBOT_TAU_V`, `KILOBOT_L_SCALE`, `KILOBOT_K_SEP`, `KILOBOT_TAU_SEP`, and `KILOBOT_SEED`. To search a different set, edit `suggest_params` in `sweep.py`.

## How a run reports progress

When `KILOBOT_SUMMARY` is set to a path, `launch.py` writes a small JSON file there and rewrites it every iteration. It holds the per-iteration history (coverage, entropy, explained variance, losses) and a few derived numbers the sweep uses: the initial and final coverage, the maximum coverage, and the fitted slope of entropy across the run. The write is atomic, so the sweep can read it at any time while the run is in progress. You can point `KILOBOT_SUMMARY` at a file yourself if you want this record for a manual run.

## The score

Each trial is scored by:

```
score = coverage_gain - entropy_penalty * max(0, entropy_climb) + small ev bonus
```

`coverage_gain` is the final coverage minus the initial coverage. `entropy_climb` is the fitted total rise in entropy over the run, and only a rise is penalized, since falling entropy is what you want. The optional explained-variance bonus nudges ties toward configs whose critic is actually learning. The entropy penalty weight is `--ent-penalty` (default 0.5); raise it if you want to be stricter about entropy staying flat. This single score is high exactly when a run trains and keeps entropy in check, which is why the same sweep serves both phases.

## Running the sweep

The harness launches `python launch.py` once per trial with the sampled variables, polls the summary while it trains, stops a trial early when it is clearly behind its peers at the same point, and records the score. Install Optuna first (`pip install optuna`), then run from the `python/` directory.

Phase one, explore on a cheap proxy task (few shapes, short runs) to find configurations that train:

```
python sweep.py --trials 40 --iters 40 --formations 8 --workers 2 --device cuda
```

Phase two, refine. Point at the same study and storage, raise the budget and the task difficulty, and keep going. The sampler will exploit what phase one learned:

```
python sweep.py --trials 30 --iters 120 --formations 64 --device cuda
```

Confirm the winner across seeds before trusting it, since reinforcement learning is noisy from seed to seed:

```
python sweep.py --trials 0 --confirm-seeds 3 --iters 120 --formations 64 --device cuda
```

This step re-runs `study.best_params` in full -- all ten swept parameters, not a subset (phase 17, `docs/tuning.md`: `r_pack`/`pack_range` were previously dropped silently here, so confirmation runs never actually tested the true best config for those two).

Useful flags: `--formations` (how many target shapes to use, the proxy task size), `--formations-path` (the folder of target images, default `../data/formations`, used to set `KILOBOT_FORMATIONS` for each trial), `--ent-penalty` (how hard to punish climbing entropy), `--warmup` (iterations before early stopping may trigger), `--trial-timeout` (a hard per-trial wall-clock cap), `--out` (where per-trial logs, summaries, and the study database go), and `--device` (`cpu` or `cuda`).

The sweep sets `KILOBOT_FORMATIONS` for each trial from `--formations-path` (resolved to an absolute path), so both the Python image pool and Unity's `ImageLibrary` find the shapes. If you ever run the build by hand without that variable set, Unity falls back to the path stored on the `ImageLibrary` component in the scene, so set `KILOBOT_FORMATIONS` for manual runs too.

## Running several searchers at once

The study lives in a database, so several searchers can share it. By default that database is a SQLite file under `--out`, which is fine for one machine. To have several machines cooperate on one search, give them all the same `--study-name` and the same `--storage` URL pointing at a shared database (Optuna supports a relational database such as PostgreSQL or MySQL for this). Each searcher then runs one trial at a time and they jointly fill in the study. Run one searcher per machine rather than many on one machine, because each trial drives its own Unity instances and they will contend for cores otherwise.

## Reading the results

At the end, the sweep prints the best score and its parameters and writes them to `best.json` under `--out`. Each trial also stores its score breakdown (coverage gain, entropy trend, and so on), which you can inspect through the Optuna study or in the per-trial folders. To turn a winning configuration into a full run, set the corresponding `KILOBOT_*` variables and launch training normally on the full task.

## Picking the budget

A few practical defaults. The bottleneck is the simulation, not the learner, so keep the proxy task small during search: a handful of shapes, a modest swarm, and only as many iterations as you need to see a trend (a few tens, not hundreds). Use early stopping (it is on by default) so bad trials die quickly. Spend the saved time on more trials and on confirming finalists across seeds, which matters more than running any single trial longer.

## If a trial runs out of GPU memory

The critic is a graph network that runs over every step's interaction graph. On a small-memory GPU, doing the whole rollout in one pass can exceed VRAM (you will see `CUDA out of memory` during the critic update). The critic now processes the rollout in chunks of `KILOBOT_CRITIC_CHUNK` graph snapshots (default 64), which bounds memory without changing the result. If you still hit out of memory with a large swarm, lower it (for example `KILOBOT_CRITIC_CHUNK=16`), and as a secondary measure set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. Reducing `--rollout-steps` or `--arenas` also helps but costs signal.

## The packing reward

An on-shape robot is rewarded for having a nearby on-shape neighbor at good spacing. On-shape status of neighbors is read from their true distance feature, which is privileged information used only in the training reward; the deployed actor never sees it, so execution stays decentralized. On a thin stroke (the Quick, Draw! targets are thin doodles) the on-shape neighbors lie along the curve, so this strings robots out along the stroke and binds the packing bonus to the on-shape cluster, which the off-shape robots cannot collect. `r_pack` sets the peak bonus and `pack_range` how far out it decays. The bonus is only active when a robot's nearest-neighbor distance falls in roughly `[tau_sep, tau_sep + pack_range]`, so if `reward/pack_mean` logs near zero throughout a run, the range does not match the swarm's actual spacing and `pack_range` should be raised. Watch `reward/pack_mean` against `rollout/mean_coverage`: the packing term is what gives a gradient to fill the interior, which the distance field cannot.
