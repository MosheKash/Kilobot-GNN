# Code overview

A fuller description of each script than the README summary. The Python side under `python/` is the training code. The Unity side under `Assets/Scripts/` is the simulator.

> Code comments describe **what** a thing does and what it must stay
> consistent with. The **why** -- measurements, failed alternatives, the bug
> that motivated a flag -- lives in [`code-history.md`](code-history.md),
> indexed by symbol, and in [`tuning.md`](tuning.md), the same material
> ordered chronologically by phase.

## Python

### Module map

Every module states its own purpose in its docstring; this is the same
information as one table. If a module's contents drift from this description,
that is a bug in the layout, not in the table.

| module | what it owns |
| --- | --- |
| **entry points** | |
| `launch.py` | KILOBOT_* vars in, a configured run out; connects to Unity and dispatches on mode |
| `run_bc_monitored.py` | the instrumented BC driver: held-out validation, plots, per-state reporting |
| `run_bc_simple_oracle.py` | the plain BC driver: clone the oracle, export an actor |
| `sweep.py` | hyperparameter search; each trial is a real run in its own process |
| **the loop** | |
| `trainer.py` | the rollout/update cycle; orchestration only |
| `parallel.py` | the same collection across several worker processes |
| `buffer.py` | `RolloutBuffer` -- storage and batching, no computation |
| **learning** | |
| `kilobot_gnn.py` | the networks: three actor families, the critic, and the width constants both sides agree on |
| `policy.py` | actor -> sampled actions; the tanh squash and its log-det |
| `ppo.py` | the clipped-surrogate update |
| `gae.py` | advantage estimation |
| `bc.py` | behaviour cloning: both the gradient step and the training loop |
| `bc_replay.py` | the per-oracle-state reservoir BC fits from |
| `graph_batch.py` | collate per-arena graphs into one critic forward pass |
| **what the robot knows** | |
| `observation.py` | worker observation -> actor input tensors |
| `actor_io.py` | `act()`: one decision batch in, one action batch out |
| `belief.py` | the per-robot particle filter over pose |
| `kinematics.py` | dead reckoning: commanded motors -> estimated motion |
| **the task** | |
| `simple_oracle.py` | **the oracle**: a five-state machine, and the BC teacher |
| `spatial_hash.py` | decentralised target assignment (pure math) |
| `formations.py` | the target shapes |
| `reward.py` | the reward function and `coverage` |
| `images.py` / `encoder.py` | formation files -> tensors -> latents |
| **talking to Unity** | |
| `unity_env.py` | build a Unity-backed `EnvWorker` from explicit settings |
| `env_worker.py` | one player, plus the per-robot state Python keeps about it |
| `channels.py` | the `CriticChannel` protocol (must match the C# side) |
| **measurement** | |
| `metrics.py` | logging and aggregation |
| `diagnostics.py` | probes and reporting; nothing here trains |
| `val_tape.py` | recorded held-out BC scoring, no simulation |
| `val_probe.py` | the cold-start check |
| `checkpoint.py` | save/load and actor export (atomic writes) |
| `config.py` | every tunable, in one dataclass |
| `python/tools/` | one-off scripts nothing imports |

Three things moved to make the names true:

- **`bc_train` was in `diagnostics.py`.** It is the project's main warm-start
  path, imported by all three BC entry points, and it trains — it is not
  instrumentation. It now sits in `bc.py` next to the update it drives.
- **Dead reckoning was in `kilobot_gnn.py`.** `dead_reckon`,
  `split_tick_motion` and the `split_track_*` anchor trackers are pure torch
  kinematics with no network involved, consumed by `belief.py`,
  `observation.py` and `simple_oracle.py`. They are now `kinematics.py`.
- **Eval log formatting was in `launch.py`.** `run_eval` stayed — it is a
  dispatch mode and reads launch's resolved configuration — but the 115-line
  printer it called is pure formatting, and moved to `diagnostics.py`.

### The simulator

**Unity is the only simulator.** The Python replica (`replica_env.py`) has been
removed, along with everything that only existed to drive it:
`run_replica_experiments.py`, `rl_driver.py`, `eval_visual.py` and
`tools/measure_oracle.py`. The BC drivers no longer take `--sim`; they always
launch headless players via `unity_env.make_unity_worker`.

Two Unity-side changes made that possible:

- `SwarmManager.Awake` now reads `KILOBOT_MIN_BOTS`/`KILOBOT_MAX_BOTS`, falling
  back to its Inspector values. Swarm size was previously fixed at build time,
  which made `--min-bots/--max-bots` silently inert and meant no test could ask
  for a small, reasoned-about swarm.
- `Assets/Editor/BuildPlayer.cs` builds the player headlessly, so the artifact
  in `Builds/` is reproducible instead of hand-made:

  ```
  <editor> -batchmode -quit -nographics -projectPath . \
           -executeMethod BuildPlayer.BuildLinux -logFile -
  ```

What is fixed at player launch, and what is not: `SwarmManager` reads swarm
size, heartbeat, seed layout and formations once in `Awake`, and
`SceneBootstrap` reads `num_arenas` in `Awake`/`Start` — **arena count included**.
Pushing a new `num_arenas` down the parameters channel and resetting does *not*
restripe a running player. Anything in that list needs a fresh process.

### Where the output goes

A Unity player inherits the launching process's stdout, so by default everything
it printed landed in the middle of the trainer's own numbers. Measured on a
1-iteration, 2-arena run: **339 console lines, ~320 of them the player's**. It is
now 17, all of them training output.

| stream | destination |
|---|---|
| trainer output (iterations, timings, warnings) | console |
| the player's own output (engine banner, physics, every `Debug.Log`) | `results/unity-logs/Player-<worker_id>.log` |
| test players | `results/unity-logs/tests/Player-<worker_id>.log` |
| metrics | `results/tb/run_<timestamp>/` (unchanged) |

One file per worker, so parallel players don't interleave. `results/` is
gitignored. Files are overwritten on the next run.

Three things changed to get there:

- **`unity_env.configure_player_logging`** passes `-logFile` to the player and,
  because that misses the boot.config echo written before it takes effect, also
  flips ml-agents' subprocess stdout to `DEVNULL`. ml-agents already has that
  branch — it just tests `logger.level`, which stays `0` (NOTSET) until
  something sets it, so it never fired. Only applied when redirecting, so a
  player that dies on startup can still be watched.
- **`KILOBOT_DEBUG_WALL_SEEDS`** now gates `WALL_SEED_DUMP` and
  `WALL_SCAN_LIVE`. The dump is one line per wall seed plus a header -- 105 per
  arena -- of static geometry, so a 16-arena run opened with ~1700 lines of it
  before a single training number. Replaced by a one-line per-arena
  summary that is always printed: `SwarmManager arena 0: 53 kilobots, 4 seeds,
  104 wall seeds, layout=corners, heartbeat=48`.
- **`cfg.log_formations`** re-gates the per-reset `arena K: formation N` print.
  `KILOBOT_LOG_FORMATIONS` was supposed to control it, but phase 65 made
  `image_names` unconditional (it translates the local pool index into the
  absolute folder index Unity looks up) and the print keyed off that, so it had
  been firing on every reset of every run since.

To get any of it back: `KILOBOT_DEBUG_WALL_SEEDS=1` for the wall dumps,
`KILOBOT_LOG_FORMATIONS=1` for per-reset shapes, `KILOBOT_UNITY_LOG_DIR=""` to
put the player back on the console.

### Testing against Unity

`tests/conftest.py` owns a session-scoped player factory. A player boots in
~0.7s and resets in ~0.03s, so players are cached for the whole session, keyed
by every launch-fixed setting, and reset (with per-robot state cleared) between
tests. Cleanup runs from `pytest_sessionfinish`, **not** `atexit`:
`UnityEnvironment.close()` can block indefinitely when it first runs during
interpreter shutdown.

Build test fixtures with `conftest.unity_cfg()` and `conftest.unity_trainer()`.
`unity_cfg` starts from `Config()`'s own defaults, which are the values
calibrated against the real player, and overrides only what a test needs — so a
test never silently runs against physics constants the build does not use.

#### The two hooks that make a real player testable

The replica had, for free, two things a real player does not: a seeded RNG and
direct write access to robot positions. `SwarmManager` now provides both, and
`test_unity_hooks.py` covers them.

**Swarm RNG.** Named for the swarm, not "seed": this project already uses that
word for the landmark robots. `KILOBOT_SEED_LAYOUT` places the corner seeds, and
neither those nor the wall seeds are affected by anything here — `SpawnSeeds`
and `SpawnWallSeeds` use fixed coordinates and draw no randomness. What this
varies is exactly the four draws `SpawnKilobots` makes: population count, each
robot's position, each robot's cardinal heading, and the heartbeat phase
stagger. Those are also the only `UnityEngine.Random` calls anywhere in
`Assets/Scripts/`.

Two knobs, because they do two different jobs:

| | reaches | effect |
|---|---|---|
| `KILOBOT_SWARM_RNG` env var | the first spawn and every later one | makes a **run** replayable: same sequence of arenas, but episodes still differ from one another (mixed with a per-arena respawn counter) |
| `swarm_rng` parameters-channel float | every respawn after launch | **pins** the next spawn exactly: same value, same arena, every time |

The env var is the only one that can reach the *first* spawn — `SpawnInitial`
runs from `SceneBootstrap.Start`, before the parameters channel has necessarily
delivered anything, the same problem that makes `KILOBOT_NUM_ARENAS` exist. The
pin is what tests use, via `UnityWorkerFactory.get(swarm_rng=...)`: the fixture
reuses one player for a whole session, so anything mixed with a respawn counter
would hand two tests asking for the same seed two different arenas. The factory
pushes the pin on *every* `get()` so it cannot leak into a later test.

It is deliberately **not** `KILOBOT_SEED`, which `launch.py` uses for torch and
numpy — the player inherits this process's environment, so sharing the name
would silently seed the spawn RNG for anyone who only wanted reproducible
network init. Drivers offset it by worker id (`--swarm-rng` on both BC
drivers, `KILOBOT_SWARM_RNG` in `launch.py`) for the same reason: one shared
value would give every parallel player the identical stream.

> **Measured, and pre-existing:** unseeded spawns are *not* independent across
> processes. Four fresh unseeded players, launched back to back, produced
> byte-identical arenas for 3 of 4 generations in one pair and for 1 of 4 in two
> others. `UnityEngine.Random`'s default state is evidently not process-unique,
> so parallel players have been collecting partly-correlated arenas all along.
> Passing `--swarm-rng` (which offsets per worker) is the fix; nothing about
> the defaults was changed here.

**Pose setting.** `CriticChannel` kind 3 teleports named robots to exact poses:
`worker.send_poses(arena, [(local_index, x, z, heading), ...])`, or
`conftest.place(...)` which also flushes it. Coordinates are arena-local raw
units (what `belief.py` works in), *not* the normalized `node[:, 0:2]` pair;
heading is radians in python's convention, direction `(cos h, sin h)` in
`(x, z)`. Poses queue and apply in `ApplyPending` **after** any reset in the
same packet, so "reset this arena, then put the robots exactly here" works and
indices refer to the new robots. Verified exact to 1e-4 on both position and
heading. Out-of-range indices are dropped with a warning.

Coverage the replica took with it, and why:

- **Exact dead-reckoning checks.** In the replica, physics *was*
  `split_tick_motion`, so heading tracking was exact. Against a real player it
  is approximate — that is what `prop_max_speed` calibration exists for. Still
  gone; the hooks do not help here.
- **Cross-run determinism.** *Restored* by the swarm-RNG pin.
  `test_flag_does_not_change_rewards_or_base_features` is back to comparing two
  full rollouts step for step, and the base features now match to 0.0 exactly
  (checked against a mismatched-seed control, which fails immediately).
- **Position planting.** *Restored* by the pose command. `test_heartbeat` plants
  its two robots at ±60 — clear of each other, of the wall seeds at ±95 and of
  the corner seeds at ±90, since a wall or seed sighting suppresses the
  heartbeat exactly like a neighbour message does — instead of spawning and
  skipping when the spawn did not cooperate.
- **Replica-fidelity tests** (`test_replica_env_*`) are moot by construction:
  Unity is the reference now.

### Note on the sections below

Everything from here down predates the Unity-only migration and is kept as the
project's own record of how each piece came to be. The entries for modules that
no longer exist have been removed, but the surviving prose still *mentions*
`replica_env.py`, `ReplicaWorker`, `OracleCoordinator`, `Arena._assign_targets`,
`rl_driver.py`, `eval_visual.py`, `oracle.py`, `run_bc_real_formations.py`,
`tools/measure_oracle.py`, `CenterSeedRobot.cs` and the `--sim` flag in passing.
**None of those exist** — see "The simulator" above for what replaced them, and
the module map at the top for what is actually here. Read these paragraphs for
rationale, not for the current file layout.

### Entry and orchestration

`launch.py` is the entry point. It reads every `KILOBOT_*` environment variable, builds the connection to the Unity player, loads the encoder and the target images, constructs the actor and critic, and then dispatches: evaluation if `KILOBOT_EVAL` is set, one of the alternative modes in `diagnostics.py` (`bc`, `watch_oracle`, `probe`, `reward_probe`, `audit`, `control`) if `KILOBOT_MODE` selects one, the multi-process learner and workers if `KILOBOT_NUM_WORKERS` is two or more, otherwise the single-process trainer for real RL training. It also wires in checkpointing, resume, and the startup coverage check, and warns up front (rather than after a 600-second hang) if `KILOBOT_NO_GRAPHICS=false` is combined with more than a couple of arenas, since graphics mode was built and tested for watching one arena at a time, not for running full training visibly. It builds a real formation pool for `simple_oracle` when a BC run needs one, conditionally, since the file I/O and image processing would otherwise land on every ordinary training run.

`trainer.py` is the single-process trainer, and the thing every other training entry point (`parallel.py`'s workers, the BC drivers) is ultimately built on. Its `collect` method runs a rollout: it steps the simulator, reads observations, records each arena's graph and reward via `_record_snapshots`, and asks the actor for actions via `_act` (a thin wrapper around `actor_io.act`, see below). Its `run` method is the training loop that alternates collection and the PPO update, logs metrics (including, since phase 8, the fully-resolved `Config` as a tensorboard text summary via `Logger.log_hparams`, called once at the start), and saves checkpoints. `_record_snapshots` is also where the two belief-filter-derived reward bonuses get added: `critic_belief_features` appends belief state to the critic's own node inputs directly (kept here since it is about what the critic sees, not a reward term), while `belief_confidence_bonus`, `seed_wall_find_reward`, and the `reward_shaping` potential term (all from `reward.py` or computed inline) get called here because this is the one place that has both the freshly-computed base reward and the per-robot worker state (belief, pending find-reward, previous distance) those bonuses need -- each one's actual contribution is separately accumulated and logged (`reward/belief_conf_bonus_mean`, `reward/shaping_mean`, phase 8) rather than only visible folded into the total. `belief_population_stats` (phase 9, see `belief.py` below) is called unconditionally alongside these for the `belief/*` diagnostic tags, whether or not `belief_conf_bonus` is active. `_init_globals` also owns two startup warnings that print rather than silently letting a common misconfiguration through: `heartbeat_ticks=0` with `gru_split_observation` (phase 5), and `belief_conf_bonus` used without an anneal horizon (phase 8). The module also has `_reset_arena`, the startup coverage check, and `make_critic`/`critic_extra_features`, which build the critic with or without the belief-filter node features appended.

`actor_io.py` owns the mechanics of building actor inputs from observations and applying actor outputs, for all three actor types -- extracted from `trainer.py` since it was a genuinely separate concern from orchestration, not because `trainer.py` needed to be shorter for its own sake. `split_obs`/`gather_databases` parse the raw Unity/replica observation vectors. `gather_nodes`, `gather_gru_state`, and `gather_split_state` are the per-actor-type state gathering: the last of these advances the split-observation actor's two persistent odometry trackers (one anchored to the robot's last neighbor event, one to its last seed event, landmark or wall) by the tick's motion and runs the belief filter's predict/update against the seed and wall observations, returning the combined eight-tracker-value-plus-eleven-belief-value proprioception. `sample_split_event` draws one event per robot from a single flat pool holding every individual sender -- each of the four corner-seed slots, each of the four center-cluster slots (phase 11), each of the four wall sides, and every neighbor row -- weighted by signal strength, narrowed to exactly one winner regardless of source (phase 10, `docs/tuning.md`: wall and landmark slots previously let more than one through as a combined category bucket, which a corner's simultaneous two-side range could and did exploit; fixed to match neighbor's existing one-winner treatment, since a real Kilobot has one IR receiver regardless of which of the three broadcasts it hears). `gather_split_state`'s belief-filter fusion consumes this exact same draw rather than independently resampling, so `Tc` and the belief filter can never disagree about which single event happened that tick. `act()` is the dispatcher itself: it calls the right gathering function for the configured actor type, calls the policy, and applies the result, including the per-robot bookkeeping (which tracker to reset, which `pending_find_reward` entry to set) for whichever event actually fired. It returns an event-count dict rather than mutating `Trainer`'s own rollout counters directly, so `Trainer._act` accumulates them instead of `act()` reaching into `self`.

`scripted_motors` (also `actor_io.py`) is the privileged oracle controller used by `cfg.motor_override` -- a proportional "turn toward the nearest target pixel, drive forward" policy, used both as a diagnostic ceiling and as a behavior-cloning teacher. `OracleCoordinator` (phase 16) is a worker-agnostic version of the coordination-aware assignment `replica_env.py`'s `Arena._assign_targets` computes natively (see below): it works from `worker.snapshot()`/`worker.image_id` alone, so it runs against real Unity's `EnvWorker` as well as the replica, with no Unity-side changes needed. `act()`'s wiring prefers a worker's own native `oracle_assigned_direction` when present and falls back to `OracleCoordinator` otherwise, gated behind `cfg.oracle_coordinated`.

**A second, separate, later oracle exists and is not otherwise described in this file: `simple_oracle.py` (phase 90, 2026-07-22).** A complete, from-scratch rebuild, not a modification of `scripted_motors` above -- a minimal five-state design (`go_north` -> `turning` -> `wall_following` -> `navigating` -> `arrived`) using no privileged information at all, selected by `cfg.motor_override="simple_oracle"`. This is what `run_bc_monitored.py` (a later CLI entry point for behavior cloning, also not otherwise described in this file) actually trains against, and what this project's more recent work (`docs/tuning.md` phases 90 onward, especially 142-147) has been built around. If you are working on oracle behavior or BC quality, start there.

A later, multi-phase effort (`docs/tuning.md` phases 70-76) removed every privileged shortcut the oracle and the actor's own observations had been relying on, and rebuilt non-privileged replacements. The global Hungarian assignment `Arena._assign_targets` used to provide was removed entirely, replaced by the already-existing hash-based local navigation as the sole target-selection path; a "beeline to nearest corner via true position" exploration fallback was replaced by the same reactive, non-privileged controller used elsewhere; and true-position-based occupancy/crowding checks were replaced by a reception-gated claim-broadcast mechanism -- a committed robot's chosen target is broadcast over dedicated message slots, and every occupancy/crowding check now reads only from `worker.oracle_received_claims`, populated purely from what a robot has actually, physically received. `stop_on_arrival` (both here and in `replica_env.py`) was found to check only distance, letting a robot declare arrival on a belief the particle filter's own confidence already flagged as untrustworthy -- fixed to also require `belief_conf >= LOCALIZED_CONF_THRESHOLD`, the same bar already required to commit. The arrived-claim broadcast (message slots 6-7, an arrived flag plus the sender's own confidence, distinct from the ordinary slot-3-5 claim) and its extraction (`oracle_received_arrived_claims`) feed `belief.py`'s arrived-claim injection described above; `oracle_arrived_claim_cooldown_ticks`, applied where the claim gets selected in `act()`, throttles how often one robot's broadcast can trigger another's injection, added after real-scale testing found the unthrottled version fired on the large majority of decisions for a cold robot and prevented convergence rather than helping it.

`gather_split_state`'s `belief_init`/`belief_predict`/`belief_update` calls (phase 77) all thread `oracle_known_start_heading` (`config.py`) through to `belief.py`'s corresponding parameters -- see that module's own description above for what this does and why. `watch_oracle.sh` enables it by default (phase 77 addendum); `Config`'s own default stays off, unaffected by that script's choice. `act()`'s own `belief_heading` construction (phase 78) additionally caches each robot's last heading reading that cleared `HEADING_CONCENTRATION_MIN` (`worker._stable_belief_heading`) and falls back to it below that bar -- built as the first fix for a real-Unity oscillation/confidence-collapse bug, tested and found to make outcomes worse than the actual fix (phase 79, `HEADING_NOISE_SCALE=0.0` in `belief.py`), and kept only as an inexpensive backstop that should not trigger now that particles stay synchronized by construction.

`diagnostics.py` holds `launch.py`'s alternative modes: `control_probe` (characterizes the raw action-to-motion mapping with a fixed motor command), `audit_run`/`audit_replay_ratio` (checks that re-evaluating a stored decision reproduces its collection-time log-probability, catching odometry/replay mismatches), `reward_probe` (breaks the reward down by component), `probe_run` (records node features and actions for offline analysis), `bc_train` (the behavior-cloning training loop, built on `bc.bc_update`), and `watch_oracle` (phase 15: forces `motor_override="oracle"` and loops collection under `no_grad` indefinitely, no training step at all -- for `watch_oracle.sh`'s open-ended visual observation, nothing more). None of these run real RL training; they characterize the pipeline, the reward, or a policy's behavior, clone an oracle by supervised learning instead of PPO, or just drive the environment for someone to watch. Kept separate so `launch.py` itself stays focused on connecting to Unity and training.

`bc.py` holds `bc_update`, the actual behavior-cloning gradient step (supervised regression of the actor's motor output toward stored oracle targets). It is not part of `ppo.py` despite being another way to train the actor, because it is a genuinely different algorithm used independently by three callers: the Unity path (`diagnostics.bc_train`), the replica path's own BC loop (`run_replica_experiments.py`), and `run_bc_real_formations.py` (phase 14, BC against real, diverse formation images and the real trained encoder rather than the replica's single default shape) -- bundling it into either PPO or the Unity-specific diagnostics module would have been the wrong home for the other callers.

`parallel.py` is the multi-process trainer. A worker process (`worker_loop`) builds its own Unity instance and actor on CPU, then loops: receive the latest weights, collect a rollout, pack it, and send it back. The `ParallelTrainer` is the learner. It broadcasts weights to all workers, gathers their packed rollouts, merges them, runs the PPO update once on the combined data, checkpoints, and (phase 8) logs the resolved config via `Logger.log_hparams` once at the start the same way the single-process trainer does. This file also holds `pack_buffer` and `unpack_buffer`, which convert a rollout into a small number of large tensors and back, so that moving data between processes is fast, and `merge_buffers`, which combines several workers' rollouts while keeping trajectory identifiers unique. The gather loop detects a dead or stalled worker and aborts cleanly rather than hanging, bounded by `collect_max_wait` (formalized as a real `Config` field in phase 7; previously read only via `getattr` with no way to override it outside a code edit).

`env_worker.py` is a thin wrapper around a single Unity environment, its side channel, and the per-arena state (the target latent, the current image, the neighbor databases, step counts, and episode reward). The trainer holds one of these per environment. `replica_env.py` implements the same interface without Unity (see below), so `trainer.py` runs unchanged against either.

`belief.py` is the per-robot particle filter (`BELIEF_PARTICLES`, currently 256, over x, y, heading) that gives the split-observation actor a pose estimate from only physically available signals. `belief_predict` advances each particle by the executed motor chord in its own frame with motion-proportional noise. `belief_update` fuses observations against the current particles: `_range_log_w` for a corner seed (a full 2D range against a known point, with the ring/mirror ambiguity a single beacon genuinely has), and `_wall_log_w` for a signed, one-axis residual against the interior side of a wall, since it has no rotation ambiguity to preserve; plus an optional peer-ranging term gated behind `KILOBOT_BELIEF_COMMS` (off by default; see `docs/tuning.md`'s 2026-07-06 entry for why). **The center-cluster fusion this paragraph originally also described here (phase 11, sharing `_wall_log_w`'s own mechanism) was removed from the Python side in phase 104, confirmed unused by `simple_oracle.py` -- not corrected inline throughout this paragraph, but do not expect a center-cluster code path to exist.** A cold cloud's first contact with a corner or wall seed gets a matching injection of fresh hypotheses (a ring for a corner seed, a band for a wall reading, tracked through `best_dir_strength`/`best_dir_val` per axis) into the low-weight half of the particles, because a uniform cold cloud has near-zero density anywhere near the true answer and plain resampling cannot manufacture a better candidate than what the initial draw happened to include. `belief_read` produces the eleven-value base proprioception described in `docs/architecture.md`, plus (when given a `target=` argument) three further target-relative values -- fourteen in total from this function alone (phase 106; `docs/architecture.md`'s own "Key shapes and constants" has the full, current breakdown against `SPLIT_ODOM_SIZE`'s real total of 22). `belief_population_stats` (phase 9) is a separate, lighter population-level readout for logging only -- summed `conf_pos`/`conf_x`/`conf_y` and a count above `LOCALIZED_CONF_THRESHOLD`, for `trainer.py`'s `belief/*` tensorboard tags -- computed unconditionally whenever the split-observation actor is in use, not gated behind `belief_conf_bonus`, since the point is visibility into localization quality even in runs that keep the bonus at zero. `set_layout` switches the corner-seed table (`SEED_LAYOUTS`) between `corners` and `cluster`.

Phase 77 (`docs/tuning.md`) added `KNOWN_START_HEADING`/`HEADING_NOISE_SCALE` and threaded a `heading_noise_scale` parameter through `belief_predict` and `belief_update`, gated behind `oracle_known_start_heading` (`config.py`): a robot that genuinely, physically spawns at a known heading can have its filter told this directly (`belief_init`'s `known_start_heading`) and reproduces `true_heading`'s own exactness through legitimate, already-exact `dtheta` accumulation rather than a privileged readout. Getting this to actually hold through the real pipeline required finding and fixing four separate places that were each unconditionally overwriting an already-accurate heading with fresh random noise: `belief_update`'s resample-jitter step, and all three of its particle-injection sites (the corner-seed ring, the arrived-claim ring, and the wall/center band) -- each now checks `heading_noise_scale is not None` and keeps the already-tracked heading instead of redrawing it. Phase 78 subsequently found and fixed two unrelated bugs in the *test methodology* used to validate this at scale (not in this file) -- see `docs/tuning.md` for the full account -- and traced an apparent tail-error case directly to genuine, verified physical isolation rather than any defect in this module.

Phase 79 (`docs/tuning.md`) changed `HEADING_NOISE_SCALE` from a small nonzero value to exactly `0.0`, after a real Unity run (not just the replica) surfaced something phase 78's validation could not have caught: even small, independent per-particle heading noise let particles disagree on heading over time, and since `belief_predict` rotates each particle's own position update by its own heading, disagreement there was inflating position spread -- and therefore collapsing `belief_conf` -- for a reason unrelated to genuine position noise, on top of making `belief_read`'s circular-mean steering readout (`sin_m/r`, `cos_m/r`) swing wildly once concentration (`r`) degraded. At exactly zero, every particle shares one, identical, deterministically-tracked heading, so neither failure mode has a source left to draw from. Substantially strengthened phase 78's own real-scale result (arrivals 11/20 and 7/20 baseline -> 17/20 and 18/20; mean heading error 87.8/69.4 degrees -> 2.0/11.0 degrees, same two seeds). A first fix attempt -- freezing the steering heading onto the last reading above a concentration threshold (`actor_io.py`'s `worker._stable_belief_heading`, `HEADING_CONCENTRATION_MIN`) rather than removing the noise source -- was tested before being trusted and found to make real outcomes worse, not better; its code was kept as an inexpensive defensive backstop rather than reverted, since particles staying synchronized by construction should mean it no longer triggers, but wasn't the actual fix.

Two more fusion mechanisms were added later (`docs/tuning.md` phases 73, 75-76), both gated behind their own `Config` flag and off unless the caller opts in. `_wall_along_log_w`, fed by `wall_seed_xy` (the specific nearest wall seed's own known position, not just aggregate band strength), constrains the along-wall axis a plain wall reading structurally never could -- directly validated as the fix for a wrong-heading particle being otherwise undiscriminable, since that axis previously had nothing to be inconsistent with. Arrived-claim injection mirrors the existing cold-cloud ring injection, centered on a nearby, confidence-gated arrived robot's broadcast position instead of a static seed; validated to meaningfully rescue a genuinely lost robot in isolation, but found -- when tested at real swarm scale rather than a small controlled case -- to fire far too often (the upstream extraction it depends on scans retained message history rather than gating on fresh reception), repeatedly disrupting robots before they could converge. A per-robot cooldown throttles this and is a confirmed improvement over the original, uncapped version specifically, but not yet a confirmed net improvement over not having the mechanism at all -- both fusion additions remain off by default for these reasons, see `config.py`'s own comments on `oracle_wall_seed_position` and `oracle_arrived_claim_injection` for the full evidence.

### Fast iteration: the replica


`Arena._assign_targets` (phase 13) is the replica's native coordination-aware oracle support: a one-time, per-episode optimal assignment (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`) between robots and a discretized set of target points, recomputed only at episode start via the same `spawn()` hook every reset already runs through. Deliberately kept separate from `Stroke`/`Formation`'s own `dist_dir`, which the reward function's coverage measurement depends on and which this must never influence -- coverage still means "reached the nearest actual on-shape point," not "reached my assigned point." `ReplicaWorker.oracle_assigned_direction` looks this up per robot for `actor_io.scripted_motors`; `actor_io.OracleCoordinator` is the worker-agnostic version of the same computation that also works against real Unity (see `actor_io.py` above).


`watch_actor.sh` and `watch_oracle.sh` (`python/`, phases 12 and 15) both drive one visible, single-arena real Unity window with no training at all, for open-ended visual inspection -- the former a trained checkpoint via `KILOBOT_EVAL`, the latter the oracle via `KILOBOT_MODE=watch_oracle`. Neither needs a meaningful checkpoint: the actor starts randomly initialized in the oracle case, since the oracle overrides whatever it would have produced anyway. Both require a separate, headed Unity build -- the normal training build is a headless server build with no rendering pipeline compiled in at all, so pointing either script at it opens no window.

### Learning

`ppo.py` is the PPO update. It computes normalized advantages, runs several epochs over the decisions with the clipped policy objective for the actor and a value-regression loss for the critic, and returns the logged quantities. It contains the device handling that moves the critic inputs and the actor batch onto the learner's device, so the critic can run on a GPU while the rollout itself was collected on CPU. `_stack_decisions`, which turns a list of buffer decisions into batched tensors, lives here since `ppo_update` itself uses it, but it is shared: `bc.py` and `diagnostics.py`'s `audit_replay_ratio` both import it from here rather than duplicating it. Behavior cloning (`bc_update`) is not in this file -- it is a different algorithm, in its own `bc.py`.

`buffer.py` is the rollout buffer. It stores two kinds of things: per-step graph snapshots for the critic, and per-decision records for the actor (the target latent, the seed, the neighbor messages, the chosen action, and the log-probability). It computes the critic's old values, runs the returns and advantages, and assembles the batched inputs the PPO update consumes.

`gae.py` is generalized advantage estimation, kept separate so it is easy to read and test. It takes rewards and value estimates with episode boundaries and produces advantages and returns.

`policy.py` wraps the actor as a Gaussian policy. During collection it samples an action and reports the log-probability. During the update it evaluates the log-probability and entropy of stored actions. It also provides a deterministic mode that returns the distribution mean, which is what evaluation uses, and it clamps actions to the valid range. Each actor type has its own act/evaluate pair (`act_batch`/`evaluate_batch` for DeepSet, `act_batch_gru`/`evaluate_batch_gru` for the recurrent actor, `act_batch_split`/`evaluate_batch_split` for the split-observation actor), all sharing the same squashed-Gaussian machinery.

`kilobot_gnn.py` holds the model definitions and the shared constants. Three decentralized actors live here: the set-based DeepSet network described in the architecture doc, the recurrent GRU actor, and the event-driven split-observation actor (`SplitObservationActor`), each with its own batched forward function (`actor_forward_batch`, `recurrent_forward_batch`, `split_forward_batch`). The critic is the graph attention network. The constants here (latent size, message size, action size, node feature count, neighbor database capacity, and so on) are imported across the rest of the code, so this file is the single source of truth for tensor shapes.

`graph_batch.py` takes the list of per-arena graphs for a step and concatenates them into one batched graph, offsetting edge indices so the arenas stay separate, which is the form the critic runs on.

`reward.py` computes the per-step reward and the shape coverage, including the geometry of how close the swarm is to the target. `belief_confidence_bonus` and `seed_wall_find_reward` are the two split-observation-actor-only additive terms, both taking their state explicitly (a belief dict, a pending-reward dict) rather than reaching into `Trainer`/worker internals: the first pays `belief_conf_bonus * conf_pos` per step as localization scaffolding, meant to be annealed to zero; the second is a one-time nudge (`seed_find_bonus` the tick after a decision was triggered by a landmark seed, `-wall_find_penalty` after a wall seed), deliberately much smaller than the on-shape reward, to bias exploration toward landmark contact without competing with the task itself. Both are called from `trainer.py`'s `_record_snapshots`, the one place that has both the freshly-computed base reward and the relevant worker state in scope.

### Simulator interface and support

`channels.py` defines the ML-Agents side channels. The important one receives each arena's interaction graph from Unity every step and parses it into tensors, reshaping the flat float list Unity sends against `NODE_FEATURES` imported from `kilobot_gnn.py` (phase 17: previously an independently hardcoded `19` -- Unity's own message carries no width at all, so this was a one-sided assumption with no cross-check). It also carries timing information used for the performance logging.

`encoder.py` loads the pretrained image encoder and checks that its output dimension matches the expected latent size.

`images.py` handles image loading and preprocessing and builds the pool of target images that arenas draw from.

`metrics.py` owns both computing and reporting training metrics. `rollout_stats`, `aggregate_payloads` (multi-worker merge), `build_histograms`, and `merge_stats` turn the raw counters `trainer.py`/`parallel.py` accumulate during a rollout into named values (`reward/on_bonus_mean`, `rollout/mean_coverage`, and so on) -- moved here from `trainer.py` since computing a metric from already-gathered data is a different job than gathering it. `Logger` writes those values to TensorBoard and prints the per-iteration console line, degrading gracefully to console only if TensorBoard is not installed; `Logger.log_hparams` (phase 8) additionally writes the fully-resolved `Config` as a TensorBoard text summary once at the start of a run, using `vars(cfg)` rather than `dataclasses.asdict(cfg)` so it works on both a real `Config` and the `SimpleNamespace` stubs some tests use in its place. `RunSummary` writes a small JSON record of a run (per-iteration coverage, entropy, explained variance, and a few derived numbers like the entropy slope), rewritten atomically each iteration so the hyperparameter sweep can read progress while a run is still going.

`config.py` is the `Config` dataclass with the hyperparameters.

`checkpoint.py` handles persistence. It saves and loads full checkpoints (actor, critic, both optimizers, and the iteration), with atomic writes so a crash mid-save cannot corrupt a checkpoint, and it handles moving optimizer state onto the right device when resuming. It also exports just the actor as `actor_final.pt`, which is the deployable policy, and provides the loader that evaluation uses to read either a full checkpoint or an exported actor.

### Hyperparameter search

`sweep.py` is the search harness, built on Optuna. Each trial launches `launch.py` as a subprocess with a sampled set of `KILOBOT_*` settings, polls the run's summary JSON, stops a trial early when it is clearly behind its peers at the same iteration, and scores the finished run on coverage gained minus any climb in entropy. It writes the best configuration to `best.json` and can re-run that configuration across several seeds (`confirm_best`) to check it holds up -- which requires carrying forward every one of the ten swept parameters, not a subset (phase 17: `r_pack`/`pack_range` were silently dropped, so confirmation runs never actually tested the true best config for those two). The scoring function is a plain function so it can be tested without Optuna or Unity. The full workflow is in `docs/sweep.md`.

### Tests

All test files live under `python/tests/` (moved there 2026-07-07; a `conftest.py` in that folder puts `python/` on `sys.path` so imports resolve regardless of how `PYTHONPATH` is set). Run with `python -m pytest -q` from `python/`, or from `python/tests/` directly.

`test_kilobot.py` is the main suite: model shapes, the reward and coverage math, GAE, the batched critic inputs, the PPO loss, the deterministic action path, the startup coverage check, and (phase 6) a guard that a plain `Config()` stays pinned to the calibrated kinematics (`prop_max_speed`/`prop_wheelbase`) and that the paired scale constants keep dead-reckoned proprioception near unit range for both the split and GRU actors, so the two cannot silently drift apart again the way they once did. `test_parallel.py` covers the multi-process pieces: buffer merging and uniqueness, the pack and unpack round trip, the worker orchestration, and the dead-worker abort. `test_checkpoint.py` covers save, load, resume, and the actor export. `test_sweep.py` covers the search scoring, the run-summary writer, and (phase 17) that `confirm_best` carries forward every parameter `suggest_params` actually sweeps, not a subset. `test_split_actor.py` covers the split-observation actor's model and wiring: parameter count against budget, the single-robot/batched forward paths, the two-tracker dead-reckoning geometry, the event sampling (weighted across every individual landmark slot, wall side, and neighbor row in one flat pool as of phase 10, narrowed to exactly one winner), the reset logic per tracker, and the device-threading fixes. `test_split_actor_trains.py` is a convergence test, not a unit test: it runs the real PPO pipeline on a small synthetic task to confirm the actor can actually learn from it, not just execute without error. `test_images.py` covers the formation image pool, including the device-placement bug documented in `docs/tuning.md`; two of its tests need a CUDA device and skip otherwise. `test_additional.py` is the catch-all for wiring gaps found by auditing the codebase and by later feature work: `bc_update` and the audit replay-consistency check for every actor type, the `KILOBOT_INIT_ACTOR` warm-start path, the diagnostic-mode dispatch fix, and the launch.py helper functions among others.

`test_fixes.py` covers the three phase-1 defects (executed-vs-sampled motor odometry, weighted event sampling carrying neighbor strength, seed-only sighting eligibility) directly. `test_belief.py` covers the particle filter's core physics: a single beacon exposes range but stays humble about full position, three beacons (visited sequentially, not simultaneously) collapse to a confident absolute fix, peer ranging cannot mint anchor confidence on its own, and `belief_read`'s shape/bounds. `test_conf_bonus.py` covers the confidence formula, its annealing schedule, and that non-split actors are unaffected. `test_heartbeat.py` covers forced event-less decisions and their staggering. `test_critic_belief.py` covers `critic_belief_features`: the flag toggles the critic's input width correctly, appended values are in valid ranges, PPO still runs end to end with it on, and base node features/rewards are bit-identical whether it is on or off. `test_seed_find_reward.py` covers `seed_find_bonus`/`wall_find_penalty`: a landmark event sets a positive pending reward and a wall event a negative one (a neighbor-only event sets neither), and the pending reward gets applied with the correct sign and magnitude on the next tick and cleared -- each check isolates the added bonus against an identical-seed baseline run, since two robots' base on/off-shape rewards otherwise differ with their (randomly drawn) positions. `test_wall_seeds.py` covers the wall-seed band fusion specifically: a wall observation pulls one axis and leaves the other genuinely unconstrained, it alone never reaches full position confidence the way a landmark beacon does, a landmark-and-wall hit on the same cold tick defers correctly to the landmark's ring injection, the wall-spacing constant actually guarantees no gap along any wall (checked numerically, not just asserted by construction), and the four true corners are within range of both adjacent walls. Since phase 10 (`docs/tuning.md`), a corner constrains both axes over two separate ticks, never one -- `test_corner_needs_two_ticks_to_constrain_both_axes_via_real_pipeline` drives an actual corner through the real pipeline (`sample_split_event` and `belief_update` together, not a hand-built observation) and checks both the hard invariant (never two axes in one tick) and the statistical outcome (most robots do eventually reach both axes confident, just never simultaneously); `test_belief_update_still_fuses_two_simultaneous_wall_axes_if_given_them` (renamed from a test that used to assert the corner double-fire as correct pipeline behavior) confirms `belief_update`'s own fusion math still handles a hand-built two-axis observation if directly given one, which can no longer happen via the real pipeline but is worth keeping as a regression guard on that function's own generality.

None of the tests above need Unity; they run on CPU except the two CUDA-conditional cases in `test_images.py`. Everything that is not a test but also not part of the maintained pipeline -- one-off diagnostic scripts, historical shell run-scripts, and old checkpoints/result histories -- lives under `python/temp_test_material/` (see its own `README.md`), not under `tests/`.

## Unity (`Assets/Scripts/`)

`SceneBootstrap.cs` runs at scene start and spawns the requested number of arenas. It reads the arena count from the environment so one build can run different counts.

`SwarmManager.cs` manages a single arena's swarm: spawning the robots, the corner seeds, the wall-lining seeds, and (phase 11) the center-cluster seeds, holding the target shape, and coordinating the arena's lifecycle. `SpawnWallSeeds()` places 26 seed robots per wall at the current `WALL_SPACING`, spaced so no point on any wall is farther than `IR_RANGE` from the nearest one; `SpawnCenterSeeds()` places the four center-cluster seeds near the origin the same defensive way (`AddCenterSeed`, kinematic rigidbody and trigger collider regardless of prefab configuration). The per-kilobot scan loop computes each wall's and each center seed's strength the same way it computes corner-seed strength, and decision eligibility counts a wall-only or center-only sighting the same as a corner one. If `KILOBOT_SHOW_RADIUS` is set, every spawned robot (kilobot, corner seed, wall seed, center seed) also gets a `CommRadiusIndicator` child.

`KilobotAgent.cs` is the ML-Agents agent for one robot. It gathers the robot's observations (`seedObs` for the four corner seeds, `centerObs` for the four center-cluster channels, `wallObs` for the four wall-side channels, and what it has heard from neighbors) and applies the action it receives from Python.

`KilobotMovement.cs` is the robot's movement model, turning motor commands into motion under the physics.

`SeedRobot.cs` is the corner seed: a fixed, precisely-known-position beacon (`SeedType`: four corner positions, `UpperLeft` through `LowerRight`, depending on `KILOBOT_SEED_LAYOUT`; the fifth, origin, position was removed in phase 11, replaced by the center cluster below) that gives the swarm a shared, exact reference point.


`WallSeedRobot.cs` (new 2026-07-07) is the wall-lining seed: unlike a landmark seed it does not have an individually known position from the actor's point of view, only a `WallSide` (north/east/south/west). Many of these line each wall so a robot can never go a whole episode without hearing from one; see `docs/architecture.md`'s "Wall seeds" section for the fusion math this enables on the Python side. Spawned wall seeds are forced kinematic with a trigger collider in `SwarmManager.AddWallSeed`, independent of the prefab's own configuration, because two of them share almost the same position at every corner by design.

`CriticChannel.cs` is the side channel that builds and sends each arena's interaction graph (nodes and edges) to Python every step. This is the privileged information the critic trains on.

`ImageLibrary.cs` loads the target shape images into the scene so arenas can be assigned shapes.

`StepDriver.cs` and `HeartbeatAgent.cs` keep the simulation stepping at a regular cadence in sync with the Python side.

`CommRadiusIndicator.cs` (new 2026-07-07) is an opt-in debug visualization, gated behind `KILOBOT_SHOW_RADIUS`: a translucent disc at a robot's true IR radius, parented under it so the robot stays visible in the center. Built entirely at runtime from a procedural circle mesh and a built-in shader, so it needs no prefab or material asset; the mesh and one material per robot kind are cached rather than rebuilt per instance. Counter-scales against the parent's `lossyScale` so the circle renders at the correct absolute size regardless of how the robot prefab itself is scaled -- a child's scale otherwise multiplies with its parent's in Unity, which was a real bug here (a circle rendering several robot-widths too large) before the counter-scale was added.

`PostBuildCopyGrpc.cs` is an editor build step that copies a needed gRPC library into the build output. `FlyCam.cs` is a debug camera for watching a scene in the editor. `Readme.cs` and `ReadmeEditor.cs` are the Unity sample readme asset and are not part of the simulation.
