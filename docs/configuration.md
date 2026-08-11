# Configuration

Everything is set through environment variables read by `python/launch.py`. The Docker image sets sensible defaults for several of them; when you run the Python directly, the code defaults apply. Where those differ, both are noted below.

`config.py` holds the numeric hyperparameters (learning rates, GAE settings, rollout length, and so on). The environment variables below are the knobs you change from run to run.

## Run mode and devices

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_MODE` | `rl` | `rl` trains, `bc` behaviour-clones; `probe`, `reward_probe`, `audit`, `control` and `watch_oracle` run diagnostics. In the container, `train`/`smoke`/`eval` are additional aliases that set `KILOBOT_SMOKE`/`KILOBOT_EVAL` for you; any other value is passed through to `launch.py` unchanged. |
| `KILOBOT_DEVICE` | `cpu` (image sets `cuda`) | Device for the learner's PPO update and the critic. Workers always run the actor on CPU regardless of this. |
| `KILOBOT_SMOKE` | `1` in code, `0` in the image | Run a short validation loop instead of full training. |
| `KILOBOT_EVAL` | `0` | Load a saved policy and run deterministic evaluation instead of training. |
| `KILOBOT_ITERATIONS` | `100` | Number of training iterations. Ignored in smoke mode. |

Note on smoke: the code default for `KILOBOT_SMOKE` is on, so a bare `python launch.py` runs a smoke loop. The Docker image turns it off so the default container run is real training. Set it explicitly when you want to be sure.

## Parallelism

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_NUM_WORKERS` | `1` | Worker processes, each driving one Unity instance. `1` is single process. `2` or more uses the learner and worker split. |
| `KILOBOT_NUM_ARENAS` | `9` | Arenas inside each Unity instance. Keep this at roughly 16 or fewer; the graph side channel sends every arena's graph each step and there is a size ceiling on that channel. |
| `KILOBOT_NUM_ENVS` | `1` | Threaded single-process environments. This is an older path; leave it alone when using workers. |
| `KILOBOT_BASE_PORT` | `5005` | Worker i connects on `BASE_PORT + i`. These ports must be free. |
| `KILOBOT_COLLECT_MAX_WAIT` | `1200.0` | Seconds `ParallelTrainer` waits, in total across retries, for every worker to report a rollout before treating the run as stalled and aborting (latest checkpoint retained). Only read by `ParallelTrainer` (`KILOBOT_NUM_WORKERS>1`); has no effect single-process. Formalized as a real `Config` field on 2026-07-10; previously read only via `getattr(cfg, "collect_max_wait", 1200.0)` in `parallel.py` with no way to override it short of a code edit. |

Total environment steps of data per iteration is roughly `NUM_WORKERS x NUM_ARENAS x rollout_steps`. To scale throughput, add workers up to about your core count and let a GPU absorb the larger PPO batch, rather than pushing arenas per instance past the channel ceiling.

## Checkpointing and evaluation

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_CKPT_EVERY` | `10` | Save a checkpoint every N iterations into the run folder under `KILOBOT_LOGDIR`. A final checkpoint and an exported actor are always written at the end. |
| `KILOBOT_RESUME` | unset | Path to a `ckpt.pt` to continue from. Restores the actor, critic, optimizers, and iteration count, then trains up to `KILOBOT_ITERATIONS`. |
| `KILOBOT_INIT_ACTOR` | unset | Path to a `ckpt.pt` or an `export_actor` file (such as behavior cloning's `KILOBOT_BC_OUT`) to warm-start the actor's weights from. Unlike `KILOBOT_RESUME`, this loads only the actor (and `log_std`); the critic and optimizers start fresh and training begins at iteration 0. Applies to every mode, not just `rl`. Meant for continuing PPO from a behavior-cloned actor rather than resuming an interrupted run. |
| `KILOBOT_EVAL_WEIGHTS` | unset | Path to a `ckpt.pt` or `actor_final.pt` to evaluate. Falls back to `KILOBOT_RESUME` if unset. |
| `KILOBOT_EVAL_ITERS` | `5` | Number of deterministic rollouts to aggregate during evaluation. |
| `KILOBOT_MIN_START_COV` | `0` | If greater than zero, abort when the first iteration's mean coverage is below this. Useful for catching a bad run where the encoder or images did not load. |

Evaluation always runs on CPU with a single environment, so it ignores `KILOBOT_DEVICE`.

## Paths

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_BUILD_PATH` | `../Builds/Kilobot.x86_64` | The headless player to launch. The image points this at `/app/Builds/Kilobot.x86_64`. |
| `KILOBOT_ENCODER_PATH` | `../data/image_encoder.pt` | The pretrained target-image encoder. |
| `KILOBOT_FORMATIONS` | `../data/formations` | Folder of target shape images. |
| `KILOBOT_LOGDIR` | `../results/tb` | Where run folders, TensorBoard logs, and checkpoints are written. |
| `KILOBOT_BEHAVIOR` | auto | Override the ML-Agents behavior name if auto-detection picks the wrong one. |

## Simulator

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_NO_GRAPHICS` | `1` | Run Unity headless. Leave on for real training; set to `false` to open a normal window (needs a display), useful for visually sanity-checking spawns and behavior. |
| `KILOBOT_TIME_SCALE` | `20` | Unity physics time scale. Higher runs faster but can destabilize physics. Set to `1` when graphics are on, or it runs too fast to watch. |
| `KILOBOT_SHOW_RADIUS` | `0` | With graphics on, draw each kilobot/landmark-seed/wall-seed's IR communication radius as a translucent disc (built at runtime, no prefab needed). Purely visual; has no effect headless. |
| `KILOBOT_SHOW_TARGET_FLOOR` | `1` | Texture each arena's floor with its current target formation image, so it is visually obvious whether kilobots are sitting on the shape. Requires assigning a `Renderer` to `floorRenderer` on that arena's `SwarmManager` in the Inspector -- defaults on, but does nothing until that per-arena assignment is made. Set to `0` to disable while keeping the reference assigned. |
| `KILOBOT_FLOOR_ROTATION_STEPS` | `2` | Number of 90-degree CCW rotations (0-3) applied to the floor texture before display, purely visual -- does not touch `BakeImage` or any reward/oracle geometry. Purely visual: it does not touch `BakeImage`, the reward, or the oracle. |
| `KILOBOT_TIMEOUT` | `600` | Seconds to wait for the Unity handshake. |
| `KILOBOT_MAX_FORMATIONS` | `256` | How many images to load from the formations folder. |
| `KILOBOT_BAKE_ROTATION_STEPS` | `1` | Number of 90-degree CCW rotations (0-3) applied to `ImageLibrary`'s own `onPoints` geometry inside `BakeImage`, before the distance/direction field is built from it. Unlike `KILOBOT_FLOOR_ROTATION_STEPS`, this affects the actual reward (`coverage`/`on_bonus`/`off_penalty`) and the uncoordinated oracle directly, not just what gets displayed. Unlike `KILOBOT_FLOOR_ROTATION_STEPS`, this affects the actual reward geometry, so changing it changes what the swarm is scored against. |
| `KILOBOT_LOG_FORMATIONS` | `0` | Print which formation file each arena is showing whenever it's set (initial spawn and every episode reset) -- e.g. `arena 0: formation 12 (000013.png)`. Off by default: a normal, many-arena training run resets constantly and this would flood the log; `watch_actor.sh`/`watch_oracle.sh` opt in. Added phase 30, `docs/tuning.md`. |
| `KILOBOT_IMAGE_MODE` | `L` | Image mode used when loading targets (`L` is grayscale). |
| `KILOBOT_IMAGE_SIZE` | `28` | Target image is resized to this square size before encoding. |
| `KILOBOT_ROLLOUT_STEPS` | from `config.py` | Override the rollout length. |
| `KILOBOT_MAX_EPISODE_STEPS` | from `config.py` | Override the maximum episode length. |
| `KILOBOT_SUCCESS_THRESHOLD` | from `config.py` (`0.85`) | Coverage fraction that ends an episode as a success. No override existed before phase 21; added the same way other real, previously-un-overridable settings have been. A value above `1.0` makes success mathematically unreachable (`coverage()` is the mean of a boolean tensor, structurally bounded to `[0,1]`), useful for disabling success-triggered resets entirely -- see `watch_oracle.sh`. |
| `KILOBOT_TORCH_THREADS` | `1` | Caps Torch CPU threads. Set at process start, before Torch is imported. |
| `KILOBOT_MIN_BOTS` / `KILOBOT_MAX_BOTS` | Inspector values (`20`/`50`) | Swarm size per arena; the player draws uniformly between them. Read once in `SwarmManager.SpawnInitial`, so they must be set before the player launches. Without these, swarm size is fixed at build time and `--min-bots`/`--max-bots` on the BC drivers are inert. |
| `KILOBOT_SWARM_RNG` | unset | Seed the player's spawn RNG, making a run's sequence of arenas replayable. Episodes still differ from one another. Deliberately NOT `KILOBOT_SEED`, which seeds torch/numpy -- the player inherits this process's environment, so one name for both would silently seed spawns for anyone who only wanted reproducible network init. Each worker gets this plus its worker id. The `swarm_rng` parameters-channel float is the separate, test-facing knob that pins ONE spawn exactly. |
| `KILOBOT_UNITY_LOG_DIR` | `results/unity-logs` | Where each player writes its own log (`Player-<worker_id>.log`). The player otherwise inherits this process's stdout and interleaves ~320 lines of engine output with the trainer's numbers. Set to `""` to put it back on the console, which is what you want when a player is failing to start. |
| `KILOBOT_DEBUG_WALL_SEEDS` | `0` | Re-enable the `WALL_SEED_DUMP` (104 lines per arena of static wall-seed geometry) and `WALL_SCAN_LIVE` dumps. Off by default; a one-line per-arena summary is always printed instead. |

## Hyperparameters in config.py

The reinforcement-learning settings live in the `Config` dataclass: learning rates for the actor and critic, the number of PPO epochs and the minibatch handling, the GAE discount and lambda, the PPO clip range, the rollout length, the episode length, the success threshold for coverage, and the initial policy standard deviation. Edit them there. A few are also exposed as environment variables above for convenience.

## Hyperparameter overrides

Each of these overrides the matching `config.py` field for one run, without editing the file. Unset variables keep the default. This is what the hyperparameter sweep sets per trial, and you can set any of them by hand for a single run.

| Variable | Config field | Default |
|---|---|---|
| `KILOBOT_ACTOR_LR` | `actor_lr` | 3e-4 |
| `KILOBOT_CRITIC_LR` | `critic_lr` | 1e-3 |
| `KILOBOT_ENTROPY_COEF` | `entropy_coef` | 0.01 |
| `KILOBOT_CLIP` | `clip` | 0.2 |
| `KILOBOT_PPO_EPOCHS` | `ppo_epochs` | 4 |
| `KILOBOT_MINIBATCH` | `minibatch` | 1024 |
| `KILOBOT_GAMMA` | `gamma` | 0.99 |
| `KILOBOT_GAE_LAMBDA` | `gae_lambda` | 0.95 |
| `KILOBOT_MAX_GRAD_NORM` | `max_grad_norm` | 0.5 |
| `KILOBOT_LOG_STD_INIT` | `log_std_init` | -0.5 |
| `KILOBOT_CRITIC_CHUNK` | `critic_chunk_steps` | 64 |
| `KILOBOT_R_ON` | `r_on` | 1.0 |
| `KILOBOT_K_POS` | `k_pos` | 1.0 |
| `KILOBOT_TAU_V` | `tau_v` | 0.05 |
| `KILOBOT_L_SCALE` | `l_scale` | 1.0 |
| `KILOBOT_K_SEP` | `k_sep` | 1.0 |
| `KILOBOT_TAU_SEP` | `tau_sep` | 0.08 |
| `KILOBOT_R_PACK` | `r_pack` | 1.0 |
| `KILOBOT_PACK_RANGE` | `pack_range` | 0.20 |
| `KILOBOT_SEED` | `seed` | 0 |

The seed is applied to Torch and NumPy at startup, so runs are reproducible given the same configuration.

## Actor architecture

The actor defaults to the DeepSet aggregator. Set `KILOBOT_ACTOR=gru` to use the recurrent actor instead, or `KILOBOT_ACTOR=gru_split_observation` for the event-driven split-observation actor. The `prop_*` variables set the kinematic constants of the dead-reckoned proprioception the GRU is given, and are calibrated against the real robot (`prop_max_speed`/`prop_wheelbase`, fitted by `calibrate_kinematics.py` -- see the row below; these were placeholders before 2026-07-10, when a regression that had silently reverted them to those placeholders for three phases was found and fixed, `docs/tuning.md`). The `split_*` variables are the split-observation actor's equivalents: `split_prop_scale`/`split_prop_time_scale` condition its odometry inputs the same way `prop_scale`/`prop_time_scale` do for the GRU actor, while `prop_max_speed`/`prop_wheelbase` (the physical robot constants) are shared between both.

| Variable | Config field | Default | Notes |
|---|---|---|---|
| `KILOBOT_ACTOR` | `actor_type` | `deepset` | `deepset`, `gru`, or `gru_split_observation`. Selects the actor architecture. See `docs/architecture.md`. |
| `KILOBOT_GRU_HIDDEN` | `gru_hidden` | 128 | Hidden size of the GRU actor. Ignored otherwise. |
| `KILOBOT_SPLIT_UPSCALE_HIDDEN` | `split_upscale_hidden` | 40 | Width of the split-observation actor's Tc/odometry upscale MLP. Ignored otherwise. |
| `KILOBOT_SPLIT_GRU_HIDDEN` | `split_gru_hidden` | 59 | Hidden size of the split-observation actor's GRU. Ignored otherwise. |
| `KILOBOT_SPLIT_HEAD_HIDDEN` | `split_head_hidden` | 40 | Hidden size of the split-observation actor's policy head. Ignored otherwise. |
| `KILOBOT_SPLIT_ACTIVATION` | `split_activation` | `relu` | Hidden activation at `up1` and `head1` of the split-observation actor: `relu`, `elu`, `silu`, `leaky_relu` or `tanh`. Holds no parameters, so it changes no checkpoint's shape -- but a checkpoint trained under one value and evaluated under another is a different function, so set it to whatever the checkpoint's `meta["activation"]` says (`bc_offline.py` records it; `tools/eval_closed_loop.py` reads it automatically). `relu` is the historical value and is what died in phase 154. |
| `KILOBOT_SPLIT_PROP_SCALE` | `split_prop_scale` | 0.04 | Scale on the split-observation actor's distance-to-anchor channel, applied to both the neighbor-anchored and seed-anchored odometry trackers. Targets the p90 of measured anchor-tracker distance (~21-27 raw units at prop_max_speed=1.55, heartbeat_ticks=48, 50-100 bots, cluster layout) toward O(1); re-derive against the replica if prop_max_speed or the population/heartbeat regime changes materially (docs/tuning.md 2026-07-10 entry). |
| `KILOBOT_SPLIT_PROP_TIME_SCALE` | `split_prop_time_scale` | 0.02 | Scale on the split-observation actor's elapsed-time-since-anchor channel, applied to both odometry trackers. Targets the p90 of measured anchor-tracker elapsed time (~44-80s at max_episode_steps=2048) toward O(1); depends on max_episode_steps, not on prop_max_speed. |
| `KILOBOT_SPLIT_SEED_WEIGHT_BOOST` | `split_seed_weight_boost` | 1.0 | Multiplies the seed sighting's weight in the split-observation actor's event pool before sampling. 1.0 leaves the weighted, no-priority-between-kinds sampling exactly as specified; raise it to sample seed events more often if the seed event rate below turns out low. |
| `KILOBOT_PROP_MAX_SPEED` | `prop_max_speed` | 4.0 | Max wheel speed used to scale dead-reckoned odometry. Shared by the GRU and split-observation actors. DERIVED in `config.py` from Unity's own moveSpeed / turnSpeed / framesPerStep / Fixed Timestep rather than fitted, so it cannot drift out of sync; `tools/check_dead_reckoning.py` verifies the result against a live player and `tests/test_kilobot.py` asserts the derivation. |
| `KILOBOT_PROP_WHEELBASE` | `prop_wheelbase` | 4/pi = 1.27324 | Wheelbase used to turn differential wheel speeds into a heading change. Shared by the GRU and split-observation actors. DERIVED in `config.py` from Unity's own moveSpeed / turnSpeed / framesPerStep / Fixed Timestep rather than fitted, so it cannot drift out of sync; `tools/check_dead_reckoning.py` verifies the result against a live player and `tests/test_kilobot.py` asserts the derivation. |
| `KILOBOT_PROP_SCALE` | `prop_scale` | 20.0 | Scale on the GRU actor's odometry inputs so they are the right magnitude for the network. The GRU actor decides almost every tick in a dense swarm, so this targets the single-tick chord, not a multi-tick interval. |
| `KILOBOT_PROP_TIME_SCALE` | `prop_time_scale` | 20.0 | Scale on the GRU actor's elapsed-time proprioception channel. Same single-tick basis as prop_scale. |
| `KILOBOT_PROP_CUM_SCALE` | `prop_cum_scale` | 0.02 | Scale on the GRU actor's cumulative distance-travelled channel. GRU only; the split-observation actor has no cumulative channel. Calibrated at max_episode_steps=2048; re-derive if that changes materially. |
| `KILOBOT_ACTOR_PRIV_MODE` | `actor_priv_mode` | `none` | Privileged columns appended to the seed input for debugging: `none`, `dir`, `heading`, `dir_heading`, `pose`, `full`. Deployment uses `none`; the others hand the actor otherwise-unavailable information to isolate whether it can use it. DeepSet and GRU only; not supported by the split-observation actor. |
| `KILOBOT_DIRECT_MOTOR` | `direct_motor` | `0` | DeepSet only. Adds a motor head that bypasses part of the aggregation. |
| `KILOBOT_SEED_LAYOUT` | `seed_layout` | `corners` | Landmark-seed placement, applied consistently to the belief filter's landmark table and Unity's `SpawnSeeds()`. `corners` (four far corners) is the only supported layout; `cluster` exists in `belief.SEED_LAYOUTS` for backward compatibility only and is marked DEPRECATED, DO NOT USE. The player reads this variable itself at startup. Independent of it, seed robots line all four walls, so a robot can never go a whole episode without hearing something near the boundary; see `docs/architecture.md`. `IR_RANGE` (7.0 cm, matching the real Kilobot's short-range IR) is not an environment variable -- it is a constant in `belief.py` and `SwarmManager.cs`, kept in sync by hand. |
| `KILOBOT_BELIEF_COMMS` | `belief_comms` | `0` | Experimental, off by default. Floats 0-2 of the broadcast message carry the belief beacon (x, y, conf) so peers can range off localized neighbors. Every disciplined variant tried so far degraded accuracy by double-counting correlated information (see `docs/tuning.md`); leave off unless implementing covariance intersection or hop-count trust. |
| `KILOBOT_BELIEF_CONF_BONUS` | `belief_conf_bonus` | 0.0 | Split-observation actor only. Adds `bonus * conf_pos` to each robot's per-step reward, paying for holding a collapsed pose belief. Localization scaffolding: it makes visiting beacons rewarding before navigation itself pays, breaking the policy-localization bootstrap (`docs/tuning.md` 2026-07-06). Meant to be annealed. |
| `KILOBOT_BELIEF_CONF_BONUS_ITERS` | `belief_conf_bonus_iters` | 0 | Anneal horizon for the bonus: it decays linearly to zero over this many iterations, so the final objective is the unmodified task reward. 0 keeps the bonus constant. Annealing is applied by both the single-process and multi-process run loops. |
| `KILOBOT_SEED_FIND_BONUS` | `seed_find_bonus` | 0.01 | Split-observation actor only. One-time reward paid the tick after a decision was triggered by a landmark seed (not a wall seed), encouraging robots to seek them out specifically. Deliberately much smaller than `r_on * dt_fixed` (~0.05, the on-shape reward), so it cannot compete with actually finishing the task. Unlike the confidence bonus, this is not annealed -- it is meant to be a permanent, small part of the reward structure. |
| `KILOBOT_WALL_FIND_PENALTY` | `wall_find_penalty` | 0.01 | Split-observation actor only. The mirror of `seed_find_bonus`: a one-time penalty the tick after a decision was triggered by a wall seed. Wall seeds exist so a robot is never permanently lost, not as a destination in themselves. |
| `KILOBOT_HEARTBEAT_TICKS` | `heartbeat_ticks` | 0 | GRU and split-observation actors only (launch refuses it with `deepset`). A robot that has gone this many decision ticks without an event still receives a decision, carrying an all-zero event, so an isolated robot can re-steer instead of coasting ballistically forever. Heartbeat phases are staggered per robot. 0 disables it (the historical semantics). The Unity binary reads this variable itself at startup; **python and the binary must be launched with the same value**, because the python side must command event-less deciders (a zero action would stop the robot). The replica reads `cfg.heartbeat_ticks` and mirrors the same behavior. Requires a build from the current sources; pre-phase-3 builds never emit heartbeat decisions. |

The split-observation actor also reports a per-iteration `rollout/split_seed_fraction`: the fraction of its recorded decisions where the sampled event was a seed sighting rather than a neighbor message. The seed half of `Tc` is this actor's only source of directional target information, so a low fraction here means it rarely gets a training example carrying direction at all, independent of anything else in the setup. It appears in `KILOBOT_SUMMARY`'s per-iteration history under the key `split_seed_fraction` when the actor is `gru_split_observation`, and is absent otherwise. With `KILOBOT_HEARTBEAT_TICKS > 0`, heartbeat (event-less) decisions are counted separately and excluded from that fraction; `rollout/split_heartbeat_fraction` reports the share of all split decisions that were heartbeats.

## Diagnostic modes and reward variants

These change what a run does, for debugging rather than training. Most are described in context in `docs/tuning.md`.

| Variable | Default | Notes |
|---|---|---|
| `KILOBOT_MODE` | `rl` | `rl` trains normally. `bc`, `watch_oracle`, `probe`, `reward_probe`, `audit`, and `control` run diagnostics instead: `bc` clones the actor to the oracle controller by supervised regression rather than PPO; `watch_oracle` drives one arena via the oracle with no training at all, for `watch_oracle.sh`'s open-ended visual observation; `audit` checks the action path (executed action versus stored action, ratio); `control` forces fixed motor commands to measure control authority. `audit` and behavior cloning support all three actor types. All diagnostic modes run single-process regardless of `KILOBOT_NUM_WORKERS` (only `rl` uses the multi-worker path); see `docs/tuning.md`'s bug log if this ever routes into a full training run instead of a diagnostic. |
| `KILOBOT_MOTOR_OVERRIDE` | `none` | Drives the environment from a scripted controller instead of the actor's policy. `simple_oracle` is the supported one -- `simple_oracle.py`'s five-state machine, and the behaviour-cloning teacher. `oracle` selects `actor_io.scripted_motors`, a separate belief-steered controller. `forward` and `fixed` (with `KILOBOT_FORCE_MOTOR`) are diagnostic. |
| `KILOBOT_ORACLE_KNOWN_START_HEADING` | `0` (off); `1` in `watch_oracle.sh` specifically | Every robot spawns at the same, known heading instead of the usual random one, and the belief filter is told this directly rather than starting genuinely uncertain about heading. Confirmed directly to reproduce `true_heading`'s own exactness (removed as privileged, phase 70) through a legitimate, non-privileged mechanism instead -- a physically-enforceable setup convention plus already-exact `dtheta` tracking, not a live ground-truth readout -- as long as the real spawn heading genuinely matches (`Arena.spawn` on the replica side; `SwarmManager.cs`'s own `knownStartHeading` field, unverified from this environment, on the real-Unity side). Off by default in `Config` itself: validated in isolation and substantially at the real-pipeline level, but not yet at real scale the way phases 75-76 showed is necessary before trusting a belief-filter change generally. Added `docs/tuning.md` phase 77. |
| `KILOBOT_ORACLE_ORBIT_AXIS_TRUST_THRESHOLD` | `0.3` | While orbiting a corner, discounts whichever axis (`conf_x`/`conf_y`, already computed from the particle cloud) is still below this per-axis confidence when computing the orbit direction, rather than trusting an axis with essentially no evidence behind it as much as one that's genuinely resolved. Confirmed directly this was needed: a robot arriving at a corner straight from wall-following had one axis resolved and one untouched, and trusting both equally centered its true orbit 15.9 raw units from the actual corner. Added `docs/tuning.md` phase 46.
| `KILOBOT_ORACLE_SEND_VISUAL_STATE` | `false` | When true, sends a per-robot visual-state code over the existing `CriticChannel` side channel each tick, purely for a human watching real Unity to tell each kilobot's current oracle state at a glance -- never touches observations, actions, or reward. `watch_oracle.sh` turns this on by default; off everywhere else, including real training, where nothing listens for it and the cost is a no-op. States map to `simple_oracle.py`'s machine: 0=go_north (ivory), 1=turning (amber), 2=wall_following (gold), 3=navigating (deep red), 4=arrived (green). Seeds stay in a cool blue/teal family so infrastructure is never mistaken for a robot.
| `KILOBOT_BC_OUT` | unset | With `KILOBOT_MODE=bc`, where to save the cloned actor (`export_actor` format -- see `KILOBOT_INIT_ACTOR` above for warm-starting `rl` from it). |
| `KILOBOT_BC_EPOCHS` | `4` | With `KILOBOT_MODE=bc`, gradient epochs per collected rollout. |
| `KILOBOT_BC_CHECKPOINT_EVERY` | `1` | With `KILOBOT_MODE=bc`, saves to `KILOBOT_BC_OUT` every N iterations during the run, not just once at the end -- the same path each time, so `KILOBOT_INIT_ACTOR` pointed at that same path always picks up the most recent progress after an interruption. Added phase 36, `docs/tuning.md`, after an unattended run's interruption lost several hours of unrecoverable progress under the old, save-once-at-the-end behavior. Defaults to every iteration -- verified empirically that the save itself costs about 1.3ms against a multi-minute real iteration, so there's no real cost to accept any additional risk for. |
| `KILOBOT_REWARD_MODE` | `normal` | `normal` is the shape-coverage reward. `speed` replaces it entirely with one proportional to raw displacement, isolating whether reinforcement learning can move the motors at all with no navigation or credit assignment involved. `steer` replaces it with reward for directed motion toward the target (signed displacement along the direction to the nearest stroke pixel), isolating whether the actor can turn the wheels as a function of direction. `steer_blend` adds that same steering term on top of the normal shape reward (and any `reward_shaping`) rather than replacing it. |
| `KILOBOT_SPEED_WEIGHT` | `0.0` | Weight on the speed reward when `KILOBOT_REWARD_MODE=speed`. |
| `KILOBOT_STEER_WEIGHT` | 0.0 | Weight on the steering reward when `KILOBOT_REWARD_MODE` is `steer` or `steer_blend`. |
| `KILOBOT_REWARD_SHAPING` | 0.0 | Potential-based shaping weight, `k*(prev_dist - gamma*dist)`, added to the normal shape reward every tick regardless of `KILOBOT_REWARD_MODE`. Off by default, and the only source of reward gradient for a robot farther than `l_scale` from the target -- `off_penalty` (the base shape reward's off-shape term) saturates at a fixed value beyond that distance and stays exactly flat no matter how much farther the robot gets. A real-Unity run with this left off showed episode reward climbing while `success_rate` stayed at exactly 0.0 and coverage stayed flat for 200 iterations, traced to this exact gap, not a bug in the reward math (`docs/tuning.md` phase 8). `5.0` is the standing recommendation for any run meant to actually train, not just smoke-test the pipeline. |
| `KILOBOT_ORACLE_DEBUG_WALL_LOG` | `0` | The project-wide verbose-oracle toggle. Prints the raw `wallObs` slice as `split_obs` returns it, every `simple_oracle.py` state transition, and the per-robot `SIMPLE_ORACLE_SPAWN_CHECK` against Unity's real heading. |
| `KILOBOT_ORACLE_PERFECT_HEADING` | `0` | Diagnostic ablation: overwrite every robot's belief HEADING with the true value each decision, leaving position untouched, to isolate heading-tracking accuracy from localization. Leaks privileged information; never combine with training. |
| `KILOBOT_CRITIC_BLIND` | `none` | Blind the critic to a group of privileged node columns, to measure what it actually uses. |
| `KILOBOT_EVAL_LOG` | `0` | With `KILOBOT_EVAL=1`, print per-decision detail of what the actor computes and sends. |
| `KILOBOT_ACTOR_TYPE` | from `config.py` | Alias for `KILOBOT_ACTOR`. |
| `KILOBOT_FORCE_MOTOR` | unset | With `KILOBOT_MODE=control`, the fixed `"L,R"` motor command to apply, for example `1.0,1.0`. |
| `KILOBOT_LOG_STD_MIN` / `KILOBOT_LOG_STD_MAX` | from `config.py` | Clamp range for the policy log-std, read at module load. |

## Behaviour cloning

`KILOBOT_MODE=bc` reads these; `run_bc_monitored.py` exposes the same settings as CLI flags (see its own section below).

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_BC_TEACHER` | `oracle` | Which controller supplies the targets. `simple_oracle` is the supported teacher and what the BC drivers pass; `oracle` is `actor_io.scripted_motors`. |
| `KILOBOT_BC_REPLAY_CAPACITY` | `0` (off) | Per-oracle-state reservoir capacity. A rollout window sits inside one phase of a very long episode, so fitting it alone teaches the current phase and unlearns the rest. Capacity is per state, not total. |
| `KILOBOT_BC_REPLAY_BALANCED` | `1` | Give every non-empty state an equal share of each minibatch. Sampling proportional to how much data each state has leaves `turning` (0.4% of an episode) essentially never drawn. |
| `KILOBOT_BC_REPLAY_MAX_AGE` | `0` (unbounded) | Drop samples older than this many iterations; their stored `prev_hidden` came from an older actor. |
| `KILOBOT_BC_REPLAY_EVICT` | `random` | `random` keeps a uniform sample of everything ever collected for a state, `fifo` the most recent. |
| `KILOBOT_BC_REPLAY_MIN_SAMPLES` | `512` | A state reaches its full equal share only once it holds this many samples, ramping in below it -- otherwise a state appearing for the first time takes its full share as a few hundred duplicates. |
| `KILOBOT_BC_REPLAY_PERSIST` | `0` | Save the reservoir beside the checkpoint and reload on resume. It is training state, not a cache: rare states take a whole run to accumulate. |
| `KILOBOT_BC_REPLAY_SAVE_INTERVAL` | `20` | How often to write it; a full reservoir is roughly 420 bytes per sample. |
| `KILOBOT_BC_ACTOR_EVAL_INTERVAL` | `1` | `bc_train` runs two collects per iteration -- one for training data, one purely to report actor coverage. Raising this trades readout frequency for iteration speed. |
| `KILOBOT_BC_MOTOR_SKIP_ARRIVED` | `0` | Drop `arrived` rows from the motor loss. `arrived` is 86.3% of an episode's data and its target `[0, 0]` sits on `squash_action`'s tanh asymptote, so fitting it drives pre-activations toward -inf. **Requires** `KILOBOT_USE_ARRIVED_HEAD`. |
| `KILOBOT_BC_ARRIVED_NATURAL_PRIOR` | `0` | Reweight the arrived head's BCE back to the reservoir's real class prior; a balanced minibatch is deliberately not a sample of the true distribution. |
| `KILOBOT_VAL_TAPE` | unset | Path to a recorded validation tape. Replayed every `KILOBOT_VAL_TAPE_INTERVAL` iterations to report held-out per-state imitation error -- what BC actually optimises -- with no simulation cost. |
| `KILOBOT_VAL_TAPE_INTERVAL` | `5` | How often to replay it. |

## Arrived head and turn anchor

Both change the actor's architecture, so a checkpoint trained with one will not load without it.

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_USE_ARRIVED_HEAD` | `0` | Give the actor a dedicated "I have arrived" head instead of learning to hold a zero motor command, which drives the GRU's recurrent state into a drift over long arrived stretches. |
| `KILOBOT_ARRIVED_CONFIDENCE_THRESHOLD` | `0.95` | High-confidence-only: a false positive strands a robot permanently, a false negative only wastes compute. |
| `KILOBOT_ARRIVED_RELEASE_THRESHOLD` | `0.0` (off) | Confidence at or below which a switched-off robot switches back on. Strictly lower than the threshold above, so this is hysteresis rather than chatter around one value. Worth pairing with `KILOBOT_BC_MOTOR_SKIP_ARRIVED`, where the gate is the only thing that stops a robot. |
| `KILOBOT_USE_TURN_ANCHOR` | `0` | Append sin/cos of the rotation since a wall-triggered heading anchor to `prop`, so the network reads its progress through a turn directly instead of reconstructing it from hidden state. |
| `KILOBOT_TURN_ANCHOR_LATCH` | `1` | Latch the anchor until rotation reaches the oracle's turn target, instead of re-arming on the tick refractory alone. A no-op on the training distribution; without it a deployed actor whose turn overruns can re-anchor mid-turn and spin indefinitely. |
| `KILOBOT_ACTOR_RECURRENT` | `1` | Ablation: `0` swaps the GRUCell for a parameter-matched feedforward stand-in, isolating recurrence from capacity. |

`arrived_loss_weight`, `cold_start_injection_prob` and `turning_duplicate_factor` are `Config` fields with no environment-variable override; reach them through `run_bc_monitored.py`'s `--arrived-loss-weight`, `--cold-start-injection-prob` and `--turning-duplicate-factor`.

## The oracle-form motor head

`Config` fields with no environment-variable override, reached through
`bc_offline.py`'s flags. All three change the **deployed** forward pass without
changing any tensor's shape, so a checkpoint trained with them loads cleanly into
an actor built without them and silently computes a different function -- which is
why `bc_offline.save_actor` records them in `meta` and every loader reads them
back. See [`tuning.md`](tuning.md) phase 160.

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `use_oracle_head` | `--oracle-head` | `False` | Build the motor command as a soft mixture over the oracle's own five commands, each in closed form from quantities the actor already observes, weighted by the state head's posterior. Costs **no parameters and no inputs** -- it reuses `head_state`, `head_wall` and `head_motor`. Requires `--state-head-weight` and `--wall-head-weight` above 0, since the mixture weights are those two heads. |
| `oracle_residual` | `--oracle-residual` | `0.05` | Bound, in motor units, on the learned COMMON-mode correction to that mixture. Nonzero so the closed form is a prior rather than a cage. |
| `oracle_residual_turn` | `--oracle-residual-turn` | `0.0` | The same residual's DIFFERENTIAL half. Zero, and measured to belong there: swept on a trained network the median `wall_following` steering error is 0.00850 at 0.003 and 0.00058 at 0, with the motor MSE unchanged. |

Two more `bc_offline.py` flags exist for the same problem and are diagnostics rather than
recommendations:

| Flag | Default | Meaning |
|---|---|---|
| `--steer-weight` | `0.0` | Extra loss on the wheel pair's DIFFERENTIAL, on top of the plain MSE. The unstructured way to make the loss see the steering channel; it helps (median error 0.0403 -> 0.0112) and does not come close to computing the command. |
| `--select` | `balanced` | Which held-out number picks `actor_best.pt`. `balanced` is the per-state mean motor MSE every run before phase 160 used; `steer` and `turn_bias_wall_following` are the steering channel's rms and its persistent per-robot component. |

## The closed-form arrival gate

Deployment-side, with no training to redo. The learned arrived head is accurate
on the tape's distribution (0.99 recall at the 0.95 threshold) but under-fires on
the deployment localisation shift, and a robot the gate never turns off is an
unfinished robot. These `Config` fields swap that head's decision for the
oracle's own arrival rule -- `closed_form_arrived` in `kilobot_gnn.py`: filter
distance to the robot's own assigned target below a radius, confidence past the
localisation floor, target actually assigned -- computed in real time from the
actor's observation. No tensor shape changes, so any checkpoint loads under
them. Reached through `tools/eval_closed_loop.py`, never through the architecture
flags, because they select the *deployed* decision only. See the report in
`../results/hybrid_cloning/` for the closed-loop comparison.

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `use_closed_form_arrived` | `--closed-form-arrived` | `False` | Gate the actor's stop on the oracle's own rule computed from the actor's filter, instead of the learned arrived head. The rule's slots in the observation's property vector: `PROP_CONF_POS` (12), `PROP_SIN_T`/`PROP_COS_T` (19, 20), `PROP_DIST_T` (21). Terminal, like the oracle's arrived state. |
| `closed_form_arrival_dist` | `--closed-form-dist` | `0.0` (uses `cfg.tau_v`, `0.05`) | Arrival radius in normalised units. The actor's own filter under-reports closeness (empirically `d_target` ~1.5&times; the true distance at arrival), so `tau_v` stops almost nobody and the swarm orbits; `0.08` is the value measured to park robots where the filter certifies arrival. |
| `closed_form_hybrid` | `--closed-form-hybrid` | `False` | With `use_closed_form_arrived`, run the closed-form rule in OR with the learned head instead of replacing it: the head under-fires on low-confidence near robots, the closed form certifies only tight arrivals, and the two miss different robots, so the OR is a strict superset of either branch alone. The latch is terminal either way. |

## Run summary

| Variable | Default | Meaning |
|---|---|---|
| `KILOBOT_SUMMARY` | unset | If set to a path, write a small JSON record of the run (per-iteration coverage, entropy, explained variance, and derived scalars), rewritten every iteration with an atomic replace. The hyperparameter sweep uses this; you can also point it at a file to keep a compact record of a manual run. |

For the full sweep workflow, see `sweep.md`.

## Run recipes

These assume you are in `python/` and have the dependencies installed. Adjust `KILOBOT_FORMATIONS` to your path.

Single-process smoke test on CPU:

```
env KILOBOT_NUM_WORKERS=1 KILOBOT_NUM_ARENAS=9 KILOBOT_SMOKE=1 \
    KILOBOT_DEVICE=cpu KILOBOT_FORMATIONS=../data/formations python launch.py
```

Multi-process smoke test on CPU:

```
env KILOBOT_NUM_WORKERS=2 KILOBOT_NUM_ARENAS=9 KILOBOT_SMOKE=1 \
    KILOBOT_DEVICE=cpu KILOBOT_FORMATIONS=../data/formations python launch.py
```

Smoke test or training run with the Unity window visible (single worker, single arena; see the README for the full explanation of why):

```
env KILOBOT_SMOKE=1 KILOBOT_DEVICE=cpu KILOBOT_NUM_WORKERS=1 KILOBOT_NUM_ARENAS=1 \
    KILOBOT_NO_GRAPHICS=false KILOBOT_TIME_SCALE=1 KILOBOT_SHOW_RADIUS=true \
    KILOBOT_FORMATIONS=../data/formations python launch.py
```

Full training run with four workers on a GPU machine:

```
env KILOBOT_NUM_WORKERS=4 KILOBOT_NUM_ARENAS=9 KILOBOT_SMOKE=0 \
    KILOBOT_ITERATIONS=200 KILOBOT_DEVICE=cuda \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_SEED_LAYOUT=cluster KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_REWARD_SHAPING=5.0 KILOBOT_ENTROPY_COEF=0.001 \
    KILOBOT_FORMATIONS=../data/formations python launch.py
```

The actor/layout/heartbeat/shaping/entropy variables above are not this tool's defaults (the bare defaults fall back to the plain `deepset` actor, no wall-seed/belief-filter work, and a reward with no navigation gradient once shaping is off) -- they are the current, validated combination the wall-seed and belief-filter work targets and phases 8-9 (`docs/tuning.md`) found necessary for a real run to make progress at all. Drop them to run the simpler baseline actor instead.

Resume from the most recent checkpoint:

```
env KILOBOT_RESUME=../results/tb/run_YYYYMMDD_HHMMSS/ckpt.pt \
    KILOBOT_NUM_WORKERS=4 KILOBOT_SMOKE=0 KILOBOT_ITERATIONS=400 \
    KILOBOT_ACTOR=gru_split_observation KILOBOT_SEED_LAYOUT=cluster KILOBOT_HEARTBEAT_TICKS=48 \
    KILOBOT_REWARD_SHAPING=5.0 KILOBOT_ENTROPY_COEF=0.001 \
    KILOBOT_FORMATIONS=../data/formations python launch.py
```

`KILOBOT_RESUME` only restores the actor, critic, optimizer state, and iteration count from the checkpoint -- it does not restore which environment variables produced that checkpoint's `Config`. Every other `KILOBOT_*` variable, including the actor/layout/heartbeat/shaping/entropy combination above, is rebuilt fresh from whatever the resuming invocation sets, same as any other run. Omitting one here silently reverts that field to its bare default rather than continuing with whatever the original run used -- for `KILOBOT_ACTOR` in particular this is not just a worse setting but a shape mismatch: `load_checkpoint` calls PyTorch's `load_state_dict` with its default strict matching, so resuming a `gru_split_observation` checkpoint without also setting `KILOBOT_ACTOR=gru_split_observation` raises a `RuntimeError` listing mismatched keys between the checkpoint and a freshly-built `deepset` network, rather than silently loading anything wrong.

Evaluate an exported actor:

```
env KILOBOT_EVAL=1 \
    KILOBOT_EVAL_WEIGHTS=../results/tb/run_YYYYMMDD_HHMMSS/actor_final.pt \
    KILOBOT_EVAL_ITERS=3 KILOBOT_FORMATIONS=../data/formations python launch.py
```

The Docker equivalents use `KILOBOT_MODE` and the in-image paths. See the README quickstart.

## `run_bc_monitored.py`: a separate entry point, CLI flags not environment variables

**Added 2026-07-27.** Everything above this point in this file covers `launch.py`, configured entirely through `KILOBOT_*` environment variables. `run_bc_monitored.py` is a separate, later entry point for behavior cloning against `simple_oracle.py` (phase 90 onward, `docs/tuning.md`) and is configured through ordinary CLI flags instead -- the two are not interchangeable, and this section covers only the flags most directly relevant to that BC path, not an exhaustive listing.

| Flag | Default | Meaning |
|---|---|---|
| `--cold-start-injection-prob` | `0.0` | Fraction of decisions where a robot's own hidden state is reset, to force more training exposure to the otherwise-rare "just came from a cold hidden state" situation. |
| `--turning-duplicate-factor` | `0` | Each real, naturally-occurring turning-state BC example gets included this many additional times in the same update, weighting it more heavily in the averaged loss without ever synthesizing an observation or target. Replaces an earlier `--turning-injection-prob` flag, removed entirely (phase 144, `docs/tuning.md`) after direct measurement found it firing regardless of a robot's own current state, repeatedly interrupting genuine progress and pairing its own fixed target with an inauthentic observation the large majority of the time. A calibration sweep (phase 145) found real, empirically-confirmed diminishing returns past small values; 20 was the reasoned starting point that sweep settled on, not a universal default -- see that phase's own entry for the full, numeric account. |
| `--use-arrived-head` | off | Adds a second, trained output head that flips a high-confidence "arrived" flag, forcing the motor to zero and freezing the GRU's own hidden state from that point on (phase 142). Expands nothing about input width; a checkpoint trained without this flag still loads with it off. |
| `--arrived-confidence-threshold` | `0.95` | Only used when `--use-arrived-head` is set. |
| `--use-turn-anchor` | off | Adds two values to the actor's own proprioception: sin/cos of the heading change since the robot's own real wall reading last went from zero to nonzero (phase 146). Expands the actor's own first-layer input width by 2 when set -- a checkpoint trained with this flag will not load into an actor built without it, or vice versa. |
| `--heartbeat` | `48` | Ticks a robot can go without an event before a forced, empty-event decision, so an isolated robot can still re-steer. |

The actor's own architecture size (`split_gru_hidden`, `split_upscale_hidden`, `split_head_hidden`) is not exposed as a CLI flag on this entry point -- it is set directly via `config.py`'s own defaults (`split_gru_hidden` is 59 as of phase 147, against a stated 24KB/24576-byte hard parameter budget; see `docs/architecture.md`'s "Key shapes and constants" for the current, full account) or by editing `Config` directly for an experimental run.

For the full, dated record of every flag above -- including the real bugs found while building several of them, and the direct, empirical measurements behind the specific default values chosen -- see `docs/tuning.md`'s phase 142-147 entries.

## Docker build arguments

The image is built for CUDA 12.1 by default. You can retarget it at build time without editing the Dockerfile:

| Build arg | Default |
|---|---|
| `CUDA_IMAGE` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| `TORCH_VERSION` | `2.3.1` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu121` |
| `MLAGENTS_REF` | `release_23` |

For example:

```
docker build -t kilobot-gnn \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  --build-arg TORCH_VERSION=2.5.1 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
```
