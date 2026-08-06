"""Every tunable in one dataclass.

Comments here say what a field does and what it must stay consistent with.
For why a value is what it is -- the measurements, the failed alternatives, the
bugs that motivated a flag -- see docs/code-history.md, which is indexed by
field name.
"""

import math
from dataclasses import dataclass

# Unity's own motion constants, mirrored here so the kinematics below can be
# derived rather than measured. These MUST match, in order:
#   Assets/Scripts/KilobotMovement.cs   moveSpeed, turnSpeed
#   Assets/Scripts/SceneBootstrap.cs    framesPerStep
#   ProjectSettings/TimeManager.asset   Fixed Timestep
# tests/test_kilobot.py asserts the derived values; nothing else keeps these in
# sync, because the Unity side does not read them.
_UNITY_MOVE_SPEED = 1.0
_UNITY_TURN_SPEED_DEG = 45.0
_UNITY_FRAMES_PER_STEP = 4
_UNITY_FIXED_DT = 0.02


@dataclass
class Config:
    # ─── episode / loop ──────────────────────────────────────────────────────
    num_arenas: int = 9
    rollout_steps: int = 4096
    max_episode_steps: int = 2048
    success_threshold: float = 0.85

    # ─── reward ──────────────────────────────────────────────────────────────
    r_on: float = 1.0
    k_pos: float = 1.0
    tau_v: float = 0.05
    l_scale: float = 1.0
    k_sep: float = 1.0
    tau_sep: float = 0.08
    # Must equal Unity's Fixed Timestep (ProjectSettings/TimeManager.asset).
    dt_fixed: float = 0.02
    reward_shaping: float = 0.0   # potential-based shaping weight k*(prev_dist - gamma*dist)
    # Peak bonus and decay range for the packing reward: an on-shape robot is
    # rewarded for being densely surrounded, down to the separation floor, so
    # isolated robots are pulled into the filled cluster.
    r_pack: float = 1.0
    pack_range: float = 0.20

    # ─── actor ───────────────────────────────────────────────────────────────
    actor_type: str = "deepset"   # "deepset", "gru", or "gru_split_observation"
    gru_hidden: int = 128
    # Kinematics, DERIVED from Unity's own constants rather than measured, so
    # they cannot drift out of sync again. KilobotMovement.cs applies
    #     forwardSpeed = (L+R)/2 * moveSpeed      moveSpeed  = 1
    #     turnRate     = (L-R)   * turnSpeed      turnSpeed  = 45 deg/s
    # once per FixedUpdate, and StepDriver runs UNITY_FRAMES_PER_STEP of those
    # per environment step. split_tick_motion counts `steps` in environment
    # steps against dt_fixed, which pins both exactly:
    #     prop_max_speed = moveSpeed * UNITY_FIXED_DT * FRAMES / dt_fixed = 4.0
    #     prop_max_speed / prop_wheelbase = radians(turnSpeed) * FRAMES = pi
    # Change any of the four Unity values and these follow automatically;
    # tools/calibrate_kinematics.py re-measures both against a live player and
    # should reproduce them to measurement noise.
    prop_max_speed: float = _UNITY_MOVE_SPEED * _UNITY_FIXED_DT * _UNITY_FRAMES_PER_STEP / 0.02
    prop_wheelbase: float = prop_max_speed / (math.radians(_UNITY_TURN_SPEED_DEG)
                                              * _UNITY_FIXED_DT * _UNITY_FRAMES_PER_STEP / 0.02)
    # Scale dead-reckoned quantities toward O(1). prop_scale/prop_time_scale are
    # on a single-tick basis; prop_cum_scale assumes max_episode_steps=2048 and
    # needs re-deriving if that changes materially.
    prop_scale: float = 20.0
    prop_time_scale: float = 20.0
    prop_cum_scale: float = 0.02
    reward_mode: str = "normal"   # "normal", "speed", "steer", or "steer_blend"
    speed_weight: float = 0.0     # weight for the pure-speed isolation reward
    steer_weight: float = 0.0     # weight for the steering reward (displacement toward target)

    # gru_split_observation only. Layer widths are jointly constrained by a 24KB
    # int8 budget; the current set measures 24129 bytes. Changing any of them
    # breaks checkpoint compatibility.
    split_upscale_hidden: int = 40   # Tc/odom upscale MLP
    split_gru_hidden: int = 59       # GRU hidden size
    split_head_hidden: int = 40      # policy head
    # Scales on the interval distance and elapsed-time channels, each targeting
    # the p90 of the measured anchor-tracker value toward O(1). Re-derive
    # split_prop_scale if prop_max_speed or the population/heartbeat regime
    # changes materially, split_prop_time_scale if max_episode_steps does.
    split_prop_scale: float = 0.04
    split_prop_time_scale: float = 0.02
    split_seed_weight_boost: float = 1.0  # multiplies the seed's pool weight before sampling
    # False swaps the GRUCell for a parameter-matched feedforward stand-in
    # (kilobot_gnn.MemorylessAggregator, 17859 params against the GRU's 17877),
    # isolating recurrence from capacity. Ablation switch only.
    actor_recurrent: bool = True

    # ─── environment ─────────────────────────────────────────────────────────
    # Landmark seed placement. "corners" is the only supported layout;
    # "cluster" exists in belief.SEED_LAYOUTS for backward compatibility only
    # and has not been run in a long time. Wall-lining seeds spawn regardless.
    seed_layout: str = "corners"
    # A robot that has gone this many decision ticks without an event still gets
    # a decision, with an all-zero event, so isolated robots can re-steer
    # instead of coasting ballistically. 0 disables. gru / gru_split_observation
    # only, and must match the KILOBOT_HEARTBEAT_TICKS the player was launched
    # with.
    heartbeat_ticks: int = 0

    # ─── message channel ─────────────────────────────────────────────────────
    # Floats 0-2 of the broadcast message carry the belief beacon (x, y, conf)
    # so peers can range off localized neighbours. Experimental: every variant
    # tried still double-counts correlated information, so seed-only filtering
    # is the supported mode.
    belief_comms: bool = False
    # Floats 3-5 carry a committed robot's claimed navigation target (x, y,
    # valid) -- not its position, and not belief_comms's slots, so the two never
    # collide. On by default: it is the non-privileged replacement for occupancy
    # checks that used to read every robot's true position directly, and
    # collision avoidance during target selection is core behaviour.
    oracle_claim_broadcast: bool = True

    # ─── reward shaping for gru_split_observation ────────────────────────────
    # Per-step bonus of belief_conf_bonus * conf_pos, paid for holding a
    # collapsed pose belief; scaffolding to break the policy<->localization
    # bootstrap, meant to be annealed to zero over belief_conf_bonus_iters
    # (0 keeps it constant).
    belief_conf_bonus: float = 0.0
    belief_conf_bonus_iters: int = 0
    # One-time reward the tick after a decision triggered by a landmark seed,
    # and its mirror for a wall seed. Both are deliberately much smaller than
    # r_on*dt_fixed so they cannot compete with finishing the task; wall seeds
    # exist so a robot is never permanently lost, not as a destination.
    seed_find_bonus: float = 0.01
    wall_find_penalty: float = 0.01
    # Append each robot's 9 belief_read values to the CRITIC's node inputs, so
    # the value baseline can represent per-robot localization state. Required
    # for PPO to learn from belief_conf_bonus at all. The actor is unchanged.
    critic_belief_features: bool = False

    # ─── returns / ppo ───────────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    ppo_epochs: int = 4
    minibatch: int = 1024
    max_grad_norm: float = 0.5
    log_std_init: float = -0.5
    # Graph snapshots per critic forward pass. Bounds GPU memory; lower it if
    # you hit CUDA out of memory.
    critic_chunk_steps: int = 64

    # ─── run plumbing ────────────────────────────────────────────────────────
    # ParallelTrainer only: seconds to wait for every worker to report a rollout
    # before treating the run as stalled and aborting (checkpoint retained).
    collect_max_wait: float = 1200.0
    # Print "arena K: formation N (name.png)" on every arena reset. Off by
    # default because a many-arena run resets constantly; the watch scripts turn
    # it on via KILOBOT_LOG_FORMATIONS.
    log_formations: bool = False
    device: str = "cpu"
    seed: int = 0

    # ─── diagnostics and probes ──────────────────────────────────────────────
    actor_priv_mode: str = "none"
    # "none", "simple_oracle", "forward", or a probe mode. "simple_oracle" is
    # simple_oracle.py's five-state machine and the behaviour-cloning teacher.
    motor_override: str = "none"
    direct_motor: bool = False
    force_motor: tuple = ()       # (left, right) constant motor command for the control probe

    # ─── oracle_* prefix ─────────────────────────────────────────────────────
    # Historical prefix only. The lock-based controller these were named for is
    # gone; every field below is read by simple_oracle.py, observation.py,
    # actor_io.py or belief.py. Do not add more with this prefix.

    # While orbiting a corner, discount whichever axis is still below this
    # per-axis confidence (conf_x/conf_y from belief_read). Deliberately below
    # LOCALIZED_CONF_THRESHOLD: this gates "trust this axis for steering", a
    # lower bar than "confident enough to commit".
    oracle_orbit_axis_trust_threshold: float = 0.3
    # Send a per-robot visual-state code over the side channel, so a human
    # watching Unity can see which state each kilobot is in. Never affects the
    # observation or the reward.
    oracle_send_visual_state: bool = False
    # Print the raw wallObs slice exactly as split_obs returns it -- before
    # sample_split_event or anything downstream -- alongside the robot's true
    # position, whenever wallObs is nonzero.
    oracle_debug_wall_log: bool = False
    # Diagnostic ablation: overwrite every robot's belief HEADING with the true
    # value each decision, leaving position untouched. Leaks privileged
    # information; never combine with training.
    oracle_perfect_heading: bool = False
    # Odometry-based heading triangulation: accumulate range/wall readings since
    # a confident reference position and solve for heading directly. Correct in
    # isolation but not reliable in full-system testing at any configuration
    # tried; belief.belief_triangulate has the mechanism.
    oracle_heading_triangulation: bool = False
    # Feed the nearest wall seed's own known position into belief_update's
    # along-wall likelihood term (belief._wall_along_log_w), rather than only
    # aggregate wall-band strength. Off means wall contact constrains one axis
    # only, so a robot fully localizes at a corner rather than on first wall
    # contact -- the two-tier corner/wall distinction the design intends, at the
    # cost of a longer path to localization. Purely additive when wall_seed_xy
    # is unavailable, e.g. against an older player.
    oracle_wall_seed_position: bool = False
    # An arrived, stopped robot broadcasts its claimed position, and a nearby
    # lost robot's particle cloud is ring-injected around it. The sending-side
    # gate (arrived, not merely committed) is load-bearing: a wrong claim that
    # gets through is worse than no claim. Off by default -- validated as safer
    # than the uncapped original, not yet validated as an improvement.
    oracle_arrived_claim_injection: bool = False
    # Minimum ticks between successive injections for one robot. 0 reproduces
    # the uncapped behaviour, which is confirmed harmful at scale.
    oracle_arrived_claim_cooldown_ticks: int = 400
    # Every robot spawns at a known heading and the particle filter is told so
    # (belief_init's known_start_heading, belief_predict's heading_noise_scale)
    # rather than starting uniformly uncertain about heading. Legitimate only as
    # long as the actual spawn heading physically matches belief.KNOWN_START_HEADING
    # everywhere this runs -- it is a setup convention, not a per-tick readout.
    oracle_known_start_heading: bool = False

    # ─── behaviour cloning ───────────────────────────────────────────────────
    # Force a random fraction of decisions to see h_prev=0, during oracle-driven
    # BC collection only. A robot's first decision is ~0.27% of BC's data, so
    # the actor otherwise never learns a correct cold start. Everything else
    # about the robot stays real and continuous; only the cached hidden state is
    # dropped. Keep it low -- the GRU still needs long coherent sequences.
    cold_start_injection_prob: float = 0.0
    # Include each naturally-occurring `turning` example this many extra times
    # in the same BC update, to offset how rare the state is (0.4% of an
    # episode). Nothing is synthesized. Subsumed by balanced replay sampling;
    # set to 0 when bc_replay_capacity is on rather than stacking the two.
    turning_duplicate_factor: int = 0

    # Per-oracle-state reservoir that persists across iterations, so a fit draws
    # from every state the run has seen rather than the current rollout window
    # alone -- a window sits inside one phase of a very long episode, and the
    # actor otherwise fits that phase and unlearns the rest. Capacity is PER
    # STATE, not total. 0 disables.
    bc_replay_capacity: int = 0
    # Equal share of every minibatch to every non-empty state. Sampling
    # proportional to how much data each state has leaves `turning` worse than
    # no replay at all, since at 0.4% of the data it is essentially never drawn.
    bc_replay_balanced: bool = True
    # Drop samples older than this many iterations. The stored prev_hidden came
    # from an older actor, and a GRU may change what its hidden state means.
    # 0 is unbounded.
    bc_replay_max_age: int = 0
    # "random" keeps a uniform sample of everything ever collected for a state;
    # "fifo" keeps the most recent. random by default, since the point is to
    # retain phases the run has moved past.
    bc_replay_evict: str = "random"
    # A state reaches its full equal share of a minibatch only once it holds
    # this many samples, ramping in linearly below it. Without the ramp a state
    # appearing for the first time takes its full share as a few hundred copies
    # of a handful of samples.
    bc_replay_min_samples: int = 512
    # Persist the reservoir alongside the checkpoint and reload it on resume.
    # It is training state, not a cache: rare states take a whole run to
    # accumulate, so an interrupted run that comes back empty has thrown away
    # what this mechanism exists to build.
    bc_replay_persist: bool = False
    bc_replay_save_interval: int = 20
    # Set at runtime by whichever entry point is driving (run_bc_monitored.py
    # from --out-dir, launch.py from KILOBOT_BC_OUT). Declared here because
    # tests check that every cfg field launch.py touches is a real field.
    bc_replay_path: str = ""
    # bc_train runs two collects per iteration: one oracle-driven, which is the
    # training data, and one actor-driven, which produces none and exists only
    # to report actor_eval_cov and the arrived_agreement census. Both cost the
    # same simulation time. Raising this trades progress-readout frequency for
    # iteration speed; the val tape is what gates checkpoints.
    bc_actor_eval_interval: int = 1
    # A recorded validation tape (val_tape.py): observations plus oracle
    # targets, replayed every val_tape_interval iterations to report held-out
    # per-state imitation error, which is what the BC phase actually optimises.
    # Needs no simulation, so a tape recorded anywhere scores any checkpoint.
    val_tape_path: str = ""
    val_tape_interval: int = 5
    # Drop `arrived` rows from the motor loss -- masked out, not given a
    # substitute target. arrived is 86.3% of an episode's BC data and its target
    # is exactly [0, 0], which sits on squash_action's tanh asymptote, so
    # fitting it drives the motor head's pre-activations toward -inf. With this
    # on the motor head only learns to keep moving and stopping is handled by
    # the arrived head. REQUIRES use_arrived_head; run_bc_monitored.py refuses
    # the combination without it.
    bc_motor_skip_arrived: bool = False
    # Reweight the arrived head's BCE term back to the class prior in the
    # reservoir. A balanced minibatch is deliberately not a sample of the real
    # distribution, and for a binary head the batch composition is the prior it
    # calibrates to. Affects the arrived term only.
    bc_arrived_natural_prior: bool = False

    # ─── arrived head ────────────────────────────────────────────────────────
    # Give the actor a dedicated head for "I have arrived" instead of learning
    # to hold a zero motor command, which drives the GRU's recurrent state into
    # a drift over the long arrived stretches. Gates whether
    # SplitObservationActor constructs the head at all, so an existing
    # checkpoint still loads while this is off.
    use_arrived_head: bool = False
    # High-confidence-only: a false positive here permanently strands a robot,
    # a false negative only wastes a little compute. Not independently tuned.
    arrived_confidence_threshold: float = 0.95
    arrived_loss_weight: float = 1.0   # BCE weight relative to the motor MSE, which stays at 1.0
    # Confidence at or below which a switched-off robot switches back on.
    # Strictly lower than arrived_confidence_threshold, so this is hysteresis
    # and a robot cannot chatter around a single threshold. 0.0 keeps the
    # original behaviour, where switching off was permanent for the episode.
    # Worth pairing with bc_motor_skip_arrived, where the gate is the only thing
    # that stops a robot.
    arrived_release_threshold: float = 0.0

    # ─── turn anchor ─────────────────────────────────────────────────────────
    # A second heading anchor, (re-)established when the actor's wall reading is
    # nonzero and at least actor_io.TURN_ANCHOR_REFRACTORY_TICKS have passed.
    # gather_split_state then appends sin/cos of (heading_now - heading_anchor)
    # to prop, so the network reads its progress through a turn directly instead
    # of having to reconstruct it from an absolute heading held in hidden state.
    # Widens the actor's first layer by TURN_ANCHOR_SIZE (kilobot_gnn.py), so a
    # checkpoint trained with this will not load without it, or vice versa.
    use_turn_anchor: bool = False
    # Latch the anchor until the rotation since it was set reaches the oracle's
    # turn target, instead of re-arming on the tick refractory alone. On by
    # default: it is a no-op on the training distribution and prevents a
    # deployed actor whose turn overruns the refractory from re-anchoring
    # mid-turn and spinning indefinitely.
    turn_anchor_latch: bool = True
