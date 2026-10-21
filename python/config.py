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

    # ─── RL fine-tune reward (docs/tuning.md phases 164-165) ─────────────────
    # Per-robot distance to its OWN assigned target point
    # (observation.ensure_target, python-side geometry -- NOT Unity's baked
    # distance field, which is 90-degree-rotated at the player's default
    # KILOBOT_BAKE_ROTATION_STEPS=1). F = k * (prev_d - gamma * d) per
    # robot, in normalized units. Theory-safe (policy-invariant), so unlike
    # reward_shaping this is not annealed. Robots without a resolved target
    # get zero.
    target_shaping: float = 0.0
    # One-time bonus paid the tick after the arrival gate latches a robot
    # (actor_io._arrived_head_gate's False->True transition, actor-driving
    # modes only), GRADED by the true distance to the robot's own assigned
    # target: bonus * exp(-(d/stop_radius)^2). Full bonus only for a tight
    # arrival; a gate false positive far from the pixel earns ~nothing. The
    # gate itself stays frozen during RL (phase 161's hybrid already beats
    # the oracle's stopping, and an absorbing wrong stop is the measured
    # failure mode).
    stop_bonus: float = 0.0
    # Kernel width of the graded stop bonus, in normalized units (0.05 =
    # 5 raw units, the settle metric's primary bar).
    stop_radius: float = 0.05
    # Shared arena-level potential for sender credit: F = k * (prev_pot -
    # gamma * pot), added EQUALLY to every robot in the arena. This is what
    # makes broadcasting learnable under per-robot GAE (buffer.py groups
    # trajectories per robot): a message that helps a neighbour settle
    # raises the shared potential, and every sender's own return carries
    # that improvement. Without it the sender of a helpful message gets no
    # gradient at all, since GAE never routes one robot's reward into
    # another robot's advantage.
    global_shaping: float = 0.0
    # Width of the SMOOTH kernel behind that shared potential:
    # pot = mean_i exp(-(d_i/sigma)^2) over robots with resolvable targets.
    # A threshold (fraction within a radius) is flat almost everywhere and
    # would only credit a sender at the exact tick a neighbour crosses the
    # bar; the kernel has gradient at every distance, including mid-
    # approach -- which is where a helpful message actually does its work.
    global_potential_sigma: float = 0.10
    # reward_mode "target": replaces the any-pixel shape income entirely.
    # Per-robot accuracy income w_acc * exp(-(d_i/sigma_acc)^2) * dt -- pays
    # for BEING at the robot's own assigned pixel, tightest arrivals most;
    # localization is rewarded instrumentally (a robot that has not
    # figured out where it is cannot find the pixel). Speed is rewarded
    # twice over: the gamma discount makes earlier income worth more, and
    # target_time_cost subtracts w_time * dt per step from every robot with
    # a resolvable target, so lingering anywhere has a carrying cost. The
    # separation term (anti-clumping) is kept; r_on/off_penalty/pack --
    # which reward occupying ANY pixel, the exact failure phase 159
    # identified -- are dropped. success_metric "settled" makes the
    # success/reset condition the settled fraction (same bar as the stop
    # bonus) instead of coverage.
    target_acc_weight: float = 1.0
    target_acc_sigma: float = 0.05
    target_time_cost: float = 0.0
    success_metric: str = "coverage"   # "coverage" or "settled"
    # Per-robot potential-based shaping on the robot's OWN belief error
    # (|particle-filter mean position - true position|, normalized units --
    # privileged, like every reward quantity; the actor never sees it).
    # F = k * (prev_e - gamma * e), positive exactly when the filter's
    # position estimate moves toward the truth. This is NOT the phase-2/9
    # belief_conf bonus: conf pays for a collapsed cloud, converged or not,
    # and was measured to make confidently-wrong profitable; the error pays
    # only for CORRECT localization, which no wrong convergence can fake.
    # Dense from the first landmark fix (the phase the whole settle curve
    # spends its first ~3000 ticks in), where the own-target terms are still
    # flat -- the accuracy kernel is exp(-(d/0.05)^2), zero until the robot
    # is nearly AT its pixel, and target shaping's gradient is weak far out.
    belief_shaping: float = 0.0
    # Kernel width of the belief-error band inside the shared arena potential
    # (see global_shaping): sigma deliberately much wider than the accuracy
    # kernel's 0.05 because the belief error is large until a robot has
    # genuinely localized (spawn e ~ 0.4-0.7, localized e ~ 0.05), and the
    # sender credit must move while that error is still being reduced --
    # which is exactly when a helpful broadcast does its work.
    belief_potential_sigma: float = 0.20
    # Relative weight of the belief-error band against the settled band
    # inside the shared arena potential. pot = mean_i [ kernel(d_i, sigma_g)
    # + belief_potential_weight * kernel(e_i, belief_potential_sigma) ].
    belief_potential_weight: float = 1.0
    # Per-tick penalty on a robot whose BROADCAST conf (message slot 2,
    # read from the transmission node column) exceeds its own honest filter
    # confidence (conf_pos * r over its own particles): -w * relu(claimed -
    # honest) * dt. The honesty constraint on the learned beacon -- see
    # reward.beacon_overclaim_penalty for the measured failure it exists to
    # close (PPO pushed the conf gate to 2.4x honest; the receiver fuses the
    # MOST confident row, so overclaimed beacons with ~50-unit-wrong
    # positions always win the slot). 0 disables.
    beacon_honesty: float = 0.0
    # Zero the split actor's message head after KILOBOT_INIT_ACTOR loads it,
    # so broadcasts start at exactly zero -- the distribution the BC tapes'
    # receivers were trained on (oracle-driven tapes broadcast zeros; the
    # message head is never supervised by BC, so a warm-started head_msg is
    # random noise that would corrupt the clone's navigation input from
    # iteration 0). Pair with msg_log_std for quiet exploration.
    quiet_msg_head: bool = False
    # Overwrite the loaded per-dimension log-std on the MESSAGE dimensions
    # only (motor dims keep the checkpoint's). 0.0 keeps the loaded value.
    # The clamp floor is KILOBOT_LOG_STD_MIN, so a value below it needs that
    # lowered too (e.g. -4.0). Start quiet (near-zero message noise) and let
    # the entropy bonus grow exploration over the run.
    msg_log_std: float = 0.0
    # Weight of ppo's auxiliary supervision of message slots 0-2 toward the
    # honest beacon (the robot's own belief mean x, y and conf_pos*r, read
    # from the same prop columns actor_io's honest-beacon override uses).
    # This is the cold-start ladder for a learned beacon: the receiver-side
    # fusion gate (PEER_CONF_MIN=0.85) needs the squashed conf past 0.85,
    # i.e. a pre-tanh mean above ~1.73 -- unreachable from a quiet start by
    # entropy exploration alone, so the useful-protocol basin has no gradient
    # path from the silent one without a supervised prior. A prior, not a
    # cage (the oracle_residual pattern): where the RL advantage consistently
    # disagrees, the mean can move away from the honest beacon. 0 disables.
    msg_supervision: float = 0.0
    # The message head reads cat([g, prop]) instead of g alone (see
    # kilobot_gnn.SplitObservationActor's own comment for the measurement
    # behind this). The beacon is then a linear readout of the robot's own
    # filter state -- the honest protocol as the head's INITIALIZATION
    # (launch.apply_warm_start_message_policy's beacon readout), refined by
    # msg_supervision and free to be modulated by the RL gradient. Without
    # it, the head must pull absolute position through the motor-trained
    # GRU trunk, which measured at a 0.5-normalized-error plateau head-only
    # and wrecked the warm start when trained through (kl 7/iteration).
    msg_head_prop_input: bool = False
    # Same override for the MOTOR dimensions. Load-bearing for the RL
    # fine-tune: a bc_offline checkpoint's saved log_std is a placeholder
    # (Config.log_std_init = -0.5, std 0.61) that BC never trains -- the MSE
    # fit only shapes the MEAN. Deterministic eval (squash of the mean) never
    # touches it, so the clone scores 0.63 settled<5u on the closed-loop
    # harness, but the very first STOCHASTIC rollout executes 0.5*(tanh(N(
    # mean, 0.61))+1) per wheel -- effectively random wheel speeds -- and the
    # warm start is dead at iteration 0, before any PPO update runs (measured:
    # settled 0.002 from iter 0, conf_pos 0.004, std_m/i 0.05/0.61 in
    # results/rl_v1_console.log). Setting this (e.g. -2.5 -> std 0.082) starts
    # motor exploration quiet instead; the entropy bonus can reopen it if the
    # value of exploring outweighs the income a precise command earns.
    motor_log_std: float = 0.0

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
    reward_mode: str = "normal"   # "normal", "speed", "steer", "steer_blend", or "target"
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
    # Hidden activation at up1 and head1 (kilobot_gnn.SPLIT_ACTIVATIONS).
    # "relu" is what every checkpoint before 2026-08-06 was trained with, and
    # what dies: phase 154 measured 13 of head1's 40 units stuck at exactly
    # zero for every decision, with no gradient path back. Holds no parameters,
    # so it changes no checkpoint's shape -- but a checkpoint trained under one
    # value and run under another is a different function, which is why
    # bc_offline.py records it in the checkpoint's meta.
    split_activation: str = "relu"
    # False swaps the GRUCell for a parameter-matched feedforward stand-in
    # (kilobot_gnn.MemorylessAggregator, 17859 params against the GRU's 17877),
    # isolating recurrence from capacity. Ablation switch only.
    actor_recurrent: bool = True

    # ─── environment ─────────────────────────────────────────────────────────
    # Landmark seed placement. "corners" is the only supported layout;
    # "cluster" exists in belief.SEED_LAYOUTS for backward compatibility only
    # and has not been run in a long time. Wall-lining seeds spawn regardless.
    seed_layout: str = "corners"
    # mid-arena landmark beacons (phase 175, owner sign-off 2026-10-21):
    # KILOBOT_LANDMARKS=1 in the player; their rows arrive in the neighbour
    # message channel (renewal_oracle.LANDMARK_POS)
    landmarks: bool = False
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
    # With belief_comms on, skip actor_io's honest-beacon override of message
    # slots 0-2 (which force-writes the sender's own filter mean position and
    # conf there) and let the POLICY's message head learn the beacon content.
    # The receiver side is unchanged: belief_update.add_peer_range fuses
    # whatever (x, y, conf) the broadcast carries, range from signal strength,
    # gated at PEER_CONF_MIN with sigma inflated by the claimed conf. A
    # squashed conf below ~0.85 makes the filter ignore the row, so the
    # learned policy holds an explicit on/off gate: broadcast a true position
    # with conf past the bar and every receiver in IR range gains a real
    # range-bearing fix; broadcast garbage with high conf and receivers'
    # belief errors GROW (the arena potential drops, the sender's own return
    # with it). With the quiet message start the head outputs zero ->
    # conf 0 -> nothing is fused, so a warm-started run begins exactly at
    # the no-messaging behaviour and grows the protocol from there.
    belief_comms_learned: bool = False
    # Floats 3-5 carry a committed robot's claimed navigation target (x, y,
    # valid) -- not its position, and not belief_comms's slots, so the two never
    # collide. On by default: it is the non-privileged replacement for occupancy
    # checks that used to read every robot's true position directly, and
    # collision avoidance during target selection is core behaviour.
    oracle_claim_broadcast: bool = True
    # Zero the drawn neighbour row's message content (slots 0-8) in Tc when the
    # row came from a PEER robot (senderId >= 1), keeping only the received
    # signal strength. Wall-seed rows (negative senderId, real landmark
    # positions) are untouched. With this on, a live beacon reaches a receiver
    # only through its belief filter (add_peer_range) and never through the
    # motor trunk's Tc input -- the BC tapes' peer rows carried zeros in slots
    # 0-2, so any beacon content is a measured input-distribution shift the
    # clone pays for (docs/tuning.md phase 168, driver D); this routes the
    # channel around that shift entirely. The policy's own broadcasts are
    # unaffected: Tc is the RECEIVER's observation, not the sender's action.
    tc_peer_zero: bool = False
    # The honest-beacon override writes slots 0-2 only when the robot's own
    # arrival latch has fired (or fires this tick); moving robots stay silent.
    # Settled robots make accurate (~settle-radius) landmark beacons that do
    # not drift, and their beacons appear exactly as settling progresses --
    # the wavefront protocol -- instead of semi-localized gossip whose claims
    # measure ~0.11 normalized off (the measured dominant ring-bias source).
    # Pairs with peer_innovation_min on the receiver side.
    beacon_arrived_only: bool = False
    # Stage 2b sender gate: with belief_comms on, a robot broadcasts the
    # honest beacon when it is anchored this episode (one direct seed/wall
    # sighting) AND its own belief confidence is at or above this bar --
    # instead of only after the arrival latch fired. Measured reason (S2a):
    # the arrived-only sender pool is geometrically separated from the
    # receivers that need it -- senders cluster at the shape, receivers in
    # need are mid-arena, and 7u IR cannot bridge it; widening the receiver
    # gate (0.85 -> 0.70/0.60) moved nothing. Anchored mid-arena robots put
    # senders out where the receivers are. Content stays the honest readout;
    # receiver sigma derives from sender confidence. Takes precedence over
    # beacon_arrived_only when > 0; 0 keeps the historical behaviour.
    beacon_anchored_min_conf: float = 0.0
    # The robust hop-count protocol (the measured honest-beacon space is
    # exhausted: position claims need accurate means, and the only senders
    # with good means are geometrically unreachable by receivers in need --
    # S2a/S2b). A robot maintains its communication-graph distance in IR hops
    # to each corner seed (min-relaxation flood over received message slots,
    # observation.hop_flood_update; direct seed sighting = hop 0) and
    # broadcasts it in message slots 0-3. Receivers fuse the hop counts as
    # SOFT UPPER BOUNDS: euclidean dist(p, seed_s) <= k * IR range, so the
    # intersection of several seeds' bounds prunes the particle cloud without
    # any sender-mean error entering it (integer minima self-correct; a
    # stale larger k is still a valid bound). This is the information
    # mid-arena robots lack. Flooding is engaged, not learned -- a stopped
    # robot keeps relaying. Default off.
    hop_comms: bool = False
    # The learned RECEIVER (the last untried branch of the transmission
    # investigation). The policy's own message slot 3 -- free real estate:
    # the honest beacon writes slots 0-2 and the supervision prior targets
    # slots 0-2 only -- is the robot's receiver-trust output: post-squash
    # [-1,1] maps to g = 0.5*(v+1) in [0,1], stored per-robot between
    # decisions, and next tick scales THAT robot's fused-ring sigma by 1/g
    # (belief.add_peer_range). g=1 reproduces the ungated receiver exactly;
    # g -> 0 defuses rings smoothly instead of binary-gating them. PPO
    # prices the gate through the correctness-grounded reward (a fusion
    # that worsens the robot's own belief error costs advantage on the
    # gate). The readout init plants the gate at ~0.98 (status-quo
    # receiver) via head_msg's slot-3 bias, so training starts at today's
    # behavior and learns whether to close it. Default off.
    recv_gate: bool = False
    # Innovation gate for add_peer_range: a peer ring is fused only if at
    # least this fraction of the receiver's particles lie within 2*sigma of
    # the ring. A ring inconsistent with the whole cloud is either a liar's
    # claim or a rescue case; fusing it anyway drags the mean every decision
    # tick (the measured false-arrival mechanism). 0 disables. Values around
    # 0.05 separate cleanly: a spread cloud covers any ring band densely,
    # a confidently-wrong collapsed cloud covers it ~never.
    peer_innovation_min: float = 0.0
    # The receiver's usability gate for a peer beacon (belief.PEER_CONF_MIN).
    # Measured: the honest beacon's confidence has median 0.80 for settled
    # robots, so the historical 0.85 filtered out most of the honest sender
    # pool (96% of receiver updates saw zero usable peers -- a threshold
    # mismatch, not sender scarcity). Default keeps 0.85 for compatibility.
    peer_conf_min: float = 0.85
    # The arrival latch (closed-form and learned head alike, hybrid included)
    # requires the robot to have had at least one DIRECT seed/wall sighting
    # this episode. A robot that has never seen the map geometry has no
    # absolute anchor except chained peer claims, and letting pure hearsay
    # stop a robot terminally is the hop-by-hop error-propagation path. The
    # flag the filter itself tracks is decision-tick sightings, which is also
    # what the filter ever sees, so reward and gate cannot disagree.
    arrival_min_anchored: bool = False
    # The arrival condition must hold on this many CONSECUTIVE decisions
    # before the latch fires; 1 keeps the current behaviour. Inflated
    # confidence from a freshly fused ring is exactly a transient -- a robot
    # whose belief genuinely sits on its pixel keeps satisfying the
    # condition, one whose conf was just pumped through it does not.
    arrival_persist: int = 1
    # Gate-1 encounter probe (eval instrumentation only, off everywhere else):
    # per decision tick, record whether this robot heard any peer at all and
    # whether any peer row carried beacon content at or above each confidence
    # bar -- i.e. whether an EVIDENCE HOLDER was in IR range. Counters live
    # per robot on the worker and are split into lost-phase (own conf < 0.4)
    # vs localized-phase, which is the window-matching question the
    # evidence-sharing design stands or falls on. Reads only data already in
    # the row pool; fuses nothing; writes nothing to the wire.
    encounter_probe: bool = False
    # ---- The evidence-sharing channel (the measured redesign; see
    # evidence.py's docstring for the full design and docs/tuning.md for the
    # Gate 1/2 measurements behind every choice). Master switch. Implies the
    # honest beacon writer under the anchored-conf sender gate and REPLACES
    # single-ring argmax fusion with the claim store + belief.add_peer_fix
    # consensus path. Incompatible with hop_comms, recv_gate and
    # claim_broadcast: those measured out, and their slot usage collides.
    evidence_comms: bool = False
    # The beacon/disclosure sender gate's confidence bar. Gate 1 measured the
    # anchored-0.7 pool at a 57% lost-phase hit rate vs 19% for arrived-only,
    # with 34% of localized robots hearing evidence BEFORE localizing (vs
    # 0.7%). The conf distribution is bimodal (~0.8+ parked or post-fix,
    # ~0.0-0.3 lost), so 0.7 admits the honest sender population.
    evidence_beacon_conf: float = 0.7
    # A referral's effective recency decays over this many ticks in the
    # RECEIVER's clock (from its own reception tick); at zero the referral is
    # dropped. Should comfortably exceed a mid-arena crossing (~2000 ticks at
    # full speed) so referrals survive the approach.
    disclosure_recency_ticks: int = 2000
    # Claims retained per receiver, keyed by sender (latest wins), oldest
    # evicted. 8 senders is generous against scatter-density neighbor counts.
    claim_store_capacity: int = 8
    # Stored-claim sigma aging: odometry variance accumulates as
    # rate * path (normalized units) under the belief code's 0.02 variance
    # convention, so a claim heard 1.0 normalized-units of travel ago carries
    # conf * exp(-0.004/0.02) ~ 0.82 of its original weight. Guards the
    # consensus fix against stale-claim bias compounding with drift.
    claim_odo_var_rate: float = 0.004
    # The consensus pre-gate (evidence.consensus_filter): a claim is
    # materialized for the fix only when a DISTINCT sender's claim agrees
    # with it on the receiver's own cloud -- the fraction of particles within
    # BOTH rings' 2-sigma bands must reach this. The measured T1a harm
    # (0.6383/0.6649 vs 0.6915 at fs7) was correlated wrong-latch pairs
    # passing the separation cap; this is the veto a spread cloud can
    # actually apply. 0 disables the pre-gate (bare materialization).
    consensus_agreement_min: float = 0.10
    # T1b: the scripted homing override (Gate 2's controller at pipeline
    # scale) replaces the policy's motors for LOST robots (own conf < 0.4)
    # holding a fresh referral with a trustworthy strength-law fit. Policy
    # motors are untouched for everyone else. The law's state machine
    # (evidence.update_homing): declining gradient -> one perpendicular
    # reorientation (20-tick refractory); out of contact past
    # homing_grace_ticks -> sweep back through the disc; out past
    # homing_budget_ticks -> ABANDON the referral and return the robot to
    # its own policy. The measured T1b catastrophe (settled 0.1815 at fs7)
    # was the naive port WITHOUT this state machine: a stale fit captured
    # robots indefinitely and wall-following -- their only localization
    # path -- never happened.
    referral_homing: bool = False
    homing_grace_ticks: int = 120
    homing_budget_ticks: int = 400
    # T2: append the +3 homing proprioception (distance, sin, cos to the
    # fitted referral beacon) to the split actor's proprioception. CHANGES THE
    # ACTOR'S INPUT WIDTH: BC tapes for a homing_proprio warm start must be
    # recorded with these columns dark (zeros), and the warm-start loader
    # must tolerate the first-layer width change.
    homing_proprio: bool = False
    # T2: the policy's message slot 6 becomes a broadcast veto -- post-squash
    # sample < 0 silences the robot's whole row. Content stays env-written;
    # the policy can only make the swarm quieter, never lie louder. Off in
    # T1 (the writer zeroes 6-8 explicitly; the wire carries protocol content
    # only).
    learned_veto: bool = False
    # The veto's initial pre-squash bias (launch.warm_start_message_policy
    # applies it; the resume paths re-apply it so an exploration run can
    # start a trained veto from a new bias). Default +2.0 = the T1 writer
    # (P(veto) ~ 4e-4 at the msg std); the t2ax exploration run used +0.5.
    veto_init: float = 2.0
    # T2b (docs/t2b_design.md): the policy's message slot 7 carries a graded
    # self-trust annotation for its OWN claim. Receiver-side, one-sided:
    # multiplier = 1 + min(0, t) -- non-negative samples are exact no-ops
    # (no amplification, and a legacy writer's zeroed slot reads as full
    # trust), t=-1 drops the claim at materialization. Slots 0-6 unchanged:
    # the honest beacon and the disclosure stay env-written, the veto stays
    # the row-level kill. Requires evidence_comms.
    learned_self_trust: bool = False
    # Phase 170: replace single-ring peer fusion with the multi-beacon
    # position fix. belief.add_peer_fix solves a Gauss-Newton least-squares
    # fix over the usable beacons' rings (initialised at the cloud's own
    # mean), vets >= 3-beacon sets by their own over-determination residual,
    # bounds the exactly-2-beacon case by claimed-position separation, and
    # delivers the fix as a HYPOTHESIS -- injected at bounded mass into the
    # worst-weight half of clouds whose spread is past peer_inject_min_spread,
    # never reweighted into log_w. The hypothesis then has to earn weight
    # against the receiver's own evidence; on a tick with no direct
    # measurement, floor_peer_only_spread keeps spread above 0.1, so conf
    # cannot reach the arrival gate's 0.4 bar on peer claims alone. Measured
    # motivation (docs/tuning.md phase 169): single-ring fusion costs a flat
    # ~-0.10 settled regardless of gating, because a spread cloud cannot vet
    # one ring and the fused cloud collapses onto an arbitrary point of it.
    peer_fix: bool = False
    # Beacons in the fix (top-K usable rows by signal strength -- the nearest
    # senders, whose ranges are the most accurate).
    peer_fix_topk: int = 6
    # Weighted-RMS LSQ residual, in sigma units, above which a >= 3-beacon
    # fix is refused: the over-determination consistency test. Two-beacon
    # fixes are vacuously self-consistent and are instead bounded by
    # PEER_PAIR_MAX_SEP. 0 disables the gate.
    peer_fix_resid_max: float = 1.5
    # Only receivers whose cloud spread is at least this receive fix
    # hypotheses: a structured cloud's own seed/wall evidence is better than
    # a neighbour's claim, and phase 169 measured that telling good receivers
    # things they already know more accurately is the harm.
    peer_inject_min_spread: float = 0.2
    # A robot receives at most one fresh fix hypothesis per this many ticks
    # (the oracle_arrived_claim_injection cooldown pattern; without it the
    # injection churned cold clouds on nearly every decision, measured at
    # 71% of belief_update calls and real arrivals 4/24 -> 1/24).
    peer_inject_cooldown_ticks: int = 400

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
    # One observation event per tick (docs/tuning.md phase 162). A Kilobot has
    # one IR receiver, so the tick has ONE ground-truth event; historically the
    # BC capture drew it twice -- the actor pipeline's draw (which becomes the
    # tape's stored tc) and a second, wall-only draw inside simple_oracle (which
    # decided what the oracle's belief fused and steered from). Measured: the
    # two draws disagree on 48.5% of ticks and ~5% of training labels (22.3% of
    # navigating) were computed from an event the input does not encode. On:
    # the oracle consumes the actor pipeline's draw in bc_capture/DAgger paths,
    # and in oracle-only modes (watch) it draws once itself from the FULL pool
    # (seeds + walls + neighbours) instead of a wall-only one. The filter
    # instances stay separate -- only the observation is shared, because it is
    # the same physical radio reception.
    oracle_shared_event: bool = True
    # The oracle's particle filter fuses the event's seed observation (range to
    # the known-position corner beacons) exactly as the actor's own filter does.
    # simple_oracle's filter was wall-only by construction (seed_obs hard-zeroed),
    # which only worked because its private draw awarded it the strongest wall on
    # ~93% of ticks -- more reception than the one-receiver model allows. Under
    # oracle_shared_event that crutch disappears; seed fusion is the compensation
    # that keeps the oracle a strong localizer while making it an honest
    # single-event physical agent. Turning this off with oracle_shared_event on
    # measurably degrades the oracle (twin audit, phase 162).
    oracle_fuse_seeds: bool = True

    # ─── renewal oracle v2 (the amended-task baseline; simple_oracle.py stays
    # frozen as the exact-motion baseline) ──────────────────────────────────
    # Selected by KILOBOT_MOTOR_OVERRIDE=renewal_oracle; design in
    # renewal_oracle.py's module docstring and
    # docs/figures/renewal_oracle_v2_flow.tex. Distances are raw arena units.
    # belief_conf = exp(-var / 0.02) in normalized units: 0.5 is a ~12u
    # spread, 0.97 ~2u -- the settle bar is 5u, so the exit must be tight.
    renewal_wall_exit_conf: float = 0.4   # edge_follow -> navigate position bar (hysteresis top);
                                          # 0.97 with the private filter
    renewal_conf_min: float = 0.3         # navigate -> seek_wall renewal bar (hysteresis bottom)
    renewal_heading_noise: float = 0.0    # filter heading noise (0 = shared heading, as today)
    # seed-pass controller (renewal_oracle.py loop 1). Offsets are the
    # perpendicular distance from the followed side's wall-seed line, which
    # sits 5u inside the wall; a robot's centre stops ~2.1u short of the
    # wall, so the outer side reads at most ~2.9u -- the setpoint stays
    # inside the seed line, clear of that gap.
    # "seed_pass": the seed-pass controller (pivot per seed); "tangent": the
    # frozen oracle's tangent steering by believed heading, with the heading
    # kept right by passive seed-pass fixes (no per-seed pivots)
    # the simple peer protocol (owner sign-off 2026-10-21): arrived robots
    # broadcast believed x, y, conf (slots 0-2); travelling robots fuse a
    # heard beacon as a range landmark behind an innovation gate
    renewal_peer_comms: bool = False
    renewal_peer_conf_min: float = 0.85   # sender gate (and receiver floor) on the beacon's conf
    renewal_peer_gate: float = 0.1        # innovation gate: fraction of own particles within 2 sigma
    renewal_controller: str = "tangent"
    renewal_tangent_offset_set: float = 5.5  # tangent mode's setpoint offset (u): where tangent
                                             # following naturally rides, clear of the wall gap
    renewal_tangent_inward: float = 0.07     # rad toward the wall before seed-pass feedback exists
    renewal_sigma_construct_max: float = 10.0  # tangent mode trusts sigma-by-construction only
                                               # under this heading budget (rad); else the corner.
                                               # 10 = always: 0.35 measured worse (s1 10->6, s2 3->0;
                                               # waiting for corners costs more than rare wrong fixes)
    renewal_track_enter: float = 5.0      # approach -> track once the windowed offset is this close
    renewal_offset_set: float = 4.0       # tracking setpoint offset
    renewal_conv_len: float = 16.0        # path over which a pivot aims to reach the setpoint
    renewal_alpha_cmd_max: float = 0.35   # largest commanded angle off the tangent (rad)
    renewal_pivot_max: float = 0.785      # largest single pivot (rad); also the no-peak recovery
    renewal_pivot_min: float = 0.02       # smaller corrections are skipped (rad)
    renewal_peak_margin: float = 0.5      # d must climb this far above the running min to call a peak
    renewal_peak_min_gap: float = 4.0     # path between peaks (u): one peak per 8u seed
    renewal_nopeak_path: float = 20.0     # path with no peak -> pushing / veering recovery pivot
    # seed counting: each seed pass = exactly 8u along the wall; the cloud's
    # along displacement between peaks is corrected to the count
    renewal_count_max_path: float = 0.0   # longer segments are not counted (u); 0 = counting OFF
                                          # (helped s2 tails, hurt s0/s1 by ~one seed; phase 175)
    renewal_count_shift_max: float = 24.0 # largest single along correction (u)
    renewal_arc_inner: float = 0.6        # inner wheel of the approach arc (outer = 1)
    renewal_trend_alpha: float = 0.3      # EMA weight for per-side strength and followed-side distance
    renewal_stale_ticks: int = 60         # followed side unheard this long -> back to approach
    renewal_lost_ticks: int = 600         # ... this long -> give up the wall (seek_wall)
    renewal_flip_turn: float = 4.712389   # response test: approach-arc turn (rad, 1.5 pi) before sigma flips
    # heading re-anchor from the per-segment angles
    renewal_fix_peaks: int = 3            # consecutive segment corrections per fix
    renewal_fix_blend: bool = False       # Kalman-blend fixes against the dead-reckoned prior
                                          # (measured: no clear gain, phase 175); False = hard reset
    renewal_fix_meas_floor: float = 0.04  # per-segment heading measurement floor (rad), blend only
    renewal_contact_sigma: float = 0.15   # heading sigma (rad) after a pressed / no-peak recovery
    renewal_fix_spread_max: float = 0.17  # they must agree within this (rad, ~10 deg)
    renewal_fix_alpha_max: float = 0.35   # segments steeper than this (rad) are not used
    renewal_gap_lo: float = 2.7           # the wall-gap offset window (a robot pressed on the
    renewal_gap_hi: float = 3.05          # wall reads ~2.9): pressed test + no fixes from slides
    renewal_anchor_dist: float = 12.0     # path (u) since the last fix that still counts as anchored
    renewal_anchor_heading_max: float = 0.1   # ...or the heading budget (rad) is still this small
                                              # (with the seed-pass controller this lost at s0: its
                                              # contact rotates robots; tangent mode keeps heading)
    # the oracle's private filter step (renewal_belief_update)
    renewal_pos_noise: float = 0.05       # position process noise per unit of travel (fraction)
    renewal_meas_sigma: float = 0.01      # wall/seed point-range likelihood sigma (normalized; 1u)
    renewal_private_filter: bool = False  # True: the oracle-private filter step (better for the
                                          # seed-pass controller; worse in tangent geometry)
    # odometry calibration (WheelBiasEstimator): per-robot wheel biases
    # from pairs of consecutive segment heading measurements, removed from
    # dead reckoning
    renewal_calib_sigma_c: float = 0.03   # per-pair heading measurement noise (rad)
    renewal_calib_sigma_b: float = 0.0    # ridge prior on each wheel bias; 0 = calibration OFF
                                          # (measured net-negative at 0.002/0.01, phase 175)
    renewal_calib_jump_max: float = 0.35  # larger pair changes are contact, not slip (rad)
    renewal_calib_min_n: int = 4          # pairs before the budget uses the residual drift
    renewal_calib_residual: float = 0.3   # once calibrated, budget drift = this * renewal_heading_drift
    # heading budget (rad) under which the turn's heading confirms sigma
    # without a corner. 0 = off: measured net-negative (phase 175) -- the
    # corner also pins the ALONG-wall coordinate (near-corner seeds broadcast
    # positions; side-only seeds are ambiguous modulo the 8u spacing), and
    # without it robots exited mid-wall on a confidently wrong period.
    renewal_sigma_from_heading_max: float = 0.0
    # arrival
    renewal_arrive_dist: float = 1.5      # believed distance (u) that counts as arrived
    renewal_leg_err_max: float = 1e9      # under drift: keep riding the wall while the predicted
                                          # leg miss drift*L^2/2 (u) exceeds this and the target is
                                          # ahead. Off: 3.0 measured no gain (s1 10->8, s2 3->3)
    renewal_near_dist: float = 10.0       # slow down inside this (u)
    # heading budget (navigate renewal): sigma_theta = fix_sigma + drift * path since fix
    renewal_heading_fix_sigma: float = 0.03
    renewal_heading_drift: float = 0.0    # rad per unit of path; 0 = never renew on heading
    renewal_heading_max: float = 0.35     # rad; renew when sigma_theta exceeds it
    renewal_renew_miss: float = 4.0       # ...and sigma_theta * remaining distance (u) exceeds this
    renewal_max_renewals: int = 2         # per robot; past it a robot commits to its best estimate
    # DIAGNOSTIC ONLY -- reads true pose: "heading" / "full" hand a navigating
    # robot its true heading / heading+position (an information upper bound)
    renewal_cheat: str = "off"
    renewal_commit_tick: int = 10 ** 9    # past this tick, unfinished robots head for their point on
                                          # their best estimate and stop (no seeking, no renewals)
    # seek timeout: pinned against parked peers, odometry lies (no collision
    # detection) -- past this, turn away and widen the cloud
    renewal_seek_timeout: int = 10 ** 9   # ticks in seek_wall without hearing a wall; off --
                                          # 600 fired 233-322x/run, no gain (pinned robots re-pin)
    renewal_seek_escape_turn: float = 2.0 # rad of the escape turn
    renewal_seek_spread: float = 15.0     # u of position jitter added (position unknown)
    # peer sidestep: pressed against a parked peer (heard at contact range,
    # persistently) in seek_wall / navigate -> quarter-turn + short straight
    # landmark detour (seed_layout "quarters"; under drift only): route a long
    # leg through a mid-arena landmark -- range fixes position, the touch
    # fixes heading from the leg's own displacement
    renewal_detour: bool = True
    renewal_detour_max: float = 15.0      # largest extra distance (u) a detour may add: ON THE WAY
                                          # only (owner, 2026-10-21; 40u made detours that never finished)
    renewal_detour_gain: float = 20.0     # it must shorten the final leg by at least this (u)
    renewal_detour_leg_min: float = 40.0  # only legs longer than this (u) detour
    renewal_touch_d: float = 5.0          # landmark range (u) that counts as touched (a kilobot is
                                          # ~3.3u across: centres never get closer than ~3.3)
    renewal_search_ticks: int = 500       # spiral search for an unheard landmark, then give up
    renewal_sidestep: bool = False         # off: fired 160-185x/run near the shape (peers there are
                                          # always at contact range) and cost 36->23 at s0, 9->6 at s1
    renewal_sidestep_dist: float = 3.5    # peer range (u) that counts as contact
    renewal_sidestep_count: float = 6.0   # close hearings (with decay) before a sidestep
    renewal_sidestep_len: float = 6.0     # u of straight after the quarter turn

    # ─── true odometry slip (Unity-side dials; the amended task's core
    # amendment) ─────────────────────────────────────────────────────────────
    # Python-side mirrors of the player env vars SwarmManager/MotionNoise
    # parse from the same names, recorded on cfg so a launch is auditable and
    # reproducible from its logs. sigma_bias: per-robot systematic wheel
    # mismatch, drawn once per robot at spawn -- the linear-drift component
    # only feedback loops can absorb. sigma_tick: per-FixedUpdate random slip
    # -- the random-walk component. slip_seed: pins the noise stream (a
    # dedicated System.Random, never the spawn RNG). All 0 = the historical
    # exact-motion build, bit-identical.
    slip_sigma_bias: float = 0.0
    slip_sigma_tick: float = 0.0
    slip_seed: int = 0
    # The Python filter's motion-noise model: launch sets belief.MOTION_NOISE
    # to this (sig_xy = MOTION_NOISE * move in belief_predict), so every
    # particle filter honestly widens as error accumulates and conf decays.
    # Under slip this must be at least the effective per-decision slip scale
    # -- an optimistic filter here is exactly the old invisible-divergence
    # failure mode KilobotMovement.cs's history comment warns about.
    motion_noise: float = 0.0

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
    # Deploy the oracle's OWN arrival rule, computed closed-form from the
    # actor's observation, instead of the learned arrived head gate. The learned
    # head is accurate on the tape's distribution (0.99 recall at 0.95) but
    # under-fires on the deployment localisation shift; this rule cannot drift.
    # The oracle stops when its filter's distance to the robot's assigned target
    # falls below cfg.tau_v (after it has localized, conf_pos >=
    # LOCALIZED_CONF_THRESHOLD), and belief_read already emits exactly
    # d_target + conf_pos, so the gate becomes the teacher's own decision on the
    # actor's own filter -- terminal, like the oracle's. Off keeps the learned
    # head path unchanged.
    use_closed_form_arrived: bool = False
    # The closed-form arrival radius, in normalized units, when
    # use_closed_form_arrived is on. 0 (the default) means cfg.tau_v, the
    # oracle's own rule -- but the actor's OWN filter is more conservative than
    # the oracle's (it under-reports closeness; empirically d_target ~1.5x the
    # true distance at arrival), so tau_v stops almost nobody and the robots
    # orbit the point forever. Set a larger radius to park them where this
    # filter actually certifies arrival.
    closed_form_arrival_dist: float = 0.0
    # With use_closed_form_arrived, run the closed-form rule in OR with the
    # learned arrived head instead of replacing it: the two miss different
    # robots (the head under-fires on low-confidence near robots; the closed
    # form certifies only tight arrivals), so either stopping is strictly a
    # superset of the two alone. The latch is terminal either way, like the
    # oracle's arrived state.
    closed_form_hybrid: bool = False
    # With use_closed_form_arrived, run the closed-form rule in AND with the
    # learned arrived head: stop only when BOTH branches agree the robot is
    # done. This is the anti-oracle of closed_form_hybrid -- it stops nobody
    # the two disagree about, trading stopping rate for stopping confidence.
    closed_form_and_gate: bool = False
    # Remove the closed-form latch's terminality (default off = the
    # historical terminal behaviour, bit-for-bit). Release v3 wakes a latched
    # robot only on STRONG REPUDIATION: its own belief confidence below
    # release_repudiate_conf AND believed target distance beyond
    # release_repudiate_dist; otherwise it stays parked whatever the head
    # says. Measured reasons this rule and not others (docs/tuning.md):
    # 59% of wrong-stopped robots no longer satisfied the gate at final
    # (S1b: conf median 0.20 vs 0.80 for settled-right) -- parked on a latch
    # their own belief would refuse; but plain closed-form re-testing
    # churned 55,890 releases/episode and disturbed 29 settled robots per 12
    # wrong stops fixed (moving 0.043 -> 0.262: a released robot's conf
    # cannot re-reach the parked latch floor, a poverty trap); and the
    # learned head saturates at p=1.000 when parked (786/786 debug lines),
    # so it can never vote to release. The strong-repudiation bar targets
    # the S1b diagnostic's 16-35u believed-dist population and structurally
    # cannot touch the borderline-conf settled robots.
    closed_form_release: bool = False
    release_repudiate_conf: float = 0.15
    release_repudiate_dist: float = 0.12
    # Whether a switched-off robot's GRU hidden state stops advancing. True is
    # the original behaviour (phase 142): with nothing having trained the
    # network on the long arrived stretches, letting the recurrence run over
    # thousands of near-identical observations drifted it somewhere it could not
    # come back from. False is correct once training DOES cover those stretches
    # -- bc_offline.py rolls each robot's whole episode, arrived included -- and
    # is then strictly better, because a frozen hidden state is a recurrence the
    # training never saw. The motor is forced to zero either way; this only
    # decides whether the hidden state keeps updating while it is.
    arrived_freeze_hidden: bool = True
    # Build the auxiliary oracle-state head (kilobot_gnn). Training-only: no
    # deployed code path reads it, so it changes no behaviour by itself -- it
    # exists so behaviour cloning can supervise the recurrent representation
    # with the teacher's own state, instead of leaving that latent to be
    # inferred from motor commands alone. Must be set to match a checkpoint,
    # like use_arrived_head, or load_state_dict rejects it.
    use_state_head: bool = False
    # Build the auxiliary last-wall head (kilobot_gnn). Training-only, like
    # use_state_head, and for the same reason in a sharper form: which wall a
    # robot is following is the latent that wall_following's command is a
    # function of, and an unsupervised guess at it shows up as a constant
    # steering bias rather than as noise.
    use_wall_head: bool = False
    # Build the motor command as a soft mixture over the ORACLE's own five
    # commands, each computed in closed form from quantities the actor already
    # observes, weighted by the state head's posterior (kilobot_gnn.
    # oracle_form_motor). Adds no parameters -- it reuses head_state, head_wall
    # and head_motor -- and adds no inputs, but it changes the deployed forward
    # pass, so a checkpoint must be loaded with the same setting. Requires
    # use_state_head and use_wall_head.
    #
    # The measurement that motivates it: the oracle's wheel pair is almost all
    # common mode, and its differential -- the only channel that steers -- has a
    # spread of 0.0093 during wall_following, 0.1% of the target variance there.
    # A two-output head fitted by MSE reproduces the common mode and misses the
    # differential by 0.0895 rms, R^2 = -109 on HELD-OUT ORACLE DATA, worsening
    # monotonically across every run of phases 156-159 while the headline MSE
    # improved. See docs/tuning.md phase 160.
    use_oracle_head: bool = False
    # Magnitude of the learned residual added to that mixture, in motor units.
    # Nonzero so the closed form is a prior rather than a cage: `navigating` in
    # particular is NOT exactly reproducible -- the teacher steers by its own
    # private particle filter, and the observation reproduces its direction with
    # median offset 0.75 degrees but 56 degrees of spread.
    oracle_residual: float = 0.05
    # The same residual's DIFFERENTIAL half. Zero, and measured to belong there:
    # the closed form's steering is already exact wherever it CAN be exact, so a
    # learned correction on that channel has no legitimate job and the fit spends
    # it hedging, which lowers MSE and worsens the command. Swept on a trained
    # network -- median wall_following steering error 0.00850 at 0.003, 0.00290
    # at 0.001, 0.00058 at 0, with the motor MSE unchanged to three figures. The
    # knob is kept because that sweep is the evidence, not because a nonzero
    # value is useful; head_motor's second output is inert while it is 0.
    oracle_residual_turn: float = 0.0
    # Feed the motor head the wall-gated alignment with the wall tangent --
    # sin/cos of (tangent - heading), mixed over the wall head's own posterior.
    # Requires use_wall_head. Costs 4 parameters and changes the deployed
    # forward pass, so a checkpoint must be loaded with the same setting.
    use_steer_feature: bool = False

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
