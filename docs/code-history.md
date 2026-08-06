# Why the code looks like this

The rationale that used to live in long inline comments. Code comments now
describe what a thing does and what it must stay consistent with; everything
here is the history behind those choices — the measurements, the failed
alternatives, the bugs that motivated a flag.

This is indexed **by symbol**, so you can look up the line you are reading.
`docs/tuning.md` is the same material organised **chronologically by phase**,
with the full experimental write-ups. Where an entry names a phase, that is the
tuning.md section with the complete account.

---

## `config.py`

### `dt_fixed = 0.02`

Was 0.05, which did not match Unity's actual Fixed Timestep
(`ProjectSettings/TimeManager.asset`: 0.02). `prop_max_speed` and
`prop_wheelbase` had been calibrated against the wrong value, so they were
re-measured against the corrected one (`calibrate_kinematics.py`, against a live
build).

The two errors had been cancelling: `dtheta`'s computed value came out within
0.5% of the old self-consistent-but-wrong pair. The correction was kept anyway,
because other computations referencing `dt_fixed` were never checked for the
same cancellation. (phase 83)

### `prop_max_speed`, `prop_wheelbase` — derived, not measured

Both are now computed from Unity's own constants rather than fitted, so they
cannot drift out of sync again:

```
prop_max_speed = moveSpeed * UNITY_FIXED_DT * FRAMES_PER_STEP / dt_fixed = 4.0
prop_max_speed / prop_wheelbase = radians(turnSpeed) * FRAMES_PER_STEP  = pi
prop_wheelbase = 4/pi = 1.273240
```

with `moveSpeed = 1`, `turnSpeed = 45`, `framesPerStep = 4` and Fixed Timestep
`0.02` mirrored into `config.py` as `_UNITY_*`. `tests/test_kilobot.py` asserts
the derivation, so changing any of the four in Unity without mirroring it fails
the suite.

**They were 3.1% low.** The configured pair was `3.875 / 1.2333`, which
`tools/calibrate_kinematics.py` re-measures as `4.002 / 1.274`. The *ratio* was
right — `omega = (vR - vL)/wheelbase` and both velocities scale with
`prop_max_speed`, so scaling the pair together leaves `dtheta` untouched — which
is why heading tracking was exact and only distance ran short, by 3.17% of every
unit travelled between position fixes. Over a full arena traverse that is 6.35
units against an arrival tolerance of 5.0.

`3.875 / 4.0 = 0.96875 = 31/32` exactly, which is the signature of averaging
displacement over 32 samples where one contributed zero.
`calibrate_kinematics.drive` now guards against that, discarding a `warm` prefix
and requiring the robot to have been commanded on the previous tick too.

### The travel-direction error: an arc where the simulator walks a polygon

Correcting the constants left a second, independent error the constants could
not express. `split_tick_motion` integrated a continuous circular arc:
magnitude `2(v/omega)sin(dtheta/2)`, travel direction `dtheta/2`.

`KilobotMovement.FixedUpdate` **rotates first and then translates along the new
heading**, so an interval of `M = steps * framesPerStep` substeps is `M` equal
segments laid at headings `1d, 2d, ... Md` with `d = dtheta/M`. Summing that
geometric series gives exactly

```
|displacement| = v * (t/M) * sin(dtheta/2) / sin(dtheta/(2M))
direction      = (M+1)/(2M) * dtheta
```

The arc form is the `M -> infinity` limit of both. At the real `M` the direction
is off by `dtheta/(2M)`, one half-substep. Measured against a live player at
`M = 4`, the ratio of travel direction to `dtheta` was **0.625 = 5/8 in Unity
against 0.5 in the model**, on every turning pair: 0.45 degrees per step at full
spin. Magnitude was unaffected at float32 resolution, so this was invisible to
any check of distance or heading -- it walked the belief filter's position
*sideways*.

Corrected in both `split_tick_motion` and `dead_reckon`. Verified two ways:
against a direct replay of `FixedUpdate` for motor pairs across interval lengths
1-48 steps (worst 1.7e-5% magnitude, 2e-6 degrees), and against a live player
via `tools/check_dead_reckoning.py` (travel-direction error 0.4502 -> 0.0008
degrees, the residual being Unity's own float noise on straight-line pairs).

Dead reckoning is now exact to float32 measurement resolution, which is the
right target: the command is known, the constants are known, and no noise is
injected on either side, so any residual is model mismatch rather than
uncertainty.

**This invalidates trained checkpoints.** Both functions feed the actor's `prop`
observation, so a checkpoint trained before the fix saw systematically different
inputs. Re-run BC.

#### The "6% gap at moderate-asymmetry motor pairs" was a misreading

An earlier entry recorded `prop_wheelbase` moving 1.313 -> 1.2333 because
`calibrate_kinematics.py` "only ever drove the (1,0) extreme" and therefore
"underpredicts omega" at `TURN_MOTOR`'s (0.9, 0.15), citing a measured ratio of
"exactly 1.064622".

Unity's turn response has no nonlinearity to miss: `turnRate = (left - right) *
turnSpeed` is exactly linear in the differential, by construction. With
`wheelbase = 1.313` the model is 6.06% low at **every** motor pair -- (1.0, 0.0),
(0.9, 0.15), (0.5, 0.1) and (0.6, 0.4) alike. And `pi / (3.875/1.313) =
1.064493`, matching that measured 1.0646 to four figures: it was simply the
correction needed to restore `ms/wb = pi`.

The cross-validation cited as support -- "median ratio 1.0638, tight clustering
despite wildly different durations and motor pairs" -- is the tell. A
pair-specific effect would vary across pairs; a flat scale error is identical at
all of them, which is what was observed.

The change itself was right. The reasoning recorded for it was not.

### `split_gru_hidden = 59` and the layer-width budget

Against a stated 24KB hard budget — tighter than an earlier 32KB figure, not
additional room on top of it. With both `use_arrived_head` and `use_turn_anchor`
enabled (the realistic configuration), the actor already measured 18772 bytes at
int8, leaving ~5.8KB of genuine headroom.

Swept on real layer objects rather than hand-derived formulas: growing
`split_gru_hidden` alone, with `split_upscale_hidden` and `split_head_hidden` at
their defaults, **59 is the largest value that fits** (24129 bytes, 98.2% of
24576) — 60 goes 76 bytes over.

The GRU specifically, not the upscale or head, received the headroom: the actor
has to hold a remembered value in hidden state and combine it with a current one
to track state-machine-like transitions (`turn_accum` reconstruction), so
recurrent capacity is the most directly motivated place to spend it. ~450 bytes
of slack remain; not spent on the other two, neither of which had a comparable
justification.

Breaks checkpoint compatibility, as `SPLIT_ODOM_SIZE`'s own changes did in
phases 104 and 106.

### `split_prop_scale = 0.04`

Targets the p90 of measured anchor-tracker distance, ~21–27 raw units at
`prop_max_speed=1.55`, `heartbeat_ticks=48`, 50–100 bots, cluster layout.
Cluster is deprecated — that is a note on the conditions this was derived under,
not a recommendation. There is no live script for re-deriving it; inspect
`buffer.decisions[i]["prop"]` from a rollout.

### `oracle_claim_broadcast = True`

Floats 3–5 of the broadcast message. Replaced privileged occupancy/crowding
checks that read every robot's true position directly, regardless of range or
reception. A robot now only learns of another's claim if it actually received a
message conveying it, through the same IR-range-limited, one-receiver-per-tick
channel as everything else. On by default because it is the non-privileged
replacement for always-on behaviour, not an optional addition. (phase 72)

### `oracle_orbit_axis_trust_threshold = 0.3`

A robot arriving at a corner lock straight from wall-following has one axis
well-resolved (the wall) and the other essentially untouched by evidence.
Trusting both equally centres the orbit on a phantom point offset from the true
corner by roughly the unresolved axis's error — confirmed at **15.9 raw units**
in one traced case, producing oscillating rather than steadily improving
confidence that some robots never recovered from within a 750-second window.
(phase 46)

### `oracle_heading_triangulation = False`

Built to close the gap left when `true_heading` — a confirmed
privileged-information leak into the actor's observations — was removed.
Verified correct in isolation, including the geometry resolving triangulation's
inherent single-reading mirror ambiguity, and refined against Trawny &
Roumeliotis (ICRA 2010) to require 3 readings rather than 2. But not reliable in
full-system testing at any configuration tried.

The investigation moved to reactive, heading-free steering instead — closer to
how the real Kilobot platform solves this. Left off so that `true_heading`'s
removal (an unambiguous fix) is not entangled with a still-experimental
replacement. (phase 70)

### `oracle_wall_seed_position` — a flag that changed default four times

Feeds the specific nearest wall seed's known position into `belief_update`'s
along-wall likelihood term. Validated directly with a hand-built test: a
wrong-heading particle's relative likelihood is penalised by a growing factor
(1.08× at 40 ticks, 10.3× at 400) as its dead-reckoned along-wall position
diverges from a real reading — discrimination a plain wall reading structurally
cannot provide. (phase 73)

- **phase 107, off.** With it on, every wall contact silently resolved both axes
  at once, no different in practice from a landmark fix — exactly what the
  two-tier corner/wall design exists to avoid conflating.
- **phase 111, back on.** Off had a measured cost once combined with phase 108's
  four-cardinal-heading pool: with wall seeds unable to constrain the along-wall
  axis, a robot cannot cross `LOCALIZED_CONF_THRESHOLD` until it physically
  reaches a corner seed. Confirmed heading-independent against a real log (mean
  `wall_following` duration within ~10% across all four starting headings), so
  the earlier north-only default never had a shorter localization path — it only
  had a single wall/corner for the behaviour to concentrate on.
- **phase 113, off again.** After phase 112 fixed the dominant source of test
  flakiness (an unseeded formation choice in `images.py`, unrelated to wall
  seeding). The phase-111-era measurement showed a heavier error tail for this
  setting (std 8.08 vs 1.30, max 37.05 vs 8.63, against an arrival tolerance of
  5.0) — but that measurement predates the phase-112 fix and its own instability
  may have been part of what it measured. Not re-verified against the fixed
  combination.

### `oracle_arrived_claim_injection = False`

An arrived, stopped robot broadcasts its claimed position; a nearby lost robot's
cloud is ring-injected around it, mirroring cold-cloud seed injection.

Validated in isolation: **5.3× error reduction** for a genuinely lost robot when
the claim is correct, but **~2× worse than doing nothing** when a wrong claim
gets through ungated. The sending-side gate is load-bearing, not a nicety. A
single peer-rescue, and repeated exposure to the same claim, was tested and
found never to manufacture enough confidence to pass the gate and re-broadcast
(the ring's angular uncertainty cannot resolve without an independent
axis-constraining reading), so no "was this earned or relayed" tracking exists —
a tested simplification, not a guarantee. (phase 75)

At real scale (24 robots, thousands of ticks) the two-robot test had missed
something serious: extraction scans the whole retained message history, not
fresh single-tick reception, so injection fired on **71% of all `belief_update`
calls**, repeatedly discarding a cold robot's progress before it could converge.
A/B on one seed: arrivals dropped 4/24 → 1/24 with injection uncapped. The
cooldown closed that regression (4/24 → 7/24) but a second seed went the other
way (14/24 → 9/24, though with better accuracy on the arrivals that happened:
5.4 vs 8.0 mean error). Net effect on arrival count is genuinely unresolved with
two seeds and no way to hold the trajectory fixed — injection consumes extra
randomness, so the conditions diverge into different trajectories rather than
the same trajectory plus a fix. Off by default for that reason: a validated-safer
version of a mechanism confirmed capable of real harm, not a validated
improvement. (phase 76)

`oracle_arrived_claim_cooldown_ticks = 400` is the value tested above; not
claimed to be tuned or optimal.

### `oracle_known_start_heading = False`

Fixes a static, physically-enforceable **setup convention** and tells the filter
about it. Not the same category as `true_heading` (removed in phase 70), which
read a live per-tick ground-truth value — legitimate only because, and as long
as, the actual spawn heading physically matches what the filter is told
everywhere this runs.

Confirmed: given a genuinely matching start and exact `dtheta` tracking, this
reproduces `true_heading`'s exactness bit for bit — a legitimate route to
identical numbers, not an approximation. Also confirmed: a small unmodeled
`dtheta` error (wheel slip, an un-replicated physics detail) would produce a
confidently **wrong** belief at exactly zero noise, with no way for
`COLD_SPREAD`'s rescue to notice, since spread never moves. That risk is
accepted, not avoided. (phase 77)

Validated at scale (3 seeds, 20 robots, 2000 ticks) through a harness built to
rule out two real methodology bugs — frozen `step_count` under a direct-loop
pattern, and silent cross-episode corruption from undisabled resets — that had
made an earlier version wrongly report the benefit eroding at longer durations.
With those fixed: heading error improved in every seed, arrivals
improved-or-tied in every seed. One large-error case traced to genuine physical
isolation (no messages, no landmark, nearest robot beyond `IR_RANGE` for the
whole gap) freezing the belief, which then snapped back to within ~0.1° of true
the instant the next decision fired — the expected behaviour of an event-driven
system with nothing to react to. (phase 78)

A real Unity run then surfaced what replica-only validation could not:
`HEADING_NOISE_SCALE`, though small, let particles independently diverge on
heading. `belief_predict` rotates each particle's position update by its own
heading, so particles disagreeing on heading turned identical physical motion
into different positions, inflating position uncertainty for a reason unrelated
to position noise. That was the single cause of both the visible steering
oscillation and the confidence collapse reported alongside it. Fixed by removing
the source: `HEADING_NOISE_SCALE` is now exactly 0.0. Re-validated on the same
two seeds, arrivals went 11/20 and 7/20 → **17/20 and 18/20**, mean heading error
87.8°/69.4° → **2.0°/11.0°**. An earlier attempt — freezing the steering heading
onto the last reading clearing a concentration threshold — was tested and made
outcomes worse on both seeds (stable-but-stale lost to noisy-but-fresh); its
code remains only as a cheap backstop (`HEADING_CONCENTRATION_MIN`) that should
not trigger now that particles stay synchronised by construction. (phase 79)

Real Unity testing then found a more fundamental problem underneath all of it:
`KNOWN_START_HEADING` was simply the wrong number. Measured from real position
data, with no privileged `true_heading` needed, every robot's true starting
heading sat within a few degrees of **+π/2**, not the 0.0 assumed — a plain,
constant, universal mismatch between what Unity's spawn rotation does and what
the code assumed. Correcting the constant to π/2 fully explains both real-Unity
symptoms that phase 79's fix alone left unresolved. (phase 81)

### `cold_start_injection_prob = 0.0`

The actor's GRU hidden state starts at `h_prev=0` exactly once per robot per
episode, which at `max_episode_steps=18000` / `heartbeat_ticks=48` is ~0.27% of
the decisions BC ever sees. `simple_oracle.py` does not read `h_prev`, so its
target is correct regardless — but the actor, trained where a cold `h_prev` is
almost never seen, learned a cold-start behaviour nowhere near it. Verified
against a real trained checkpoint: `go_north`'s correct target is `[1.0, 1.0]`
(full speed straight), while the actor's raw motor logits at a genuine
`h_prev=0` were deeply saturated negative, squashing to essentially `[0, 0]` —
the opposite end of the range. (phase 138)

Only the cached hidden state is dropped; position, belief state and sensor
history stay continuous, and the oracle's target is computed from the robot's
real situation. So these are genuinely correctly-labelled pairs, not synthetic
data.

### `turning_duplicate_factor` and the removed `turning_injection_prob`

Entering `wall_following` requires physically completing a turn, driven by the
fixed `TURN_MOTOR = (0.9, 0.15)` accumulated until `TURN_TARGET_RAD` (π/2).
`turning` is a short transient state — rarer in BC's distribution than
`wall_following`, itself already a small minority — and BC's average-MSE loss
barely notices getting a brief, infrequent state wrong. Verified: a real
checkpoint's motor differential during an early turn-like moment (0.11–0.45) was
far below the oracle's required 0.75, consistent with robots never completing
enough rotation to reach `wall_following` at all, left in a partially-turned
limbo producing the slow wide-arc looping that was reported.

**`turning_injection_prob` was removed entirely rather than defaulted off.** It
forced `worker.simple_state` to `"turning"` for a fraction of decisions. A bug
(unconditional firing regardless of current state) meant **98.4% of firings
landed on a robot already mid-turn**, resetting its rotation progress before it
could finish. Gating on `go_north` fixed that, but a deeper problem surfaced:
even restricted to `go_north`, only **1.7% of firings** coincided with a tick
where the robot's real wall reading was nonzero. The other 98.3% paired the
fixed `TURN_MOTOR` target with an observation that did not show a wall at all,
training the actor on a state label decoupled from its own input. Restricting
injection to ticks with a nonzero wall reading would make it a complete no-op,
since that is the oracle's own natural trigger. A known-flawed mechanism left
dormant is a real risk of being silently re-enabled without this context.

`turning_duplicate_factor` replaced it with something structurally different: it
duplicates BC's real, naturally-occurring `turning` examples — authentic in both
observation and target — within the same update. A duplicated example
contributes to the averaged loss as many times as it appears, mathematically the
same as a per-example weight, without touching the loss computation and with no
possibility of observation/target mismatch.

### `bc_replay_*` — the reservoir

BC's fit only ever saw the rollout window it had just collected, and a window
sits inside exactly one phase of a very long episode. Measured over one full
18000-tick episode, labelling every BC decision by the oracle state producing
its target: **go_north 1.2%, turning 0.4%, wall_following 9.0%, navigating 3.1%,
arrived 86.3%** — and past roughly halfway it is 100% arrived and stays there.
The actor fit whichever phase was in front of it and unlearned the rest.

Confirmed as catastrophic forgetting rather than slow learning by training on
one fixed pooled dataset with `prev_hidden` frozen, so no out-of-distribution
hidden state could explain it: `go_north` validation MSE improved to 0.103 then
degraded monotonically to 0.635 as arrived-only data accumulated;
`wall_following` 0.038 → 0.235. More training made both strictly worse, which
also rules out under-training.

**`bc_replay_balanced = True`** is what the measurement supports, not replay on
its own. At matched gradient steps over three seeds, replay with proportional
sampling fixed four states but left `turning` **worse than no replay at all**
(0.252/0.253/0.256 vs 0.189/0.206/0.143) — at 0.4% of the data, proportional
sampling still essentially never draws a turn. Balanced was the only setting
that learned all five (turning 0.055/0.032/0.048, go_north ~0.015, navigating
~0.012, arrived ~0.001).

**`bc_replay_max_age = 0`.** The ordering experiment above held `prev_hidden`
frozen and so does not measure this either way; see `docs/tuning.md` for the
head-to-head that set it.

**`bc_replay_min_samples = 512`** was found the hard way. In a real head-to-head
the first time `arrived` appeared it had 32 stored samples, immediately took a
fifth of every minibatch as ~200 copies each, and the fit loss jumped from 0.06
to 1.30 while cold-start error rose for fifteen iterations before recovering.
Balance is meant to stop a state being ignored, not to manufacture a batch from
a handful of duplicates.

**`bc_replay_persist`.** Measured on a real restart: `arrived` came back at 416
instead of 11515, `turning` at 2841 instead of 9664, and the arrived head's
held-out recall collapsed from 0.414 to 0.039 within ten iterations, because
`arrived` fell back below `bc_replay_min_samples` and the ramp starved it. A
full reservoir is roughly 420 bytes per sample — about 4GB at capacity 2000000
across five states — which is why `bc_replay_save_interval` exists; the actual
size and duration are printed on every save.

### `bc_actor_eval_interval = 1`

Simulation is essentially all of BC's wall clock: ~114s per iteration on a real
16-arena run, roughly half of it the actor-driven eval collect. Running it every
iteration mattered more when coverage was the checkpoint criterion; since phase
157 that belongs to the val tape, which replays a recording and needs no
simulation. Raising it is a real trade-off, not free: fewer ticks per iteration
means fewer episode boundaries and fewer distinct formations. Acceptable only
because formation diversity was measured not to be the binding constraint (10
formations already generalised in a smoke run), and because a persisted
reservoir carries the diversity of every earlier run.

### `bc_motor_skip_arrived = False`

Two separate costs, both measured rather than argued: `arrived` is the dominant
class in the motor regression at 86.3%, and its target `[0.0, 0.0]` sits exactly
on `squash_action`'s tanh asymptote, so a motor head fitting it must drive its
pre-activations toward −∞ — the pathology already seen as `motor_pre = -6.87` in
a real checkpoint and as raw logits of −7 to −13 that never self-recovered
(phase 137.5).

Measured on a fixed pooled dataset, two seeds, balanced sampling held constant:
every remaining state improved (go_north .0133→.0102 and .0148→.0129, turning
.0314→.0278 and .0288→.0258, wall_following .0086→.0054 and .0108→.0085,
navigating .0114→.0096 and .0104→.0082) and the motor pre-activation range
compressed from −2.98/−3.04 to −1.32/−1.10, with the deep-negative tail gone.

### `bc_arrived_natural_prior = False`

Measured on a real run: the arrived head fired 1866 times at iteration 19 when
no robot had arrived at all, and only **40.2% of its arrival claims at iteration
39 were correct**, while `arrived` was 3.4% of the reservoir against roughly 20%
of every balanced minibatch.

### `arrived_release_threshold = 0.0`

0.0 keeps the original behaviour, where switching off was permanent for the rest
of the episode — which turned a single false positive into a robot frozen for
every remaining tick, the symptom phase 154 opens with.

### `use_arrived_head = False`

From a direct request: "rework the actor to flip a flag when it thinks it has
arrived… instead of learning to sit there which could destroy weights."
`arrived` is exactly the kind of long, near-unchanging observation that phase
137.5's cold-start test showed drives this GRU's recurrent state into
progressively worse drift rather than a stable point — and arrived robots can
sit for a large fraction of an 18000-tick episode. A real architecture change,
not a training-data-only fix, which is why the flag gates construction of the
head rather than just its use.

### `use_turn_anchor = False`

The actor's raw `dtheta` never survives into `tc` or `prop` as a standalone
value — `gather_split_state` folds it into `belief_predict`'s particle
integration before anything reaches the network, leaving only an absolute
`(sin_m/r, cos_m/r)` heading pair. Recovering "how far into this turn am I" from
that requires holding a past heading in hidden state and differencing it every
decision, a genuinely bilinear operation (`sin(a-b) = sin_a·cos_b - cos_a·sin_b`)
that a GRU's affine-plus-gate structure does not compute for free. Compounding
it, the oracle's target during `turning` is the fixed `TURN_MOTOR` regardless of
`turn_accum`, so no per-tick gradient in that state rewards learning the
difference — the only tick carrying signal is the one where the target
discontinuously switches.

**A plain rising edge on `walls_b` was the first version and is not what ships.**
`walls_b` is reception-lottery-subject, so a single tick where a neighbour
message won the lottery instead of the wall reading looked like "wall gone, then
rediscovered", re-anchoring mid-turn — confirmed at **83 of 116 real turns** in
one rollout, biasing tracked rotation to undershoot by a mean of 32°. The
refractory period closes most of it: 86% of turns land within 5° of ground truth
in the same rollout.

An honest limitation: this tracks a relative angle between two heading
snapshots, which is not quite `turn_accum` (a running sum of `|dtheta|`). They
coincide only because rotation stays in one direction under the fixed turn
motor; a mid-turn collision knocking heading backward would make `turn_accum`
keep climbing while this dips. That, not remaining re-anchor timing error, is
the likely source of the rollout's remaining worst case of 34° — doubling the
refractory did not shrink it.

### `turn_anchor_latch = True`

An oracle turn is exactly 35 ticks against a 40-tick refractory, so BC never
once sees a mid-turn re-anchor. A deployed actor whose turn overruns 40 ticks
gets re-anchored by the next wall reading, its implied rotation snaps to zero,
and it spins forever — directly observed in live Unity and reproduced on-policy
(net/path 0.156, seventeen full rotations). On by default because it is a no-op
on the training distribution, so it corrects an already-trained actor without
retraining.

---

## `belief.py`

### `KNOWN_START_HEADING = pi/2`, `HEADING_NOISE_SCALE = 0.0`

See `config.py`'s `oracle_known_start_heading` above — that entry has the full
arc across phases 77, 78, 79 and 81, including the measurement that established
π/2 and the arrivals/heading-error numbers that came with removing the noise.

### `CARDINAL_HEADINGS` — three of four entries are derived, not measured

Only the first (`KNOWN_START_HEADING`, north) was measured against real position
data. The other three follow from it by rotation math: Unity's left-handed
Y-rotation turns a forward-facing object from +Z toward +X, and this project's
kinematics fix heading=0 along +X and π/2 along +Y counterclockwise — so Unity's
+Z is this project's +Y, and each successive 90° Unity rotation **subtracts**
π/2 from the Python heading rather than adding it.

Verify with `SIMPLE_ORACLE_SPAWN_CHECK` (`KILOBOT_ORACLE_DEBUG_WALL_LOG=1`),
which exists specifically to catch this class of mismatch per robot at spawn —
it is what caught the original phase-80 bug. (phase 108)

### `BELIEF_PARTICLES = 256`

Increased from 32. A correct single-axis (wall) reading fused against a prior
that has drifted — which it does over a long committed journey; predict-only
motion tracks true distance correctly, so this is not a dead-reckoning bug —
produces a large, essentially random resampling jump in the axis the
measurement says nothing about. At 32 particles that jump measured **~41 units**
on a realistic case, shrinking monotonically with more particles: ~17 at 100, ~8
at 300, ~1 at 1000. Classic Monte Carlo resampling noise from too few particles
representing a wide prior.

This is also the confirmed mechanism behind committed robots stopping at wildly
wrong locations, sometimes still visibly on the arena wall: a real wall reading
fires, the spurious jump in the other axis happens to land the belief near the
assigned target, `dist` drops below the stop threshold, and the robot halts
where it physically was.

Compute cost is favourable — the operation is dominated by fixed per-call
overhead, not particle count, so 8× the particles cost only ~63% more wall
clock, still under 8ms/cycle for 100 robots doing a full predict+update.
(phase 58)

### `_matched_generator`

A GPU run crashed twice inside this file's `generator=` call sites (generator and
tensor on different devices) despite two traced fixes at the source: the policy
never being moved to `cfg.device` (phase 128), and the shared generator's
construction being made explicit and defensive (phase 129). Both were real fixes
for things code-reading confirmed *would* cause this, but the crash recurred at
the identical line.

This is a different kind of fix: a defensive normalization at the point of use,
so correctness no longer depends on that construction being right at all. It
prints once, the first time a mismatch is actually caught — real evidence, not a
fourth guess, if this ever needs revisiting. (phase 130)

### `TRIANGULATE_MIN_READINGS = 5`

Raised from the theoretical minimum of 3 after inspecting the raw per-candidate
likelihood grid: a short, realistic 3-reading baseline can leave the true
candidate and a spurious one nearly exactly tied, log-weight difference ~0.001.

Measured across a real multi-corner test: 3 readings at
`ANCHOR_MIN_DISPLACEMENT=0.06` gave median heading error 6.2–10.9°; 5 readings
plus a higher `ANCHOR_MIN_DISPLACEMENT` gave 6.6–8.4°. A real if modest
improvement, and not a full fix for the underlying oscillation — a good estimate
can still be disrupted by a later ambiguous encounter.

### `resample`'s `best_raw_log_w`

Uses the MEAN raw importance weight (Thrun/Fox's aMCL formula), not the max. Max
was tried and rejected: it lets one lucky particle mask a cloud whose centre is
still far from truth. A collapsed-but-wrong test cloud kept showing "improving"
fit under a max-based signal purely because resample jitter occasionally placed
one particle near the true value, even while the cloud's mean stayed put — which
starves `inject_random_particles` exactly when it is needed most.

Computed by log-sum-exp because `exp(log_w).mean()` underflows to exactly 0.0 in
float32 for every particle when `log_w` runs to −1000s, silently destroying the
information the signal exists to capture. (phase 59)

### `inject_random_particles`' heading handling

Two rounds. First (phase 82), an audit of everywhere heading can get randomized
found this call missing `known_start_heading`, so the rescue could scramble an
accurate heading back to uniform-random the moment it fired.

That fix was itself incomplete and was then shown to *cause* the large
oscillating heading jumps chased through phases 83–85. This injection replaces
only a fraction of particles, so resetting that fraction to the fixed
`known_start_heading` splits the population between old-tracked-heading and
fresh-known-start-heading particles whenever the robot has rotated since spawn.
A controlled test forcing the rescue on a robot already rotated ~180° left the
population no longer sharing one heading — the exact invariant the zero-noise
design depends on — and a later resample can land on either group. That is the
~180° back-and-forth jump real logs showed, correlated with wall readings (100%
of large jumps had `has_wall=True`) because a wall's tight band constraint is
more likely than other measurements to make an already-wrong belief look bad
enough to trigger the rescue.

Fixed by preserving the CURRENT tracked heading on fresh particles, matching the
other three injection sites. (phase 86)

### `inject_wall_band`'s per-axis gate

`COLD_SPREAD` was calibrated against the combined `sqrt(var_x + var_y)`, so one
axis locking in tight could pull the combined number under threshold and
silently stop injection for the other axis before it was ever separately
measured. Confirmed empirically: `belief/mean_conf_x` and `mean_conf_y` climbed
at a practical pace in a real run while `belief/mean_conf_pos`, which needs
both, climbed 6–10× slower. (phase 12)

---

## `kilobot_gnn.py`

### `dead_reckon` / `split_tick_motion` sign: `(vR - vL)`

Phase 49 changed this to `(vL - vR)`, verified against an *assumed* Unity
`turnRate=(left-right)*turnSpeed` formula — the same assumption later proven
backwards by direct real-Unity evidence when fixing the steering law (phases
51/53, where real robot headings measured opposite to their target direction
under it).

Phase 56 reverted to the original, which had been correct all along. The wrong
sign introduces a systematic, *shared* heading-drift bias into every particle's
dead-reckoning — they all use this same formula, so they drift together, staying
tightly clustered (i.e. confident) while walking away from the truth. That is
the mechanistic explanation for the belief filter's median ~90-unit position
error with near-zero correlation to reported confidence.

### `SPLIT_GRU_HIDDEN` duplicated with `config.split_gru_hidden`

Two constants for one concept: this module's is what `SplitObservationActor()`
uses when constructed with no arguments (tests), `config.py`'s is what
`build_actor(cfg)` passes at real construction. Divergence is the
duplicate-constant pattern this project has repeatedly found real bugs in
(`SEED_SIZE`, `NODE_FEATURES`) — caught here because a test asserting the
actor's parameter budget was silently checking a stale 48-hidden configuration
long after config's default had changed to 59. (phase 147)

---

## `actor_io.py`

### The steering law's un-negated `cross`

Phase 51 negated it, verified against a Python simulation of Unity's
`turnRate=(left-right)*turnSpeed` rather than the running `KilobotMovement.cs`.
Phase 53 reverted that on overwhelming real-Unity evidence
(`WALL_DEBUG_MOTOR` logging, 2390 committed-robot samples): median
`dot(heading, assigned_dir) = -0.982`, with **79% of samples clearly opposite**
(<−0.8) and only 0.5% matching (>0.8). Heading was converging on the exact
opposite of the target direction, which uniformly explained every symptom
reported after the phase-52 seedObs fix — counterclockwise wall-following,
backward corner-orbiting, and committed robots heading toward the seed they had
just left — as one "converges 180° from where it should" failure expressed
differently depending on which `g` fed it.

### The reacquire magnitude and gain `k`

The original `k=0.9` was tuned implicitly assuming much more frequent
re-evaluation than the architecture provides: commands are deliberately
ballistic between decisions, held for up to a full heartbeat interval (48 ticks).
Sign-corrected but untuned, it settled ~29° from target at every tested starting
heading — a stable bounded oscillation, not convergence. Both values were swept
together against the same simulation over many starting headings at the real
held-interval duration, and verified to converge within a few degrees from every
start. (phases 50/51)

---

## `formations.py`

### `Formation`'s orientation

This has been got wrong more than once and is worth re-deriving carefully rather
than trusting the algebra alone.

- The **row flip** is real and required: `Texture2D.GetPixels()` is bottom-up
  (row 0 = bottom, Unity's OpenGL-derived convention) while PIL is top-down.
  Without the flip the two produce mirrored `nz` for identical input. An earlier
  comment claimed no flip was needed; that was wrong. (phase 23)
- The **rotation** is a net identity. Phase 32's change measured 180° against the
  phase-25/26 state rather than the intended 90 — two different 90° turns off the
  same starting point are 180° apart from *each other*, not from where either
  started. Phase 33 composed a further 90° CW step onto phase 32's transform,
  which works out to exactly `(nx, nz)` again: CW undoing the CCW step, leaving
  only the row-flip baseline, matching `ImageLibrary.BakeImage`.

---

## `observation.py`

### `TURN_ANCHOR_REFRACTORY_TICKS = 40`

A tick-based refractory rather than a count of consecutive zero-decisions: no
fixed decision count is safe against a long enough streak of lost
reception-lottery draws, whereas an elapsed-time requirement only has to outlast
a real turn, whose duration is fixed by physics (~33 ticks for π/2 at
`TURN_MOTOR`'s rate). 40 clears that while staying under `heartbeat_ticks`' 48.

Measured against a decision-count debounce of 20, in a real-scale rollout: mean
error −2.5 → −2.0°, std 31.5 → 5.5, worst case 124 → 34°, with 86% of turns
landing within 5° of ground truth. Doubling to 60 did not improve it (84% within
5°, same 34° worst case) — the residual is `use_turn_anchor`'s documented
signed-angle-vs-`turn_accum` limitation, not a re-anchoring timing issue.

### The relative-target vector's known gap

`formation_pool` (`cfg._oracle_formation_pool`) is only set by BC entry points.
When it or an arena's `image_id` is unavailable the vector degrades to zeros for
those robots, rather than feeding a placeholder position into the
bearing/distance math. Wiring it into every `gru_split_observation` entry point
is outstanding work. (phase 106)

---

## `simple_oracle.py`

### `wall_following`'s approach slowdown

The state had no awareness of approaching localization, so it drove at full
speed straight through a corner seed's range while `belief_conf` — its sole exit
condition — took several ticks of real particle-filter convergence to cross the
threshold. By the time the state exited, the robot had driven measurably past
where it should have.

This got worse as a direct consequence of phase 107 making wall seeds
single-axis by default: before that, a plain wall reading could localize a robot
well before it neared a corner, so the overshoot had no chance to manifest. With
a corner seed the only way to cross the threshold, the lag's distance cost is no
longer negligible. Slowing begins at half `LOCALIZED_CONF_THRESHOLD` and floors
at 0.15× rather than a full stop, so a robot whose conf hovers just under keeps
making progress instead of looking frozen. (phase 110)

### `sample_split_event` rather than a local argmax

A "strongest wall wins" argmax correctly enforces one wall side per tick, but
not with the same selection *rule* the trained-policy pipeline uses for the
identical situation — that draws a single winner by strength-weighted
`torch.multinomial`. A real IR receiver's collision behaviour could plausibly go
either way; the two were made to match so the actor under BC training sees
decisions generated under the same selection process its own observations are
drawn from. Reusing the function rather than reimplementing the draw keeps the
two from drifting apart if the weighting is ever retuned. (phase 117)

The local argmax itself existed to fix a crash: feeding `belief_update` the raw
`(n, WALL_SIZE, 2)` table throws `RuntimeError` the first time any robot's wall
channel is nonzero. That only ever fired against real Unity, the only caller
that populates `wall_seed_xy`. (phase 92)

---

## `trainer.py`

### `self.sample_rng` construction

A GPU run crashed downstream of this line with "found at least two devices,
cuda:0 and cpu" inside `belief_predict`, which reads this generator. The data
side was traced exhaustively first and found clean: `split_obs`,
`gather_split_state`, `split_track_read` and `belief_read` all either take
`cfg.device` explicitly or derive it from their input tensor.

This generator is the one thing every one of belief.py's `generator=` sites
shares, and the one thing plausible to end up on the wrong device without
raising at construction — `torch.Generator`'s device argument is accepted even
before a CUDA context is fully established, unlike a real tensor `.to()`, which
forces that context to exist. Hence the explicit `torch.device` and the tensor
allocation that forces the context first.

Never confirmed directly (no GPU available), which is why
`belief._matched_generator` exists as a second line of defence. (phases 128–130)

---

## Unity (`Assets/Scripts/`)

### `KilobotAgent.SetVisualState`'s palette

Redesigned from an ad-hoc palette (white/orange/magenta/red/black — magenta
chosen only because it was the unused slot, black for "done" reading as
off/dead) into two deliberate families: every seed in cool blue/teal, so
anything cool-hued is infrastructure and never a robot; a kilobot's body a warm
progression tracking task state, breaking to green — the one universally "done"
colour — only at genuine completion. No hue is reused between families. (phase
105)

The ring originally stayed a fixed neutral grey on the reasoning that a changing
ring would be redundant with the body. Direct feedback was that a visibly
unchanging ring read as a bug rather than a choice, so it now tracks the body.
(phase 107)

### The ring alpha bug

Every `Color` literal in `SetVisualState`'s switch omits its 4th (alpha)
argument, which C# defaults to 1.0. Correct for a solid body — but writing the
same colour straight onto the ring overwrote whatever alpha
`CommRadiusIndicator.Attach` gave it with full opacity on **every call**. The
ring was therefore always opaque in practice regardless of `ALPHA`, and no
amount of lowering that constant could ever have fixed it. Preserving the ring's
existing alpha and changing only RGB is what actually keeps it translucent.
(phase 110)

### `KilobotMovement` has no domain randomization

The file previously carried a per-robot fixed motor bias sampled at spawn, a
Gaussian noise term, and a low-pass filter on the commanded motors. It was the
confirmed root cause of a heading-drift investigation: a robot commanded
`[1, 1]` drifted at a constant, per-robot-specific rate for as long as it held
that command, traced to `leftBias`/`rightBias` rather than anything Python-side.

The defaults had already been zeroed once, in phase 82, for exactly this failure
mode — but a `[SerializeField]` value already saved on the prefab does not pick
up a later change to the script's default, and hadn't, which is how it fired
again. Removing the mechanism outright means nothing on the prefab can carry a
stale nonzero value into a future build. `belief.MOTION_NOISE` is 0 because of
this. (phase 97)

### `SwarmManager`'s spawn rotation of 0

Does NOT correspond to `belief.py`'s heading=0. Measured from real position data
(no privileged `true_heading` needed — comparing known local-frame motion against
observed global displacement), every one of 53 robots' true starting heading sat
within a few degrees of +π/2, with a low-noise estimate of −90.02° mean /
−90.07° median offset from the assumed 0. The fix was entirely Python-side; this
rotation is unchanged and correct. (phases 80/81)
