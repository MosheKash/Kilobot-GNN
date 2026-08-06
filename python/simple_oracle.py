"""simple_oracle.py -- a minimal, from-scratch oracle.

Genuinely separate from actor_io.scripted_motors in
actor_io.py -- shares only already-independently-verified primitives
(split_tick_motion's kinematics, belief_predict/belief_update/belief_conf/
belief_read for position tracking, the existing wall observation channel,
hilbert_order/mix_hash for hash-based target assignment). None of the old
oracle's own state machine, coordination, or injection-site history is
reused, so none of its bugs can leak in either.

State machine, per robot, in order:
  1. GO_NORTH: motors=[1,1], no course correction of any kind, until a wall
     signal is received. Not influenced by other kilobots at all -- no
     peer/seed reception is read anywhere in this state.
  2. TURNING: a fixed, dead-reckoned 90-degree clockwise turn (motors=
     (0.9, 0.15), verified empirically this phase to produce a negative
     dtheta -- i.e. clockwise -- under this project's own kinematics), so
     the robot ends up moving clockwise along the wall it just hit.
  3. WALL_FOLLOWING: drive along WALL_TANGENT[wall_name] (already the
     clockwise-along-the-perimeter direction -- see WALL_TANGENT's own
     comment below) using the robot's own, independently-tracked heading to
     correct any drift. Continues until belief_conf (position spread only)
     crosses LOCALIZED_CONF_THRESHOLD.
  4. NAVIGATING: steer directly toward the hash-assigned target using the
     belief filter's own position estimate and the robot's own,
     independently-tracked heading.
  5. ARRIVED: stop, terminal for the episode.

Heading is tracked ENTIRELY separately from the particle filter's own
internal heading (particles[:, :, 2]) -- a single scalar per robot, starting
at KNOWN_START_HEADING (a physically-legitimate setup convention: every
robot is placed at the same known starting orientation, the same real-world
justification this project's own oracle_known_start_heading flag already
rests on -- NOT a live read of Unity's ground truth) and updated every tick
purely from the robot's own commanded motor values via split_tick_motion,
the same kinematic formula this project has independently, repeatedly
verified against real Unity data (docs/tuning.md phases 81-86). This is
deliberate: the particle filter's own internal heading has been the
specific, repeated source of the injection-site bugs phases 77-89 spent
this whole project chasing down one at a time. This oracle's own steering
NEVER reads particles[:, :, 2] for anything. belief_predict is still fed
this same, independently-computed dtheta (heading_noise_scale=0) purely to
keep the particle filter's own internal bookkeeping self-consistent for
position tracking; nothing this oracle does depends on what that internal
value ends up holding.

NO ground truth or privileged information anywhere in this file: no true
position, no true heading, no other-robot coordination, no occupancy-
checking. Position is genuinely unknown at spawn (belief_init's own uniform-
random spread) and narrowed only by real, received wall contact -- the
existing wall observation channel plus, when available, the specific wall
seed's own broadcast position (wall_seed_xy, the existing phase-73
mechanism) for the along-wall axis a plain wall reading alone cannot
constrain.
"""
import math

import numpy as np
import torch

from belief import (belief_init, belief_predict, belief_update, belief_read, belief_conf,
                    KNOWN_START_HEADING, HEADING_NOISE_SCALE, ARENA_HALF, BELIEF_PARTICLES,
                    LOCALIZED_CONF_THRESHOLD, SEED_SIZE)
from kilobot_gnn import MOTOR_SIZE, MESSAGE_SIZE, WALL_SIZE
from kinematics import split_tick_motion
from actor_io import ensure_target, sample_split_event

WALL_NAMES = ["north", "east", "south", "west"]
# already the clockwise-along-the-perimeter direction -- this dictionary's
# own comment in actor_io.py ("LEFT while traveling") and direct geometric
# tracing both confirm this: wall on the
# traveling robot's left, at every one of the four walls in turn, is
# clockwise motion around the inside of the arena. Reused verbatim from
# actor_io.py (not recomputed) since it is already verified there
# (verified by rotation math, not assumed).
WALL_TANGENT = {"north": np.array([1.0, 0.0]), "east": np.array([0.0, -1.0]),
                "south": np.array([-1.0, 0.0]), "west": np.array([0.0, 1.0])}

# Verified empirically, not assumed -- left motor
# faster than right produces a negative dtheta (clockwise) under this
# project's own split_tick_motion kinematics. Matches the sign the existing
# oracle's own stuck-recovery turn already uses (actor_io.py's
# exploring_motor, motor=[0.9, 0.15]), an independent point of agreement.
TURN_MOTOR = (0.9, 0.15)
TURN_TARGET_RAD = math.pi / 2.0

# Slow down as belief_conf approaches LOCALIZED_CONF_THRESHOLD (0.4), which is
# wall_following's sole exit condition. The filter needs several ticks to
# converge past it, so a robot at full speed drives measurably past the corner
# seed before the state exits. Slowing begins at half the threshold, leaving
# room to shed speed, and floors at 0.15x rather than stopping, so a robot whose
# conf hovers just under keeps making progress. See docs/code-history.md.
APPROACH_SLOWDOWN_CONF = 0.5 * LOCALIZED_CONF_THRESHOLD
APPROACH_SLOWDOWN_MIN_SCALE = 0.15

# The same dot/cross steering law
# scripted_motors' own "oracle" branch already uses (actor_io.py), reused
# directly rather than re-derived, since getting this sign wrong is exactly
# this project's own, repeated historical failure mode (phases 51/53).
REACQUIRE_TURN = 0.45
STEER_BASE = 0.9
STEER_GAIN = 0.35


def _steer(h_vec, g_vec):
    hn = h_vec / (np.linalg.norm(h_vec) + 1e-6)
    gn = g_vec / (np.linalg.norm(g_vec) + 1e-6)
    cross = float(hn[0] * gn[1] - hn[1] * gn[0])
    dot = float(np.dot(hn, gn))
    turn = (REACQUIRE_TURN if cross >= 0 else -REACQUIRE_TURN) if dot < 0 else cross
    left = float(np.clip(STEER_BASE - STEER_GAIN * turn, 0.0, 1.0))
    right = float(np.clip(STEER_BASE + STEER_GAIN * turn, 0.0, 1.0))
    return left, right


# Diagnostic-only true pose, shared by the
# spawn check, the per-transition check, the periodic go_north check, and
# the arrival on-shape check below -- one try/except-guarded
# worker.snapshot() read rather than four copies of the same pattern. Same
# worker-agnostic mechanism actor_io.py's own TRUE_HEADING_DEBUG already
# uses (real Unity's critic side channel; works against a real EnvWorker,
# not just the replica). Never called outside an oracle_debug_wall_log
# check, and its return value is never used for anything but a print --
# this oracle's actual steering never calls this function. Returns
# (true_pos, true_heading_deg); either or both may be None if snapshot
# data isn't available here.
def _true_pose(worker, a, l):
    try:
        snap = worker.snapshot(a)
        if snap is not None and snap.get("node") is not None and l < snap["node"].shape[0]:
            node_arr = snap["node"]
            node_arr = node_arr.cpu().numpy() if hasattr(node_arr, "cpu") else np.asarray(node_arr)
            true_pos = node_arr[l, 0:2] * ARENA_HALF
            true_heading_deg = math.degrees(math.atan2(node_arr[l, 3], node_arr[l, 2])) % 360.0
            return true_pos, true_heading_deg
    except Exception:
        pass
    return None, None


def _true_heading_deg(worker, a, l):
    return _true_pose(worker, a, l)[1]


def simple_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, formation_pool,
                         belief_attr = "belief"):
    # walls: (n, WALL_SIZE) raw wall observation strengths -- the same received
    # transmission channel the rest of the pipeline reads, never privileged.
    # wall_seed_xy: (n, WALL_SIZE, 2) or None, every wall side's broadcast
    # position; narrowed below to the single side that fired, (n, 2), which is
    # the shape belief_update expects.
    #
    # belief_attr lets this oracle's particle filter live under a worker
    # attribute other than worker.belief. Needed when this runs alongside
    # gather_split_state in the same tick (BC capture), which reads and writes
    # worker.belief for the trained actor's own proprioception -- the two track
    # different things and would otherwise overwrite each other every tick.
    # Every other worker attribute this file uses is already prefixed uniquely.
    #
    # State is gathered into one batched tensor so split_tick_motion,
    # belief_predict, belief_update, belief_conf and belief_read each run ONCE
    # rather than once per robot; per-call overhead paid n times was measured to
    # be the bottleneck. Only the state-machine branching remains a per-robot
    # Python loop.
    n = len(arena_ids)
    device = walls.device
    motor = torch.zeros(n, MOTOR_SIZE, device = device)

    # Enforce "at most one wall side per tick" using sample_split_event itself,
    # the same strength-weighted draw the trained-policy pipeline uses, rather
    # than a local "loudest wins" argmax -- so the actor under BC training sees
    # decisions generated under the same selection process its own observations
    # are drawn from. Reusing the function rather than reimplementing the draw
    # keeps the two from drifting apart if its weighting is ever retuned.
    #
    # Its pool also spans seeds and neighbour messages, which this wall-only
    # oracle has no logic for, so those are passed in genuinely empty: they
    # carry zero weight and can never win the draw, leaving wall selection the
    # only thing this call decides. wall_part comes back already masked to one
    # winner, and wall_seed_xy already narrowed to (n, 2).
    dummy_seeds = torch.zeros(n, SEED_SIZE, device = device)
    dummy_rows = torch.zeros(n, 1, MESSAGE_SIZE + 2, device = device)
    dummy_valid = torch.zeros(n, 1, dtype = torch.bool, device = device)
    _tc, _seed_part, walls, wall_seed_xy = sample_split_event(
        dummy_seeds, walls, dummy_rows, dummy_valid, cfg, rng, wall_seed_xy = wall_seed_xy)

    if not hasattr(worker, "simple_heading"):
        worker.simple_heading = {}
        worker.simple_state = {}
        worker.simple_turn_accum = {}
        worker.simple_wall_name = {}
    if not hasattr(worker, belief_attr):
        setattr(worker, belief_attr, {})
    belief_dict = getattr(worker, belief_attr)

    # --- pass 1: cheap per-robot init/lookup, gathered into batched tensors ---
    last_motor_batch = torch.zeros(n, MOTOR_SIZE, device = device)
    steps_batch = torch.zeros(n, device = device)
    particles_batch = torch.zeros(n, BELIEF_PARTICLES, 3, device=device)
    # Robots whose arena has a reset pending
    # (worker.reset_pending(a), if the caller exposes it) at the exact
    # moment they'd otherwise be freshly initialized below -- see that
    # method's own comment for the full race this closes. Pass 2 skips
    # these entirely, rather than committing any heading (even the
    # KNOWN_START_HEADING fallback) that would itself already be stale the
    # moment arena.spawn() actually runs, next tick.
    skip_tick = set()

    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        worker.simple_heading.setdefault(a, {})
        worker.simple_state.setdefault(a, {})
        worker.simple_turn_accum.setdefault(a, {})
        worker.simple_wall_name.setdefault(a, {})
        belief_dict.setdefault(a, {})

        if l not in worker.simple_heading[a]:
            # Checked before anything else commits: a robot whose arena has a
            # reset pending is skipped entirely this tick rather than
            # initialized with any value. Any value written now is wrong the
            # moment the real spawn happens, and the once-only "if l not in
            # worker.simple_heading[a]" check below would never revisit it.
            # Leaving the robot out of every worker.simple_* dict is what lets
            # that check fire again, correctly, next tick.
            if hasattr(worker, "reset_pending") and worker.reset_pending(a):
                skip_tick.add(i)
                continue
            # Known start plus self-integration: a physically legitimate setup
            # convention, never a live read of Unity's internal state.
            # worker.spawn_heading(a, l) reports which of the four cardinal
            # headings (belief.CARDINAL_HEADINGS) this robot was actually given,
            # falling back to KNOWN_START_HEADING when nothing is available --
            # known_start_heading off, or a player predating the per-robot
            # observation column.
            start_heading = KNOWN_START_HEADING
            if hasattr(worker, "spawn_heading"):
                sh = worker.spawn_heading(a, l)
                if sh is not None:
                    start_heading = sh
            worker.simple_heading[a][l] = start_heading
            worker.simple_state[a][l] = "go_north"
            worker.simple_turn_accum[a][l] = 0.0
            image_id = worker.image_id.get(a)
            # The hash-based, launch-decentralized target
            # selection (spatial_hash.py) -- a robot computes its own
            # target from (the shared formation, its own local index)
            # alone, no coordination or occupancy-checking with any other
            # robot. Lives in observation.ensure_target,
            # shared with the actor's own relative target vector, rather
            # than kept as a copy inline here -- same cache
            # (worker.simple_target), so the two can never disagree.
            ensure_target(worker, a, l, image_id, formation_pool)
            belief_dict[a][l] = belief_init(1, rng, particles=BELIEF_PARTICLES, device=device,
                                            known_start_heading=True).squeeze(0)
            if belief_dict[a][l].shape[0] > 0:
                belief_dict[a][l][:, 2] = start_heading

            # This state machine assumes every robot genuinely spawns at its
            # reported spawn_heading in Unity. If that is false -- e.g.
            # SwarmManager's knownStartHeading is off in the build being run --
            # every downstream use of the tracked heading is silently wrong, and
            # the per-transition logging cannot reveal it, since that logging
            # only ever shows the assumed heading. Hence this check.
            if getattr(cfg, "oracle_debug_wall_log", False):
                true_heading_deg = _true_heading_deg(worker, a, l)
                assumed_deg = math.degrees(start_heading) % 360.0
                if true_heading_deg is None:
                    print(f"SIMPLE_ORACLE_SPAWN_CHECK arena={a} robot={l} assumed_deg={assumed_deg:.1f} "
                         f"true_deg=unavailable (worker.snapshot has no usable node data here)")
                else:
                    err = ((true_heading_deg - assumed_deg) + 180.0) % 360.0 - 180.0
                    flag = " <-- MISMATCH" if abs(err) > 5.0 else ""
                    print(f"SIMPLE_ORACLE_SPAWN_CHECK arena={a} robot={l} assumed_deg={assumed_deg:.1f} "
                         f"true_deg={true_heading_deg:.1f} err_deg={err:.1f}{flag}")

        # this robot's own, real motion since its last decision -- gathered
        # here, computed in one batched call below (not per robot)
        last_motor = worker.last_motor.get(a, {}).get(l)
        step_count = worker.step_count.get(a, 0)
        steps_since = step_count - worker.last_dec_step.get(a, {}).get(l, step_count)
        if last_motor is not None and steps_since > 0:
            lm = last_motor if isinstance(last_motor, torch.Tensor) else torch.tensor(last_motor)
            last_motor_batch[i] = lm.to(torch.float32)
            steps_batch[i] = float(steps_since)
        # else: leave at zero -- steps=0 forces dtheta=0 regardless of
        # motor value (a robot's very first decision, before it has ever
        # moved), so the placeholder motor value here is never read

        # The spawn gate above only initializes belief_dict[a][l] on a robot's
        # first tick under this exact belief_attr. A second belief_attr reaching
        # a robot some other one already spawned skips that block entirely --
        # simple_heading is shared across belief_attrs and already has it -- yet
        # never gets a particle set of its own, raising KeyError below. Checked
        # independently here: the already-known simple_heading seeds a particle
        # set the same way the spawn path does, on whichever later tick this
        # belief_attr first reaches the robot.
        if l not in belief_dict[a]:
            belief_dict[a][l] = belief_init(1, rng, particles=BELIEF_PARTICLES, device=device,
                                            known_start_heading=True).squeeze(0)
            if belief_dict[a][l].shape[0] > 0:
                belief_dict[a][l][:, 2] = worker.simple_heading[a][l]
        particles_batch[i] = belief_dict[a][l].to(device)

    # --- batched, expensive tensor operations: ONE call each across all n robots ---
    x_local, y_local, dtheta, _t = split_tick_motion(
        last_motor_batch, steps_batch, cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed)

    # keep the particle filter's own internal heading in exact sync with
    # this same, independently-computed dtheta (heading_noise_scale=0)
    # purely for its own internal consistency -- this oracle's steering
    # never reads particles[:, :, 2] itself, see this file's own module
    # docstring for why
    particles_batch = belief_predict(particles_batch, x_local, y_local, dtheta, rng,
                                     heading_noise_scale=HEADING_NOISE_SCALE)
    seed_obs = torch.zeros(n, SEED_SIZE, device=device)
    particles_batch = belief_update(particles_batch, seed_obs, rng, wall_obs=walls,
                                    wall_seed_xy=wall_seed_xy, heading_noise_scale=HEADING_NOISE_SCALE)
    conf_batch = belief_conf(particles_batch)
    br_batch = belief_read(particles_batch)

    # --- pass 2: cheap per-robot state machine, using the already-batched results ---
    for i in range(n):
        if i in skip_tick:
            # This robot was never initialized
            # in pass 1 above (its arena's reset was still pending) -- every
            # worker.simple_* dict genuinely has no entry for it this tick,
            # so touching any of them here would raise, not just read a
            # stale value. motor[i] stays at its pre-allocated zero (a
            # harmless stop for exactly one tick), and next tick's pass 1
            # re-checks reset_pending, hopefully now false.
            continue
        a = int(arena_ids[i]); l = int(locals_[i])
        belief_dict[a][l] = particles_batch[i].detach().clone()

        worker.simple_heading[a][l] += float(dtheta[i])
        heading = worker.simple_heading[a][l]
        h_vec = np.array([math.cos(heading), math.sin(heading)])
        state = worker.simple_state[a][l]

        if state == "go_north":
            motor[i] = torch.tensor([1.0, 1.0])
            if float(walls[i].sum()) > 0:
                wall_idx = int(walls[i].argmax())
                worker.simple_wall_name[a][l] = WALL_NAMES[wall_idx]
                worker.simple_state[a][l] = "turning"
                worker.simple_turn_accum[a][l] = 0.0
            elif getattr(cfg, "oracle_debug_wall_log", False):
                # Samples every decision made while still in go_north, to
                # distinguish a smooth heading-error ramp (ongoing drift under a
                # constant motor command) from discrete jumps (collisions).
                # Not throttled beyond the decision cadence: an event-quiet
                # robot only decides once per heartbeat_ticks anyway.
                true_heading_deg = _true_heading_deg(worker, a, l)
                if true_heading_deg is not None:
                    assumed_deg = math.degrees(heading) % 360.0
                    h_err = ((true_heading_deg - assumed_deg) + 180.0) % 360.0 - 180.0
                    print(f"SIMPLE_ORACLE_GONORTH arena={a} robot={l} tick={worker.step_count.get(a, 0)} "
                         f"heading_err_deg={h_err:.1f}")

        elif state == "turning":
            worker.simple_turn_accum[a][l] += abs(float(dtheta[i]))
            motor[i] = torch.tensor(TURN_MOTOR)
            if worker.simple_turn_accum[a][l] >= TURN_TARGET_RAD:
                worker.simple_state[a][l] = "wall_following"

        elif state == "wall_following":
            g_vec = WALL_TANGENT[worker.simple_wall_name[a][l]]
            conf_i = float(conf_batch[i])
            if conf_i <= APPROACH_SLOWDOWN_CONF:
                speed_scale = 1.0
            else:
                frac = min((conf_i - APPROACH_SLOWDOWN_CONF) /
                          (LOCALIZED_CONF_THRESHOLD - APPROACH_SLOWDOWN_CONF), 1.0)
                speed_scale = 1.0 - frac * (1.0 - APPROACH_SLOWDOWN_MIN_SCALE)
            motor[i] = torch.tensor(_steer(h_vec, g_vec)) * speed_scale
            if conf_i >= LOCALIZED_CONF_THRESHOLD:
                worker.simple_state[a][l] = "navigating"

        elif state == "navigating":
            est_pos = np.array([float(br_batch[i, 0]), float(br_batch[i, 1])]) * ARENA_HALF
            target = worker.simple_target[a][l]
            delta = target - est_pos
            dist = float(np.linalg.norm(delta))
            if dist < cfg.tau_v * ARENA_HALF:
                worker.simple_state[a][l] = "arrived"
                motor[i] = torch.tensor([0.0, 0.0])
                # Arrival is decided from belief alone (est_pos vs target),
                # correct by design since the oracle has no other position
                # source -- so it cannot distinguish "arrived and correct" from
                # "arrived on a belief that was never quite right". This reads
                # Unity's real position diagnostically, to report that gap,
                # never to steer. Reuses formations.Formation.dist_dir and
                # cfg.tau_v, the same on-shape threshold reward.py uses, rather
                # than inventing a separate notion of close enough.
                if getattr(cfg, "oracle_debug_wall_log", False):
                    true_pos, _true_h = _true_pose(worker, a, l)
                    if true_pos is not None:
                        target_err = float(np.linalg.norm(true_pos - target))
                        image_id = worker.image_id.get(a)
                        form = formation_pool[image_id % len(formation_pool)]
                        shape_dist_norm, _dir = form.dist_dir(true_pos.reshape(1, 2))
                        shape_dist = float(shape_dist_norm[0]) * ARENA_HALF
                        on_shape = float(shape_dist_norm[0]) <= cfg.tau_v
                        flag = "" if on_shape else " <-- OFF-SHAPE"
                        print(f"SIMPLE_ORACLE_ARRIVAL_CHECK arena={a} robot={l} "
                             f"true_pos=({true_pos[0]:.1f},{true_pos[1]:.1f}) "
                             f"target=({target[0]:.1f},{target[1]:.1f}) target_err={target_err:.1f} "
                             f"dist_to_shape={shape_dist:.1f}{flag}")
            else:
                g_vec = delta / dist if dist > 1e-9 else h_vec
                motor[i] = torch.tensor(_steer(h_vec, g_vec))

        elif state == "arrived":
            motor[i] = torch.tensor([0.0, 0.0])

        # config.py's own bc_replay_capacity has the full rationale. This is
        # the state that actually PRODUCED motor[i], which is not always the
        # state the robot is in once the branch above finishes: go_north and
        # turning both command their own motor and only then transition, so
        # reading worker.simple_state after this point labels those two ticks
        # with the state they are leaving for, not the one whose command is
        # being learned. navigating is the one case where the transition also
        # replaces the motor (with arrived's own [0, 0]), so it is the one
        # case where the post-transition state is the correct label.
        label = state
        if state == "navigating" and worker.simple_state[a][l] == "arrived":
            label = "arrived"
        if not hasattr(worker, "simple_motor_state"):
            worker.simple_motor_state = {}
        worker.simple_motor_state.setdefault(a, {})[l] = label

        # Under oracle_debug_wall_log, this project's general verbose-oracle
        # toggle. Logs every state TRANSITION, not every tick: a five-state
        # machine changes state a handful of times per robot per episode, which
        # stays readable over a watch session where per-tick output would not.
        #
        # heading_err_deg is read at every transition rather than once, so a
        # roughly constant error confirms the gap opened before this file got a
        # decision to make and the accumulation from there is sound, while a
        # growing one points to a bug in the tracking itself.
        if getattr(cfg, "oracle_debug_wall_log", False):
            new_state = worker.simple_state[a][l]
            if new_state != state:
                est_pos = np.array([float(br_batch[i, 0]), float(br_batch[i, 1])]) * ARENA_HALF
                msg = (f"SIMPLE_ORACLE arena={a} robot={l} tick={worker.step_count.get(a, 0)} "
                      f"{state}->{new_state} heading_deg={math.degrees(heading) % 360.0:.1f} "
                      f"est_pos=({est_pos[0]:.1f},{est_pos[1]:.1f}) conf={float(conf_batch[i]):.2f}")
                arenas = getattr(worker, "arenas", None)
                if arenas is not None:
                    true_pos = arenas[a].pos[l]
                    msg += (f" true_pos=({true_pos[0]:.1f},{true_pos[1]:.1f}) "
                           f"err={float(np.linalg.norm(true_pos - est_pos)):.1f}")
                try:
                    true_heading_deg = _true_heading_deg(worker, a, l)
                    if true_heading_deg is not None:
                        assumed_deg = math.degrees(heading) % 360.0
                        h_err = ((true_heading_deg - assumed_deg) + 180.0) % 360.0 - 180.0
                        msg += f" heading_err_deg={h_err:.1f}"
                except Exception:
                    pass
                print(msg)

        if getattr(cfg, "oracle_send_visual_state", False):
            if not hasattr(worker, "oracle_visual_state"):
                worker.oracle_visual_state = {}
            # mirrors the existing 0..4 convention (KilobotAgent.
            # SetVisualState: ivory/amber/gold/
            # deep red/green). turning is off go_north's
            # color (0) onto its own slot (2) -- sharing a color made the
            # two states indistinguishable when watching a real Unity run,
            # which is specifically what this flag is for
            vstate = {"go_north": 0, "turning": 2, "wall_following": 1,
                     "navigating": 3, "arrived": 4}[worker.simple_state[a][l]]
            worker.oracle_visual_state.setdefault(a, {})[l] = vstate

    # A periodic population-wide summary alongside the
    # per-transition lines above -- throttled (every 50 calls, not every
    # one) since this scans every robot the worker has ever tracked, unlike
    # the per-transition log which only prints on a real change
    if getattr(cfg, "oracle_debug_wall_log", False):
        worker._simple_oracle_debug_calls = getattr(worker, "_simple_oracle_debug_calls", 0) + 1
        if worker._simple_oracle_debug_calls % 50 == 1:
            counts = {}
            for robots in worker.simple_state.values():
                for st in robots.values():
                    counts[st] = counts.get(st, 0) + 1
            print(f"SIMPLE_ORACLE_SUMMARY call={worker._simple_oracle_debug_calls} state_counts={counts}")

    return motor
