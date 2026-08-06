"""Turning a worker's raw observation into what the actor and oracle read.

Everything between "ml-agents handed us a decision batch" and "the actor has
its input tensors" lives here: unpacking the flat observation, sampling the
one neighbour message a single-receiver robot could actually have heard,
advancing each robot's belief filter, and assembling the per-actor state
(database rows for the DeepSet actor, the split observation for the recurrent
one). actor_io.act() drives all of it; simple_oracle.py reuses the same
primitives so its own view can never disagree with the actor's.

Nothing here decides anything -- no steering, no targets, no motor commands.
Those live in actor_io.py (the act
loop). Names still resolve from actor_io for backward compatibility.
"""

import math
import numpy as np
import torch

from belief import (belief_init, belief_predict, belief_update, belief_read, belief_track_anchor,
                    ANCHOR_STATE_SIZE, BELIEF_PARTICLES, ARENA_HALF, belief_conf,
                    LOCALIZED_CONF_THRESHOLD, SEED_LAYOUTS, TRIANGULATE_MIN_READINGS,
                    HEADING_NOISE_SCALE, IR_RANGE, WALL_SPACING, _matched_generator)
from kilobot_gnn import SEED_SIZE, WALL_SIZE, MESSAGE_SIZE, MOTOR_SIZE, DB_ROW_SIZE, DB_CAPACITY, NODE_FEATURES, priv_cols, SPLIT_SEED_OFFSET, SPLIT_WALL_OFFSET
from kinematics import dead_reckon, split_tick_motion, split_track_update, split_track_read
from spatial_hash import hilbert_order, mix_hash, assign_target_index


STRENGTH_COL = MESSAGE_SIZE + 1


# Number of points sampled from a formation
# before ordering them along the Hilbert curve for target assignment.
# Duplicated from simple_oracle.py's own original constant of the same
# name and value (300), not re-derived -- ensure_target below replaces
# that module's own inline copy of this whole mechanism, so the constant
# moved here with it.
TARGET_POOL_SIZE = 300


def ensure_target(worker, a, l, image_id, formation_pool):
    # The one shared, stateful path for a robot's assigned target point. Both
    # simple_oracle.py's steering and gather_split_state's relative target
    # vector read the same worker.simple_target cache, so the two can never
    # disagree about which point a robot was assigned. Resolved once per robot
    # per episode, then reused from the cache.
    if not hasattr(worker, "simple_target"):
        worker.simple_target = {}
    if not hasattr(worker, "simple_hilbert_order"):
        worker.simple_hilbert_order = {}
    worker.simple_target.setdefault(a, {})
    if l in worker.simple_target[a]:
        return worker.simple_target[a][l]
    form = formation_pool[image_id % len(formation_pool)]
    points = form.sample_points(TARGET_POOL_SIZE)
    cached = worker.simple_hilbert_order.get(a)
    if cached is None or len(cached) != len(points):
        worker.simple_hilbert_order[a] = hilbert_order(points)
    order = worker.simple_hilbert_order[a]
    idx = assign_target_index(order, l, image_id)
    worker.simple_target[a][l] = points[idx]
    return worker.simple_target[a][l]



def split_obs(obs, device):
    vector = None
    rows = None
    for ob in obs:
        if ob.ndim == 2:
            vector = ob
        elif ob.ndim == 3:
            rows = ob
    vector = torch.tensor(vector, dtype = torch.float32, device = device)
    rows = torch.tensor(rows, dtype = torch.float32, device = device)
    return vector, rows


def gather_databases(worker, arena_ids, locals_, device):
    n = arena_ids.shape[0]
    rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE, device = device)
    valid = torch.zeros(n, DB_CAPACITY, dtype = torch.bool, device = device)
    for i in range(n):
        entry = worker.databases[int(arena_ids[i])].get(int(locals_[i]))
        if entry is not None:
            rows[i] = entry[0].to(device)
            valid[i] = entry[1].to(device)
    return rows, valid


def gather_nodes(worker, arena_ids, locals_, n, device):
    # Per decision-robot node feature row, aligned with decision order.
    out = torch.zeros((n, NODE_FEATURES))
    cache = {}
    for idx in range(n):
        a = int(arena_ids[idx])
        if a not in cache:
            snap = worker.snapshot(a)
            cache[a] = snap["node"] if snap is not None else None
        nd = cache[a]
        l = int(locals_[idx])
        if nd is not None and l < nd.shape[0]:
            out[idx] = nd[l]
    return out.to(device)


def gather_gru_state(worker, arena_ids, locals_, cfg):
    # Returns h_prev (n,HIDDEN), prop_b (n,PROP_SIZE) dead-reckoned from the command each
    # robot held since its last decision, and raw_path (n,) the interval path length used
    # to advance each robot's cumulative-distance odometer.
    n = arena_ids.shape[0]
    device = cfg.device
    hidden = cfg.gru_hidden
    h_prev = torch.zeros(n, hidden, device=device)
    last_motor = torch.zeros(n, 2, device=device)
    steps = torch.zeros(n, device=device)
    cum = torch.zeros(n, device=device)
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        hv = worker.hidden[a].get(l)
        if hv is not None:
            h_prev[i] = hv.to(device)
        lm = worker.last_motor[a].get(l)
        if lm is not None:
            last_motor[i] = lm.to(device)
        steps[i] = float(worker.step_count[a] - worker.last_dec_step[a].get(l, worker.step_count[a]))
        cum[i] = worker.odometer[a].get(l, 0.0)
    prop_b = dead_reckon(last_motor, steps, cum, cfg.prop_max_speed,
                         cfg.prop_wheelbase, cfg.dt_fixed, cfg.prop_scale,
                         cfg.prop_time_scale, cfg.prop_cum_scale)
    v = 0.5 * (last_motor[:, 0] + last_motor[:, 1]) * cfg.prop_max_speed
    raw_path = v * steps * cfg.dt_fixed
    return h_prev, prop_b, raw_path


# Minimum ticks between one anchor event and the next, so gather_split_state's
# rising-edge check treats a nonzero walls_b reading as a genuine new wall
# approach rather than the same reception-lottery-subject signal blipping to
# zero mid-turn.
#
# Ticks rather than a count of consecutive zero-decisions: no fixed count is
# safe against a long enough streak of lost lottery draws, whereas an elapsed
# time only has to outlast a real turn, whose duration is fixed by physics
# (~33 ticks for pi/2 at TURN_MOTOR's rate). 40 clears that comfortably while
# staying under heartbeat_ticks' 48. Raising it to 60 does not improve matters;
# the residual error is config.py's documented signed-angle-vs-turn_accum
# limitation, not a re-anchoring timing issue.
TURN_ANCHOR_REFRACTORY_TICKS = 40
# The tick refractory alone is not sufficient. An oracle turn takes exactly 35
# ticks, so during BC the anchor is latched for every turn it ever sees -- it
# physically cannot re-fire mid-turn, and every training sample carries a clean
# monotonic rotation ramp.
#
# At deployment nothing ends the turn at 35 ticks. A turn overrunning 40 gets
# re-anchored by the next wall reading, its implied rotation snaps back toward
# zero, and the actor sees "just started turning" again -- a one-way trap the
# first overshoot makes inescapable.
#
# The latch re-arms only once rotation since the anchor reaches the oracle's
# turn target, which is what simple_oracle.py itself does: it checks the wall
# condition once on the way OUT of go_north and never re-checks it inside
# turning. Deliberately a no-op on the BC training distribution -- it diverges
# from the tick-only rule only for a turn past 40 ticks, which the oracle never
# produces -- which is what makes it safe on an already-trained actor.
# See docs/code-history.md.
TURN_ANCHOR_LATCH_RAD = math.pi / 2.0
# Escape hatch. If the belief heading is bad enough that the implied rotation
# never reaches the target, a pure rotation latch would hold the anchor
# forever and a genuine later turn could never anchor at all. After this many
# ticks the anchor re-arms regardless. Generous on purpose -- roughly five
# times a real turn -- since it exists only to bound a pathological case.
TURN_ANCHOR_MAX_LATCH_TICKS = 200


def gather_split_state(worker, arena_ids, locals_, seeds_b, walls_b, cfg, rng, rows = None, valid = None,
                       wall_seed_xy = None, arrived_claim = None):
    n = arena_ids.shape[0]
    device = cfg.device
    hidden = cfg.split_gru_hidden
    h_prev = torch.zeros(n, hidden, device=device)
    last_motor = torch.zeros(n, 2, device=device)
    steps = torch.zeros(n, device=device)
    track_neighbor = torch.zeros(n, 4, device=device)
    track_seed = torch.zeros(n, 4, device=device)
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        hv = worker.hidden[a].get(l)
        if hv is not None:
            h_prev[i] = hv.to(device)
        lm = worker.last_motor[a].get(l)
        if lm is not None:
            last_motor[i] = lm.to(device)
        steps[i] = float(worker.step_count[a] - worker.last_dec_step[a].get(l, worker.step_count[a]))
        tn = worker.track_neighbor[a].get(l)
        if tn is not None:
            track_neighbor[i] = tn.to(device)
        ts = worker.track_seed[a].get(l)
        if ts is not None:
            track_seed[i] = ts.to(device)
    particles = torch.zeros(n, BELIEF_PARTICLES, 3, device = device)
    known_start_heading = getattr(cfg, "oracle_known_start_heading", False)
    fresh = belief_init(n, rng, device = device, known_start_heading = known_start_heading)
    if not hasattr(worker, "belief_fit"):
        worker.belief_fit = {}
    if not hasattr(worker, "belief_anchor"):
        worker.belief_anchor = {}
    fit_batch = torch.ones(n, 2, device = device)   # 1.0 = uninitialized sentinel, see belief_update
    anchor_batch = torch.zeros(n, ANCHOR_STATE_SIZE, device = device)   # [valid, anchor_x, anchor_y, af_x, af_y, rel_theta, n_readings, probe_accum...] -- see belief_triangulate
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        bp = worker.belief[a].get(l)
        if bp is None and known_start_heading and hasattr(worker, "spawn_heading"):
            # belief_init's known_start_heading path sets every row to the
            # same KNOWN_START_HEADING; this robot-specific overwrite is what
            # applies the per-robot draw from belief.CARDINAL_HEADINGS instead,
            # mirroring simple_oracle.py's overwrite for its own cloud. Left at
            # belief_init's single value when spawn_heading reports nothing.
            sh = worker.spawn_heading(a, l)
            if sh is not None:
                fresh[i, :, 2] = sh
        particles[i] = fresh[i] if bp is None else bp.to(device)
        bf = worker.belief_fit.get(a, {}).get(l)
        if bf is not None:
            fit_batch[i] = bf.to(device)
        ba = worker.belief_anchor.get(a, {}).get(l)
        if ba is not None:
            anchor_batch[i] = ba.to(device)
    x_local, y_local, dtheta, t = split_tick_motion(last_motor, steps, cfg.prop_max_speed,
                                                     cfg.prop_wheelbase, cfg.dt_fixed)
    track_neighbor = split_track_update(track_neighbor, x_local, y_local, dtheta, t)
    track_seed = split_track_update(track_seed, x_local, y_local, dtheta, t)
    anchor_batch = belief_track_anchor(anchor_batch, x_local, y_local, dtheta)
    debug_on = getattr(cfg, "oracle_debug_wall_log", False)
    if debug_on:
        pre_predict_mean = particles[:, :, :2].mean(dim=1).clone()
        pre_predict_heading = particles[:, :, 2].clone()
    particles = belief_predict(particles, x_local, y_local, dtheta, rng,
                               heading_noise_scale = HEADING_NOISE_SCALE if getattr(cfg, "oracle_known_start_heading", False) else None)
    if debug_on:
        post_predict_mean = particles[:, :, :2].mean(dim=1).clone()
        post_predict_heading = particles[:, :, 2].clone()
        # Diagnostic for whether the filter's tracked heading matches reality,
        # not merely internal consistency. Reconstructs what heading would have
        # been needed at the start of this tick's motion to turn the known
        # local-frame displacement into the observed global displacement.
        # Comparing that against the filter's pre-tick heading measures
        # accumulated drift using true POSITION alone, differenced across two
        # ticks -- no privileged true_heading required. Every particle shares
        # one heading when heading_noise_scale is active, so
        # pre_predict_heading[i, 0] IS that shared value.
        import math as _math
        if not hasattr(worker, "_drift_debug_last_true_pos"):
            worker._drift_debug_last_true_pos = {}
            worker._drift_debug_cum_dtheta = {}
        for i in range(n):
            a = int(arena_ids[i]); l = int(locals_[i])
            true_pos_now = None
            try:
                snap = worker.snapshot(a)
                if snap is not None and snap.get("node") is not None and l < snap["node"].shape[0]:
                    node_arr = snap["node"]
                    node_arr = node_arr.cpu().numpy() if hasattr(node_arr, "cpu") else np.asarray(node_arr)
                    true_pos_now = node_arr[l, 0:2] * ARENA_HALF
            except Exception:
                true_pos_now = None
            cache = worker._drift_debug_last_true_pos.setdefault(a, {})
            cum = worker._drift_debug_cum_dtheta.setdefault(a, {})
            cum[l] = cum.get(l, 0.0) + abs(float(dtheta[i]))
            if true_pos_now is not None and l in cache:
                true_delta = true_pos_now - cache[l]
                true_speed = float(np.linalg.norm(true_delta))
                xl, yl = float(x_local[i]), float(y_local[i])
                local_speed = _math.hypot(xl, yl)
                # both must be genuinely moving -- near-zero vectors make
                # atan2 numerically meaningless, not just noisy
                if true_speed > 0.05 and local_speed > 1e-4:
                    true_angle = _math.atan2(true_delta[1], true_delta[0])
                    local_angle = _math.atan2(yl, xl)
                    implied_heading = true_angle - local_angle
                    filter_heading = float(pre_predict_heading[i, 0])
                    drift_err = ((filter_heading - implied_heading) + _math.pi) % (2 * _math.pi) - _math.pi
                    is_turning = bool(getattr(worker, "oracle_ex_turning", {}).get(a, {}).get(l, False))
                    lock = getattr(worker, "oracle_lock", {}).get(a, {}).get(l)
                    conf_now = float(belief_conf(particles[i].unsqueeze(0))[0])
                    print("HEADING_DRIFT_DEBUG tick=%s arena=%d robot=%d filter_heading=%.4f implied_heading=%.4f "
                          "drift_err_deg=%.2f dtheta_this=%.4f cum_dtheta_mag=%.3f true_speed=%.3f local_speed=%.3f "
                          "is_turning=%s lock=%s belief_conf=%.4f" % (
                              int(worker.step_count.get(a, 0)), a, l, filter_heading, implied_heading,
                              _math.degrees(drift_err), float(dtheta[i]), cum[l], true_speed, local_speed,
                              is_turning, lock, conf_now))
            if true_pos_now is not None:
                cache[l] = true_pos_now
        # A second, independent diagnostic, deliberately not built on the same
        # mechanism as the one above: that one reconstructs heading indirectly
        # from two position snapshots and was shown to give wildly inconsistent
        # results on otherwise-clean examples. This reads true heading DIRECTLY
        # (node[:, 2:4]) -- no reconstruction, no two-snapshot timing to get
        # wrong. Purely diagnostic, never fed into steering or the belief
        # update. Reads the same snapshot object the diagnostic above already
        # fetched, so the two cannot disagree about which tick they describe.
        if not hasattr(worker, "_true_heading_debug_last"):
            worker._true_heading_debug_last = {}
        if true_pos_now is not None:
            true_heading_now = float(_math.atan2(node_arr[l, 3], node_arr[l, 2]))
            tcache = worker._true_heading_debug_last.setdefault(a, {})
            filter_heading = float(pre_predict_heading[i, 0])
            total_err = ((filter_heading - true_heading_now) + _math.pi) % (2 * _math.pi) - _math.pi
            per_tick_err = None
            if l in tcache:
                # true_heading_now and tcache[l] are both atan2 outputs,
                # each independently wrapped to [-pi, pi] -- their naive
                # difference can be off by any multiple of 2*pi from the
                # true rotation that happened between the two reads.
                # Simply wrapping that difference to [-pi, pi] would only
                # be correct for a single-tick-sized rotation; dtheta_this
                # itself can legitimately be many full turns (a robot can
                # go a long time between decisions while continuously
                # turning), so use it as a prior to pick the multiple of
                # 2*pi closest to what was actually commanded, rather than
                # assuming the shortest path is always the right one
                raw_diff = true_heading_now - tcache[l]
                k = round((float(dtheta[i]) - raw_diff) / (2 * _math.pi))
                true_dtheta_since_last = raw_diff + k * 2 * _math.pi
                per_tick_err_rad = ((float(dtheta[i]) - true_dtheta_since_last) + _math.pi) % (2 * _math.pi) - _math.pi
                per_tick_err = _math.degrees(per_tick_err_rad)
            tcache[l] = true_heading_now
            print("TRUE_HEADING_DEBUG tick=%s arena=%d robot=%d filter_heading=%.4f true_heading=%.4f "
                  "total_err_deg=%.2f dtheta_this=%.5f per_tick_err_deg=%s" % (
                      int(worker.step_count.get(a, 0)), a, l, filter_heading, true_heading_now,
                      _math.degrees(total_err), float(dtheta[i]),
                      ("%.4f" % per_tick_err) if per_tick_err is not None else "None"))
    peer_pos = None
    peer_conf = None
    peer_strength = None
    if rows is not None and getattr(cfg, "belief_comms", False):
        rows_d = rows.to(device)
        peer_pos = rows_d[:, :, 0:2]
        peer_conf = rows_d[:, :, 2].clamp(0.0, 1.0) * valid.to(device)
        peer_strength = rows_d[:, :, STRENGTH_COL] * valid.to(device)
    ac_pos = ac_conf = ac_valid = ac_strength = None
    if arrived_claim is not None:
        ac_pos, ac_conf, ac_valid, ac_strength = arrived_claim
    particles = belief_update(particles, seeds_b.to(device), rng,
                              peer_pos = peer_pos, peer_conf = peer_conf, peer_strength = peer_strength,
                              wall_obs = walls_b.to(device),
                              fit_ema = fit_batch,
                              anchor = anchor_batch if getattr(cfg, "oracle_heading_triangulation", False) else None,
                              wall_seed_xy = wall_seed_xy.to(device) if wall_seed_xy is not None else None,
                              arrived_claim_pos = ac_pos, arrived_claim_conf = ac_conf,
                              arrived_claim_valid = ac_valid, arrived_claim_strength = ac_strength,
                              heading_noise_scale = HEADING_NOISE_SCALE if getattr(cfg, "oracle_known_start_heading", False) else None)
    if debug_on:
        post_update_mean = particles[:, :, :2].mean(dim=1).clone()
        post_update_heading = particles[:, :, 2].clone()
        if not hasattr(worker, "_predict_update_debug_seen"):
            worker._predict_update_debug_seen = {}
        for i in range(n):
            a = int(arena_ids[i]); l = int(locals_[i])
            if not (hasattr(worker, "oracle_ever_localized") and worker.oracle_ever_localized.get(a, {}).get(l, False)):
                continue
            pukey = (a, l)
            pulast = worker._predict_update_debug_seen.get(pukey)
            cur_tick = int(worker.step_count[a])
            if pulast is not None and (cur_tick - pulast) < 25:
                continue
            worker._predict_update_debug_seen[pukey] = cur_tick
            predict_delta = float((post_predict_mean[i] - pre_predict_mean[i]).norm()) * ARENA_HALF
            update_delta = float((post_update_mean[i] - post_predict_mean[i]).norm()) * ARENA_HALF
            # Heading analogue of update_delta above, added in response to
            # TRUE_HEADING_DEBUG
            # showing per-tick dtheta accuracy is mostly fine (small,
            # confirmed via positive control) while total_err_deg still
            # jumps in large, discrete, non-accumulating steps -- pointing
            # at belief_update's own injection/rescue mechanisms rather
            # than the kinematic dtheta computation. This isolates exactly
            # that: how much did THIS tick's belief_update call change
            # heading, independent of belief_predict's own dtheta
            # contribution (already accounted for by comparing against
            # post_predict_heading, not pre_predict_heading)
            heading_update_delta_rad = float(((post_update_heading[i, 0] - post_predict_heading[i, 0]) + _math.pi) % (2 * _math.pi) - _math.pi)
            has_seed = bool((seeds_b[i] > 0).any())
            has_wall = bool((walls_b[i] > 0).any())
            has_peer = bool(peer_pos is not None and (peer_strength[i] > 0).any() and (peer_conf[i] > 0.85).any())
            peer_note = ""
            if has_peer:
                usable_i = (peer_strength[i] > 0) & (peer_conf[i] > 0.85)
                best_i = int(torch.where(usable_i, peer_conf[i], torch.full_like(peer_conf[i], -1.0)).argmax())
                bx = float(peer_pos[i, best_i, 0]) * ARENA_HALF
                by = float(peer_pos[i, best_i, 1]) * ARENA_HALF
                bstr = float(peer_strength[i, best_i])
                d_meas = (1.0 / max(bstr, 1e-6) - 1.0)
                peer_note = " peer_broadcast_pos=[%.2f, %.2f] peer_strength=%.4f peer_d_meas=%.2f" % (bx, by, bstr, d_meas)
            print("PREDICT_UPDATE_DEBUG tick=%s arena=%d robot=%d predict_delta=%.3f update_delta=%.3f "
                  "heading_update_delta_deg=%.2f has_seed=%s has_wall=%s has_peer=%s%s" % (
                      cur_tick, a, l, predict_delta, update_delta, _math.degrees(heading_update_delta_rad),
                      has_seed, has_wall, has_peer, peer_note))
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        # Diagnostic ablation (oracle_perfect_heading): overwrites ONLY
        # heading with the true value, leaving position untouched, so the robot
        # still spawns uniformly uncertain and still needs genuine landmark
        # contact to narrow. That isolates heading-tracking accuracy from
        # position localization. Overwriting position too would make belief_conf
        # -- which depends only on position spread -- read 1.0 from the first
        # tick, skipping exploration-until-localized entirely.
        #
        # Applied after both belief_predict and belief_update, so it is
        # unconditionally the true value regardless of what either did to
        # heading. Falls back to whatever they computed when true state is not
        # yet available, rather than writing garbage.
        if getattr(cfg, "oracle_perfect_heading", False):
            import math as _po_math
            try:
                snap = worker.snapshot(a)
                if snap is not None and snap.get("node") is not None and l < snap["node"].shape[0]:
                    node_arr = snap["node"]
                    node_arr = node_arr.cpu().numpy() if hasattr(node_arr, "cpu") else np.asarray(node_arr)
                    true_heading = float(_po_math.atan2(node_arr[l, 3], node_arr[l, 2]))
                    particles[i, :, 2] = true_heading
            except Exception:
                pass
        worker.track_neighbor[a][l] = track_neighbor[i].detach().clone()
        worker.track_seed[a][l] = track_seed[i].detach().clone()
        worker.belief[a][l] = particles[i].detach().clone()
        worker.belief_fit.setdefault(a, {})[l] = fit_batch[i].detach().clone()
        worker.belief_anchor.setdefault(a, {})[l] = anchor_batch[i].detach().clone()
    scale = cfg.split_prop_scale
    time_scale = cfg.split_prop_time_scale
    prop_neighbor = split_track_read(track_neighbor, scale, time_scale)
    prop_seed = split_track_read(track_seed, scale, time_scale)
    # Each robot's relative bearing/distance to its assigned target point,
    # resolved once via the shared ensure_target and recomputed each tick into
    # an egocentric, per-particle-averaged form by belief_read's target= path.
    #
    # KNOWN GAP: formation_pool (cfg._oracle_formation_pool) is only set by the
    # BC entry points. When it or an arena's image_id is unavailable this
    # degrades to zeros for those robots rather than feeding a misleading
    # placeholder position into the bearing/distance math. Wiring it into every
    # gru_split_observation entry point is outstanding work.
    formation_pool = getattr(cfg, "_oracle_formation_pool", None)
    target_batch = torch.zeros(n, 2, device = device)
    has_target = torch.zeros(n, dtype = torch.bool, device = device)
    if formation_pool:
        for i in range(n):
            a = int(arena_ids[i]); l = int(locals_[i])
            image_id = worker.image_id.get(a) if hasattr(worker, "image_id") else None
            if image_id is None:
                continue
            tgt = ensure_target(worker, a, l, image_id, formation_pool)
            target_batch[i, 0] = float(tgt[0]) / ARENA_HALF
            target_batch[i, 1] = float(tgt[1]) / ARENA_HALF
            has_target[i] = True
    belief_out = belief_read(particles, target = target_batch)
    belief_out[:, -3:] = torch.where(has_target.unsqueeze(1), belief_out[:, -3:],
                                     torch.zeros_like(belief_out[:, -3:]))
    prop_b = torch.cat([prop_neighbor, prop_seed, belief_out], dim = 1)
    if getattr(cfg, "use_turn_anchor", False):
        # config.py's own use_turn_anchor has the full rationale. sin_now,
        # cos_now here are belief_out's own columns 2 and 3 exactly --
        # the same (sin_m/r, cos_m/r) absolute-heading pair already read
        # above, not a second, separate belief_read call.
        if not hasattr(worker, "turn_anchor"):
            worker.turn_anchor = {}
            worker.turn_anchor_set_tick = {}
        sin_now = belief_out[:, 2]
        cos_now = belief_out[:, 3]
        sin_anchor = torch.zeros(n, device = device)
        cos_anchor = torch.ones(n, device = device)
        for i in range(n):
            a = int(arena_ids[i]); l = int(locals_[i])
            wall_nonzero_now = bool(walls_b[i].sum() > 0)
            current_tick = float(worker.step_count.get(a, 0))
            last_set_tick = worker.turn_anchor_set_tick.get(a, {}).get(l)
            past_refractory = last_set_tick is None or (current_tick - last_set_tick) >= TURN_ANCHOR_REFRACTORY_TICKS
            # walls_b is the same reception-lottery-subject signal the oracle
            # uses for its go_north -> turning check, but the oracle examines
            # that condition once, on the way OUT of go_north, and never
            # re-checks it inside turning. A raw rising edge on every decision
            # re-anchors on any tick where a neighbour message merely won the
            # lottery instead of the wall reading. See
            # TURN_ANCHOR_REFRACTORY_TICKS and config.py's turn_anchor_latch.
            armed = past_refractory
            if armed and getattr(cfg, "turn_anchor_latch", True):
                prev_anc = worker.turn_anchor.get(a, {}).get(l)
                if prev_anc is not None:
                    ps, pc = prev_anc
                    rel_s = float(sin_now[i]) * pc - float(cos_now[i]) * ps
                    rel_c = float(cos_now[i]) * pc + float(sin_now[i]) * ps
                    rotated = abs(math.atan2(rel_s, rel_c))
                    elapsed = current_tick - (last_set_tick if last_set_tick is not None else current_tick)
                    armed = (rotated >= TURN_ANCHOR_LATCH_RAD
                             or elapsed >= TURN_ANCHOR_MAX_LATCH_TICKS)
            if wall_nonzero_now and armed:
                # rising edge -- (re-)anchor to this tick's own real
                # heading. Stored as the (sin, cos) pair directly, not a
                # raw angle via atan2 -- the angle-difference identity
                # below needs only this pair, so there is no wraparound
                # to handle.
                worker.turn_anchor.setdefault(a, {})[l] = (float(sin_now[i]), float(cos_now[i]))
                worker.turn_anchor_set_tick.setdefault(a, {})[l] = current_tick
            anc = worker.turn_anchor.get(a, {}).get(l)
            if anc is not None:
                sin_anchor[i], cos_anchor[i] = anc
            # else: no rising edge has ever fired for this robot yet --
            # sin_anchor/cos_anchor stay at their (0, 1) default, which
            # the identity below turns into a relative angle of exactly
            # zero, matching config.py's own documented default exactly.
        sin_rel = sin_now * cos_anchor - cos_now * sin_anchor   # sin(now - anchor)
        cos_rel = cos_now * cos_anchor + sin_now * sin_anchor   # cos(now - anchor)
        prop_b = torch.cat([prop_b, torch.stack([sin_rel, cos_rel], dim = 1)], dim = 1)
    return h_prev, prop_b


def _resolve_wall_seed_xy(worker, arena_ids, locals_, cfg, device, wall_seed_xy_unity = None):
    # wall_seed_xy_unity comes from the message channel already masked: with
    # oracle_wall_seed_position off, SwarmManager only transmits positions for
    # its near-corner seeds. Used as-is rather than re-masked here, so the two
    # sides cannot disagree about which points qualify.
    if not getattr(cfg, "oracle_wall_seed_position", False):
        return wall_seed_xy_unity
    if hasattr(worker, "_wall_seed_xy"):
        return torch.stack([
            torch.as_tensor(worker._wall_seed_xy[int(a)][int(l)], dtype = torch.float32, device = device)
            for a, l in zip(arena_ids.tolist(), locals_.tolist())
        ])
    return wall_seed_xy_unity


def sample_split_event(seeds, walls, rows, valid, cfg, rng, wall_seed_xy = None):
    # One event per robot, drawn from a pool of every seed slot, every wall
    # side and every valid neighbour message, weighted by strength, with no
    # priority between kinds unless split_seed_weight_boost != 1.0. Real
    # hardware has one IR receiver, so a robot hears at most one transmitter per
    # tick whatever its kind. Returns the narrowed seed_part and wall_part
    # alongside Tc, so gather_split_state's belief_update fuses the same single
    # winner Tc reports rather than an independently resampled one.
    #
    # wall_seed_xy (n, WALL_SIZE, 2), optional: the nearest wall seed's known
    # position per band. Kept outside the ml-agents-negotiated
    # seeds/walls/rows tensors -- widening those needs a matched Unity rebuild
    # -- and masked by this same draw, so it is only ever known on a tick where
    # the wall channel genuinely won. None for callers without the data.
    n = seeds.shape[0]
    boost = getattr(cfg, "split_seed_weight_boost", 1.0)
    seed_weight = seeds * boost
    wall_weight = walls * boost
    neighbor_weight = rows[:, :, STRENGTH_COL] * valid
    pool_weight = torch.cat([seed_weight, wall_weight, neighbor_weight], dim = 1)
    ssum = pool_weight.sum(dim = 1, keepdim = True)
    weights = torch.where(ssum > 0, pool_weight, torch.ones_like(pool_weight))
    chosen = torch.multinomial(weights, 1, generator = _matched_generator(rng, weights.device)).squeeze(1)

    is_seed = chosen < SEED_SIZE
    is_wall = (chosen >= SEED_SIZE) & (chosen < SEED_SIZE + WALL_SIZE)
    is_neighbor = ~is_seed & ~is_wall

    seed_idx = chosen.clamp(max = SEED_SIZE - 1)
    seed_mask = torch.arange(SEED_SIZE, device = seeds.device).unsqueeze(0) == seed_idx.unsqueeze(1)
    seed_part = torch.where(is_seed.unsqueeze(1) & seed_mask, seeds, torch.zeros_like(seeds))

    wall_idx = (chosen - SEED_SIZE).clamp(min = 0, max = WALL_SIZE - 1)
    wall_mask = torch.arange(WALL_SIZE, device = walls.device).unsqueeze(0) == wall_idx.unsqueeze(1)
    wall_part = torch.where(is_wall.unsqueeze(1) & wall_mask, walls, torch.zeros_like(walls))

    wall_seed_xy_part = None
    if wall_seed_xy is not None:
        gathered_xy = wall_seed_xy[torch.arange(n, device = wall_seed_xy.device), wall_idx]
        wall_seed_xy_part = torch.where(is_wall.unsqueeze(1), gathered_xy, torch.zeros_like(gathered_xy))

    neighbor_idx = (chosen - SEED_SIZE - WALL_SIZE).clamp(min = 0)
    chosen_rows = rows[torch.arange(n, device = rows.device), neighbor_idx]
    # message content plus the received signal strength: the strength is the
    # robot's only ranging measurement of the sender and must reach the network,
    # not just weight the sampling draw
    neighbor_content = torch.cat([chosen_rows[:, :MESSAGE_SIZE],
                                  chosen_rows[:, STRENGTH_COL:STRENGTH_COL + 1]], dim = 1)
    actor_part = torch.where(is_neighbor.unsqueeze(1), neighbor_content, torch.zeros_like(neighbor_content))

    tc = torch.cat([actor_part, seed_part, wall_part], dim = 1)
    return tc, seed_part, wall_part, wall_seed_xy_part


