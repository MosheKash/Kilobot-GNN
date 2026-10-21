"""renewal_oracle.py -- the renewal-cycle oracle for the amended
(noisy-odometry) task, v2: heading-free edge following + heading
re-anchoring at the wall.

Selected by cfg.motor_override == "renewal_oracle"
(KILOBOT_MOTOR_OVERRIDE=renewal_oracle). simple_oracle.py stays frozen as
the exact-motion baseline; this module is the amended-task baseline, and it
reuses only simple_oracle's already-verified constants and helpers
(WALL_TANGENT, TURN_MOTOR, _steer, _true_pose) by import, never by copy.
The design is drawn in docs/figures/renewal_oracle_v2_flow.tex.

Why it exists (docs/tuning.md phases 174-175): with exact dead reckoning a
robot anchors once at a corner and is accurate forever, so information
never goes stale. The amendment makes true odometry noisy (wheel slip,
MotionNoise.cs). Under slip the old oracle's states break in measured ways
(phase 175's turn diagnostic): the robot meets its first wall at an angle,
so a fixed dead-reckoned 90-degree turn leaves it 9-13 degrees off the
tangent at the median (19% of turns > 45 degrees off), and its own
dead-reckoned heading is wrong by up to ~47 degrees (p90, s2) with nothing
ever correcting it. v1's band controller also steered by that heading
(a world-frame goal through _steer), so the error fed straight back in.

v2 closes both loops on receptions the robot already reads (no privilege;
the same sensors the trained actor gets):

  1. EDGE FOLLOWING WITHOUT A WORLD HEADING, SAMPLED AT SEED PASSES.
     Wall seeds sit WALL_SEED_INSET inside the wall, WALL_SPACING (8u)
     apart; d = 1/s - 1 is the distance to the nearest seed of the
     followed side W. Between seeds d ripples between the perpendicular
     offset y and sqrt(y^2 + 16) -- larger, close in, than anything the
     robot's own motion does to it, so no continuous band/trend controller
     survives there (phase 175's traces: robots slid along the wall or
     orbited a seed). The clean measurement is the closest approach at each
     seed pass (peak_step): the offset y_k. Two consecutive peaks give the
     segment's angle off the tangent from geometry alone,
     alpha_k = atan((y_k - y_{k-1}) / (8 * dk)) (segment_angle), and at
     each peak the robot makes one short dead-reckoned pivot toward the
     angle that reaches the setpoint offset over renewal_conv_len
     (pivot_command), then drives straight to the next seed. Phases:
       approach  far from the seed line (no usable peaks; the ripple is
                 small there): trend-gated arcs toward sigma
                 (approach_motor) until y_hat <= renewal_track_enter
       track     straight; pivot at each peak; if odometry covers
                 renewal_nopeak_path with no peak, the robot is pushing
                 something (no collision detection) or veering -- pivot
                 away from the wall
       pivot     the dead-reckoned rotation, then back to track
     At a corner the new wall is dead ahead: a quarter-turn pivot away from
     it, in the direction the corner proved.
  2. HEADING RE-ANCHOR. Every segment is straight, so each peak's alpha_k
     is also a heading measurement: the confirmed tangent minus
     sigma * alpha_k is the segment's absolute heading, and its difference
     from the dead-reckoned heading at the peak is the frame correction.
     When renewal_fix_peaks consecutive corrections agree within
     renewal_fix_spread_max (consistent_correction), their mean is applied
     to the dead-reckoned heading and to every particle's heading.

The wall-side bit sigma (owner decision 2026-10-20: oracle-private state,
never shared with the actor or the estimator). The IR receiver is
omnidirectional, so which side of the robot the wall is on -- and so the
travel direction (wall on the left = clockwise) -- is not sensed. The turn
GUESSES sigma = left; the arcs then physically pull the robot into
wall-on-sigma travel (a wrong-side robot told "arc away = right" turns into
and through the wall-facing direction until the wall is on its left), the
response test flips sigma if arcing toward the assumed side for a long
turn never brings the wall closer, and the CORNER confirms it: the next
wall heard is the clockwise neighbour (sigma = left) or the
counter-clockwise one (sigma = right). The heading fix and the exit to
navigate both wait for confirmation, because a wrong sigma makes the wall
tangent 180 degrees off.

State machine, per robot:

  seek_wall    steer toward the nearest boundary point by believed position
               until any wall reception; sigma resets (unconfirmed).
  turn         fixed dead-reckoned ~90-degree clockwise turn -- a rough
               initial guess only; its error is absorbed by edge_follow.
  edge_follow  (1) + (2) above, plus the corner step; position comes from
               the unchanged belief_update. Drawn in
               docs/figures/renewal_oracle_v2_flow.tex. Exits to navigate when conf >=
               renewal_wall_exit_conf AND sigma confirmed AND the heading
               was re-anchored within renewal_anchor_dist of path; to
               seek_wall when side W is unheard for renewal_lost_ticks.
  navigate     steer to the hash-assigned target by believed position and
               the anchored heading; arrives at believed dist < tau_v;
               renews (seek_wall) when conf < renewal_conf_min OR the
               heading budget is spent: sigma_theta = fix_sigma +
               renewal_heading_drift * path since the fix > renewal_heading_max.
  arrived      stop, terminal. Error is frozen while parked.

Heading tracking, spawn conventions, batching, reset hygiene and the
shared_event/single-receiver rules are copied structurally from
simple_oracle.py (phase 162's one-draw semantics included) -- that file's
docstrings are the authority for those mechanisms.
"""

import math

import numpy as np
import torch

from belief import (belief_init, belief_predict, belief_read, belief_conf,
                    KNOWN_START_HEADING, ARENA_HALF, BELIEF_PARTICLES, SEED_SIZE,
                    WALL_SIZE, WALL_AXIS, WALL_VAL, WALL_SPACING, _Update, _matched_generator,
                    _range_from_strength)
from kilobot_gnn import MOTOR_SIZE, MESSAGE_SIZE
from kinematics import split_tick_motion
from actor_io import ensure_target, sample_split_event
from simple_oracle import (WALL_NAMES, WALL_TANGENT, TURN_MOTOR, TURN_TARGET_RAD, _true_pose,
                           APPROACH_SLOWDOWN_MIN_SCALE, _steer)

# Clockwise perimeter order (the WALL_TANGENT convention: north's tangent
# is +x, so clockwise travel along north reaches the north-east corner and
# continues south along east).
CW_NEXT = {"north": "east", "east": "south", "south": "west", "west": "north"}
CCW_NEXT = {v: k for k, v in CW_NEXT.items()}

# Physical wall-seed geometry (SwarmManager.cs WALL_SPACING): one seed per
# this many units along every wall, so a window of exactly this much path
# parallel to a wall contains exactly one seed passage.
PEAK_WINDOW = 8.0

# The physical box a robot's centre can occupy (normalized): the wall sits at
# ARENA_HALF and a robot's centre stops ~2.1u short of it (measured 97.9 in
# phase 175's traces). Odometry of a robot pushing a wall (no collision
# detection) carries particles through it, so out-of-box particles are
# penalized as a LIKELIHOOD in the private update (resampling removes them).
# A hard clamp at this limit was tried first and was wrong: an along-wall
# cloud still spread over the whole wall got stacked onto the end wall's
# line, its spread collapsed, and the filter reported conf ~1.0 up to 100u
# from the truth (phase 175). SAFETY_LIMIT only stops runaway particles.
ARENA_LIMIT = 97.9 / ARENA_HALF
SAFETY_LIMIT = 1.05
OUT_OF_BOX_LOG_W = -10.0

# Every wall seed's along-wall coordinate (normalized), one side's worth:
# SwarmManager.SpawnWallSeeds places them at -half .. +half in WALL_SPACING
# steps, on the line WALL_VAL (WALL_SEED_INSET inside the wall).
_ALONG = torch.arange(-ARENA_HALF, ARENA_HALF + 0.01, WALL_SPACING) / ARENA_HALF

# Mid-arena landmark beacons (KILOBOT_LANDMARKS=1; SwarmManager.SpawnLandmarks,
# LANDMARK_FRAC 0.45): known map geometry, like the corner seeds' positions.
# Their broadcasts arrive as neighbour rows: slots 0-1 position (normalized),
# slot 2 conf 1.0, slot 8 = 1 (the landmark flag).
LANDMARK_POS = np.array([[-45.0, 45.0], [45.0, 45.0], [-45.0, -45.0], [45.0, -45.0]])
LANDMARK_FLAG_SLOT = 8

SIGMA_LEFT = 1      # wall on the robot's left  -> clockwise travel
SIGMA_RIGHT = -1    # wall on the robot's right -> counter-clockwise travel


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def tangent_heading(wall_name, sigma):
    """World heading of travel along wall_name with the wall on side sigma.

    sigma = left is clockwise travel, exactly WALL_TANGENT's direction;
    sigma = right is the reverse.
    """
    t = WALL_TANGENT[wall_name]
    th = math.atan2(t[1], t[0])
    return th if sigma == SIGMA_LEFT else _wrap(th + math.pi)


def resolve_corner(old_wall, new_wall):
    """The travel direction a wall handoff proves, or None if it proves none.

    Following old_wall and next hearing its clockwise neighbour means the
    robot was travelling clockwise (wall on its left); its counter-clockwise
    neighbour means counter-clockwise (wall on its right). The opposite wall
    proves nothing (not a corner).
    """
    if CW_NEXT[old_wall] == new_wall:
        return SIGMA_LEFT
    if CCW_NEXT[old_wall] == new_wall:
        return SIGMA_RIGHT
    return None


# Dead-reckoned pivots: TURN_MOTOR (left faster) is the verified clockwise
# turn; its mirror turns counter-clockwise. Short pivots (<= renewal_pivot_max)
# are the only turns the tracking controller makes, and over that little
# rotation dead reckoning is accurate even under slip.
PIVOT_CW = TURN_MOTOR
PIVOT_CCW = (TURN_MOTOR[1], TURN_MOTOR[0])


def arc_motor(toward_left, inner):
    """A constant-curvature arc. (inner, 1) turns counter-clockwise (left):
    the right wheel faster, _steer's own sign convention (left = base - gain
    * turn for a CCW goal) and TURN_MOTOR's (left faster = clockwise)."""
    return (inner, 1.0) if toward_left else (1.0, inner)


def approach_motor(trend, sigma, inner):
    """The far phase (no usable peaks yet): straight while the followed
    side's smoothed distance is falling, else arc toward the wall side
    sigma. A constant arc alone only circles; gating it on the trend makes
    it a heading-free gradient approach. Far from the seed line the seed
    ripple (sqrt(y^2 + 16) - y) is small against the motion signal, which
    is why this runs only until the robot is within renewal_track_enter.
    Returns (motor, branch)."""
    if trend < 0:
        return (1.0, 1.0), "far_approach"
    return arc_motor(sigma == SIGMA_LEFT, inner), "far_arc"


def peak_step(det, path, d, heading, margin, min_gap, extra = None):
    """Online seed-pass detector over the followed side's distance samples.

    det: {"min": (path, d, heading) or None, "last_peak_path": float or None},
    mutated in place. The running minimum becomes a peak once d has climbed
    `margin` above it (the robot has just passed a seed) and it lies at
    least `min_gap` of path past the previous peak (one peak per seed).
    Returns the peak (path, offset, heading, extra) or None. offset is the
    closest approach to that seed, i.e. the perpendicular offset from the
    seed line; extra is whatever the caller attached to that sample (the
    oracle attaches its cloud's along-wall mean, so seed counting measures
    between closest approaches, not between detections -- detection lags the
    minimum by a variable amount of path).
    """
    m = det["min"]
    if m is None or d < m[1]:
        det["min"] = (path, d, heading, extra)
        return None
    if d - m[1] >= margin:
        det["min"] = (path, d, heading, extra)
        lp = det["last_peak_path"]
        if lp is None or m[0] - lp >= min_gap:
            det["last_peak_path"] = m[0]
            return m
    return None


def segment_angle(prev_peak, peak):
    """The straight segment's angle off the wall tangent between two
    consecutive peaks, from GEOMETRY: seeds are PEAK_WINDOW apart along the
    wall, so the along-wall advance is PEAK_WINDOW * dk, dk = seeds passed
    (odometric path / PEAK_WINDOW, rounded, >= 1 -- covers missed peaks;
    odometry only has to be right to the nearest seed, so slip and
    wall-pushing cannot corrupt the angle). alpha > 0: drifting away from
    the wall. Returns (alpha, dk)."""
    dpath = peak[0] - prev_peak[0]
    dk = max(1, int(round(dpath / PEAK_WINDOW)))
    return math.atan2(peak[1] - prev_peak[1], PEAK_WINDOW * dk), dk


def pivot_command(alpha, offset, offset_set, conv_len, alpha_cmd_max, pivot_max, sigma):
    """The rotation (rad, world sense, + = counter-clockwise) to pivot at a
    peak: from the measured angle alpha to the commanded
    alpha* = atan((offset_set - offset) / conv_len), clamped to
    +-alpha_cmd_max -- i.e. head for the setpoint offset over conv_len of
    travel. alpha > 0 is "away from the wall", a clockwise rotation when the
    wall is on the left (sigma = left), hence the -sigma. Clamped to
    +-pivot_max."""
    a_star = max(-alpha_cmd_max, min(alpha_cmd_max, math.atan2(offset_set - offset, conv_len)))
    rot = -sigma * (a_star - alpha)
    return max(-pivot_max, min(pivot_max, rot))


def approach_angle(ctrl, min_samples = 4, min_span = 2.0):
    """The approach's angle off the tangent (rad, > 0 = away) from the slope
    of distance over path in the control window: d(d)/d(path) = sin(alpha)
    for straight travel. Only used at the approach -> track handoff, where
    the robot has been closing in on the seed line from >= renewal_track_enter
    and the seed ripple is modest; it buys the first pivot before the robot
    can cross the seed line (two peaks would come too late at a steep
    approach). Returns None on too little data."""
    if len(ctrl) < min_samples:
        return None
    ell = np.array([c[0] for c in ctrl], dtype=float)
    d = np.array([c[1] for c in ctrl], dtype=float)
    if float(ell.max() - ell.min()) < min_span:
        return None
    b = float(np.polyfit(ell, d, 1)[0])
    return math.asin(max(-1.0, min(1.0, b)))


def in_wall_gap(offset, gap_lo, gap_hi):
    """The physical wall is a hard constraint 5u outside the seed line and a
    robot's centre stops ~2.1u short of it, so a robot pushed against it
    reads a peak offset in a narrow window around ~2.9u (measured
    2.75-2.95 in phase 175's traces)."""
    return gap_lo <= offset <= gap_hi


def pressed_against_wall(offsets, gap_lo, gap_hi, flat_tol = 0.12):
    """True when the last two peak offsets both sit in the wall-gap window
    and agree within flat_tol.

    A robot pushing the wall (no collision detection: odometry still says it
    moves) slides there with its geometric alpha reading ~0 -- the MOTION is
    parallel, the heading is not -- so neither the controller nor the heading
    fix can see the problem from alpha. Phase 175 tried two looser tests:
    "3 offsets flat" missed slip-jittered slides, "2 offsets <= 3.0" fired
    on healthy inner-side tracking dipping below the setpoint and threw
    robots off the wall; the window + flatness pair is the narrow middle."""
    if len(offsets) < 2:
        return False
    o1, o2 = offsets[-2], offsets[-1]
    return in_wall_gap(o1, gap_lo, gap_hi) and in_wall_gap(o2, gap_lo, gap_hi) and abs(o1 - o2) < flat_tol


def blend_gain(prior_sigma, corrs, meas_floor):
    """Kalman gain for a heading fix, and the posterior sigma.

    prior_sigma: the dead-reckoned heading's current uncertainty (rad).
    The measurement's sigma is the spread of the agreeing per-segment
    corrections over sqrt(n), floored. A hard reset (gain 1) was the first
    version: without slip it REPLACED a near-exact dead-reckoned heading
    with the fixes' ~2-3 degree noise floor, and on 150u legs that alone
    missed the 5u settle bar (phase 175). Returns (gain, posterior_sigma).
    """
    n = len(corrs)
    c0 = corrs[0]
    rel = np.array([_wrap(c - c0) for c in corrs])
    meas = max(float(rel.std()), meas_floor) / math.sqrt(n)
    pv, mv = prior_sigma * prior_sigma, meas * meas
    gain = pv / (pv + mv)
    return gain, math.sqrt(pv * mv / (pv + mv))


def consistent_correction(corrs, spread_max):
    """The mean of the per-segment heading corrections if they agree within
    spread_max (rad), else None. Each segment's correction is the offset
    between the dead-reckoned frame and the geometric one, so on a clean
    wall pass they are all the same number up to measurement noise; a
    disagreement means a contaminated segment (a peer blocking receptions,
    a pivot inside the segment, the outer side of the seed line)."""
    if not corrs:
        return None
    c0 = corrs[0]
    rel = [_wrap(c - c0) for c in corrs]
    if max(rel) - min(rel) > spread_max:
        return None
    return _wrap(c0 + sum(rel) / len(rel))


def wall_point_log_w(p, w, d_meas, seed_xy, vis, sigma, outer_penalty = 2.0):
    """The correct likelihood of one wall reception: a range to a POINT.

    A wall reception's strength is the range to the nearest seed of that side
    (SwarmManager: wallBestDist over that side's seeds), not to the seed
    LINE. The shared filter (belief._wall_log_w) models it as the
    perpendicular distance, which is right only beside a seed: between seeds
    the true range is sqrt(y^2 + along^2), so every reading pulls the cloud
    inward by up to sqrt(y^2 + 16) - y -- the confidently-wrong exits of
    phase 175 (conf ~1.0, 4-7u off). Here the predicted range is the
    distance to the specific broadcast seed when the event carries its
    position (seed_xy nonzero: near-corner seeds), else to the nearest seed
    of that side. Same reception, same information; only the oracle's own
    inference changes, and the shared filter is untouched.
    p: (n, k, 3) normalized; d_meas: (n,) normalized; seed_xy: (n, 2) raw
    or None; vis: (n,) bool. Returns (n, k) log weights.
    """
    axis = WALL_AXIS[w]
    along_axis = 1 - axis
    line = WALL_VAL[w]
    perp = p[:, :, axis] - line                                   # (n, k)
    along = _ALONG.to(p.device)
    da = p[:, :, along_axis].unsqueeze(2) - along.view(1, 1, -1)   # (n, k, m)
    d_near = torch.sqrt(perp.unsqueeze(2) ** 2 + da ** 2).min(dim=2).values
    d_pred = d_near
    if seed_xy is not None:
        known = seed_xy.abs().sum(dim=1) > 0
        if bool(known.any()):
            sx = (seed_xy[:, 0] / ARENA_HALF).view(-1, 1)
            sy = (seed_xy[:, 1] / ARENA_HALF).view(-1, 1)
            d_known = torch.sqrt((p[:, :, 0] - sx) ** 2 + (p[:, :, 1] - sy) ** 2)
            d_pred = torch.where(known.view(-1, 1), d_known, d_near)
    err = (d_pred - d_meas.view(-1, 1)) / sigma
    lw = -0.5 * err * err
    # A range to a point is symmetric across the seed line, but the outer
    # side is only the ~2.9u wall gap, and the controller holds robots on
    # the inner side (pressed episodes are brief and detected). Without
    # this prior the cloud settled on the mirror image against the wall
    # (phase 175: exits estimated at the clamp, 97.9, truly at ~91).
    outward = perp * (1.0 if line > 0 else -1.0)
    lw = lw - outer_penalty * (outward > 0).float()
    return torch.where(vis.view(-1, 1), lw, torch.zeros_like(err))


def renewal_belief_update(p, seed_obs, walls, wall_seed_xy, generator, heading_noise_scale,
                          meas_sigma):
    """The oracle's private filter step on the tick's receptions.

    The shared belief_update's phases, composed with the wall phase swapped
    for wall_point_log_w: corner-seed ranges (unchanged), the point-range
    wall likelihood, resample, then the shared re-seeding injections. The
    wall band bookkeeping (best_dir_*) is kept so a cold cloud's band
    injection still seeds hypotheses near the wall; the corrected
    likelihood then resolves them. No peer terms -- this oracle does not
    consume messages.
    """
    generator = _matched_generator(generator, p.device)
    u = _Update(p, generator, heading_noise_scale)
    u.add_seed_ranges(seed_obs)
    if walls is not None:
        for w in range(WALL_SIZE):
            strength = walls[:, w]
            vis = strength > 0
            if not bool(vis.any()):
                continue
            u.any_meas = u.any_meas | vis
            axis = WALL_AXIS[w]
            better = vis & (strength > u.best_dir_strength[axis])
            u.best_dir_strength[axis] = torch.where(better, strength, u.best_dir_strength[axis])
            u.best_dir_val[axis] = torch.where(better, torch.full_like(u.best_dir_val[axis], WALL_VAL[w]),
                                               u.best_dir_val[axis])
            d_meas = _range_from_strength(strength)
            u.log_w = u.log_w + wall_point_log_w(u.p, w, d_meas, wall_seed_xy, vis, meas_sigma)
    out_box = (u.p[:, :, 0].abs() > ARENA_LIMIT) | (u.p[:, :, 1].abs() > ARENA_LIMIT)
    u.log_w = u.log_w + torch.where(u.any_meas.view(-1, 1) & out_box,
                                    torch.full_like(u.log_w, OUT_OF_BOX_LOG_W), torch.zeros_like(u.log_w))
    u.seed_meas = u.any_meas.clone()
    if not u.has_measurement():
        return p
    u.resample()
    u.inject_seed_ring()
    u.inject_wall_band()
    u.inject_random_particles(None)
    return u.out


class WheelBiasEstimator:
    """Per-robot wheel-slip calibration from the oracle's own measurements.

    Under the slip model (KilobotMovement.cs) a commanded wheel speed m
    executes as m * (1 + b) with b this robot's persistent bias, so the true
    turn rate is pi * (L (1 + bL) - R (1 + bR)) against dead reckoning's
    pi * (L - R): the heading error grows as pi * (bL * int L dt - bR * int R dt).
    That is linear in (bL, bR) with regressors the robot knows (its own
    commands), so every pair of heading measurements -- consecutive
    segment corrections on a wall -- gives one equation. Pivots in both
    directions and straight driving make it well-conditioned; a ridge prior
    keeps it sane before they have. A per-unit-path drift rate (tried first,
    phase 175) cannot represent pivot error, which scales with rotation.
    """

    def __init__(self, sigma_c, sigma_b):
        self.S = np.eye(2) / (sigma_b * sigma_b)
        self.v = np.zeros(2)
        self.inv_var_c = 1.0 / (sigma_c * sigma_c)
        self.b = np.zeros(2)
        self.n = 0

    def add(self, d_int_l, d_int_r, d_c):
        x = math.pi * np.array([d_int_l, -d_int_r])
        self.S += self.inv_var_c * np.outer(x, x)
        self.v += self.inv_var_c * x * d_c
        self.b = np.linalg.solve(self.S, self.v)
        self.n += 1

    def heading_rate(self, left, right):
        """Predicted heading-error rate (rad/s) for a wheel command."""
        return math.pi * (self.b[0] * left - self.b[1] * right)


def peer_ring_update(p, peer_xy, peer_conf, d_meas, use, generator, gate_frac, jitter):
    """Fuse one arrived peer's beacon as a range landmark (oracle-private).

    The simple peer protocol (owner sign-off 2026-10-21: "relatively simple,
    no communication state machine"): an arrived robot broadcasts its
    believed position and confidence; a robot still travelling that hears it
    treats the claim as a point landmark at range d_meas = 1/s - 1. The
    ring's width comes from the sender's own confidence (belief_conf =
    exp(-var/0.02) -> sigma = sqrt(-0.02 ln conf)) plus a small range floor;
    the shared filter's 12u peer floor (it protects the actor) would make
    such a ring useless for a 5u target. Innovation gate: fuse only when at
    least gate_frac of the receiver's own particles already lie within 2
    sigma of the ring -- a claim the cloud disagrees with is ignored, not
    averaged in. Resamples on the ring weights; returns the new cloud.
    p: (n, k, 3) normalized; peer_xy: (n, 2) normalized; peer_conf, d_meas:
    (n,) (d_meas normalized); use: (n,) bool.
    """
    if not bool(use.any()):
        return p
    gen = generator if (generator is not None and generator.device == p.device) else None
    sig = torch.sqrt(-0.02 * torch.log(peer_conf.clamp(min=1e-3, max=0.999))) + 0.01      # (n,)
    d_pred = torch.sqrt((p[:, :, 0] - peer_xy[:, 0:1]) ** 2 + (p[:, :, 1] - peer_xy[:, 1:2]) ** 2)
    err = (d_pred - d_meas.view(-1, 1)) / sig.view(-1, 1)
    agree = (err.abs() < 2.0).float().mean(dim=1)
    use = use & (agree >= gate_frac)
    if not bool(use.any()):
        return p
    lw = -0.5 * err * err
    w = torch.exp(lw - lw.max(dim=1, keepdim=True).values)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    n, k, _ = p.shape
    idx = torch.multinomial(w.clamp(min=1e-12), k, replacement=True, generator=gen)
    res = torch.gather(p, 1, idx.unsqueeze(2).expand(n, k, 3)).clone()
    res[:, :, :2] = res[:, :, :2] + torch.randn(n, k, 2, device=p.device, generator=gen) * jitter
    return torch.where(use.view(-1, 1, 1), res, p)


def choose_detour(est, target, landmarks, detour_max, detour_gain):
    """The landmark to route a long leg through, or None.

    Admissible: it leaves a final leg at least detour_gain shorter than the
    direct one, at an extra distance (|est-L| + |L-target| - |est-target|)
    of at most detour_max. Among those, the one with the shortest final leg
    -- the blind stretch the landmark exists to cut.
    """
    direct = float(np.linalg.norm(target - est))
    best, best_leg = None, None
    for k, lm in enumerate(landmarks):
        to_lm = float(np.linalg.norm(lm - est))
        leg = float(np.linalg.norm(target - lm))
        if leg <= direct - detour_gain and (to_lm + leg - direct) <= detour_max:
            if best_leg is None or leg < best_leg:
                best, best_leg = k, leg
    return best


def landmark_heading_fix(p0, dr, landmark):
    """Heading correction from a landmark touch: the angle between the true
    displacement since the leg began (to the landmark, within touch range)
    and the dead-reckoned one. A heading error theta turns the dead-reckoned
    displacement by -theta relative to the truth, so this IS the
    correction. Returns radians, or None when the leg is too short to
    resolve an angle."""
    v_true, v_dr = landmark - p0, dr - p0
    if np.linalg.norm(v_true) < 20.0 or np.linalg.norm(v_dr) < 20.0:
        return None
    return _wrap(math.atan2(v_true[1], v_true[0]) - math.atan2(v_dr[1], v_dr[0]))


def nearest_wall_point(est_x, est_y):
    """The nearest boundary point to a believed position (raw arena units).

    seek_wall steers toward it. On a garbage-wide cloud the estimate is
    ~arena-centre and every boundary is equidistant, so this degrades to an
    arbitrary fixed direction -- harmless: the arena is bounded and ANY wall
    reception starts the localization path.
    """
    c = ARENA_HALF
    ds = [("north", c - est_y), ("south", est_y + c),
          ("east", c - est_x), ("west", est_x + c)]
    name, _ = min(ds, key=lambda kv: kv[1])
    pts = {"north": (est_x, c), "south": (est_x, -c),
           "east": (c, est_y), "west": (-c, est_y)}
    return name, np.array(pts[name], dtype=float)


def _fresh_aux(now):
    """Per-robot edge_follow memory. Reset at every seek_wall entry (a new
    wall visit: sigma is a fresh guess) -- except the heading-fix record,
    which belongs to the heading, not to the visit."""
    return {
        "sigma": SIGMA_LEFT, "sigma_confirmed": False,
        "mode": "approach",                  # approach | track | pivot
        "s_ema": [None] * len(WALL_NAMES),   # per side, for the corner test
        "s_last": [None] * len(WALL_NAMES),  # tick each side was last heard
        "s_hist": [[] for _ in WALL_NAMES],  # last receptions per side, for the approach test
        "d_ema": None, "d_prev": None,       # followed side, for the trend
        "last_side_tick": now,
        "path": 0.0,                         # odometric path length this visit
        "ctrl": [],                          # (path, d) over the last PEAK_WINDOW of path
        "det": {"min": None, "last_peak_path": None},
        "prev_peak": None,                   # (path, offset, heading)
        "track_path": 0.0,                   # path at the last peak / track entry
        "corrs": [],                         # per-segment heading corrections
        "calib_prev": None,                  # last (int L, int R, raw correction) this visit
        "seg_sum": 0.0, "seg_len": 0.0,      # path-weighted dead-reckoned heading, this segment
        "pk_along": None,                    # cloud's mean along-wall coord at the last peak
        "offsets": [],                       # last peak offsets, for the pressed test
        "pivot_target": 0.0, "pivot_done": 0.0, "pivot_fresh": False,
        "far_turn": 0.0,                     # turn accumulated in far_arc, response test
        "far_d0": None,
        "goal_heading": None,                # tangent mode: heading goal set at each peak
        "prev_wall": None,                   # the side just handed off from
        "corner_path": None,                 # path at that handoff
    }


def _reset_visit(aux, now):
    keep = {k: aux[k] for k in ("path_since_fix", "fix_valid", "wheel", "int_l", "int_r", "frame_off",
                                "heading_trusted", "head_sigma", "n_renewals") if k in aux}
    aux.clear()
    aux.update(_fresh_aux(now))
    aux.update(keep)


def renewal_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng,
                          formation_pool, belief_attr = "belief", shared_event = None,
                          seeds = None, rows = None, valid = None):
    # The call signature, argument semantics and batching structure mirror
    # simple_oracle.simple_oracle_motors exactly (that file's docstring is
    # the authority): walls is the (n, WALL_SIZE) received-strength channel;
    # wall_seed_xy is (n, WALL_SIZE, 2) or the already-narrowed (n, 2);
    # shared_event is the tick's already-drawn single reception when the
    # caller owns the draw (bc_capture), else this call draws once itself
    # under cfg.oracle_shared_event from the full physical pool; belief_attr
    # keeps this oracle's filter separate from the actor's own when both
    # run in one tick.
    n = len(arena_ids)
    device = walls.device
    motor = torch.zeros(n, MOTOR_SIZE, device=device)

    event_tc = None
    if shared_event is not None:
        event_seed_obs, walls, wall_seed_xy = shared_event
    else:
        if getattr(cfg, "oracle_shared_event", False) and seeds is not None and rows is not None:
            event_tc, event_seed_obs, walls, wall_seed_xy = sample_split_event(
                seeds, walls, rows, valid, cfg, rng, wall_seed_xy=wall_seed_xy)
        else:
            dummy_seeds = torch.zeros(n, SEED_SIZE, device=device)
            dummy_rows = torch.zeros(n, 1, MESSAGE_SIZE + 2, device=device)
            dummy_valid = torch.zeros(n, 1, dtype=torch.bool, device=device)
            _tc, event_seed_obs, walls, wall_seed_xy = sample_split_event(
                dummy_seeds, walls, dummy_rows, dummy_valid, cfg, rng, wall_seed_xy=wall_seed_xy)

    if not hasattr(worker, "simple_heading"):
        worker.simple_heading = {}
        worker.simple_state = {}
        worker.simple_turn_accum = {}
        worker.simple_wall_name = {}
    if not hasattr(worker, "renewal_aux"):
        worker.renewal_aux = {}
    if not hasattr(worker, belief_attr):
        setattr(worker, belief_attr, {})
    belief_dict = getattr(worker, belief_attr)

    # --- pass 1: cheap per-robot init/lookup, gathered into batched tensors ---
    last_motor_batch = torch.zeros(n, MOTOR_SIZE, device=device)
    steps_batch = torch.zeros(n, device=device)
    particles_batch = torch.zeros(n, BELIEF_PARTICLES, 3, device=device)
    skip_tick = set()

    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        worker.simple_heading.setdefault(a, {})
        worker.simple_state.setdefault(a, {})
        worker.simple_turn_accum.setdefault(a, {})
        worker.simple_wall_name.setdefault(a, {})
        worker.renewal_aux.setdefault(a, {})
        belief_dict.setdefault(a, {})

        if l not in worker.simple_heading[a]:
            # Identical reset-race guard to simple_oracle's: a robot whose
            # arena has a reset pending is skipped entirely this tick, so
            # its once-only init fires again correctly after the real spawn.
            if hasattr(worker, "reset_pending") and worker.reset_pending(a):
                skip_tick.add(i)
                continue
            start_heading = KNOWN_START_HEADING
            if hasattr(worker, "spawn_heading"):
                sh = worker.spawn_heading(a, l)
                if sh is not None:
                    start_heading = sh
            worker.simple_heading[a][l] = start_heading
            worker.simple_state[a][l] = "seek_wall"
            worker.simple_turn_accum[a][l] = 0.0
            aux0 = _fresh_aux(int(worker.step_count.get(a, 0)))
            # The spawn heading is known exactly -- an anchored heading at
            # zero path, the same budget a fresh fix starts from.
            aux0["path_since_fix"] = 0.0
            aux0["fix_valid"] = False
            worker.renewal_aux[a][l] = aux0
            image_id = worker.image_id.get(a)
            ensure_target(worker, a, l, image_id, formation_pool)
            belief_dict[a][l] = belief_init(1, rng, particles=BELIEF_PARTICLES, device=device,
                                            known_start_heading=True).squeeze(0)
            if belief_dict[a][l].shape[0] > 0:
                belief_dict[a][l][:, 2] = start_heading

        # the same second-belief_attr fallback guard as simple_oracle's: a
        # robot already known to simple_heading but never seeded under THIS
        # belief_attr needs a particle set the same way the spawn path makes.
        if l not in belief_dict[a]:
            belief_dict[a][l] = belief_init(1, rng, particles=BELIEF_PARTICLES, device=device,
                                            known_start_heading=True).squeeze(0)
            if belief_dict[a][l].shape[0] > 0:
                belief_dict[a][l][:, 2] = worker.simple_heading[a][l]
        particles_batch[i] = belief_dict[a][l].to(device)

        last_motor = worker.last_motor.get(a, {}).get(l)
        step_count = worker.step_count.get(a, 0)
        steps_since = step_count - worker.last_dec_step.get(a, {}).get(l, step_count)
        if last_motor is not None and steps_since > 0:
            lm = last_motor if isinstance(last_motor, torch.Tensor) else torch.tensor(last_motor)
            last_motor_batch[i] = lm.to(torch.float32)
            steps_batch[i] = float(steps_since)

    # --- batched operations: ONE call each across all n robots ---
    x_local, y_local, dtheta, _t = split_tick_motion(
        last_motor_batch, steps_batch, cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed)
    move_batch = torch.sqrt(x_local * x_local + y_local * y_local)

    # The honest-filter difference from simple_oracle: the heading noise the
    # filter models is a cfg dial, not the pinned global HEADING_NOISE_SCALE
    # (0 by the kinematics tests). Default 0.0 keeps the frozen oracle's
    # filter; under slip, launch/eval set KILOBOT_MOTION_NOISE (position)
    # and renewal_heading_noise here, so the cloud honestly widens.
    heading_noise = float(getattr(cfg, "renewal_heading_noise", 0.0))
    particles_batch = belief_predict(particles_batch, x_local, y_local, dtheta, rng,
                                     heading_noise_scale=heading_noise)
    # Oracle-private process noise per unit of travel: there is ALWAYS
    # unmodelled motion (peer and wall contact, no collision detection;
    # slip under the amendment), and a cloud with none collapses onto a
    # point and then cannot follow the error -- conf ~1.0 while biased.
    pos_noise = float(getattr(cfg, "renewal_pos_noise", 0.05))
    if pos_noise > 0.0:
        gen = rng if (rng is not None and rng.device == particles_batch.device) else None
        jit = torch.randn(n, BELIEF_PARTICLES, 2, device=device, generator=gen)
        scale = (pos_noise * move_batch / ARENA_HALF).view(n, 1, 1)
        particles_batch = particles_batch.clone()
        particles_batch[:, :, :2] = particles_batch[:, :, :2] + jit * scale
    particles_batch[:, :, :2] = particles_batch[:, :, :2].clamp(-SAFETY_LIMIT, SAFETY_LIMIT)
    if getattr(cfg, "oracle_fuse_seeds", False):
        seed_obs = event_seed_obs
    else:
        seed_obs = torch.zeros(n, SEED_SIZE, device=device)
    wall_xy = wall_seed_xy if (wall_seed_xy is not None and wall_seed_xy.dim() == 2) else None
    if getattr(cfg, "renewal_private_filter", False):
        particles_batch = renewal_belief_update(particles_batch, seed_obs, walls, wall_xy, rng, heading_noise,
                                                float(getattr(cfg, "renewal_meas_sigma", 0.01)))
    else:
        from belief import belief_update
        particles_batch = belief_update(particles_batch, seed_obs, rng, wall_obs=walls,
                                        wall_seed_xy=wall_seed_xy, heading_noise_scale=heading_noise)
    particles_batch[:, :, :2] = particles_batch[:, :, :2].clamp(-SAFETY_LIMIT, SAFETY_LIMIT)
    peer_comms = bool(getattr(cfg, "renewal_peer_comms", False))
    peer_conf_min = float(getattr(cfg, "renewal_peer_conf_min", 0.85))
    landmarks_on = bool(getattr(cfg, "landmarks", False))
    lm_idx = [-1] * n            # which landmark this tick's reception is, if any
    lm_str = [0.0] * n
    if (peer_comms or landmarks_on) and event_tc is not None:
        msg = event_tc[:, :MESSAGE_SIZE]
        strength = event_tc[:, MESSAGE_SIZE]
        not_arrived = torch.tensor([worker.simple_state.get(int(arena_ids[i]), {}).get(int(locals_[i])) != "arrived"
                                    for i in range(n)], dtype=torch.bool, device=device)
        is_lm = (strength > 0) & (msg[:, LANDMARK_FLAG_SLOT] == 1.0) & (msg[:, 2] == 1.0)
        if landmarks_on:
            for i in range(n):
                if bool(is_lm[i]):
                    xy = msg[i, 0:2].cpu().numpy() * ARENA_HALF
                    lm_idx[i] = int(np.argmin(np.linalg.norm(LANDMARK_POS - xy, axis=1)))
                    lm_str[i] = float(strength[i])
        # landmarks always fuse (map geometry); peer beacons only with peer_comms
        use = (strength > 0) & not_arrived & (
            (is_lm & landmarks_on) |
            ((~is_lm) & (msg[:, 2] >= peer_conf_min) & peer_comms))
        d_meas = _range_from_strength(strength.clamp(min=1e-6))
        particles_batch = peer_ring_update(particles_batch, msg[:, 0:2], msg[:, 2], d_meas, use, rng,
                                           float(getattr(cfg, "renewal_peer_gate", 0.1)), 0.003)
        if getattr(cfg, "oracle_debug_wall_log", False) and bool(use.any()):
            print(f"RENEWAL_PEER heard={int(use.sum())} tick={int(worker.step_count.get(int(arena_ids[0]), 0))}")
    conf_batch = belief_conf(particles_batch)
    br_batch = belief_read(particles_batch)

    exit_conf = float(getattr(cfg, "renewal_wall_exit_conf", 0.4))
    renew_min = float(getattr(cfg, "renewal_conf_min", 0.3))
    track_enter = float(getattr(cfg, "renewal_track_enter", 5.0))
    offset_set = float(getattr(cfg, "renewal_offset_set", 4.0))
    conv_len = float(getattr(cfg, "renewal_conv_len", 16.0))
    alpha_cmd_max = float(getattr(cfg, "renewal_alpha_cmd_max", 0.35))
    pivot_max = float(getattr(cfg, "renewal_pivot_max", 0.785))
    pivot_min = float(getattr(cfg, "renewal_pivot_min", 0.02))
    peak_margin = float(getattr(cfg, "renewal_peak_margin", 0.5))
    peak_min_gap = float(getattr(cfg, "renewal_peak_min_gap", 4.0))
    nopeak_path = float(getattr(cfg, "renewal_nopeak_path", 20.0))
    arc_inner = float(getattr(cfg, "renewal_arc_inner", 0.6))
    trend_alpha = float(getattr(cfg, "renewal_trend_alpha", 0.3))
    stale_ticks = int(getattr(cfg, "renewal_stale_ticks", 60))
    lost_ticks = int(getattr(cfg, "renewal_lost_ticks", 600))
    flip_turn = float(getattr(cfg, "renewal_flip_turn", 1.5 * math.pi))
    fix_peaks = int(getattr(cfg, "renewal_fix_peaks", 3))
    fix_meas_floor = float(getattr(cfg, "renewal_fix_meas_floor", 0.04))
    contact_sigma = float(getattr(cfg, "renewal_contact_sigma", 0.15))
    fix_blend = bool(getattr(cfg, "renewal_fix_blend", False))
    fix_spread_max = float(getattr(cfg, "renewal_fix_spread_max", 0.17))
    fix_alpha_max = float(getattr(cfg, "renewal_fix_alpha_max", 0.35))
    count_max_path = float(getattr(cfg, "renewal_count_max_path", 0.0))
    count_shift_max = float(getattr(cfg, "renewal_count_shift_max", 24.0))
    gap_lo = float(getattr(cfg, "renewal_gap_lo", 2.7))
    gap_hi = float(getattr(cfg, "renewal_gap_hi", 3.05))
    anchor_dist = float(getattr(cfg, "renewal_anchor_dist", 12.0))
    anchor_heading_max = float(getattr(cfg, "renewal_anchor_heading_max", 0.1))
    tangent_mode = getattr(cfg, "renewal_controller", "tangent") == "tangent"
    tangent_offset_set = float(getattr(cfg, "renewal_tangent_offset_set", 5.5))
    tangent_inward = float(getattr(cfg, "renewal_tangent_inward", 0.07))
    sigma_construct_max = float(getattr(cfg, "renewal_sigma_construct_max", 10.0))
    leg_err_max = float(getattr(cfg, "renewal_leg_err_max", 1e9))
    renew_miss = float(getattr(cfg, "renewal_renew_miss", 4.0))
    max_renewals = int(getattr(cfg, "renewal_max_renewals", 2))
    cheat = str(getattr(cfg, "renewal_cheat", "off"))
    commit_tick = int(getattr(cfg, "renewal_commit_tick", 10 ** 9))
    seek_timeout = int(getattr(cfg, "renewal_seek_timeout", 10 ** 9))
    import belief as _belief_mod
    if getattr(cfg, "landmarks", False):
        landmarks = LANDMARK_POS
    elif getattr(cfg, "seed_layout", "corners") == "quarters":
        landmarks = _belief_mod.SEED_POS.numpy() * ARENA_HALF
    else:
        landmarks = np.zeros((0, 2))
    detour_on = bool(getattr(cfg, "renewal_detour", True)) and len(landmarks) > 0
    detour_max = float(getattr(cfg, "renewal_detour_max", 15.0))
    detour_gain = float(getattr(cfg, "renewal_detour_gain", 20.0))
    detour_leg_min = float(getattr(cfg, "renewal_detour_leg_min", 40.0))
    touch_d = float(getattr(cfg, "renewal_touch_d", 5.0))
    search_ticks = int(getattr(cfg, "renewal_search_ticks", 500))
    sidestep_on = bool(getattr(cfg, "renewal_sidestep", False))
    sidestep_dist = float(getattr(cfg, "renewal_sidestep_dist", 3.5))
    sidestep_count = float(getattr(cfg, "renewal_sidestep_count", 6.0))
    sidestep_len = float(getattr(cfg, "renewal_sidestep_len", 6.0))
    seek_escape_turn = float(getattr(cfg, "renewal_seek_escape_turn", 2.0))
    seek_spread = float(getattr(cfg, "renewal_seek_spread", 15.0))
    fix_sigma = float(getattr(cfg, "renewal_heading_fix_sigma", 0.03))
    drift = float(getattr(cfg, "renewal_heading_drift", 0.0))
    calib_sigma_c = float(getattr(cfg, "renewal_calib_sigma_c", 0.03))
    calib_sigma_b = float(getattr(cfg, "renewal_calib_sigma_b", 0.0))
    calib_on = calib_sigma_b > 0.0
    calib_jump_max = float(getattr(cfg, "renewal_calib_jump_max", 0.35))
    calib_min_n = int(getattr(cfg, "renewal_calib_min_n", 4))
    sigma_from_heading_max = float(getattr(cfg, "renewal_sigma_from_heading_max", 0.0))
    calib_residual = float(getattr(cfg, "renewal_calib_residual", 0.3))
    arrive_dist = float(getattr(cfg, "renewal_arrive_dist", 1.5))
    near_dist = float(getattr(cfg, "renewal_near_dist", 10.0))
    heading_max = float(getattr(cfg, "renewal_heading_max", 0.35))
    debug = bool(getattr(cfg, "oracle_debug_wall_log", False))

    # --- pass 2: cheap per-robot state machine on the batched results ---
    for i in range(n):
        if i in skip_tick:
            continue
        a = int(arena_ids[i]); l = int(locals_[i])

        state = worker.simple_state[a][l]
        now = int(worker.step_count.get(a, 0))
        aux = worker.renewal_aux[a].setdefault(l, _fresh_aux(now))
        aux.setdefault("path_since_fix", 0.0)
        aux.setdefault("fix_valid", False)
        if "wheel" not in aux:
            aux["wheel"] = WheelBiasEstimator(calib_sigma_c, calib_sigma_b if calib_on else 1.0)
            aux["int_l"], aux["int_r"], aux["frame_off"] = 0.0, 0.0, 0.0
        move_i = float(move_batch[i])
        # Odometry calibration (oracle-private, WheelBiasEstimator): the
        # executed command's integrals feed the regression, and the current
        # bias estimate is removed from dead reckoning -- heading and every
        # particle alike. frame_off tracks everything added to the dead-
        # reckoned heading beyond raw odometry, so measurements can be put
        # back in the raw frame the regression models.
        dt_i = float(steps_batch[i]) * float(cfg.dt_fixed)
        lm_l, lm_r = float(last_motor_batch[i, 0]), float(last_motor_batch[i, 1])
        aux["int_l"] += lm_l * dt_i
        aux["int_r"] += lm_r * dt_i
        drift_comp = aux["wheel"].heading_rate(lm_l, lm_r) * dt_i if calib_on else 0.0
        aux["frame_off"] += drift_comp
        worker.simple_heading[a][l] += float(dtheta[i]) + drift_comp
        if drift_comp != 0.0:
            particles_batch[i, :, 2] = particles_batch[i, :, 2] + drift_comp
        aux["path_since_fix"] += move_i
        conf_i = float(conf_batch[i])
        est_pos = np.array([float(br_batch[i, 0]), float(br_batch[i, 1])]) * ARENA_HALF
        # DIAGNOSTIC CHEAT (renewal_cheat, default "off"): an upper bound on
        # what better information could buy -- NOT a controller. While
        # crossing the arena (navigate), "heading" hands the robot its true
        # heading every tick, "full" its true heading and position. Reads
        # Unity's true pose, so it is never on in a real comparison.
        if cheat != "off" and state == "navigate":
            tp, tdeg = _true_pose(worker, a, l)
            if tdeg is not None:
                th = math.radians(tdeg)
                d_h = _wrap(th - worker.simple_heading[a][l])
                worker.simple_heading[a][l] += d_h
                particles_batch[i, :, 2] = particles_batch[i, :, 2] + d_h
            if cheat == "full" and tp is not None:
                particles_batch[i, :, :2] = torch.as_tensor(np.asarray(tp, dtype=np.float32) / ARENA_HALF,
                                                            device=device).view(1, 2)
                est_pos = np.asarray(tp, dtype=float)
                conf_i = 1.0

        # Commit deadline: past renewal_commit_tick an unfinished robot stops
        # looking for walls and heads for its assigned point on its best
        # current estimate -- one still moving at the end scores zero for
        # certain. Its own elapsed time only (step_count).
        if now >= commit_tick and state in ("seek_wall", "turn", "edge_follow") \
                and worker.simple_target.get(a, {}).get(l) is not None:
            worker.simple_state[a][l] = state = "navigate"

        if state == "seek_wall":
            heading = worker.simple_heading[a][l]
            h_vec = np.array([math.cos(heading), math.sin(heading)])
            _wn, wp = nearest_wall_point(est_pos[0], est_pos[1])
            motor[i] = torch.tensor(_steer(h_vec, wp - est_pos))
            # Seek timeout (no collision detection): a robot pinned against
            # parked peers keeps dead-reckoning toward the wall it never
            # reaches -- its belief walks out to the boundary while it stands
            # still (phase 175: 10-16 of 41 robots ended their run in
            # seek_wall, true positions mid-arena at the shape, beliefs AT or
            # beyond the wall). Longer than any honest trip to a wall, or a
            # belief at the boundary with no wall heard: turn away (alternating
            # direction on repeats) and widen the cloud -- position is unknown.
            if "seek_since" not in aux or aux.get("seek_state_tick") is None:
                aux["seek_since"], aux["seek_state_tick"] = now, now
            at_edge = max(abs(est_pos[0]), abs(est_pos[1])) > 97.0
            if now - aux["seek_since"] > seek_timeout or (at_edge and seek_timeout < 10 ** 9
                                                            and now - aux["seek_since"] > 60):
                aux["seek_escapes"] = aux.get("seek_escapes", 0) + 1
                aux["seek_since"] = now
                aux["seek_turn_left"] = seek_escape_turn
                aux["seek_turn_dir"] = 1.0 if aux["seek_escapes"] % 2 else -1.0
                particles_batch[i, :, :2] = particles_batch[i, :, :2] + torch.randn(
                    BELIEF_PARTICLES, 2, device=device) * (seek_spread / ARENA_HALF)
                if debug:
                    print(f"RENEWAL_SEEK_ESCAPE arena={a} robot={l} tick={now} n={aux['seek_escapes']}")
            if aux.get("seek_turn_left", 0.0) > 0.0:
                aux["seek_turn_left"] -= abs(float(dtheta[i]))
                motor[i] = torch.tensor(PIVOT_CCW if aux["seek_turn_dir"] > 0 else PIVOT_CW)
            if float(walls[i].sum()) > 0:
                aux["seek_state_tick"] = None
                aux["seek_turn_left"] = 0.0
                wall_idx = int(walls[i].argmax())
                worker.simple_wall_name[a][l] = WALL_NAMES[wall_idx]
                worker.simple_state[a][l] = "turn"
                worker.simple_turn_accum[a][l] = 0.0
                _reset_visit(aux, now)

        elif state == "turn":
            worker.simple_turn_accum[a][l] += abs(float(dtheta[i]))
            motor[i] = torch.tensor(TURN_MOTOR)
            if worker.simple_turn_accum[a][l] >= TURN_TARGET_RAD:
                worker.simple_state[a][l] = "edge_follow"
                _reset_visit(aux, now)
                # sigma from the heading when the heading is trustworthy: the
                # travel direction only needs heading right to within 90
                # degrees, so with the budget well inside that (spawn's known
                # heading; a renewal from navigate) the side the wall is on
                # follows from the tangent the robot now roughly faces. The
                # corner still re-checks it; without a trustworthy heading the
                # corner is what confirms it.
                sig_theta = fix_sigma + drift * aux["path_since_fix"]
                if tangent_mode:
                    # by construction: the clockwise turn from facing the wall,
                    # then tangent steering, keeps the wall on the left -- IF
                    # the heading is good enough for "facing the wall" to hold.
                    # At s2 robots arrived 65 degrees off after a long seek, and
                    # three agreeing segments under the wrong side made a
                    # confidently wrong fix (phase 175). Beyond the budget bar
                    # the corner confirms, as in seed-pass mode.
                    budget_turn = aux.get("head_sigma", fix_sigma) + drift * aux["path_since_fix"]
                    aux["sigma"] = SIGMA_LEFT
                    aux["sigma_confirmed"] = budget_turn <= sigma_construct_max
                    aux["mode"], aux["track_path"] = "track", 0.0
                elif sig_theta <= sigma_from_heading_max:
                    wn = worker.simple_wall_name[a][l]
                    h = worker.simple_heading[a][l]
                    cw = tangent_heading(wn, SIGMA_LEFT)
                    aux["sigma"] = SIGMA_LEFT if math.cos(h - cw) > 0 else SIGMA_RIGHT
                    aux["sigma_confirmed"] = True
                if debug:
                    # Diagnostic only (reads Unity's true pose, never steers
                    # on it): how far off the wall tangent the fixed turn
                    # leaves the robot, and how wrong its dead-reckoned
                    # heading is. Positive tangent error = nose CCW of the
                    # clockwise tangent, i.e. toward the wall.
                    _tp, true_deg = _true_pose(worker, a, l)
                    if true_deg is not None:
                        wn = worker.simple_wall_name[a][l]
                        tan_deg = math.degrees(tangent_heading(wn, SIGMA_LEFT))
                        tan_err = (true_deg - tan_deg + 180.0) % 360.0 - 180.0
                        dr_err = (math.degrees(worker.simple_heading[a][l]) - true_deg
                                  + 180.0) % 360.0 - 180.0
                        print(f"RENEWAL_TURN_DONE arena={a} robot={l} tick={now} "
                              f"wall={wn} tangent_err_deg={tan_err:.1f} "
                              f"deadreckon_err_deg={dr_err:.1f}")

        elif state == "edge_follow":
            wall_name = worker.simple_wall_name[a][l]
            aux["path"] += move_i
            aux["seg_sum"] += move_i * worker.simple_heading[a][l]
            aux["seg_len"] += move_i
            sigma = aux["sigma"]

            def start_pivot(rot, why):
                if tangent_mode and why != "corner":
                    return          # tangent mode steers continuously; only corners pivot
                aux["mode"] = "pivot"
                aux["pivot_target"], aux["pivot_done"], aux["pivot_fresh"] = rot, 0.0, True
                if why != "peak":
                    # a big dead-reckoned rotation (corner) or a wall-contact
                    # recovery: the last fix no longer vouches for the heading,
                    # whatever little path it took. (Keeping the anchor through
                    # the corner pivot, with a fix from the pre-corner
                    # segments, was tried and lost: exit-heading p90 3 -> 24
                    # degrees; phase 175.)
                    aux["fix_valid"] = False
                    if why in ("pressed", "no_peak"):
                        aux["heading_trusted"] = False
                        aux["head_sigma"] = max(aux.get("head_sigma", fix_sigma), contact_sigma)
                if debug:
                    print(f"RENEWAL_PIVOT arena={a} robot={l} tick={now} rot_deg={math.degrees(rot):.1f} why={why}")

            # receptions: per-side EMA (corner test) and the followed side's
            # distance sample. walls is the one-winner narrowed event, so at
            # most one side is nonzero.
            for k in range(len(WALL_NAMES)):
                s_k = float(walls[i, k])
                if s_k > 0.0:
                    aux["s_last"][k] = now
                    aux["s_hist"][k] = (aux["s_hist"][k] + [s_k])[-3:]
                    prev = aux["s_ema"][k]
                    aux["s_ema"][k] = s_k if prev is None else \
                        (1.0 - trend_alpha) * prev + trend_alpha * s_k
            side_idx = WALL_NAMES.index(wall_name)
            s_now = float(walls[i, side_idx])
            d_now = None
            if s_now > 0.0:
                d_now = 1.0 / s_now - 1.0
                aux["last_side_tick"] = now
                aux["d_prev"] = aux["d_ema"]
                aux["d_ema"] = d_now if aux["d_ema"] is None else \
                    (1.0 - trend_alpha) * aux["d_ema"] + trend_alpha * d_now
                aux["ctrl"].append((aux["path"], d_now))

            # corner step: another side out-hears the followed one. Only a
            # side heard within stale_ticks counts -- an EMA keeps its last
            # value once receptions stop, and a frozen neighbour value beat
            # the followed side's between-seed dips and handed robots back
            # with the wrong sigma (phase 175 trace, 4 of 24 corners). The
            # side just handed off from is also excluded for 2 * PEAK_WINDOW
            # of path: it is still loud right after the handoff, and as the
            # new side's counter-directional neighbour it would hand
            # straight back and flip sigma wrong. And the neighbour must be
            # getting LOUDER (its last 3 receptions rising): hearing the next
            # wall proves the travel direction only while moving toward it.
            # A robot loitering at a corner after a recovery pivot (worse
            # under slip) otherwise "confirmed" whichever way it happened to
            # face (phase 175: 2/21 corners wrong at s1, 4/23 at s2). Never
            # mid-pivot, for the same reason.
            corner_to = None
            s_w = aux["s_ema"][side_idx] or 0.0
            recently_left = (aux["corner_path"] is not None
                             and aux["path"] - aux["corner_path"] < 2.0 * PEAK_WINDOW)
            for k, nm in enumerate(WALL_NAMES):
                if k == side_idx or (recently_left and nm == aux["prev_wall"]):
                    continue
                heard = aux["s_last"][k] is not None and now - aux["s_last"][k] <= stale_ticks
                hist = aux["s_hist"][k]
                approaching = len(hist) == 3 and hist[0] < hist[1] < hist[2]
                if aux["mode"] == "pivot" or not (heard and approaching):
                    continue
                if aux["s_ema"][k] is not None and aux["s_ema"][k] > s_w:
                    if resolve_corner(wall_name, nm) is not None:
                        corner_to = nm
            if corner_to is not None:
                sig = resolve_corner(wall_name, corner_to)
                if debug:
                    _tp, true_deg = _true_pose(worker, a, l)
                    true_sig = None
                    if true_deg is not None:
                        cw = math.degrees(tangent_heading(wall_name, SIGMA_LEFT))
                        true_sig = SIGMA_LEFT if math.cos(math.radians(true_deg - cw)) > 0 \
                            else SIGMA_RIGHT
                    print(f"RENEWAL_CORNER arena={a} robot={l} tick={now} {wall_name}->{corner_to} "
                          f"sigma={sig} prior_sigma={sigma} true_sigma={true_sig}")
                aux["prev_wall"], aux["corner_path"] = wall_name, aux["path"]
                aux["s_ema"][side_idx] = None
                worker.simple_wall_name[a][l] = corner_to
                wall_name = corner_to
                side_idx = WALL_NAMES.index(corner_to)
                aux["sigma"] = sigma = sig
                aux["sigma_confirmed"] = True
                # new seed line: restart the trend, the control window, the
                # peaks and the corrections on the new side.
                aux["ctrl"], aux["prev_peak"], aux["corrs"] = [], None, []
                aux["pk_along"] = None
                aux["goal_heading"] = None
                aux["calib_prev"] = None
                aux["det"] = {"min": None, "last_peak_path": None}
                aux["d_ema"] = aux["d_prev"] = None
                s_new = aux["s_ema"][side_idx]
                if s_new:
                    aux["d_ema"] = 1.0 / s_new - 1.0
                aux["last_side_tick"] = now
                d_now = None
                # the new wall is dead ahead: a quarter turn away from it, in
                # the direction the corner just proved.
                start_pivot(-sigma * math.pi / 2.0, "corner")

            stale = (now - aux["last_side_tick"]) > stale_ticks
            aux["ctrl"] = [c for c in aux["ctrl"] if c[0] >= aux["path"] - PEAK_WINDOW]
            y_hat = None if (stale or not aux["ctrl"]) else min(c[1] for c in aux["ctrl"])
            trend = 0
            if not stale and aux["d_prev"] is not None and aux["d_ema"] is not None:
                trend = -1 if aux["d_ema"] < aux["d_prev"] else (1 if aux["d_ema"] > aux["d_prev"] else 0)

            mode = aux["mode"]
            branch = mode
            m = (1.0, 1.0)
            if mode == "pivot":
                if aux["pivot_fresh"]:
                    aux["pivot_fresh"] = False   # this tick's dtheta predates the pivot
                else:
                    aux["pivot_done"] += float(dtheta[i])
                tgt = aux["pivot_target"]
                if aux["pivot_done"] * (1.0 if tgt >= 0 else -1.0) >= abs(tgt):
                    aux["mode"] = "track"
                    aux["det"]["min"] = None     # pre-pivot samples belong to no segment
                    aux["track_path"] = aux["path"]
                    m, branch = (1.0, 1.0), "track"
                else:
                    m = PIVOT_CCW if tgt >= 0 else PIVOT_CW
            elif mode == "approach":
                if y_hat is not None and y_hat <= track_enter:
                    aux["mode"] = "track"
                    aux["det"] = {"min": None, "last_peak_path": None}
                    aux["prev_peak"], aux["track_path"], aux["offsets"] = None, aux["path"], []
                    aux["pk_along"] = None
                    m, branch = (1.0, 1.0), "track"
                    a_app = approach_angle(aux["ctrl"])
                    if a_app is not None:
                        rot = pivot_command(a_app, y_hat, offset_set, conv_len,
                                            alpha_cmd_max, pivot_max, sigma)
                        if abs(rot) >= pivot_min:
                            start_pivot(rot, "track_entry")
                            m, branch = (PIVOT_CCW if rot >= 0 else PIVOT_CW), "pivot"
                else:
                    m, branch = approach_motor(trend, sigma, arc_inner)
            else:   # track: straight between seed passes, pivot at each peak
                if stale:
                    aux["mode"] = "approach"
                    m, branch = approach_motor(trend, sigma, arc_inner)
                else:
                    pk = None
                    if d_now is not None:
                        along_ax = 1 - WALL_AXIS[WALL_NAMES.index(wall_name)]
                        pk = peak_step(aux["det"], aux["path"], d_now, worker.simple_heading[a][l],
                                       peak_margin, peak_min_gap,
                                       extra=float(particles_batch[i, :, along_ax].mean()) * ARENA_HALF)
                    if pk is not None:
                        prev_pk, aux["prev_peak"] = aux["prev_peak"], pk
                        # the segment's path-weighted mean dead-reckoned heading:
                        # alpha_k is a path-averaged direction (it includes the
                        # stretch before the last pivot, and the pivot), so the
                        # heading it is compared with must be averaged the same
                        # way -- comparing with the heading AT the peak biased
                        # every fix by (pre-pivot fraction x pivot angle), and
                        # the pivots lean one way (phase 175)
                        seg_mean = (aux["seg_sum"] / aux["seg_len"]) if aux["seg_len"] > 1e-6 \
                            else worker.simple_heading[a][l]
                        aux["seg_sum"], aux["seg_len"] = 0.0, 0.0
                        aux["track_path"] = aux["path"]
                        aux["offsets"] = (aux["offsets"] + [pk[1]])[-3:]
                        # seed counting (along-wall odometry from geometry): one
                        # seed pass = exactly PEAK_WINDOW along the wall. Wheel
                        # odometry keeps advancing while a robot is held by a
                        # peer or the wall (no collision detection), and side-
                        # only seed ranges are periodic, so they never notice a
                        # whole-spacing slide: exits landed 30-126u off ALONG
                        # the wall at s2 (phase 175). Correct the cloud's along
                        # displacement since the last peak to the seed count.
                        along_axis = 1 - WALL_AXIS[WALL_NAMES.index(wall_name)]
                        # the cloud's along mean AT the closest approach
                        mean_along = pk[3] if (len(pk) > 3 and pk[3] is not None) else \
                            float(particles_batch[i, :, along_axis].mean()) * ARENA_HALF
                        if prev_pk is not None and aux["sigma_confirmed"] and aux["pk_along"] is not None:
                            dpath = pk[0] - prev_pk[0]
                            if dpath <= count_max_path:
                                dk_c = max(1, int(round(dpath / PEAK_WINDOW)))
                                th_t = tangent_heading(wall_name, sigma)
                                dir_along = math.cos(th_t) if along_axis == 0 else math.sin(th_t)
                                expected = PEAK_WINDOW * dk_c * (1.0 if dir_along > 0 else -1.0)
                                shift = expected - (mean_along - aux["pk_along"])
                                shift = max(-count_shift_max, min(count_shift_max, shift))
                                particles_batch[i, :, along_axis] = particles_batch[i, :, along_axis] + shift / ARENA_HALF
                                mean_along += shift
                                if debug and abs(shift) > 2.0:
                                    print(f"RENEWAL_COUNT arena={a} robot={l} tick={now} wall={wall_name} "
                                          f"dpath={dpath:.1f} dk={dk_c} shift={shift:.1f}")
                        aux["pk_along"] = mean_along
                        if pressed_against_wall(aux["offsets"], gap_lo, gap_hi):
                            aux["offsets"], aux["corrs"], aux["prev_peak"] = [], [], None
                            aux["pk_along"] = None
                            aux["calib_prev"] = None
                            start_pivot(-sigma * pivot_max, "pressed")
                            m, branch = (PIVOT_CCW if -sigma >= 0 else PIVOT_CW), "pivot"
                        elif prev_pk is not None:
                            alpha, dk = segment_angle(prev_pk, pk)
                            if debug:
                                _tp, true_deg = _true_pose(worker, a, l)
                                ta = "-"
                                if true_deg is not None:
                                    ta = "%.1f" % math.degrees(-sigma * _wrap(math.radians(true_deg)
                                                               - tangent_heading(wall_name, sigma)))
                                print(f"RENEWAL_PEAK arena={a} robot={l} tick={now} wall={wall_name} "
                                      f"offset={pk[1]:.2f} dk={dk} alpha_deg={math.degrees(alpha):.1f} "
                                      f"true_alpha_now_deg={ta}")
                            # heading re-anchor (loop 2): the segment's absolute
                            # heading is the confirmed tangent minus sigma*alpha.
                            # only segments clearly on the inner side: a slide
                            # along the wall reads alpha ~0 whatever the heading.
                            inner = not (in_wall_gap(prev_pk[1], gap_lo, gap_hi)
                                         and in_wall_gap(pk[1], gap_lo, gap_hi))
                            # corrections are COLLECTED under the assumed sigma and
                            # APPLIED only once it is confirmed (here, or at the
                            # corner that confirms it)
                            if abs(alpha) <= fix_alpha_max and inner:
                                th_true = tangent_heading(wall_name, sigma) - sigma * alpha
                                c_k = _wrap(th_true - seg_mean)
                                aux["corrs"] = (aux["corrs"] + [c_k])[-fix_peaks:]
                                # calibration pair: the raw-frame correction (c_k plus
                                # everything added to the heading beyond odometry) and
                                # the command integrals at this measurement; consecutive
                                # pairs within a visit give one regression equation.
                                c_raw = c_k + aux["frame_off"]
                                cur = (aux["int_l"], aux["int_r"], c_raw)
                                prev_c = aux.get("calib_prev")
                                if prev_c is not None and calib_on:
                                    d_c = _wrap(cur[2] - prev_c[2])
                                    if abs(d_c) < calib_jump_max:     # a collision is not slip
                                        aux["wheel"].add(cur[0] - prev_c[0], cur[1] - prev_c[1], d_c)
                                        if debug and aux["wheel"].n % 5 == 0:
                                            print(f"RENEWAL_CALIB arena={a} robot={l} tick={now} n={aux['wheel'].n} "
                                                  f"bL={aux['wheel'].b[0]:.4f} bR={aux['wheel'].b[1]:.4f}")
                                aux["calib_prev"] = cur
                                corr = None
                                if aux["sigma_confirmed"] and len(aux["corrs"]) >= fix_peaks:
                                    corr = consistent_correction(aux["corrs"], fix_spread_max)
                                if corr is not None:
                                    if fix_blend:
                                        prior = aux.get("head_sigma", fix_sigma) + drift * aux["path_since_fix"]
                                        gain, post = blend_gain(prior, aux["corrs"], fix_meas_floor)
                                        corr = corr * gain
                                        aux["head_sigma"] = post
                                    if debug:
                                        _tp, true_deg = _true_pose(worker, a, l)
                                        if true_deg is not None:
                                            hh = worker.simple_heading[a][l]
                                            before = _wrap(hh - math.radians(true_deg))
                                            after = _wrap(hh + corr - math.radians(true_deg))
                                            print(f"RENEWAL_HEADING_FIX arena={a} robot={l} tick={now} "
                                                  f"wall={wall_name} alpha_deg={math.degrees(alpha):.1f} "
                                                  f"err_before_deg={math.degrees(before):.1f} "
                                                  f"err_after_deg={math.degrees(after):.1f}")
                                    aux["frame_off"] += corr
                                    worker.simple_heading[a][l] += corr
                                    aux["seg_sum"] += corr * aux["seg_len"]
                                    if aux.get("goal_heading") is not None:
                                        aux["goal_heading"] += corr
                                    particles_batch[i, :, 2] = particles_batch[i, :, 2] + corr
                                    aux["prev_peak"] = (pk[0], pk[1], pk[2] + corr)
                                    aux["corrs"] = []
                                    aux["path_since_fix"] = 0.0
                                    aux["fix_valid"] = True
                                    aux["heading_trusted"] = True
                            if tangent_mode:
                                # smooth version of the pivot: a heading GOAL that
                                # turns toward the setpoint offset, steered to
                                # continuously (no stop-and-pivot, less contact)
                                rot = pivot_command(alpha, pk[1], tangent_offset_set, conv_len,
                                                    alpha_cmd_max, pivot_max, sigma)
                                aux["goal_heading"] = worker.simple_heading[a][l] + rot
                            rot = pivot_command(alpha, pk[1], offset_set, conv_len,
                                                alpha_cmd_max, pivot_max, sigma)
                            if abs(rot) >= pivot_min:
                                start_pivot(rot, "peak")
                                m, branch = (PIVOT_CCW if rot >= 0 else PIVOT_CW), "pivot"
                    elif aux["path"] - aux["track_path"] > nopeak_path:
                        # no seed passed where odometry says several should
                        # have been: pushing something (no collision
                        # detection) or veering. Pivot away from the wall.
                        aux["track_path"] = aux["path"]
                        aux["prev_peak"], aux["pk_along"] = None, None
                        start_pivot(-sigma * pivot_max, "no_peak")
                        m, branch = (PIVOT_CCW if -sigma >= 0 else PIVOT_CW), "pivot"

            if tangent_mode and aux["mode"] != "pivot":
                # The frozen oracle's wall phase: steer along the wall tangent
                # by the believed heading, at whatever offset the robot holds.
                # What it lacked under slip is a heading that stays right --
                # supplied here by the passive seed-pass fixes above.
                aux["mode"] = "track"
                hh = worker.simple_heading[a][l]
                th_t = aux.get("goal_heading")
                if th_t is None or stale:
                    # no seed-pass feedback yet (or the side went quiet): aim a
                    # little INTO the wall rather than along the bare tangent.
                    # Right after the turn the robot sits near the 7u IR edge,
                    # and a slightly-off heading carried it out of range before
                    # two seed passes could steer it (phase 175: 37 lost-wall
                    # restarts per run at s1). Toward the wall = rotating to the
                    # wall side: +sigma.
                    th_t = tangent_heading(wall_name, aux["sigma"]) + aux["sigma"] * tangent_inward
                    if stale:
                        aux["goal_heading"] = None
                m = _steer(np.array([math.cos(hh), math.sin(hh)]),
                           np.array([math.cos(th_t), math.sin(th_t)]))
                branch = "tangent"

            if debug and l < 6 and now % 25 == 0:
                # periodic trace of the controller's inputs against truth
                tp, tdeg = _true_pose(worker, a, l)
                fmt = lambda v: "-" if v is None else "%.2f" % v
                true_s = "-" if tp is None else "%.1f,%.1f" % (tp[0], tp[1])
                print(f"RENEWAL_EF arena={a} robot={l} tick={now} wall={wall_name} mode={aux['mode']} "
                      f"branch={branch} y_hat={fmt(y_hat)} d_ema={fmt(aux['d_ema'])} stale={stale} "
                      f"path={aux['path']:.1f} sigma={aux['sigma']} conf_sig={aux['sigma_confirmed']} "
                      f"true=({true_s}) est=({est_pos[0]:.1f},{est_pos[1]:.1f}) "
                      f"pos_err={'-' if tp is None else '%.1f' % float(np.linalg.norm(est_pos - np.asarray(tp)))} "
                      f"conf={conf_i:.2f} tdeg={'-' if tdeg is None else '%.0f' % tdeg}")

            # response test: a long arc toward sigma that never brought the
            # wall closer means the wall is on the other side.
            if branch == "far_arc":
                if aux["far_d0"] is None:
                    aux["far_d0"] = aux["d_ema"]
                aux["far_turn"] += abs(float(dtheta[i]))
                if aux["far_turn"] >= flip_turn and not aux["sigma_confirmed"]:
                    if aux["far_d0"] is None or aux["d_ema"] is None or aux["d_ema"] >= aux["far_d0"]:
                        aux["sigma"] = -aux["sigma"]
                        aux["corrs"] = []        # computed under the other side
                        if debug:
                            print(f"RENEWAL_SIDE_FLIP arena={a} robot={l} tick={now} sigma={aux['sigma']}")
                    aux["far_turn"], aux["far_d0"] = 0.0, None
            else:
                aux["far_turn"], aux["far_d0"] = 0.0, None

            # simple_oracle's slowdown law, re-based on the exit bar: give the
            # filter time to collapse before the robot drives past the anchor.
            # Only when conf is the LAST missing exit condition: before sigma
            # is confirmed and the heading anchored, slowing just strands a
            # confident robot short of the corner it needs.
            # anchored = the heading is trustworthy: a recent fix, OR a heading
            # budget (fix sigma + drift * path since the last fix / spawn's
            # known heading) still small -- without slip that is always true,
            # so the robot can leave right at the corner instead of first
            # tracking three segments on the new wall (phase 175: the frozen
            # oracle's tight stops come from leaving at the corner)
            budget = fix_sigma + drift * aux["path_since_fix"]
            anchored = (aux["fix_valid"] and aux["path_since_fix"] <= anchor_dist) or \
                (budget <= anchor_heading_max and aux.get("heading_trusted", True))
            approach_conf = 0.5 * exit_conf
            if conf_i <= approach_conf or not (aux["sigma_confirmed"] and anchored):
                speed_scale = 1.0
            else:
                frac = min((conf_i - approach_conf) / max(exit_conf - approach_conf, 1e-6), 1.0)
                speed_scale = 1.0 - frac * (1.0 - APPROACH_SLOWDOWN_MIN_SCALE)
            motor[i] = torch.tensor(m) * speed_scale

            ready = conf_i >= exit_conf and aux["sigma_confirmed"] and anchored
            leave = ready
            # Slip-conditional wall riding: under drift the blind leg's
            # cross-track error grows ~ drift * L^2 / 2, so when that
            # predicts a miss and the target still lies AHEAD along this wall,
            # keep following (cheap, measured travel) instead of leaving.
            # Without slip (drift 0) this never fires. (An unconditional
            # version with the pivoting controller lost on time; phase 175.)
            tgt = worker.simple_target.get(a, {}).get(l)
            if leave and drift > 0.0 and tgt is not None:
                d_tgt = float(np.linalg.norm(np.asarray(tgt) - est_pos))
                th_tr = tangent_heading(wall_name, aux["sigma"])
                ahead = float(np.dot(np.asarray(tgt) - est_pos, [math.cos(th_tr), math.sin(th_tr)]))
                if drift * d_tgt * d_tgt / 2.0 > leg_err_max and ahead > 4.0:
                    leave = False
            if leave:
                worker.simple_state[a][l] = "navigate"
            elif (now - aux["last_side_tick"]) > lost_ticks:
                worker.simple_state[a][l] = "seek_wall"

        elif state == "navigate":
            heading = worker.simple_heading[a][l]
            h_vec = np.array([math.cos(heading), math.sin(heading)])
            target = worker.simple_target.get(a, {}).get(l)
            delta = None        # set only when this tick steers to the target itself
            # budget: the posterior heading sigma at the last fix plus drift
            # since (the residual fraction once calibrated, if calibration is on)
            eff_drift = drift * (calib_residual if aux["wheel"].n >= calib_min_n else 1.0)
            sigma_theta = aux.get("head_sigma", fix_sigma) + eff_drift * aux["path_since_fix"]
            if target is None:
                # no resolvable target (no formation pool / unknown image):
                # mirror the frozen oracle's NaN-target stance -- spend the
                # time localizing.
                worker.simple_state[a][l] = "seek_wall"
                _wn, wp = nearest_wall_point(est_pos[0], est_pos[1])
                motor[i] = torch.tensor(_steer(h_vec, wp - est_pos))
            elif (conf_i < renew_min or (sigma_theta > heading_max and
                                         sigma_theta * float(np.linalg.norm(target - est_pos)) > renew_miss)) \
                    and aux.get("n_renewals", 0) < max_renewals and now < commit_tick:
                # THE RENEWAL TRANSITION: position cloud or heading budget
                # spent -- the latter only while the REMAINING blind distance
                # is long enough for that heading error to cost a miss
                # (sigma_theta * remaining > renew_miss); a robot already in
                # reach finishes instead of re-running a ~1000-tick wall cycle
                # (phase 175: every s1/s2 renewal was the heading budget).
                # Capped per robot: past max_renewals a robot commits to its
                # best estimate -- one still moving at the end scores zero for
                # certain, one that stops may land inside the bar.
                # Hysteresis against the edge_follow exit bar (exit_conf >
                # renew_min) keeps this from oscillating.
                aux["n_renewals"] = aux.get("n_renewals", 0) + 1
                worker.simple_state[a][l] = "seek_wall"
                _wn, wp = nearest_wall_point(est_pos[0], est_pos[1])
                motor[i] = torch.tensor(_steer(h_vec, wp - est_pos))
            else:
                # Landmark detour (sparse mid-arena seeds, seed_layout
                # "quarters"): under drift a long blind leg misses the 5u bar
                # (cross-track error ~ drift * L^2 / 2), so route through a
                # landmark: its range fixes position, and the touch fixes
                # heading from the leg's own displacement.
                goal = target
                if detour_on and drift > 0.0:
                    if not aux.get("leg_started"):
                        aux["leg_started"] = True
                        aux["leg_p0"], aux["leg_dr"] = est_pos.copy(), est_pos.copy()
                        aux["detour_lm"] = None
                        if float(np.linalg.norm(target - est_pos)) > detour_leg_min:
                            aux["detour_lm"] = choose_detour(est_pos, target, landmarks, detour_max, detour_gain)
                        if debug:
                            print(f"RENEWAL_DETOUR arena={a} robot={l} tick={now} lm={aux['detour_lm']} "
                                  f"est=({est_pos[0]:.0f},{est_pos[1]:.0f}) tgt=({target[0]:.0f},{target[1]:.0f})")
                    aux["leg_dr"] = aux["leg_dr"] + move_i * h_vec
                    k = aux.get("detour_lm")
                    if k is not None:
                        lm = landmarks[k]
                        if landmarks_on:
                            s_k = lm_str[i] if lm_idx[i] == k else 0.0
                        else:
                            s_k = float(event_seed_obs[i, k]) if event_seed_obs is not None else 0.0
                        touched = s_k > 0.0 and (1.0 / s_k - 1.0) <= touch_d
                        if touched:
                            phi = landmark_heading_fix(aux["leg_p0"], aux["leg_dr"], lm)
                            if phi is not None and abs(phi) <= 0.6:
                                worker.simple_heading[a][l] += phi
                                particles_batch[i, :, 2] = particles_batch[i, :, 2] + phi
                                aux["path_since_fix"], aux["fix_valid"] = 0.0, True
                                heading = worker.simple_heading[a][l]
                                h_vec = np.array([math.cos(heading), math.sin(heading)])
                            if debug:
                                _tp, true_deg = _true_pose(worker, a, l)
                                he = "-" if true_deg is None else "%.1f" % math.degrees(
                                    _wrap(heading - math.radians(true_deg)))
                                print(f"RENEWAL_LANDMARK arena={a} robot={l} tick={now} lm={k} "
                                      f"fix_deg={'-' if phi is None else '%.1f' % math.degrees(phi)} "
                                      f"heading_err_after={he}")
                            aux["detour_lm"] = None
                        elif s_k > 0.0:
                            aux["search_t"] = None       # heard it: the range ring is homing us in
                            goal = lm
                        elif float(np.linalg.norm(lm - est_pos)) < 2.0 or aux.get("search_t") is not None:
                            # believed there, never heard it: the robot is
                            # truly > 7u off (the drift this detour exists
                            # for). Expanding spiral until heard, bounded.
                            if aux.get("search_t") is None:
                                aux["search_t"] = 0
                            aux["search_t"] += 1
                            if aux["search_t"] > search_ticks:
                                aux["detour_lm"], aux["search_t"] = None, None
                                if debug:
                                    print(f"RENEWAL_DETOUR_GIVEUP arena={a} robot={l} tick={now} lm={k}")
                            else:
                                inner = min(0.95, 0.3 + 0.65 * aux["search_t"] / search_ticks)
                                motor[i] = torch.tensor(arc_motor(True, inner))
                                goal = None
                        else:
                            goal = lm
                if goal is None:
                    delta = None        # spiral search already commanded the motor
                elif goal is not target:
                    gd = goal - est_pos
                    gdist = float(np.linalg.norm(gd))
                    motor[i] = torch.tensor(_steer(h_vec, gd / gdist if gdist > 1e-9 else h_vec))
                    delta = None
                else:
                    delta = target - est_pos
            if delta is not None:
                dist = float(np.linalg.norm(delta))
                # Arrive well inside the settle bar (tau_v, 5u): stopping the
                # moment the belief says "within 5u" puts every residual error
                # on the wrong side of the bar half the time.
                if dist < arrive_dist:
                    worker.simple_state[a][l] = "arrived"
                    motor[i] = torch.tensor([0.0, 0.0])
                else:
                    g_vec = delta / dist if dist > 1e-9 else h_vec
                    scale = 1.0 if dist >= near_dist else max(0.3, dist / near_dist)
                    motor[i] = torch.tensor(_steer(h_vec, g_vec)) * scale

        elif state == "arrived":
            motor[i] = torch.tensor([0.0, 0.0])

        if peer_comms:
            if not hasattr(worker, "renewal_beacon"):
                worker.renewal_beacon = {}
            if worker.simple_state[a][l] == "arrived" and conf_i >= peer_conf_min:
                worker.renewal_beacon.setdefault(a, {})[l] = (float(br_batch[i, 0]), float(br_batch[i, 1]), conf_i)
            else:
                worker.renewal_beacon.setdefault(a, {}).pop(l, None)

        # Peer sidestep (crowd at the shape; no collision detection): a robot
        # in seek_wall / navigate that keeps hearing a peer at contact range
        # is pressed against a parked robot -- its odometry "drives" while it
        # stands still. Detected from the reception's own strength (range =
        # 1/s - 1), the robot pivots a quarter turn (alternating sides) and
        # drives a short straight before resuming: around the parked robot,
        # not through it. Overrides only the motor; the state is untouched.
        cur_state = worker.simple_state[a][l]
        if sidestep_on and cur_state in ("seek_wall", "navigate") and event_tc is not None:
            ps = float(event_tc[i, MESSAGE_SIZE])
            close = ps > 0.0 and (1.0 / ps - 1.0) < sidestep_dist
            aux["close_n"] = min(aux.get("close_n", 0.0) + 1.0, 20.0) if close else max(aux.get("close_n", 0.0) - 0.25, 0.0)
            if aux.get("side_phase") is None and aux["close_n"] >= sidestep_count:
                aux["side_n"] = aux.get("side_n", 0) + 1
                aux["side_phase"], aux["side_rot"] = "pivot", 0.0
                aux["side_dir"] = 1.0 if aux["side_n"] % 2 else -1.0
                aux["close_n"] = 0.0
                if debug:
                    print(f"RENEWAL_SIDESTEP arena={a} robot={l} tick={now} state={cur_state}")
            if aux.get("side_phase") == "pivot":
                aux["side_rot"] += abs(float(dtheta[i]))
                motor[i] = torch.tensor(PIVOT_CCW if aux["side_dir"] > 0 else PIVOT_CW)
                if aux["side_rot"] >= math.pi / 2:
                    aux["side_phase"], aux["side_path"] = "straight", 0.0
            elif aux.get("side_phase") == "straight":
                aux["side_path"] += move_i
                motor[i] = torch.tensor([1.0, 1.0])
                if aux["side_path"] >= sidestep_len:
                    aux["side_phase"] = None
        elif aux.get("side_phase") is not None:
            aux["side_phase"] = None

        belief_dict[a][l] = particles_batch[i].detach().clone()

        # The label semantics are simple_oracle's: the state that PRODUCED
        # motor[i]. seek_wall and turn command their own motor and only then
        # transition, so the post-transition state would mislabel them;
        # navigating is the one case where the transition also replaces the
        # motor (with arrived's own [0, 0]).
        label = state
        if state == "navigate" and worker.simple_state[a][l] == "arrived":
            label = "arrived"
        if not hasattr(worker, "simple_motor_state"):
            worker.simple_motor_state = {}
        worker.simple_motor_state.setdefault(a, {})[l] = label

        if debug:
            new_state = worker.simple_state[a][l]
            if new_state != state:
                tp, tdeg = _true_pose(worker, a, l)
                pos_err = "-" if tp is None else "%.1f" % float(np.linalg.norm(est_pos - np.asarray(tp)))
                hd_err = "-" if tdeg is None else "%.1f" % math.degrees(
                    _wrap(worker.simple_heading[a][l] - math.radians(tdeg)))
                true_s = "-" if tp is None else "%.1f,%.1f" % (tp[0], tp[1])
                print(f"RENEWAL_ORACLE arena={a} robot={l} tick={now} "
                      f"{state}->{new_state} heading_deg={math.degrees(worker.simple_heading[a][l]) % 360.0:.1f} "
                      f"est_pos=({est_pos[0]:.1f},{est_pos[1]:.1f}) true_pos=({true_s}) conf={conf_i:.2f} "
                      f"pos_err={pos_err} heading_err_deg={hd_err} "
                      f"bias=({aux['wheel'].b[0]:.4f},{aux['wheel'].b[1]:.4f}) n={aux['wheel'].n}")

        if getattr(cfg, "oracle_send_visual_state", False):
            if not hasattr(worker, "oracle_visual_state"):
                worker.oracle_visual_state = {}
            # KilobotAgent.SetVisualState's own 0..4 convention (see its
            # switch): ivory, amber (sustained search), gold (brief
            # transition), deep red, green.
            vstate = {"seek_wall": 0, "turn": 2, "edge_follow": 1,
                      "navigate": 3, "arrived": 4}[worker.simple_state[a][l]]
            worker.oracle_visual_state.setdefault(a, {})[l] = vstate

    return motor
