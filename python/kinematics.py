"""Dead-reckoning kinematics: commanded motors -> estimated motion.

Pure torch math with no network or simulator dependency. One tick's motor
command, held for a known number of physics steps, integrated into the
displacement and heading change it produced.

Two consumers, at different granularities:

  dead_reckon        the `prop` vector the DeepSet/GRU actors consume -- one
                     interval summarised as distance, chord, heading change,
                     elapsed time and a running odometer
  split_tick_motion  the same physics reported as raw local-frame motion, which
                     belief.belief_predict integrates per particle and the
                     split_track_* anchor trackers accumulate

The sign of omega here decides whether the belief filter's whole particle cloud
drifts, since every particle shares this formula. See docs/code-history.md.
"""

import torch

# Substeps the simulator runs per environment step (SceneBootstrap.framesPerStep).
# The motion below is a polygon with this many segments, not a smooth arc, so
# this is part of the physics rather than a tuning knob.
FRAMES_PER_STEP = 4

PROP_SIZE = 6   # [path_len, euclid_disp, sin(dheading), cos(dheading), elapsed_time, cumulative_distance]


def dead_reckon(last_motor, steps, cum_dist, max_speed, wheelbase, dt, scale, time_scale, cum_scale):
    # Estimate the robot's own motion since the last tick from the command it held, and
    # carry two signals the network cannot otherwise recover: how much TIME elapsed this
    # interval (which vanishes if you only report displacement, e.g. when the robot is
    # stationary), and the total DISTANCE travelled so far this episode (a running
    # odometer the one-step GRU cannot integrate on its own).
    # last_motor (N,2) in [0,1] = (left, right); steps (N,) physics ticks held;
    # cum_dist (N,) running path length before this interval. Returns prop (N, PROP_SIZE) =
    # [path_len*scale, euclid*scale, sin(dtheta), cos(dtheta), elapsed_time*time_scale,
    #  cumulative_distance*cum_scale].
    L = last_motor[:, 0]
    R = last_motor[:, 1]
    vL = L * max_speed
    vR = R * max_speed
    v = 0.5 * (vL + vR)
    # (vR - vL), verified against real Unity rather than an assumed
    # turnRate=(left-right)*turnSpeed formula. Getting this sign wrong gives
    # every particle the same heading-drift bias, so the cloud stays tightly
    # clustered -- confident -- while walking away from the truth.
    # See docs/code-history.md.
    omega = (vR - vL) / wheelbase
    t = steps.to(v.dtype) * dt
    path = v * t          # arc length; identical for the polygon, being the sum of its segments
    dtheta = omega * t
    straight = v * t
    # Straight-line distance across the interval. The same M-segment polygon
    # split_tick_motion documents, not a circular arc -- see its docstring.
    m = (steps.to(v.dtype) * float(FRAMES_PER_STEP)).clamp(min = 1.0)
    half_sub = dtheta / (2.0 * m)
    small = half_sub.abs() < 1e-6
    safe_sub = torch.where(small, torch.ones_like(half_sub), half_sub)
    polygon = v * (t / m) * torch.sin(0.5 * dtheta) / torch.sin(safe_sub)
    euclid = torch.where(small, straight, polygon).abs()
    return torch.stack([path * scale, euclid * scale, torch.sin(dtheta), torch.cos(dtheta),
                     t * time_scale, cum_dist * cum_scale], dim=1)


def split_tick_motion(last_motor, steps, max_speed, wheelbase, dt,
                      frames_per_step = FRAMES_PER_STEP):
    """This interval's local-frame motion: (x_local, y_local, dtheta, t).

    (x_local, y_local) is the displacement expressed in the robot's frame at the
    START of the interval; dtheta is the total heading change.

    Models what the simulator ACTUALLY does, which is not a continuous arc.
    KilobotMovement.FixedUpdate rotates first and then translates along the NEW
    heading, so an interval of M = steps * frames_per_step substeps is M equal
    segments laid at headings 1*d, 2*d, ... M*d, where d = dtheta/M. Summing
    that geometric series gives, exactly:

        |displacement| = v * (t/M) * sin(dtheta/2) / sin(dtheta/(2M))
        direction      = (M+1)/(2M) * dtheta

    The continuous-arc form (magnitude 2(v/omega)sin(dtheta/2), direction
    dtheta/2) is the M -> infinity limit of both. At the real M the direction
    differs by dtheta/(2M) -- 0.45 degrees per step at full spin, measured
    against a live player -- which walks the belief filter's position sideways
    even while its magnitude and heading are exact. tools/check_dead_reckoning.py
    is what measures this.
    """
    L = last_motor[:, 0]
    R = last_motor[:, 1]
    vL = L * max_speed
    vR = R * max_speed
    v = 0.5 * (vL + vR)
    # Same sign convention as dead_reckon above -- see that comment. This is
    # the function that feeds the belief filter's dtheta (belief_predict), so
    # the sign here directly determines whether the whole cloud drifts.
    omega = (vR - vL) / wheelbase
    t = steps.to(v.dtype) * dt
    dtheta = omega * t

    m = (steps.to(v.dtype) * float(frames_per_step)).clamp(min = 1.0)
    half_sub = dtheta / (2.0 * m)
    straight = v * t
    # guard the 0/0 at dtheta -> 0, where both factors vanish together
    small = half_sub.abs() < 1e-6
    safe_sub = torch.where(small, torch.ones_like(half_sub), half_sub)
    polygon = v * (t / m) * torch.sin(0.5 * dtheta) / torch.sin(safe_sub)
    euclid = torch.where(small, straight, polygon)
    # (M+1)/(2M) * dtheta, i.e. the arc's dtheta/2 plus one half-substep
    travel_dir = 0.5 * dtheta + half_sub
    x_local = euclid * torch.cos(travel_dir)
    y_local = euclid * torch.sin(travel_dir)
    return x_local, y_local, dtheta, t


def split_track_update(track, x_local, y_local, dtheta, t):
    # track columns: [dx, dy, heading_accum, elapsed], dx/dy in the frame fixed at the
    # last reset (anchor). Rotate this tick's local motion by the heading accumulated so
    # far before adding it in, so dx/dy stay expressed in the anchor's fixed frame.
    dx = track[:, 0]
    dy = track[:, 1]
    heading = track[:, 2]
    elapsed = track[:, 3]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    x_anchor = x_local * cos_h - y_local * sin_h
    y_anchor = x_local * sin_h + y_local * cos_h
    new_dx = dx + x_anchor
    new_dy = dy + y_anchor
    new_heading = heading + dtheta
    new_elapsed = elapsed + t
    return torch.stack([new_dx, new_dy, new_heading, new_elapsed], dim = 1)


def split_track_read(track, scale, time_scale):
    # reads a track's current state as (distance*scale, sin(bearing), cos(bearing),
    # elapsed*time_scale), where bearing points from the robot's current position back
    # toward the anchor, expressed relative to the robot's current heading -- rotating
    # the anchor-frame position by minus the heading accumulated since the anchor.
    dx = track[:, 0]
    dy = track[:, 1]
    heading = track[:, 2]
    elapsed = track[:, 3]
    distance = torch.sqrt(dx * dx + dy * dy)
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    bx = -dx * cos_h - dy * sin_h
    by = dx * sin_h - dy * cos_h
    zero_dist = distance < 1e-6
    safe_dist = torch.where(zero_dist, torch.ones_like(distance), distance)
    cos_bearing = torch.where(zero_dist, torch.ones_like(distance), bx / safe_dist)
    sin_bearing = torch.where(zero_dist, torch.zeros_like(distance), by / safe_dist)
    return torch.stack([distance * scale, sin_bearing, cos_bearing, elapsed * time_scale], dim = 1)
