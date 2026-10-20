"""Gate-2 homing probe: can a lost robot convert intermittent strength
receptions into directed approach to a parked beacon?

Standalone -- no Unity, no network. The physics is the project's own verified
kinematics (kinematics.split_tick_motion, the M-segment polygon the simulator
actually walks, with config.py's derived prop_max_speed / prop_wheelbase) and
the project's IR strength law s = 1/(1 + d) (SwarmManager.BuildMessage).

The question this answers before any interface work: a receiver that has NO
bearing sensor -- one scalar strength per reception, exact odometry between
receptions, receptions only while within IR_RANGE and only with probability
p per decision (the single-sender lottery) -- can it reach a parked beacon at
all, how fast, and how does that degrade under contention-style reception
loss? Three laws:

  gradient   least-squares ascent over the last K receptions' (odometry
             position, strength) pairs; holds the last ascent bearing between
             receptions (the intermittent-contact case)
  chemo      bang-bang: strength rose since the last reception -> keep
             straight; fell -> turn +-90 deg
  random     random-walk baseline (no information), the floor every law must
             beat

Success = within SUCCESS_DIST of the beacon within --max-ticks decisions.
Reports success rate, median ticks-to-arrival, median receptions, and path
efficiency per (distance, reception probability, law).
"""

import argparse
import math
import numpy as np

# --- project constants (config.py derived values, belief.ARENA_HALF) ---
MAX_SPEED = 4.0          # units/s at full wheel speed
WHEELBASE = 1.2732395447351628   # 4/pi
DT = 0.02                # Unity fixed timestep
STEPS = 4                # physics steps per decision (frame skip)
FRAMES = 4               # SceneBootstrap.framesPerStep
IR_RANGE = 7.0           # SwarmManager.cs IR_RANGE
T_MAX = 8000             # decision budget, matches the eval harness horizon
SUCCESS_DIST = 1.0
CONF_LOCALIZED = 0.4     # not used here; documented for parity with the probe

T_STEP = STEPS * DT      # 0.08 s per decision
M_SUB = STEPS * FRAMES   # polygon segments per decision


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def step(pos, theta, L, R):
    """One decision's motion: exactly kinematics.split_tick_motion's polygon."""
    v = 0.5 * (L + R) * MAX_SPEED
    omega = (R - L) * MAX_SPEED / WHEELBASE
    dtheta = omega * T_STEP
    half = dtheta / (2.0 * M_SUB)
    if abs(half) < 1e-6:
        mag = v * T_STEP
    else:
        mag = v * (T_STEP / M_SUB) * math.sin(0.5 * dtheta) / math.sin(half)
    travel_dir = theta + 0.5 * dtheta + half
    return pos + mag * np.array([math.cos(travel_dir), math.sin(travel_dir)]), theta + dtheta


def steer(pos, theta, bearing, deadband=0.2):
    """Turn toward bearing (world frame), else full speed straight.
    Bang-bang with the oracle's own TURN_MOTOR differential."""
    e = wrap(bearing - theta)
    if abs(e) <= deadband:
        return 1.0, 1.0
    return (0.9, 0.15) if e < 0 else (0.15, 0.9)


def fit_beacon(receptions):
    """Nonlinear least squares for the beacon position q in odometry frame:
    s_i = 1/(1 + |p_i - q|). Gauss-Newton from 8 inits around the peak
    reception; returns (q, rms). This is the estimator the belief filter
    itself could run -- it inverts the physics instead of linearizing it."""
    P = np.array([r[0] for r in receptions])
    S = np.array([r[1] for r in receptions])
    i0 = int(np.argmax(S))
    d0 = 1.0 / max(S[i0], 1e-6) - 1.0
    best_q, best_rms = None, float("inf")
    for k in range(8):
        phi = 2.0 * math.pi * k / 8.0
        q = P[i0] + d0 * np.array([math.cos(phi), math.sin(phi)])
        for _ in range(30):
            dd = np.maximum(np.linalg.norm(P - q, axis=1), 1e-3)
            R = 1.0 + dd
            resid = 1.0 / R - S
            J = (P - q) / ((R ** 2) * dd)[:, None]
            dq, *_ = np.linalg.lstsq(J, -resid, rcond=None)
            nrm = np.linalg.norm(dq)
            if not np.isfinite(nrm):
                break
            dq = dq * min(1.0, 2.0 / max(nrm, 1e-9))
            q = q + dq
            if np.linalg.norm(dq) < 1e-4:
                break
        rms = float(np.sqrt(np.mean((1.0 / (1.0 + np.linalg.norm(P - q, axis=1)) - S) ** 2)))
        if rms < best_rms:
            best_rms, best_q = rms, q
    return best_q, best_rms


def run_trial(rng, law, dist, p_recv, heading_drift=0.0):
    ang = rng.uniform(0.0, 2.0 * math.pi)
    beacon = np.array([dist * math.cos(ang), dist * math.sin(ang)])
    start_ang = rng.uniform(0.0, 2.0 * math.pi)
    pos = beacon + dist * np.array([math.cos(start_ang), math.sin(start_ang)])
    theta = rng.uniform(0.0, 2.0 * math.pi)
    odo = np.zeros(2)          # odometry pose relative to homing start
    odo_heading = 0.0          # heading accumulated since anchor (split_track_update's track[2])
    theta0 = theta             # world heading at anchor: odo frame = world rotated by -theta0
    receptions = []            # (odo_pos, strength)
    K = 60                     # global buffer; spans passes => long baseline
    last_s = None
    prev_heading = theta       # remembered bearing between receptions
    out_ticks = 0              # decisions since last reception
    GRACE = 80                 # hold bearing this long after losing contact
    reorient_cooldown = 0      # decisions since the last decline reorientation
    n_receptions = 0
    path = 0.0
    first_rx_tick = -1         # for the conditional (post-contact) stats
    rms_ok = False
    for t in range(T_MAX):
        d = float(np.linalg.norm(beacon - pos))
        if d <= SUCCESS_DIST:
            homing_t = (t - first_rx_tick) if first_rx_tick >= 0 else -1
            return True, t, n_receptions, path / max(dist, 1e-6), homing_t
        # --- reception (parked beacon broadcasts every decision) ---
        heard = False
        rose = False
        if d <= IR_RANGE and rng.uniform() < p_recv:
            n_receptions += 1
            if first_rx_tick < 0:
                first_rx_tick = t
            s = 1.0 / (1.0 + d)
            rose = last_s is not None and s > last_s
            last_s = s
            receptions.append((odo.copy(), s))
            if len(receptions) > K:
                receptions.pop(0)
            heard = True
        if heard:
            out_ticks = 0
        else:
            out_ticks += 1
        reorient_cooldown = max(0, reorient_cooldown - 1)
        # --- choose motors ---
        if law == "random":
            theta += rng.uniform(-0.3, 0.3)
            L = R = 1.0
        elif law == "chemo":
            # pre-contact: no information, maximize encounter rate
            if not receptions:
                theta += rng.uniform(-0.3, 0.3)
                L = R = 1.0
            else:
                # in range: strength rose -> keep heading; fell -> ONE reorient
                # +-90 (refractory-guarded, no per-tick thrash). Out of range
                # past the grace window -> sweep back through the disc.
                if heard and not rose and last_s is not None and reorient_cooldown == 0:
                    prev_heading = theta + rng.choice([-1.0, 1.0]) * math.pi / 2.0
                    reorient_cooldown = 20
                if out_ticks > GRACE:
                    prev_heading = theta + math.pi + rng.uniform(-0.5, 0.5)
                    out_ticks = 0
                L, R = steer(pos, theta, prev_heading)
        else:  # gradient: invert the strength law for q, steer at it
            bearing = prev_heading
            if len(receptions) >= 4:
                q, rms = fit_beacon(receptions)
                if rms < 0.02:
                    # odo-frame bearing to q -> world frame
                    psi = math.atan2(q[1] - odo[1], q[0] - odo[0])
                    bearing = wrap(psi + (theta - odo_heading))
            if len(receptions) == 0:
                # pre-contact: no information, maximize encounter rate
                theta += rng.uniform(-0.3, 0.3)
                L = R = 1.0
            else:
                if heard and not rose and last_s is not None and reorient_cooldown == 0:
                    # fit not yet trustworthy or moving down-gradient: one
                    # perpendicular reorientation, then let the fit speak
                    bearing = theta + rng.choice([-1.0, 1.0]) * math.pi / 2.0
                    reorient_cooldown = 20
                if out_ticks > GRACE:
                    bearing = theta + math.pi + rng.uniform(-0.5, 0.5)
                    out_ticks = 0
                prev_heading = bearing
                L, R = steer(pos, theta, bearing)
        # --- act + odometry (exact position update; heading drift = the
        # real system's dominant odometry error, compounds with travel) ---
        npos, ntheta = step(pos, theta, L, R)
        # odometry sees the SAME commanded motion; each interval's local
        # motion is rotated by the heading accumulated since the anchor
        # before being added -- exactly kinematics.split_track_update.
        v = 0.5 * (L + R) * MAX_SPEED
        omega_cmd = (R - L) * MAX_SPEED / WHEELBASE
        dtheta_cmd = omega_cmd * T_STEP
        half = dtheta_cmd / (2.0 * M_SUB)
        mag = v * (T_STEP / M_SUB) * math.sin(0.5 * dtheta_cmd) / math.sin(half) \
            if abs(half) >= 1e-6 else v * T_STEP
        dir_local = 0.5 * dtheta_cmd + half
        x_loc, y_loc = mag * math.cos(dir_local), mag * math.sin(dir_local)
        cos_h, sin_h = math.cos(odo_heading), math.sin(odo_heading)
        odo = odo + np.array([x_loc * cos_h - y_loc * sin_h,
                              x_loc * sin_h + y_loc * cos_h])
        odo_heading += dtheta_cmd
        path += mag
        pos, theta = npos, ntheta + heading_drift * rng.standard_normal()
    return False, T_MAX, n_receptions, path / max(dist, 1e-6), -1


def main():
    global T_MAX
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-ticks", type=int, default=T_MAX)
    args = ap.parse_args()
    T_MAX = args.max_ticks

    print("homing probe: parked beacon, IR %.0fu, %.2f u/decision at full speed, "
          "budget %d decisions" % (IR_RANGE, MAX_SPEED * T_STEP, T_MAX))
    print("cols: success | med ticks | med recv | eff || GIVEN CONTACT: "
          "success | med homing ticks (arrival minus first reception)")
    print("%-8s %-5s %-4s | %-7s %-6s %-5s %-5s | %-7s %s" %
          ("law", "dist", "p", "succ", "ticks", "recv", "eff", "c-succ", "c-ticks"))
    for law in ("random", "chemo", "gradient"):
        for dist in (20.0, 40.0, 60.0, 80.0):
            for p in (1.0, 0.3):
                rng = np.random.default_rng(args.seed + int(dist) * 10 + int(p * 10))
                res = [run_trial(rng, law, dist, p) for _ in range(args.trials)]
                ok = [r for r in res if r[0]]
                sr = len(ok) / len(res)
                mt = int(np.median([r[1] for r in ok])) if ok else -1
                mr = int(np.median([r[2] for r in ok])) if ok else -1
                me = float(np.median([r[3] for r in ok])) if ok else -1
                contact = [r for r in res if r[2] >= 4]
                cs = len([r for r in contact if r[0]]) / len(contact) if contact else -1.0
                ct = int(np.median([r[4] for r in contact if r[0]])) if contact and any(r[0] for r in contact) else -1
                print("%-8s %-5.0f %-4.1f | %-7.3f %-6s %-5s %-5s | %-7.3f %s" %
                      (law, dist, p, sr, str(mt), str(mr), ("%.2f" % me) if me >= 0 else "-1", cs, str(ct)))


if __name__ == "__main__":
    main()