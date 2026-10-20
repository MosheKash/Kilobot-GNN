"""The evidence-sharing channel (config.evidence_comms): machinery for the
redesigned message semantics.

Design (docs/tuning.md, the evidence-sharing proposal; Gates 1-2 measured):

  Writers   slots 0-2 stay the honest beacon under the anchored-conf sender
            gate (beacon_anchored_min_conf -- the pool Gate 1 measured at 57%
            lost-phase hit rate). Slots 3-5 carry a DISCLOSURE: the sender's
            last DIRECT sighting (which landmark or wall side, its measured
            range, how fresh) -- a raw measurement record, not a posterior.
            Slot 6 is the learned broadcast veto (T2): the policy can only
            make the swarm quieter, never lie louder.

  Consumers the claim store: beacons heard at any time are RETAINED per sender
            and propagated through the receiver's own odometry, so claims met
            at different times and places can be consumed JOINTLY by
            belief.add_peer_fix -- the existing consensus machinery (pair
            separation cap, >=3 residual gate, spread floor, cooldown,
            bounded-mass injection). A claim is never fused alone; two
            independently-wrong latches rarely agree with each other AND the
            receiver's support. Sigmas age with odometry path, the same 0.02
            variance convention the peer code already uses.

  Homing    the referral store selects WHICH sender to approach (freshest
            sighting, then smallest landmark range -- a sender that was AT
            the landmark when it saw it). The homing controller tracks that
            sender's signal stream (reception strength + odometry) and
            inverts the strength law for the sender's position -- Gate 2's
            result: ~100% conversion to arrival in ~25 decisions once contact
            exists. Arrival at a referred landmark yields the receiver's own
            DIRECT fix; nothing on this path is trusted without physics.

Everything here reads only data already on the wire or in the filter's own
inputs. Nothing privileged enters the actor; nothing here writes to the wire
except through actor_io's writer block.
"""

import math
import random

import numpy as np
import torch

from belief import ARENA_HALF, PEER_SIGMA, PEER_SIGMA_FLOOR
from kilobot_gnn import MESSAGE_SIZE

# disclosure slot layout (message content columns)
DISCLOSURE_ID_COL = 3
DISCLOSURE_RANGE_COL = 4
DISCLOSURE_RECENCY_COL = 5
VETO_COL = 6
SELF_TRUST_COL = 7          # T2b: the sender's graded self-trust annotation
DISCLOSURE_SCALE = 10.0     # landmark id k encoded as (k+1)/10; 0 = silent
WALL_ID_OFFSET = 4          # ids 0-3 corners, 4-7 wall sides (SEED_SIZE=4)

CONF_VARIANCE_SCALE = 0.02  # belief's conf convention: conf = exp(-var/0.02)


def encode_disclosure(landmark_id, range_norm, recency):
    """(id, range, recency) -> the three content floats."""
    return ((landmark_id + 1) / DISCLOSURE_SCALE,
            float(min(max(range_norm, 0.0), 1.0)),
            float(min(max(recency, 0.0), 1.0)))


def decode_disclosure(v_id, v_range, v_recency):
    """Content floats -> (landmark_id, range_norm, recency) or None."""
    k = int(round(float(v_id) * DISCLOSURE_SCALE)) - 1
    if k < 0:
        return None
    return k, float(v_range), float(v_recency)


# ─── claim store ──────────────────────────────────────────────────────────────

def update_claims(worker, arena_ids, locals_, rows, valid, cfg, device):
    """Refresh the per-robot claim store from this tick's reception rows.

    A claim = one sender's latest beacon (slots 0-2). Rows with beacon content
    are stored no matter whose they are; the consumption-side PEER_CONF_MIN
    gate does the vetting, so the store holds candidates, not beliefs. Keyed
    by sender id (latest wins), capped at cfg.claim_store_capacity with
    oldest-reception eviction.
    """
    if not getattr(cfg, "evidence_comms", False) or rows is None:
        return
    store = getattr(worker, "claim_store", None)
    if store is None:
        store = {}
        worker.claim_store = store
    rows_d = rows.to(device)
    valid_d = valid.to(device) if valid is not None else torch.ones_like(rows_d[:, :, 0], dtype=torch.bool)
    cap = int(getattr(cfg, "claim_store_capacity", 8))
    for i in range(rows_d.shape[0]):
        a, l = int(arena_ids[i]), int(locals_[i])
        now = int(worker.step_count.get(a, 0))
        mine = store.setdefault((a, l), {})
        peer = valid_d[i] & (rows_d[i, :, MESSAGE_SIZE] >= 1.0)   # senderId col = MESSAGE_SIZE
        content = peer & (rows_d[i, :, 2] > 0.0)
        for k in torch.nonzero(content).flatten().tolist():
            sid = int(rows_d[i, k, MESSAGE_SIZE])
            mine[sid] = {"x": float(rows_d[i, k, 0]), "y": float(rows_d[i, k, 1]),
                         "conf": float(rows_d[i, k, 2]),
                         "strength": float(rows_d[i, k, 10]), "rx_tick": now,
                         "dx": 0.0, "dy": 0.0, "heading": 0.0, "path": 0.0,
                         "trust": float(min(max(float(rows_d[i, k, SELF_TRUST_COL]), -1.0), 1.0))}
        if len(mine) > cap:
            for sid in sorted(mine, key=lambda s: mine[s]["rx_tick"])[:len(mine) - cap]:
                del mine[sid]


def propagate_claims(worker, arena_ids, locals_, x_local, y_local, dtheta):
    """Move every stored claim's ring by this tick's own motion.

    A claim constrains the receiver's position AT RECEPTION: |p_rx - claimed|
    = r. Propagated to now it constrains |p_now - (claimed + displacement)| =
    r, so each claim carries its own (dx, dy, heading) displacement track --
    the same rotate-then-accumulate semantics as kinematics.split_track_update,
    one track per claim. path accumulates travel for the sigma aging.

    Units: split_tick_motion reports RAW arena units; the claim tracks live
    in the filter's NORMALIZED frame (same frame as the stored claimed
    positions), so the boundary divides by ARENA_HALF -- exactly what
    belief.belief_predict does with the same tensors. dtheta is radians
    either way.
    """
    store = getattr(worker, "claim_store", None)
    if not store:
        return
    for i in range(x_local.shape[0]):
        mine = store.get((int(arena_ids[i]), int(locals_[i])))
        if not mine:
            continue
        xl, yl = float(x_local[i]) / ARENA_HALF, float(y_local[i]) / ARENA_HALF
        mag = float(math.hypot(xl, yl))
        dt_ = float(dtheta[i])
        for c in mine.values():
            ch, sh = math.cos(c["heading"]), math.sin(c["heading"])
            c["dx"] += xl * ch - yl * sh
            c["dy"] += xl * sh + yl * ch
            c["heading"] += dt_
            c["path"] += mag


def materialize_claims(worker, arena_ids, locals_, cfg, device):
    """Stored claims -> (peer_pos, peer_conf, peer_strength) rows for
    belief.add_peer_fix.

    Ring center = claimed position + displacement since reception; sigma ages
    through the conf: conf_eff = conf * exp(-rate*path/0.02), the same 0.02
    variance convention the peer code's sigma formula uses, so odometry drift
    widens the claim exactly the way the filter widens its own motion model.

    T2b (cfg.learned_self_trust): the sender's broadcast self-trust
    annotation demotes its own claim, ONE-SIDED with a deadband:
    multiplier = 1.0 for t >= SELF_TRUST_DEADBAND (-0.25), else 1 + t.
    Rationale: a legacy/pre-T2b writer's slot 7 is not exactly zero -- it is
    the BC head's fit noise around zero (~+-0.01), and the first parity
    smoke measured the divergence that causes: a -0.004 sample demoted
    every claim by 0.4% and the tick lines drifted apart within 1000
    ticks. The deadband absorbs fit noise as an exact no-op (legacy wire
    content = full trust, as the design intended) while trained policies
    keep the full demotion range: t=-0.25 -> multiplier 0.75, t=-1 -> 0
    (claim dropped). No amplification either way: t > 0 changes nothing.
    Applied only when the flag is on: with it off, stored trust values are
    ignored and the T1/T2a behavior is bit-for-bit unchanged.
    """
    n = len(arena_ids)
    k = int(getattr(cfg, "claim_store_capacity", 8))
    peer_pos = torch.zeros(n, k, 2, device=device)
    peer_conf = torch.zeros(n, k, device=device)
    peer_strength = torch.zeros(n, k, device=device)
    store = getattr(worker, "claim_store", None)
    if not store:
        return peer_pos, peer_conf, peer_strength
    rate = float(getattr(cfg, "claim_odo_var_rate", 0.004))
    for i in range(n):
        mine = store.get((int(arena_ids[i]), int(locals_[i])))
        if not mine:
            continue
        for j, sid in enumerate(sorted(mine)):
            if j >= k:
                break
            c = mine[sid]
            conf_eff = c["conf"] * math.exp(-rate * c["path"] / CONF_VARIANCE_SCALE)
            if getattr(cfg, "learned_self_trust", False):
                t = c.get("trust", 0.0)
                conf_eff *= 1.0 if t >= -0.25 else 1.0 + t
            peer_pos[i, j, 0] = c["x"] + c["dx"]
            peer_pos[i, j, 1] = c["y"] + c["dy"]
            peer_conf[i, j] = min(max(conf_eff, 0.0), 1.0)
            peer_strength[i, j] = c["strength"]
    return peer_pos, peer_conf, peer_strength


def consensus_filter(peer_pos, peer_conf, peer_strength, particles, cfg):
    """The second-order discriminator, applied BEFORE add_peer_fix.

    A claim survives materialization only if a DISTINCT sender's claim agrees
    with it ON THE RECEIVER'S OWN CLOUD: the fraction of particles lying
    within BOTH rings' 2-sigma bands must reach cfg.consensus_agreement_min.
    This is phase 169's "two independently-wrong latches rarely agree with
    each other AND with the receiver's support" -- the one veto a spread
    cloud can actually apply, and the one the measured T1a harm needed: the
    wrong-latch population's errors are CORRELATED (the same filter
    mislocalization mechanisms), so wrong pairs agree with each other often
    enough that a separation cap alone cannot vet them. Lone claims are
    zeroed outright (a claim is never fused alone); disagreeing pairs are
    zeroed; agreeing pairs and consistent larger sets pass to add_peer_fix,
    whose residual gate (>=3) and pair separation cap still apply.

    sigma per claim mirrors add_peer_fix's own formula so the bands vetted
    here are the bands the fix would inject at.
    """
    agree_min = float(getattr(cfg, "consensus_agreement_min", 0.10))
    if agree_min <= 0.0 or peer_pos is None:
        return peer_pos, peer_conf, peer_strength
    n, k, _ = peer_pos.shape
    if not bool((peer_strength > 0).any()):
        return peer_pos, peer_conf, peer_strength
    # band half-width per claim: PEER_SIGMA + 2*sender_sigma, the fix's own
    # motion-independent part (its spread term varies per robot; the veto
    # only needs the claim-consistency structure)
    sender_sig = torch.sqrt(-CONF_VARIANCE_SCALE *
                            torch.log(peer_conf.clamp(min=1e-3, max=0.999)))
    sig = (PEER_SIGMA + 2.0 * sender_sig).clamp(min=PEER_SIGMA_FLOOR)
    px, py = particles[:, :, 0], particles[:, :, 1]          # (n, P)
    keep = torch.zeros(n, k, dtype=torch.bool, device=peer_pos.device)
    for j in range(k):
        cj_x = peer_pos[:, j, 0].unsqueeze(1)
        cj_y = peer_pos[:, j, 1].unsqueeze(1)
        # same inversion as belief._range_from_strength: (1/s - 1) is RAW
        # units; the cloud is normalized
        rj = ((1.0 / peer_strength[:, j].clamp(min=1e-6) - 1.0) / ARENA_HALF).unsqueeze(1)
        dist_j = ((px - cj_x) ** 2 + (py - cj_y) ** 2).sqrt()
        in_j = (dist_j - rj).abs() <= (2.0 * sig[:, j]).unsqueeze(1)
        for m in range(k):
            if m == j or not bool((peer_strength[:, j] > 0).any()):
                continue
            cm_x = peer_pos[:, m, 0].unsqueeze(1)
            cm_y = peer_pos[:, m, 1].unsqueeze(1)
            rm = ((1.0 / peer_strength[:, m].clamp(min=1e-6) - 1.0) / ARENA_HALF).unsqueeze(1)
            dist_m = ((px - cm_x) ** 2 + (py - cm_y) ** 2).sqrt()
            in_m = (dist_m - rm).abs() <= (2.0 * sig[:, m]).unsqueeze(1)
            both = (in_j & in_m).float().mean(dim=1)
            keep[:, j] |= (peer_strength[:, j] > 0) & (peer_strength[:, m] > 0) & (both >= agree_min)
    strength_kept = peer_strength * keep.float()
    return peer_pos, peer_conf, strength_kept


# ─── sightings and referrals ──────────────────────────────────────────────────

def record_sightings(worker, arena_ids, locals_, seeds_b, walls_b, cfg):
    """Record each robot's last DIRECT sighting (landmark or wall side).

    This is what the disclosure writer broadcasts and what makes a sender
    'anchored' for referral purposes: a record exists only if this robot
    personally ranged map geometry. Range normalized by the arena diagonal
    half-width so it fits the content float.
    """
    sightings = getattr(worker, "last_sighting", None)
    if sightings is None:
        sightings = {}
        worker.last_sighting = sightings
    for i in range(seeds_b.shape[0]):
        a, l = int(arena_ids[i]), int(locals_[i])
        now = int(worker.step_count.get(a, 0))
        s = seeds_b[i]
        w = walls_b[i]
        if bool((s > 0).any()):
            k = int(s.argmax())
            d_raw = 1.0 / max(float(s[k]), 1e-6) - 1.0
        elif bool((w > 0).any()):
            k = WALL_ID_OFFSET + int(w.argmax())
            d_raw = 1.0 / max(float(w.max()), 1e-6) - 1.0
        else:
            continue
        sightings[(a, l)] = (k, min(d_raw / (2.0 * ARENA_HALF), 1.0), now)


def update_referrals(worker, arena_ids, locals_, rows, valid, cfg, device):
    """Decode disclosure slots 3-5 from this tick's peer rows into the
    referral store, keyed by sender (latest wins)."""
    if not getattr(cfg, "evidence_comms", False) or rows is None:
        return
    refs = getattr(worker, "referral_store", None)
    if refs is None:
        refs = {}
        worker.referral_store = refs
    rows_d = rows.to(device)
    valid_d = valid.to(device) if valid is not None else torch.ones_like(rows_d[:, :, 0], dtype=torch.bool)
    for i in range(rows_d.shape[0]):
        a, l = int(arena_ids[i]), int(locals_[i])
        now = int(worker.step_count.get(a, 0))
        mine = refs.setdefault((a, l), {})
        peer = valid_d[i] & (rows_d[i, :, MESSAGE_SIZE] >= 1.0)
        disc = peer & (rows_d[i, :, DISCLOSURE_ID_COL] > 0.0)
        for k in torch.nonzero(disc).flatten().tolist():
            dec = decode_disclosure(float(rows_d[i, k, DISCLOSURE_ID_COL]),
                                    float(rows_d[i, k, DISCLOSURE_RANGE_COL]),
                                    float(rows_d[i, k, DISCLOSURE_RECENCY_COL]))
            if dec is None:
                continue
            sid = int(rows_d[i, k, MESSAGE_SIZE])
            mine[sid] = {"landmark": dec[0], "range": dec[1],
                         "recency": dec[2], "rx_tick": now}


def select_referral(worker, a, l, cfg, skip=()):
    """Best referral sender for this robot: freshest effective recency, then
    smallest landmark range. Returns sender id or None.

    Effective recency combines the sender's own encoded age-at-sending with
    the receiver's holding time since reception: total age = (send_tick -
    sighting_tick) + (now - rx_tick), so a referral decays in BOTH clocks and
    hits zero at cfg.disclosure_recency_ticks of total age, at which point
    the entry is dropped. `skip` removes senders the homing controller has
    abandoned (chased without progress past its grace budget) so the robot
    returns to its own policy instead of orbiting a dead target."""
    refs = getattr(worker, "referral_store", None)
    if not refs:
        return None
    mine = refs.get((a, l))
    if not mine:
        return None
    now = int(worker.step_count.get(a, 0))
    horizon = float(getattr(cfg, "disclosure_recency_ticks", 2000))
    best, best_key = None, None
    for sid in list(mine):
        if sid in skip:
            continue
        r = mine[sid]
        eff = max(0.0, r["recency"] - (now - r["rx_tick"]) / horizon)
        if eff <= 0.0:
            del mine[sid]
            continue
        key = (-eff, r["range"])
        if best_key is None or key < best_key:
            best, best_key = sid, key
    return best


# ─── homing controller (Gate 2's law, at pipeline scale) ──────────────────────

def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _fit_beacon(receptions):
    """Invert s = 1/(1 + ARENA_HALF*|p-q|) for the beacon position q.

    Receptions are (x, y, strength) in the receiver's odometry frame (same
    units as the particle filter's world). Gauss-Newton from 8 inits around
    the peak-strength reception; returns (q, rms) or (None, inf).
    """
    if len(receptions) < 4:
        return None, float("inf")
    P = np.array([(r[0], r[1]) for r in receptions], dtype=float)
    S = np.array([r[2] for r in receptions], dtype=float)
    # degenerate-fit guard: receptions bunched at one spot (a stationary
    # receiver in range) fit ANY q with a small residual -- the T1b failure
    # mode that let a garbage bearing capture a robot. Require the
    # receptions to span a minimal baseline before the fit says anything.
    if max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))) < 0.01:
        return None, float("inf")
    i0 = int(np.argmax(S))
    # 1/s - 1 = ARENA_HALF * d at the peak, so the init circle's radius is
    # that DIVIDED by ARENA_HALF -- in the receptions' normalized units
    d0 = (1.0 / max(S[i0], 1e-6) - 1.0) / ARENA_HALF
    best_q, best_rms = None, float("inf")
    for k in range(8):
        phi = 2.0 * math.pi * k / 8.0
        q = P[i0] + d0 * np.array([math.cos(phi), math.sin(phi)])
        for _ in range(30):
            dd = np.maximum(np.linalg.norm(P - q, axis=1), 1e-6)
            pred = 1.0 / (1.0 + ARENA_HALF * dd)
            resid = pred - S
            # ds/dq = A*(p-q) / ((1+A*d)^2 * d)  -- d(s)/d(d) = -A/(1+A d)^2,
            # d(d)/d(q) = (q-p)/d; the sign here decides whether Gauss-Newton
            # descends or ascends the residual, so it is pinned by derivation
            J = (P - q) * ARENA_HALF / ((1.0 + ARENA_HALF * dd) ** 2 * dd)[:, None]
            dq, *_ = np.linalg.lstsq(J, -resid, rcond=None)
            nrm = np.linalg.norm(dq)
            if not np.isfinite(nrm):
                break
            q = q + dq * min(1.0, 0.5 / max(nrm, 1e-9))
            if nrm < 1e-5:
                break
        dd = np.linalg.norm(P - q, axis=1)
        rms = float(np.sqrt(np.mean((1.0 / (1.0 + ARENA_HALF * dd) - S) ** 2)))
        if np.isfinite(rms) and rms < best_rms:
            best_rms, best_q = rms, q
    return best_q, best_rms


def _update_track(track, x_local, y_local, dtheta, mag):
    """One scalar claim/homing displacement track: rotate this interval's
    local motion by the accumulated heading, then accumulate (exactly
    kinematics.split_track_update's semantics)."""
    dx, dy, heading = track
    ch, sh = math.cos(heading), math.sin(heading)
    return (dx + x_local * ch - y_local * sh,
            dy + x_local * sh + y_local * ch,
            heading + dtheta), mag


def update_homing(worker, arena_ids, locals_, rows, valid, x_local, y_local, dtheta, cfg, device):
    """Per-tick homing bookkeeping: propagate each robot's odometry track,
    (re)select the referral target, record receptions FROM that target.

    Receptions come from the row pool whenever the target sender is in range
    -- not only when the split-event draw happens to pick it -- which is the
    same pool-level view the filter's own peer path consumes. One target at a
    time; changing targets resets the odometry track (receptions from
    different senders must not mix frames). Tracks are NORMALIZED (see
    propagate_claims' units note); dtheta is radians.
    """
    if not getattr(cfg, "evidence_comms", False):
        return
    hs = getattr(worker, "homing_state", None)
    if hs is None:
        hs = {}
        worker.homing_state = hs
    rows_d = rows.to(device) if rows is not None else None
    valid_d = None
    if rows_d is not None:
        valid_d = valid.to(device) if valid is not None else torch.ones_like(rows_d[:, :, 0], dtype=torch.bool)
    for i in range(len(arena_ids)):
        a, l = int(arena_ids[i]), int(locals_[i])
        st = hs.get((a, l))
        if st is None:
            st = {"target": None, "track": (0.0, 0.0, 0.0), "receptions": [],
                  "fit_q": None, "fit_rms": float("inf")}
            hs[(a, l)] = st
        xl, yl, dt_ = float(x_local[i]) / ARENA_HALF, float(y_local[i]) / ARENA_HALF, float(dtheta[i])
        st["track"], _ = _update_track(list(st["track"]), xl, yl, dt_,
                                       math.hypot(xl, yl))
        target = select_referral(worker, a, l, cfg, skip=st.get("abandoned", ()))
        if target != st["target"]:
            st["target"] = target
            st["track"] = (0.0, 0.0, 0.0)
            st["receptions"] = []
            st["fit_q"] = None
            st["fit_rms"] = float("inf")
            st["out_ticks"] = 0
            st["declining"] = False
            st["reverse"] = False
            st["reorient_ticks"] = 0
            st["reorient_sign"] = 0.0
        if target is None or rows_d is None:
            # out of contact with any referral: age the grace clock so a
            # stale fit cannot hold the robot forever (the measured T1b
            # failure -- capture without surrender)
            if st["target"] is None:
                st["out_ticks"] = st.get("out_ticks", 0) + 1
            continue
        peer = valid_d[i] & (rows_d[i, :, MESSAGE_SIZE] == float(target))
        heard = False
        for k in torch.nonzero(peer).flatten().tolist():
            s = float(rows_d[i, k, 10])
            if s <= 0.0:
                continue
            heard = True
            prev_s = st["receptions"][-1][2] if st["receptions"] else None
            st["receptions"].append((st["track"][0], st["track"][1], s))
            if len(st["receptions"]) > 60:
                st["receptions"].pop(0)
            # decline detection drives the reorientation (the probe's law:
            # riding a falling gradient means the pass is tangential)
            st["declining"] = bool(prev_s is not None and s < prev_s)
        st["out_ticks"] = 0 if heard else st.get("out_ticks", 0) + 1
        # the probe's law, in full:
        #   declining in range -> ONE perpendicular reorientation (refractory)
        #   out of range past the grace -> sweep back through the disc
        #   out of range past the budget -> ABANDON the referral; the robot
        #     returns to its own policy (wall-following is its localization
        #     path; chasing a stale fit forever is the measured catastrophe)
        if st["out_ticks"] == 0:
            st["reorient_ticks"] = max(0, st.get("reorient_ticks", 0) - 1)
            st["reverse"] = False
        grace = int(getattr(cfg, "homing_grace_ticks", 120))
        budget = int(getattr(cfg, "homing_budget_ticks", 400))
        if st["out_ticks"] > budget:
            st.setdefault("abandoned", []).append(target)
            st["target"] = None
            st["fit_q"] = None
            st["fit_rms"] = float("inf")
            st["receptions"] = []
            st["reverse"] = False
            continue
        st["reverse"] = st["out_ticks"] > grace
        if (st["declining"] and st.get("reorient_ticks", 0) == 0
                and st["out_ticks"] == 0):
            st["reorient_sign"] = random.choice((-1.0, 1.0))
            st["reorient_ticks"] = 20
        # refit on a 10-reception cadence (receptions arrive every tick the
        # target is in range; refitting every 4 made the per-tick loop the
        # cell-time bottleneck). The fitted position barely moves between
        # refits -- the estimator is overdetermined long before 10.
        if len(st["receptions"]) >= 4 and (len(st["receptions"]) - 4) % 10 == 0:
            q, rms = _fit_beacon(st["receptions"])
            if q is not None:
                st["fit_q"], st["fit_rms"] = q, rms


def homing_readout(worker, a, l, cfg):
    """(+3 proprioception) distance and bearing to the fitted referral beacon,
    in the robot's CURRENT heading frame -- the same convention as the
    nearest-seed trio. (dist, sin, cos); zeros when nothing is fitted."""
    st = getattr(worker, "homing_state", {}).get((a, l))
    if not st or st["fit_q"] is None or st["fit_rms"] > 0.02:
        return 0.0, 0.0, 0.0
    dx = st["fit_q"][0] - st["track"][0]
    dy = st["fit_q"][1] - st["track"][1]
    dist = math.hypot(dx, dy)
    # bearing in the odometry frame minus the accumulated heading = the error
    # in the robot's current heading frame (frames differ by a constant
    # rotation, so the DIFFERENCE is frame-invariant)
    e = math.atan2(dy, dx) - st["track"][2]
    return dist, math.sin(e), math.cos(e)


def homing_steering_error(worker, a, l, cfg):
    """The control law's heading error (radians, current frame): the fitted
    beacon's bearing PLUS the law's own offsets -- sweep-back (+pi) while
    past the grace clock out of range, one perpendicular reorientation
    (+-pi/2, refractory) while riding a declining gradient. None when the
    law has nothing to say (no target, no trustworthy fit)."""
    st = getattr(worker, "homing_state", {}).get((a, l))
    if not st or st.get("target") is None or st.get("fit_q") is None or st["fit_rms"] > 0.02:
        return None
    dx = st["fit_q"][0] - st["track"][0]
    dy = st["fit_q"][1] - st["track"][1]
    e = math.atan2(dy, dx) - st["track"][2]
    if st.get("reverse"):
        e = wrap(e + math.pi)
    if st.get("reorient_ticks", 0) > 0:
        e = wrap(e + st["reorient_sign"] * math.pi / 2.0)
    return e


def homing_motor_override(worker, a, l, cfg):
    """T1b scripted homing: (left, right) motor pair per the control law, or
    None when this robot should not home.

    Fires only for LOST robots (own conf < 0.4 -- the same bar the arrival
    gate uses) with a trustworthy fit. Turn differential is the oracle's own
    TURN_MOTOR convention. The law's full behavior -- decline reorientation,
    grace sweep-back, abandonment budget -- lives in update_homing's state
    machine and _steering-error offsets above."""
    if not getattr(cfg, "referral_homing", False):
        return None
    e = homing_steering_error(worker, a, l, cfg)
    if e is None:
        return None
    if abs(e) <= 0.2:
        return (1.0, 1.0)
    return (0.9, 0.15) if e < 0 else (0.15, 0.9)