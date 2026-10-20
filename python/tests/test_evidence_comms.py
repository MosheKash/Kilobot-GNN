"""The evidence-sharing channel (config.evidence_comms): unit tests.

Covers the codec, the claim store's insert/propagate/materialize math
(including the odometry-rotation semantics pinned by
kinematics.split_track_update), the consensus consumption gates through
belief.add_peer_fix, sighting/referral bookkeeping, and the homing
controller's estimator and steering conventions. All Unity-free: every
quantity here is tensor math on synthetic rows and clouds.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import belief
import evidence
from belief import ARENA_HALF, BELIEF_PARTICLES
from kilobot_gnn import MESSAGE_SIZE, DB_ROW_SIZE
from kinematics import split_tick_motion, split_track_update


def _row(x, y, conf, sid, strength, valid=True):
    r = torch.zeros(DB_ROW_SIZE)
    r[0], r[1], r[2] = x, y, conf
    r[MESSAGE_SIZE] = float(sid)
    r[MESSAGE_SIZE + 1] = strength
    return r, valid


def _rows_batch(rows_per_robot):
    n = len(rows_per_robot)
    k = max((len(r) for r in rows_per_robot), default=1)
    rows = torch.zeros(n, k, DB_ROW_SIZE)
    valid = torch.zeros(n, k, dtype=torch.bool)
    for i, robot_rows in enumerate(rows_per_robot):
        for j, (r, v) in enumerate(robot_rows):
            rows[i, j] = r
            valid[i, j] = v
    return rows, valid


class _FakeWorker:
    def __init__(self):
        self.step_count = {0: 100}
        self.claim_store = {}
        self.referral_store = {}
        self.homing_state = {}
        self.last_sighting = {}


def _cfg(**kw):
    import config
    c = config.Config()
    c.evidence_comms = True
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ─── codec ────────────────────────────────────────────────────────────────────

def test_disclosure_codec_round_trip():
    for k in range(8):
        enc = evidence.encode_disclosure(k, 0.37, 0.9)
        assert enc[0] > 0.0
        dec = evidence.decode_disclosure(*enc)
        assert dec[0] == k and abs(dec[1] - 0.37) < 1e-6 and abs(dec[2] - 0.9) < 1e-6


def test_disclosure_codec_zero_is_silent():
    assert evidence.decode_disclosure(0.0, 0.5, 0.5) is None
    assert evidence.encode_disclosure(0, 0.0, 1.0)[0] == pytest.approx(0.1)


# ─── claim store ──────────────────────────────────────────────────────────────

def test_claim_store_insert_and_dedupe():
    w = _FakeWorker()
    cfg = _cfg()
    r3, _ = _row(0.10, 0.20, 0.90, 3, 0.5)
    r5, _ = _row(-0.1, 0.05, 0.80, 5, 0.4)
    wall, _ = _row(0.0, 0.0, 0.0, -2, 0.9)     # wall seed: negative id, no content
    pad, v = _row(0.0, 0.0, 0.0, 0, 0.0, valid=False)
    rows, valid = _rows_batch([[ (r3, True), (r5, True), (wall, True), (pad, v) ]])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    store = w.claim_store[(0, 0)]
    assert set(store) == {3, 5}
    assert store[3]["conf"] == pytest.approx(0.9)
    # re-hear sender 3, newer: latest wins, still two entries
    r3b, _ = _row(0.11, 0.21, 0.85, 3, 0.55)
    rows, valid = _rows_batch([[ (r3b, True) ]])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    assert set(w.claim_store[(0, 0)]) == {3, 5}
    assert w.claim_store[(0, 0)][3]["conf"] == pytest.approx(0.85)


def test_claim_store_capacity_evicts_oldest():
    w = _FakeWorker()
    cfg = _cfg(claim_store_capacity=2)
    robot_rows = []
    for sid, x in ((1, 0.1), (2, 0.2), (3, 0.3)):
        r, _ = _row(x, 0.0, 0.9, sid, 0.5)
        robot_rows.append((r, True))
    rows, valid = _rows_batch([robot_rows])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    assert set(w.claim_store[(0, 0)]) == {2, 3}   # sender 1 (oldest) evicted


def test_claim_propagation_matches_split_track_update():
    w = _FakeWorker()
    cfg = _cfg()
    r1, _ = _row(0.0, 0.0, 0.9, 1, 0.5)
    rows, valid = _rows_batch([[ (r1, True) ]])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    # three intervals of a known command, run through BOTH paths
    last_motor = torch.tensor([[0.9, 0.15]])
    steps = torch.tensor([4.0])
    track = torch.zeros(1, 4)
    for _ in range(3):
        xl, yl, dth, t = split_tick_motion(last_motor, steps, 4.0, 4.0 / math.pi, 0.02)
        track = split_track_update(track, xl, yl, dth, t)
        evidence.propagate_claims(w, torch.tensor([0]), torch.tensor([0]), xl, yl, dth)
    c = w.claim_store[(0, 0)][1]
    # claim tracks are NORMALIZED; split_track_update accumulates RAW units
    assert c["dx"] == pytest.approx(float(track[0, 0]) / ARENA_HALF, abs=1e-8)
    assert c["dy"] == pytest.approx(float(track[0, 1]) / ARENA_HALF, abs=1e-8)
    assert c["heading"] == pytest.approx(float(track[0, 2]), abs=1e-6)


def test_materialized_center_is_claim_plus_displacement():
    w = _FakeWorker()
    cfg = _cfg()
    r1, _ = _row(0.2, -0.1, 0.9, 1, 0.5)
    rows, valid = _rows_batch([[ (r1, True) ]])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    xl = torch.tensor([1.0]); yl = torch.tensor([-2.0]); dth = torch.tensor([0.1])   # RAW units
    evidence.propagate_claims(w, torch.tensor([0]), torch.tensor([0]), xl, yl, dth)
    pp, pc, ps = evidence.materialize_claims(w, torch.tensor([0]), torch.tensor([0]), cfg, "cpu")
    c = w.claim_store[(0, 0)][1]
    ex = c["x"] + 1.0 / ARENA_HALF     # the boundary divides by ARENA_HALF
    ey = c["y"] - 2.0 / ARENA_HALF
    assert pp[0, 0, 0] == pytest.approx(ex, abs=1e-6)
    assert pp[0, 0, 1] == pytest.approx(ey, abs=1e-6)
    # path = hypot(1, -2)/ARENA_HALF = 0.0224; conf ages by exp(-rate*path/0.02)
    assert pc[0, 0] == pytest.approx(0.9 * math.exp(-0.004 * 0.0224 / 0.02), rel=1e-3)
    assert ps[0, 0] == pytest.approx(0.5)


def test_claim_conf_ages_with_path():
    w = _FakeWorker()
    cfg = _cfg(claim_odo_var_rate=0.004)
    r1, _ = _row(0.2, -0.1, 0.9, 1, 0.5)
    rows, valid = _rows_batch([[ (r1, True) ]])
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    # 2.0 normalized units of straight travel: conf * exp(-0.004*2.0/0.02)
    for _ in range(625):
        evidence.propagate_claims(w, torch.tensor([0]), torch.tensor([0]),
                                  torch.tensor([0.32]), torch.tensor([0.0]),
                                  torch.tensor([0.0]))   # RAW: 0.32 raw/decision = 0.0032 normalized
    pp, pc, ps = evidence.materialize_claims(w, torch.tensor([0]), torch.tensor([0]), cfg, "cpu")
    assert pc[0, 0] == pytest.approx(0.9 * math.exp(-0.004 * 2.0 / 0.02), rel=1e-2)


# ─── consensus consumption through add_peer_fix ───────────────────────────────

def _cloud(n, mean, spread, generator):
    p = torch.empty(n, BELIEF_PARTICLES, 3)
    p[:, :, 0] = mean[0] + (torch.rand(n, BELIEF_PARTICLES, generator=generator) - 0.5) * 2 * spread
    p[:, :, 1] = mean[1] + (torch.rand(n, BELIEF_PARTICLES, generator=generator) - 0.5) * 2 * spread
    p[:, :, 2] = 0.0
    return p


def test_two_agreeing_claims_inject_single_claim_does_not():
    belief.set_peer_conf_min(0.7)
    generator = torch.Generator().manual_seed(7)
    mean = (0.0, 0.0)
    # two senders whose claimed positions sit near the cloud mean, with
    # reception strengths consistent with a receiver AT the mean: the rings
    # both pass through the mean, separation well under PEER_PAIR_MAX_SEP
    c1, c2 = np.array([0.05, 0.0]), np.array([-0.05, 0.0])
    d1, d2 = np.linalg.norm(mean - c1), np.linalg.norm(mean - c2)
    s1, s2 = 1.0 / (1.0 + ARENA_HALF * d1), 1.0 / (1.0 + ARENA_HALF * d2)
    r1, _ = _row(c1[0], c1[1], 0.9, 1, s1)
    r2, _ = _row(c2[0], c2[1], 0.9, 2, s2)
    rows, valid = _rows_batch([[ (r1, True), (r2, True) ]])
    w = _FakeWorker()
    cfg = _cfg()
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    pp, pc, ps = evidence.materialize_claims(w, torch.tensor([0]), torch.tensor([0]), cfg, "cpu")

    p = _cloud(1, mean, 0.25, generator)     # spread >= min_spread 0.2
    out = belief.belief_update(p.clone(), torch.zeros(1, 4), generator,
                               peer_pos=pp, peer_conf=pc, peer_strength=ps,
                               wall_obs=torch.zeros(1, 4), peer_fix=True,
                               peer_inject_min_spread=0.2)
    injected = getattr(belief, "_last_peer_fix_injected", None)
    # belief_update does not expose the mask; assert on the EFFECT: with two
    # agreeing claims the cloud gains mass at the fix (the mean moves toward
    # the intersection region), with one claim it does not.
    two_mean = out[0, :, :2].mean(dim=0)

    # single claim: counts < 2, the consensus gate refuses
    w1 = _FakeWorker()
    rows1, valid1 = _rows_batch([[ (r1, True) ]])
    evidence.update_claims(w1, torch.tensor([0]), torch.tensor([0]), rows1, valid1, cfg, "cpu")
    pp1, pc1, ps1 = evidence.materialize_claims(w1, torch.tensor([0]), torch.tensor([0]), cfg, "cpu")
    p1 = _cloud(1, mean, 0.25, generator)
    out1 = belief.belief_update(p1.clone(), torch.zeros(1, 4), generator,
                                peer_pos=pp1, peer_conf=pc1, peer_strength=ps1,
                                wall_obs=torch.zeros(1, 4), peer_fix=True,
                                peer_inject_min_spread=0.2)
    one_mean = out1[0, :, :2].mean(dim=0)
    # two agreeing claims pull the mean toward the cloud center's fix; one
    # claim injects nothing (bounded-mass only fires on counts >= 2), so its
    # mean stays at the cloud's own
    assert torch.dist(two_mean, torch.tensor(mean)) < torch.dist(one_mean, torch.tensor(mean)) + 1e-6
    assert torch.dist(one_mean, torch.tensor(mean)) == pytest.approx(
        torch.dist(p1[0, :, :2].mean(dim=0), torch.tensor(mean)), abs=1e-4)


def test_two_far_apart_claims_do_not_inject():
    belief.set_peer_conf_min(0.7)
    generator = torch.Generator().manual_seed(7)
    mean = (0.0, 0.0)
    # claimed positions separated beyond PEER_PAIR_MAX_SEP: two honest
    # senders both within IR range of the receiver cannot disagree by that
    # much, so the pair is refused
    c1, c2 = np.array([0.30, 0.0]), np.array([-0.30, 0.0])
    s1 = 1.0 / (1.0 + ARENA_HALF * np.linalg.norm(mean - c1))
    s2 = 1.0 / (1.0 + ARENA_HALF * np.linalg.norm(mean - c2))
    r1, _ = _row(c1[0], c1[1], 0.9, 1, s1)
    r2, _ = _row(c2[0], c2[1], 0.9, 2, s2)
    rows, valid = _rows_batch([[ (r1, True), (r2, True) ]])
    w = _FakeWorker()
    cfg = _cfg()
    evidence.update_claims(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    pp, pc, ps = evidence.materialize_claims(w, torch.tensor([0]), torch.tensor([0]), cfg, "cpu")
    p = _cloud(1, mean, 0.25, generator)
    out = belief.belief_update(p.clone(), torch.zeros(1, 4), generator,
                               peer_pos=pp, peer_conf=pc, peer_strength=ps,
                               wall_obs=torch.zeros(1, 4), peer_fix=True,
                               peer_inject_min_spread=0.2)
    assert torch.dist(out[0, :, :2].mean(dim=0),
                      p[0, :, :2].mean(dim=0)) == pytest.approx(0.0, abs=1e-4)


# ─── sightings and referrals ──────────────────────────────────────────────────

def test_record_sightings_prefers_landmark_and_encodes_wall_offset():
    w = _FakeWorker()
    cfg = _cfg()
    seeds = torch.zeros(1, 4); seeds[0, 2] = 0.5     # corner 2, range 1.0 raw
    walls = torch.zeros(1, 4)
    evidence.record_sightings(w, torch.tensor([0]), torch.tensor([0]), seeds, walls, cfg)
    k, rng_norm, tick = w.last_sighting[(0, 0)]
    assert k == 2 and rng_norm == pytest.approx(1.0 / (2 * ARENA_HALF)) and tick == 100
    seeds0 = torch.zeros(1, 4)
    walls[0, 1] = 1.0 / 11.0                          # wall side 1, range 10 raw
    evidence.record_sightings(w, torch.tensor([0]), torch.tensor([0]), seeds0, walls, cfg)
    k, rng_norm, _ = w.last_sighting[(0, 0)]
    assert k == 4 + 1 and rng_norm == pytest.approx(10.0 / (2 * ARENA_HALF))


def test_referral_selection_ranks_recency_then_range():
    w = _FakeWorker()
    cfg = _cfg()
    # sender 3: fresher sighting, farther from its landmark;
    # sender 5: staler sighting, closer
    enc3 = evidence.encode_disclosure(2, 0.40, 0.9)
    enc5 = evidence.encode_disclosure(1, 0.05, 0.3)
    r3, _ = _row(0.0, 0.0, 0.9, 3, 0.5); r3[3], r3[4], r3[5] = enc3
    r5, _ = _row(0.1, 0.0, 0.9, 5, 0.5); r5[3], r5[4], r5[5] = enc5
    rows, valid = _rows_batch([[ (r3, True), (r5, True) ]])
    evidence.update_referrals(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    assert evidence.select_referral(w, 0, 0, cfg) == 3   # freshest wins
    # equal encoded recency: the range tiebreak decides (closer to the
    # landmark wins -- that sender was AT the landmark when it saw it)
    r5b, _ = _row(0.1, 0.0, 0.9, 5, 0.5)
    enc5b = evidence.encode_disclosure(1, 0.05, 0.9)
    r5b[3], r5b[4], r5b[5] = enc5b
    rows, valid = _rows_batch([[ (r5b, True) ]])
    evidence.update_referrals(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    assert evidence.select_referral(w, 0, 0, cfg) == 5
    # a NEWER reception of the same-recency referral takes over: holding
    # time decays the older entry
    r3b, _ = _row(0.0, 0.0, 0.9, 3, 0.5)
    r3b[3], r3b[4], r3b[5] = enc3
    w.step_count[0] = 100 + int(cfg.disclosure_recency_ticks * 0.75)
    rows, valid = _rows_batch([[ (r3b, True) ]])
    evidence.update_referrals(w, torch.tensor([0]), torch.tensor([0]), rows, valid, cfg, "cpu")
    assert evidence.select_referral(w, 0, 0, cfg) == 3
    # past the horizon, the STALE entry (heard once at tick 100) is dropped;
    # the refreshed one (re-heard at 1600) survives until its own total age
    # (200 send-age + 2000 holding) expires
    w.step_count[0] = 100 + cfg.disclosure_recency_ticks + 10
    assert evidence.select_referral(w, 0, 0, cfg) == 3
    assert 5 not in w.referral_store[(0, 0)]             # expired entries dropped
    w.step_count[0] = 1600 + cfg.disclosure_recency_ticks + 10
    assert evidence.select_referral(w, 0, 0, cfg) is None
    assert 3 not in w.referral_store[(0, 0)]


# ─── homing controller ────────────────────────────────────────────────────────

def _homing_state_with_receptions(q, robot_positions):
    """A homing state whose receptions were taken at robot_positions with
    strengths consistent with a beacon at q (world == odometry frame), FITTED
    so the readout/override paths are live."""
    recs = []
    for p in robot_positions:
        d = np.linalg.norm(np.array(p) - q)
        recs.append((p[0], p[1], 1.0 / (1.0 + ARENA_HALF * d)))
    fit, rms = evidence._fit_beacon(recs)
    return {"target": 7, "track": (0.0, 0.0, 0.0), "receptions": recs,
            "fit_q": fit, "fit_rms": rms}


def test_fit_beacon_recovers_position_from_two_chords():
    q = np.array([0.3, -0.2])
    # two passes: a chord below and a chord to the right of the beacon
    path = ([q + np.array([t, -0.15]) for t in np.linspace(-0.2, 0.2, 12)])
    path += [q + np.array([0.15, t]) for t in np.linspace(-0.2, 0.2, 12)]
    recs = [(p[0], p[1], 1.0 / (1.0 + ARENA_HALF * np.linalg.norm(p - q))) for p in path]
    fit, rms = evidence._fit_beacon(recs)
    assert fit is not None and rms < 1e-3
    assert np.linalg.norm(fit - q) < 0.01


def test_homing_readout_bearing_is_in_current_heading_frame():
    w = _FakeWorker()
    cfg = _cfg()
    q = np.array([0.2, 0.1])
    st = _homing_state_with_receptions(q, [q + np.array([t, -0.1]) for t in np.linspace(-0.2, 0.2, 10)])
    # robot has turned 90 degrees CCW from the odometry anchor
    st["track"] = (0.0, 0.0, math.pi / 2)
    w.homing_state[(0, 0)] = st
    dist, s, c = evidence.homing_readout(w, 0, 0, cfg)
    assert dist == pytest.approx(np.linalg.norm(q), abs=1e-3)
    # beacon at atan2(0.1, 0.2) in the anchor frame = -63.4 deg in the
    # robot's +90-degree TURNED frame
    e_true = math.atan2(0.1, 0.2) - math.pi / 2
    assert s == pytest.approx(math.sin(e_true), abs=1e-2)
    assert c == pytest.approx(math.cos(e_true), abs=1e-2)


def test_homing_motor_override_steering_signs():
    w = _FakeWorker()
    cfg = _cfg(referral_homing=True)
    q = np.array([0.2, 0.1])
    st = _homing_state_with_receptions(q, [q + np.array([t, -0.1]) for t in np.linspace(-0.2, 0.2, 10)])
    w.homing_state[(0, 0)] = st
    # beacon straight ahead in the current frame: full speed
    st["track"] = (0.0, 0.0, math.atan2(0.1, 0.2))
    assert evidence.homing_motor_override(w, 0, 0, cfg) == (1.0, 1.0)
    # beacon to the LEFT of the current heading (e > 0, CCW): right wheel faster
    st["track"] = (0.0, 0.0, math.atan2(0.1, 0.2) - 1.0)
    assert evidence.homing_motor_override(w, 0, 0, cfg) == (0.15, 0.9)
    # beacon to the RIGHT (e < 0, CW): left wheel faster
    st["track"] = (0.0, 0.0, math.atan2(0.1, 0.2) + 1.0)
    assert evidence.homing_motor_override(w, 0, 0, cfg) == (0.9, 0.15)


def test_homing_override_refuses_unfitted_and_needs_flag():
    w = _FakeWorker()
    cfg_off = _cfg(referral_homing=False)
    cfg_on = _cfg(referral_homing=True)
    q = np.array([0.2, 0.0])
    st = _homing_state_with_receptions(q, [q + np.array([t, -0.1]) for t in np.linspace(-0.2, 0.2, 10)])
    w.homing_state[(0, 0)] = st
    assert evidence.homing_motor_override(w, 0, 0, cfg_off) is None
    assert evidence.homing_motor_override(w, 0, 0, cfg_on) is not None
    st["fit_rms"] = 0.5     # untrustworthy fit: refuse even with the flag
    assert evidence.homing_motor_override(w, 0, 0, cfg_on) is None

# ─── consensus pre-gate ───────────────────────────────────────────────────────

def test_consensus_filter_zeroes_lone_and_disagreeing_claims():
    cfg = _cfg(consensus_agreement_min=0.10)
    generator = torch.Generator().manual_seed(7)
    mean = (0.0, 0.0)
    p = _cloud(1, mean, 0.25, generator)
    # two claims whose rings both pass through the cloud body -> both kept
    c1, c2 = np.array([0.05, 0.0]), np.array([-0.05, 0.0])
    s1 = 1.0 / (1.0 + ARENA_HALF * np.linalg.norm(mean - c1))
    s2 = 1.0 / (1.0 + ARENA_HALF * np.linalg.norm(mean - c2))
    pp = torch.zeros(1, 8, 2); pc = torch.zeros(1, 8); ps = torch.zeros(1, 8)
    pp[0, 0] = torch.tensor(c1, dtype=torch.float32); pc[0, 0] = 0.9; ps[0, 0] = s1
    pp[0, 1] = torch.tensor(c2, dtype=torch.float32); pc[0, 1] = 0.9; ps[0, 1] = s2
    _, _, kept = evidence.consensus_filter(pp.clone(), pc.clone(), ps.clone(), p, cfg)
    assert kept[0, 0] > 0 and kept[0, 1] > 0
    # lone claim: zeroed (never fused alone)
    pp1 = torch.zeros(1, 8, 2); pp1[0, 0] = torch.tensor(c1, dtype=torch.float32)
    pc1 = torch.zeros(1, 8); pc1[0, 0] = 0.9
    ps1 = torch.zeros(1, 8); ps1[0, 0] = s1
    _, _, kept1 = evidence.consensus_filter(pp1, pc1, ps1, p, cfg)
    assert kept1[0, 0] == 0.0
    # disagreeing pair: same claimed center, incompatible radii (one sender
    # lies about its range) -- the rings share no band on the cloud, so the
    # both-consistent fraction is ~0 and BOTH are refused
    cf = (0.05, 0.0)
    sf1 = 1.0 / (1.0 + ARENA_HALF * 0.05)      # ring through the cloud body
    sf2 = 1.0 / (1.0 + ARENA_HALF * 0.80)      # ring 0.75 away: gap > 4 sigma
    ppf = torch.zeros(1, 8, 2)
    ppf[0, 0] = torch.tensor(cf, dtype=torch.float32); ppf[0, 1] = torch.tensor(cf, dtype=torch.float32)
    pcf = torch.zeros(1, 8); pcf[0, 0] = 0.9; pcf[0, 1] = 0.9
    psf = torch.zeros(1, 8); psf[0, 0] = sf1; psf[0, 1] = sf2
    _, _, keptf = evidence.consensus_filter(ppf, pcf, psf, p, cfg)
    assert keptf[0, 0] == 0.0 and keptf[0, 1] == 0.0


def test_consensus_filter_can_be_disabled():
    cfg = _cfg(consensus_agreement_min=0.0)
    generator = torch.Generator().manual_seed(7)
    p = _cloud(1, (0.0, 0.0), 0.25, generator)
    pp = torch.zeros(1, 8, 2); pp[0, 0] = torch.tensor([0.4, 0.4])
    pc = torch.zeros(1, 8); pc[0, 0] = 0.9
    ps = torch.zeros(1, 8); ps[0, 0] = 1.0 / (1.0 + ARENA_HALF * 0.1)
    _, _, kept = evidence.consensus_filter(pp, pc, ps, p, cfg)
    assert kept[0, 0] > 0     # disabled: lone claim passes through untouched


def test_homing_steering_offsets_reverse_and_reorient():
    w = _FakeWorker()
    cfg = _cfg()
    q = np.array([0.2, 0.0])
    st = _homing_state_with_receptions(q, [q + np.array([t, -0.1]) for t in np.linspace(-0.2, 0.2, 10)])
    st["track"] = (0.0, 0.0, 0.0)     # beacon dead ahead: e ~ 0
    w.homing_state[(0, 0)] = st
    base = evidence.homing_steering_error(w, 0, 0, cfg)
    assert abs(base) < 0.05
    # sweep-back: past the grace clock out of range, the law inverts
    st["reverse"] = True
    e = evidence.homing_steering_error(w, 0, 0, cfg)
    assert abs(abs(e) - math.pi) < 1e-6
    st["reverse"] = False
    # one perpendicular reorientation while the gradient declines
    st["reorient_sign"], st["reorient_ticks"] = 1.0, 20
    e = evidence.homing_steering_error(w, 0, 0, cfg)
    assert abs(abs(e) - math.pi / 2) < 1e-6


def test_fit_beacon_refuses_degenerate_receptions():
    q = np.array([0.3, 0.0])
    s = 1.0 / (1.0 + ARENA_HALF * 0.3)
    recs = [(0.0, 0.0, s)] * 10     # stationary receiver: no baseline
    fit, rms = evidence._fit_beacon(recs)
    assert fit is None
