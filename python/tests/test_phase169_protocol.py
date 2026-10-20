"""Phase 169 (stage A): the settle-anchored wavefront protocol.

Four mechanisms, each tested at the level the docs/tuning.md phase-168
attribution measured: the peer-ring innovation gate (the receiver stops
fusing rings the cloud disagrees with), tc_peer_zero (beacon content leaves
the motor trunk's input), the arrival-latch protection (persistence +
direct-anchor requirement), and the arrived-only sender gate.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

import actor_io
from belief import belief_init, belief_update, _spread
from kilobot_gnn import (SPLIT_SEED_OFFSET, SEED_SIZE, WALL_SIZE, MESSAGE_SIZE,
                         NODE_FEATURES, SplitObservationActor)
from policy import GaussianPolicy
from config import Config
from buffer import RolloutBuffer

ARENA_HALF = 92.6


def _collapsed_cloud(seed=0, center=(0.0, 0.0), std=0.08, particles=256):
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(1, particles, 3)
    p[0, :, 0] = center[0] + torch.randn(particles, generator=g) * std
    p[0, :, 1] = center[1] + torch.randn(particles, generator=g) * std
    p[0, :, 2] = (torch.rand(particles, generator=g) - 0.5) * 0.1
    return p, g


def _strength_for_norm_dist(d_norm):
    return float(1.0 / (1.0 + d_norm * ARENA_HALF))


def test_peer_ring_innovation_gate_rejects_an_inconsistent_claim():
    # the receiver's cloud sits at the origin; a beacon claims a position 0.8
    # away with conf 0.99 and a strength implying a 0.30 range. Every particle
    # is ~0.5 out of band: with the gate the ring is simply absent (the cloud
    # is returned untouched); without it the fused ring drags the mean toward
    # the claim -- the measured false-arrival mechanism.
    p, g = _collapsed_cloud()
    claim = torch.tensor([[[0.8, 0.0]]])
    conf = torch.tensor([[0.99]])
    strength = torch.tensor([[_strength_for_norm_dist(0.30)]])
    seed_zero = torch.zeros(1, SEED_SIZE)

    gated = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                          peer_strength=strength, peer_innovation_min=0.05)
    ungated = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                            peer_strength=strength)

    assert torch.allclose(gated, p)              # rejected: nothing happened
    assert float(ungated[0, :, 0].mean()) > 0.05  # fused: mean dragged toward the claim


def test_peer_ring_innovation_gate_passes_a_consistent_beacon():
    # an accurate arrived-robot beacon: claim center at the sender's true
    # position, strength implying the true range. Most of the cloud sits on
    # the ring, a displaced minority off it; the gate must NOT block the
    # update -- fusion prunes the off-ring particles (ESS collapses, the
    # resample redraws from the on-ring cluster) and the gated run is
    # identical to the ungated one: for a consistent beacon the gate is a
    # no-op.
    g = torch.Generator().manual_seed(0)
    p = torch.zeros(1, 256, 3)
    p[0, :102, 0] = torch.randn(102, generator=g) * 0.02          # on the ring
    p[0, :102, 1] = torch.randn(102, generator=g) * 0.02
    p[0, 102:, 0] = 0.36 + torch.randn(154, generator=g) * 0.03   # displaced
    p[0, 102:, 1] = 0.30 + torch.randn(154, generator=g) * 0.03
    p[0, :, 2] = (torch.rand(256, generator=g) - 0.5) * 0.1
    sender = (0.06, 0.0)
    claim = torch.tensor([[[sender[0], sender[1]]]])
    conf = torch.tensor([[0.99]])
    strength = torch.tensor([[_strength_for_norm_dist(0.06)]])
    seed_zero = torch.zeros(1, SEED_SIZE)

    spread_before = float(_spread(p)[0])
    fused = belief_update(p.clone(), seed_zero, torch.Generator().manual_seed(11),
                          peer_pos=claim, peer_conf=conf,
                          peer_strength=strength, peer_innovation_min=0.05)
    ungated = belief_update(p.clone(), seed_zero, torch.Generator().manual_seed(11),
                            peer_pos=claim, peer_conf=conf,
                            peer_strength=strength)

    assert not torch.allclose(fused, p)      # fusion pruned the off-ring mass
    assert abs(float(fused[0, :, 0].mean())) < 0.1
    assert abs(float(fused[0, :, 1].mean())) < 0.1
    assert torch.allclose(fused, ungated)    # the gate did not change the outcome


def test_sample_split_event_tc_peer_zero_zeroes_peer_content_keeps_strength():
    cfg = SimpleNamespace(seed=0, tc_peer_zero=True)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 2, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 0.7          # beacon content
    rows[0, 0, MESSAGE_SIZE] = 3.0           # senderId >= 1: peer robot
    rows[0, 0, MESSAGE_SIZE + 1] = 0.9       # strength: wins the draw
    rows[0, 1, 0:2] = torch.tensor([0.3, -0.4])
    rows[0, 1, MESSAGE_SIZE] = -1.0          # wall-seed row (negative senderId)
    rows[0, 1, MESSAGE_SIZE + 1] = 0.1
    valid = torch.tensor([[True, True]])
    rng = torch.Generator().manual_seed(0)

    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, cfg, rng)
    # the peer row won (strength 0.9): content zeroed, strength kept
    assert torch.allclose(tc[0, :MESSAGE_SIZE], torch.zeros(MESSAGE_SIZE))
    assert abs(float(tc[0, MESSAGE_SIZE]) - 0.9) < 1e-6

    # now let the wall-seed row win: its landmark position must survive
    rows2 = rows.clone()
    rows2[0, 0, MESSAGE_SIZE + 1] = 0.1
    rows2[0, 1, MESSAGE_SIZE + 1] = 0.9
    rng2 = torch.Generator().manual_seed(0)
    tc2, _, _, _ = actor_io.sample_split_event(seeds, walls, rows2, valid, cfg, rng2)
    assert abs(float(tc2[0, 0]) - 0.3) < 1e-6
    assert abs(float(tc2[0, 1]) - (-0.4)) < 1e-6


def test_sample_split_event_tc_peer_zero_off_keeps_peer_content():
    cfg = SimpleNamespace(seed=0, tc_peer_zero=False)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 0.7
    rows[0, 0, MESSAGE_SIZE] = 3.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.9
    valid = torch.tensor([[True]])
    rng = torch.Generator().manual_seed(0)

    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, cfg, rng)
    assert abs(float(tc[0, 0]) - 0.7) < 1e-6
    assert abs(float(tc[0, MESSAGE_SIZE]) - 0.9) < 1e-6


class _ProtectionWorker(SimpleNamespace):
    pass


def _protection_cfg(persist=1, anchored=False):
    return SimpleNamespace(arrival_persist=persist, arrival_min_anchored=anchored)


def test_arrival_gate_protection_needs_consecutive_fires():
    worker = _ProtectionWorker(belief_ever_anchored={0: {2: True}})
    cfg = _protection_cfg(persist=3)
    seq = [True, True, False, True, True, True, True]
    got = [actor_io._arrival_gate_protection(worker, cfg, 0, 2, f) for f in seq]
    # fires 1-2 build the count, the False resets it, the next three fire in a row
    assert got == [False, False, False, False, False, True, True]
    # and the counter is visible for the reset test (three consecutive
    # fires plus the fourth that finally latched)
    assert worker._arrival_persist[0][2] == 4


def test_arrival_gate_protection_requires_anchor_and_resets_on_gap():
    worker = _ProtectionWorker(belief_ever_anchored={})
    cfg = _protection_cfg(persist=1, anchored=True)
    # no direct sighting ever: never latches, however many times it fires
    for _ in range(5):
        assert actor_io._arrival_gate_protection(worker, cfg, 0, 2, True) is False
    worker.belief_ever_anchored.setdefault(0, {})[2] = True
    assert actor_io._arrival_gate_protection(worker, cfg, 0, 2, True) is True
    # a gap resets the persistence counter
    worker2 = _ProtectionWorker(belief_ever_anchored={0: {2: True}})
    cfg2 = _protection_cfg(persist=2, anchored=True)
    assert actor_io._arrival_gate_protection(worker2, cfg2, 0, 2, True) is False
    assert actor_io._arrival_gate_protection(worker2, cfg2, 0, 2, False) is False
    assert actor_io._arrival_gate_protection(worker2, cfg2, 0, 2, True) is False
    assert actor_io._arrival_gate_protection(worker2, cfg2, 0, 2, True) is True


def test_arrival_gate_protection_passthrough_when_disabled():
    worker = _ProtectionWorker()
    cfg = _protection_cfg(persist=1, anchored=False)
    assert actor_io._arrival_gate_protection(worker, cfg, 0, 2, True) is True
    assert actor_io._arrival_gate_protection(worker, cfg, 0, 2, False) is False
    assert not hasattr(worker, "_arrival_persist")


def test_reset_arena_clears_arrival_protection_state():
    import trainer as T
    worker = SimpleNamespace(
        z={}, image_id={}, databases={}, hidden={}, last_motor={},
        last_dec_step={}, odometer={}, track_neighbor={}, track_seed={},
        belief={}, pending_find_reward={}, step_count={}, ep_reward={},
        simple_heading={}, simple_state={}, simple_turn_accum={},
        simple_wall_name={}, simple_target={}, arrived_switched_off={},
        simple_hilbert_order={},
        belief_ever_anchored={0: {2: True}},
        _arrival_persist={0: {2: 3}},
        sent=[])
    worker.send_reset = lambda k, i: worker.sent.append(("reset", k, i))
    worker.send_image = lambda k, i: worker.sent.append(("image", k, i))
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = Config()
    tr.image_pool = [torch.zeros(1, 28, 28)]
    tr.encoder = lambda img: torch.zeros(64)
    tr.image_names = None
    tr.formations_dir = None
    tr._reset_arena(worker, 0, send_reset=False)
    assert worker.belief_ever_anchored == {}
    assert worker._arrival_persist == {}


class _FakeDecisionSteps:
    def __init__(self, obs):
        self.obs = obs

    def __len__(self):
        return self.obs[0].shape[0]


class _FakeActWorker:
    def __init__(self, node):
        self.z = {0: torch.randn(64)}
        self.hidden = {0: {}}
        self.last_motor = {0: {}}
        self.last_dec_step = {0: {}}
        self.step_count = {0: 5}
        self.databases = {0: {}}
        self.track_neighbor = {0: {}}
        self.track_seed = {0: {}}
        self.belief = {0: {}}
        self.pending_find_reward = {0: {}}
        self._node = node
        self.set_actions_calls = []

    def snapshot(self, a):
        return {"step_index": 0, "node": self._node}

    def set_actions(self, actions):
        self.set_actions_calls.append(actions)


def _run_act(beacon_arrived_only=True, arrival_min_anchored=True, seed_visible=False):
    import trainer as T
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "none"
    cfg.belief_comms = True
    cfg.belief_comms_learned = False
    cfg.beacon_arrived_only = beacon_arrived_only
    cfg.arrival_min_anchored = arrival_min_anchored
    cfg.use_closed_form_arrived = False          # learned-head path
    cfg.arrived_confidence_threshold = -1.0      # the learned head always fires
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden=cfg.split_gru_hidden,
                                                  use_arrived_head=True))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype=np.float32)
    if seed_visible:
        vector[0, 2] = 0.5
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype=np.float32)
    rows[0, 0, :MESSAGE_SIZE] = 1.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.8
    decision_steps = _FakeDecisionSteps([vector, rows])
    worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))
    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)
    return worker.set_actions_calls[-1]


def test_act_beacon_arrived_only_zeroes_an_unanchored_robots_beacon():
    # the robot's latch is blocked by arrival_min_anchored (never saw a
    # seed/wall): its beacon slots must be exactly zero, not whatever the
    # (untrained) message head happened to output -- noise in slots 0-2 can
    # clear PEER_CONF_MIN and fuse a garbage ring into every nearby filter.
    actions = _run_act(beacon_arrived_only=True, arrival_min_anchored=True,
                       seed_visible=False)
    assert float(actions[0][0]) == 0.0
    assert float(actions[0][1]) == 0.0
    assert float(actions[0][2]) == 0.0


def test_act_beacon_arrived_only_broadcasts_once_anchored():
    # with a direct seed sighting this tick, the protection passes and the
    # honest beacon is written (slots 0-2 not all exactly zero)
    actions = _run_act(beacon_arrived_only=True, arrival_min_anchored=True,
                       seed_visible=True)
    assert not np.allclose(actions[0][0:3], np.zeros(3))


def test_act_beacon_gate_off_keeps_the_historical_override():
    # control: without beacon_arrived_only the honest write happens on every
    # decision exactly as before, anchored or not
    actions = _run_act(beacon_arrived_only=False, arrival_min_anchored=True,
                       seed_visible=False)
    assert not np.allclose(actions[0][0:3], np.zeros(3))
