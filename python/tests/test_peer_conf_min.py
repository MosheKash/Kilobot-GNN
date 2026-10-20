"""The receiver's peer-beacon usability gate (belief.PEER_CONF_MIN).

Measured (S1b diagnostic + peer-fix probe stats): the honest beacon's
confidence has median 0.80 for settled robots, so the historical 0.85 gate
filtered out most of the honest sender pool -- 96% of receiver updates saw
zero usable peers while ~96% of robots were stopped and broadcasting. The
gate is now configurable (set_peer_conf_min); default stays 0.85.
"""

import torch

import belief
from belief import belief_update, set_peer_conf_min
from kilobot_gnn import SEED_SIZE

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


def test_default_gate_still_0_85():
    assert belief.PEER_CONF_MIN == 0.85


def test_gate_below_sender_conf_lets_the_ring_fuse():
    p, g = _collapsed_cloud()
    claim = torch.tensor([[[0.8, 0.0]]])
    conf = torch.tensor([[0.80]])   # the honest settled median: below 0.85, above 0.70
    strength = torch.tensor([[_strength_for_norm_dist(0.30)]])
    seed_zero = torch.zeros(1, SEED_SIZE)

    set_peer_conf_min(0.85)
    try:
        gated = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                              peer_strength=strength)
        assert torch.allclose(gated, p)           # 0.80 < 0.85: rejected at the historical gate
    finally:
        set_peer_conf_min(0.85)

    set_peer_conf_min(0.70)
    try:
        fused = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                              peer_strength=strength)
        assert float(fused[0, :, 0].mean()) > 0.05   # 0.80 >= 0.70: the ring fuses
    finally:
        set_peer_conf_min(0.85)


def test_gate_above_sender_conf_rejects():
    p, g = _collapsed_cloud()
    claim = torch.tensor([[[0.8, 0.0]]])
    conf = torch.tensor([[0.90]])
    strength = torch.tensor([[_strength_for_norm_dist(0.30)]])
    seed_zero = torch.zeros(1, SEED_SIZE)

    set_peer_conf_min(0.95)
    try:
        gated = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                              peer_strength=strength)
        assert torch.allclose(gated, p)           # 0.90 < 0.95: rejected
    finally:
        set_peer_conf_min(0.85)
