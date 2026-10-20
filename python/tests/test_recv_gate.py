"""The learned receiver's trust gate (config.recv_gate).

The policy's own message slot 3 maps to g = 0.5*(v+1) in [0,1]; the
receiver's fused-ring sigma scales by 1/g. g=1 must reproduce the ungated
receiver exactly (the warm start's status quo); lower gates must move the
mean strictly less (the harm mode is mean bias, so trust is widened, not
binary-gated). Every configuration runs on a FRESH generator: belief_update
consumes RNG (resample jitter), so a shared generator would confound the
comparison with resample noise.
"""

import torch

from belief import belief_update
from kilobot_gnn import SEED_SIZE

ARENA_HALF = 92.6


def _fused_mean(gate):
    # the innovation-gate test's proven drag geometry: a confident sender at
    # (0.8, 0) whose strength implies a 0.30 range, cloud at the origin
    g = torch.Generator().manual_seed(0)
    p = torch.zeros(1, 256, 3)
    p[0, :, 0] = torch.randn(256, generator=g) * 0.05
    p[0, :, 1] = torch.randn(256, generator=g) * 0.05
    p[0, :, 2] = (torch.rand(256, generator=g) - 0.5) * 0.1
    claim = torch.tensor([[[0.8, 0.0]]])
    conf = torch.tensor([[0.99]])
    strength = torch.tensor([[float(1.0 / (1.0 + 0.30 * ARENA_HALF))]])
    seed_zero = torch.zeros(1, SEED_SIZE)
    trust = None if gate is None else torch.tensor([gate])
    out = belief_update(p.clone(), seed_zero, g, peer_pos=claim, peer_conf=conf,
                        peer_strength=strength, peer_trust=trust)
    return float(out[0, :, 0].mean())


def test_gate_one_reproduces_the_ungated_receiver():
    assert abs(_fused_mean(1.0) - _fused_mean(None)) < 1e-6


def test_lower_gate_moves_the_mean_less():
    # The knob's dynamic range sits in [0.5, 1]: below ~0.5 the sigma is
    # wide enough that the ring is effectively flat over the cloud and the
    # drag saturates at the no-fusion baseline -- hence the non-strict
    # ordering below the saturation point.
    drag = {g: abs(_fused_mean(g)) for g in (1.0, 0.5, 0.25, 0.06)}
    assert drag[1.0] > drag[0.5] >= drag[0.25] >= drag[0.06], drag


def test_half_gate_drag_is_materially_smaller():
    drag1 = abs(_fused_mean(1.0))
    drag_half = abs(_fused_mean(0.5))
    assert drag_half < 0.5 * drag1, (drag1, drag_half)


def test_near_zero_gate_nearly_defuses_the_ring():
    drag1 = abs(_fused_mean(1.0))
    drag_tiny = abs(_fused_mean(0.06))
    assert drag_tiny < 0.1 * drag1, (drag1, drag_tiny)
