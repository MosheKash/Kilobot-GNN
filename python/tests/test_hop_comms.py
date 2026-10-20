"""The hop-count protocol (config.hop_comms).

Two sides, each measured negative of the honest-beacon space motivated it:
position claims need accurate sender means, and the only accurate senders are
geometrically unreachable by receivers in need (S2a/S2b). Hop counts are
integer minima that self-correct, carry no mean, and bound the receiver's
position region without any sender error entering it.

- observation.hop_flood_update: the min-relaxation flood. slot v >= 1 in a
  received row means the sender is k=v-1 hops from that seed, so the
  receiver's candidate is v. Direct seed sighting resets that seed to hop 0.
- belief.add_hop_bounds: soft upper-bound likelihoods -- euclidean
  dist(p, seed) <= k * IR range -- whose intersection prunes the cloud.
"""

from types import SimpleNamespace

import torch

import observation
from belief import belief_update
from kilobot_gnn import SEED_SIZE, MESSAGE_SIZE, DB_ROW_SIZE

STRENGTH_COL = MESSAGE_SIZE + 1


def _worker():
    return SimpleNamespace(hop={})


def _rows(n=1, cap=2, size=DB_ROW_SIZE):
    return torch.zeros(n, cap, size, dtype=torch.float32), torch.zeros(n, cap, dtype=torch.bool)


def _arena_local(n=1, a=0, l=0):
    return torch.tensor([a] * n), torch.tensor([l] * n)


def test_direct_sighting_pins_hop_zero():
    w = _worker()
    rows, valid = _rows()
    direct = torch.tensor([[1.0, 0.0, 0.0, 0.0]])   # seed 0 in direct range
    out = observation.hop_flood_update(w, *_arena_local(), rows, valid, STRENGTH_COL, direct)
    assert out.tolist() == [[0, -1, -1, -1]]
    assert w.hop[0][0] == [0, -1, -1, -1]


def test_received_row_advances_a_hop():
    w = _worker()
    rows, valid = _rows(cap=1)
    # sender is 0 hops (direct) to seed 1, 1 hop to seed 2 -> broadcasts 1 and 2
    rows[0, 0, 1] = 1.0
    rows[0, 0, 2] = 2.0
    rows[0, 0, STRENGTH_COL] = 0.5
    valid[0, 0] = True
    out = observation.hop_flood_update(w, *_arena_local(), rows, valid, STRENGTH_COL, None)
    assert out.tolist() == [[-1, 1, 2, -1]]


def test_zero_strength_rows_are_ignored():
    w = _worker()
    rows, valid = _rows(cap=1)
    rows[0, 0, 0] = 1.0
    rows[0, 0, STRENGTH_COL] = 0.0     # not physically received
    valid[0, 0] = True
    out = observation.hop_flood_update(w, *_arena_local(), rows, valid, STRENGTH_COL, None)
    assert out.tolist() == [[-1, -1, -1, -1]]


def test_min_relaxation_only_ever_lowers():
    w = _worker()
    rows, valid = _rows(cap=2)
    rows[0, 0, 0] = 5.0      # sender claims a far hop count
    rows[0, 1, 0] = 2.0
    rows[0, :, STRENGTH_COL] = 0.5
    valid[0, 0] = True
    valid[0, 1] = True
    out = observation.hop_flood_update(w, *_arena_local(), rows, valid, STRENGTH_COL, None)
    assert out.tolist() == [[2, -1, -1, -1]]      # the tighter row wins
    # a later, looser row cannot raise it
    rows2, valid2 = _rows(cap=1)
    rows2[0, 0, 0] = 7.0
    rows2[0, 0, STRENGTH_COL] = 0.5
    valid2[0, 0] = True
    out2 = observation.hop_flood_update(w, *_arena_local(), rows2, valid2, STRENGTH_COL, None)
    assert out2.tolist() == [[2, -1, -1, -1]]


def test_direct_sighting_overrides_stale_hop():
    w = _worker()
    rows, valid = _rows(cap=1)
    rows[0, 0, 0] = 4.0
    rows[0, 0, STRENGTH_COL] = 0.5
    valid[0, 0] = True
    direct = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # now in range of seed 3
    out = observation.hop_flood_update(w, *_arena_local(), rows, valid, STRENGTH_COL, direct)
    assert out.tolist() == [[4, -1, -1, 0]]


def test_hop_bounds_prune_the_mirror_cluster():
    # half the cloud hugs the top-left seed (seed 0), half the bottom-left
    # (seed 2). A single claim "1 hop from seed 0" must leave the top cluster
    # weighted and drive the bottom one to ~-inf log-weight -- the euclidean
    # upper bound kills it. Tested at the _Update level, where the pure
    # measurement lives (belief_update wraps it in resample/injection logic
    # that would otherwise contaminate the assertion).
    from belief import _Update
    parts = 256
    head = parts // 2
    p = torch.zeros(1, parts, 3)
    p[0, :head, 0] = -0.9 + 0.01 * torch.randn(head)
    p[0, :head, 1] = 0.9 + 0.01 * torch.randn(head)
    p[0, head:, 0] = -0.9 + 0.01 * torch.randn(head)
    p[0, head:, 1] = -0.9 + 0.01 * torch.randn(head)
    u = _Update(p.clone(), torch.Generator().manual_seed(3), None)
    u.add_hop_bounds(torch.tensor([[1, -1, -1, -1]]))
    lw = u.log_w[0]
    top = float(lw[:head].max())
    bot = float(lw[head:].max())
    assert top > -1.0                      # inside the bound: ~neutral
    assert bot < -20.0                     # outside: massively down-weighted


def test_hop_bounds_unknown_does_nothing():
    p = torch.zeros(1, 64, 3)
    p[0, :, 0] = torch.linspace(-1, 1, 64)
    g = torch.Generator().manual_seed(4)
    out = belief_update(p.clone(), torch.zeros(1, SEED_SIZE), g, hop_k=torch.full((1, SEED_SIZE), -1))
    kept = float((out[0, :, 0] - p[0, :, 0]).abs().max())
    assert kept <= 1e-6   # no bound fired: the cloud is returned unchanged