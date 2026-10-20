"""Phase 170: the peer fix -- the multi-beacon position fix that replaces
single-ring ring fusion (docs/tuning.md phase 169's measured -0.10).

Two levels: the Gauss-Newton solver directly (accuracy and the residual
consistency test), and the full belief_update path (gates, bounded-mass
injection, spread floor).
"""

import math

import torch

from belief import belief_update, _peer_fix_gauss_newton, _spread
from kilobot_gnn import SEED_SIZE

ARENA_HALF = 92.6


def _strength_for(d_norm):
    return 1.0 / (1.0 + d_norm * ARENA_HALF)


def _spread_cloud(center, std, particles=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(1, particles, 3)
    p[0, :, 0] = center[0] + torch.randn(particles, generator=g) * std
    p[0, :, 1] = center[1] + torch.randn(particles, generator=g) * std
    p[0, :, 2] = (torch.rand(particles, generator=g) - 0.5) * 0.1
    return p, g


def _beacons(q_true, offsets, claim_errors=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
    """Rows-shaped peer tensors for beacons at q_true + offsets.

    claim_errors[i] displaces beacon i's CLAIMED centre (its strength stays
    the true range, as a real broadcast does: the receiver ranges the sender's
    TRUE position and hears the sender's CLAIMED position).
    """
    k = len(offsets)
    centers = [torch.tensor([q_true[0] + o[0], q_true[1] + o[1]]) for o in offsets]
    c = torch.zeros(1, k, 2)
    d = torch.zeros(1, k)
    for i, ct in enumerate(centers):
        e = claim_errors[i] if i < len(claim_errors) else 0.0
        claimed = ct + torch.tensor([e, -e * 0.5])
        c[0, i, 0] = claimed[0]
        c[0, i, 1] = claimed[1]
        d[0, i] = (torch.tensor(q_true) - ct).norm()
    conf = torch.full((1, k), 0.99)
    strength = _strength_for(d)
    return c, conf, strength


def test_peer_fix_gauss_newton_recovers_position_from_three_accurate_beacons():
    q_true = torch.tensor([0.1, -0.05])
    c, conf, strength = _beacons(q_true, [(0.05, 0.02), (-0.04, 0.03), (0.06, -0.04)])
    d = (1.0 / strength - 1.0) / ARENA_HALF
    sig = torch.full_like(d, 0.12)
    valid = torch.ones(1, 3, dtype=torch.bool)
    q0 = torch.tensor([[0.0, 0.0]])
    q, resid = _peer_fix_gauss_newton(q0, c, d, sig, valid)
    assert (q[0] - q_true).norm() < 5e-3
    assert float(resid[0]) < 0.5


def test_peer_fix_gauss_newton_flags_an_inconsistent_set():
    # one beacon's claim displaced by 0.3: three rings with no common point.
    # the residual gate is what a single wrong claim cannot hide.
    q_true = torch.tensor([0.1, -0.05])
    c, conf, strength = _beacons(q_true, [(0.05, 0.02), (-0.04, 0.03), (0.06, -0.04)],
                                 claim_errors=(0.0, 0.0, 0.4))
    d = (1.0 / strength - 1.0) / ARENA_HALF
    sig = torch.full_like(d, 0.12)
    valid = torch.ones(1, 3, dtype=torch.bool)
    q, resid = _peer_fix_gauss_newton(torch.tensor([[0.0, 0.0]]), c, d, sig, valid)
    assert float(resid[0]) > 1.5


def _run_peer_fix(cloud_center, cloud_std, beacons, seed=3, **kwargs):
    p, g = _spread_cloud(cloud_center, cloud_std, seed=seed)
    seed_zero = torch.zeros(1, SEED_SIZE)
    out = belief_update(p.clone(), seed_zero, g, peer_pos=beacons[0], peer_conf=beacons[1],
                        peer_strength=beacons[2], peer_fix=True, **kwargs)
    return p, out


def test_peer_fix_injects_bounded_mass_into_spread_clouds():
    # a lost robot's cloud (std 0.25, centred away from the beacons) and two
    # accurate beacons whose rings intersect near the cloud: the fix injects
    # at most half the cloud AT the fix, and the mean moves toward it.
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.03, 0.04)])
    p, out = _run_peer_fix((0.30, 0.25), 0.25, beacons)

    assert not torch.allclose(out, p)                       # something was injected
    near_fix_before = int(((p[0, :, 0:2] - torch.tensor(q_fix)).norm(dim=1) < 0.15).sum())
    near_fix_after = int(((out[0, :, 0:2] - torch.tensor(q_fix)).norm(dim=1) < 0.15).sum())
    assert near_fix_after >= 100 > near_fix_before          # bounded mass: ~half
    d_before = float((p[0, :, 0:2].mean(dim=0) - torch.tensor(q_fix)).norm())
    d_after = float((out[0, :, 0:2].mean(dim=0) - torch.tensor(q_fix)).norm())
    assert d_after < d_before


def test_peer_fix_skips_structured_receivers():
    # spread below peer_inject_min_spread: the fix is refused and the cloud is
    # returned untouched -- good receivers never hear peer claims (the
    # measured A7-vs-A2 harm)
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.03, 0.04)])
    p, out = _run_peer_fix((0.30, 0.25), 0.05, beacons)
    assert torch.allclose(out, p)


def test_peer_fix_rejects_bad_pair_separation():
    # exactly two beacons whose claims sit 0.5 apart: impossible geometry for
    # two honest senders, both within IR range of one receiver
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.45, 0.02)])
    assert abs(float((beacons[0][0, 0] - beacons[0][0, 1]).norm()) - 0.5) < 1e-3
    p, out = _run_peer_fix((0.30, 0.25), 0.25, beacons)
    assert torch.allclose(out, p)


def test_peer_fix_rejects_inconsistent_trio():
    # three beacons, one claim off by 0.3: the residual gate refuses the fix
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.03, 0.04), (0.02, -0.05)],
                       claim_errors=(0.0, 0.0, 0.4))
    p, out = _run_peer_fix((0.30, 0.25), 0.25, beacons)
    assert torch.allclose(out, p)


def test_peer_fix_reports_injected_robots_to_the_caller():
    # the in-place peer_injected mask: exactly the injected robot stamped, so
    # the caller's cooldown bookkeeping targets the right robots
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.03, 0.04)])
    p, g = _spread_cloud((0.30, 0.25), 0.25, seed=3)
    injected = torch.zeros(1, dtype=torch.bool)
    belief_update(p.clone(), torch.zeros(1, SEED_SIZE), g, peer_pos=beacons[0],
                  peer_conf=beacons[1], peer_strength=beacons[2], peer_fix=True,
                  peer_injected=injected)
    assert bool(injected[0])


def test_peer_fix_respects_the_caller_cooldown():
    # inject_ok False: the fix computes fine but is not delivered this tick
    q_fix = (0.35, 0.20)
    beacons = _beacons(q_fix, [(0.05, 0.02), (-0.03, 0.04)])
    p, g = _spread_cloud((0.30, 0.25), 0.25, seed=3)
    ok = torch.zeros(1, dtype=torch.bool)
    injected = torch.zeros(1, dtype=torch.bool)
    out = belief_update(p.clone(), torch.zeros(1, SEED_SIZE), g, peer_pos=beacons[0],
                        peer_conf=beacons[1], peer_strength=beacons[2], peer_fix=True,
                        peer_inject_ok=ok, peer_injected=injected)
    assert torch.allclose(out, p)
    assert not bool(injected[0])
