"""The two SwarmManager hooks that make a real player testable at all.

Both exist because the deleted python replica had, for free, two things a real
Unity player does not: a seeded RNG (so two runs saw the same arena) and direct
write access to robot positions (so a test could set up an exact geometry). The
swarm-RNG pin and the pose command give those back over the existing channels.

These tests cover the hooks themselves. What they enable is exercised in
test_critic_belief (two rollouts compared step for step) and test_heartbeat
(two robots planted out of everything's range).
"""

import math

import numpy as np
import pytest

import conftest
from conftest import requires_unity


def _xz(worker, k = 0):
    from belief import ARENA_HALF
    return worker.snapshot(k)["node"][:, 0:2].cpu().numpy() * ARENA_HALF


def _forward(worker, k = 0):
    return worker.snapshot(k)["node"][:, 2:4].cpu().numpy()


@requires_unity
def test_unseeded_respawns_differ():
    # The default has to stay unseeded: a training run wants a fresh arena every
    # episode, and this is the behavior every run had before the hook existed.
    #
    # Several respawns, and only "not all identical" asserted, rather than
    # comparing one pair: two spawns coinciding is improbable but not
    # impossible, and this test failing intermittently would be worse than
    # useless. (Measured separately, coincidences across separate PROCESSES are
    # not improbable at all -- see the note in docs/code-overview.md.)
    worker = conftest.players().get(min_bots = 6, max_bots = 12)
    seen = [_xz(worker)]
    for _ in range(3):
        worker.send_reset(0, 0)
        worker.reset_env()
        seen.append(_xz(worker))
    differs = any(a.shape != b.shape or not np.allclose(a, b, atol = 1e-4)
                  for a, b in zip(seen, seen[1:]))
    assert differs, "every unseeded respawn produced the identical arena"


@requires_unity
def test_pinned_seed_replays_the_same_arena():
    # The property the pin exists for: same seed, same arena, even across other
    # respawns in between and on a player that has been alive the whole session.
    worker = conftest.players().get(min_bots = 6, max_bots = 10, swarm_rng = 1234)
    pos_a, fwd_a = _xz(worker), _forward(worker)

    conftest.players().get(min_bots = 6, max_bots = 10, swarm_rng = 99)
    conftest.players().get(min_bots = 6, max_bots = 10)

    worker = conftest.players().get(min_bots = 6, max_bots = 10, swarm_rng = 1234)
    pos_b, fwd_b = _xz(worker), _forward(worker)

    assert pos_a.shape == pos_b.shape, "the pin must reproduce the population count too"
    assert np.allclose(pos_a, pos_b, atol = 1e-4)
    assert np.allclose(fwd_a, fwd_b, atol = 1e-4), "cardinal spawn headings must replay too"


@requires_unity
def test_different_seeds_give_different_arenas():
    worker = conftest.players().get(min_bots = 6, max_bots = 10, swarm_rng = 1)
    first = _xz(worker)
    worker = conftest.players().get(min_bots = 6, max_bots = 10, swarm_rng = 2)
    second = _xz(worker)
    assert first.shape != second.shape or not np.allclose(first, second, atol = 1e-4)


@requires_unity
def test_arenas_differ_under_one_seed():
    # arenaId is mixed into the seed on purpose: parallel arenas sharing one seed
    # would otherwise all run the identical swarm, which is worse than useless
    # for training throughput.
    worker = conftest.players().get(num_arenas = 2, min_bots = 6, max_bots = 10,
                                    swarm_rng = 55)
    a, b = _xz(worker, 0), _xz(worker, 1)
    assert a.shape != b.shape or not np.allclose(a, b, atol = 1e-4)


@requires_unity
def test_place_sets_position_and_heading_exactly():
    worker = conftest.players().get(min_bots = 4, max_bots = 4)
    want = [(0, -40.0, 25.0, 0.0),
            (1, 12.5, -60.25, math.pi / 2),
            (2, 0.0, 0.0, math.pi),
            (3, 33.0, 33.0, -math.pi / 2)]
    conftest.place(worker, 0, want)
    pos = _xz(worker)
    fwd = _forward(worker)
    for local, x, z, heading in want:
        assert pos[local] == pytest.approx([x, z], abs = 0.05)
        # python heading h means direction (cos h, sin h) in (x, z); the node's
        # own forward columns are that same pair, so this checks the whole
        # round trip through SwarmManager's yaw conversion, not just that
        # SOMETHING changed.
        assert fwd[local] == pytest.approx([math.cos(heading), math.sin(heading)], abs = 0.02)


@requires_unity
def test_place_leaves_other_arenas_alone():
    worker = conftest.players().get(num_arenas = 2, min_bots = 4, max_bots = 4)
    before = _xz(worker, 1)
    conftest.place(worker, 0, [(0, 5.0, 5.0, 0.0)])
    after = _xz(worker, 1)
    assert before.shape == after.shape
    # arena 1 keeps ticking, so its robots drive a little; what must not happen
    # is one of them being teleported.
    assert np.abs(before - after).max() < 2.0


@requires_unity
def test_place_ignores_out_of_range_index():
    # A stale index (a pose aimed at a robot count the arena no longer has) must
    # be dropped with a warning, not take the player down mid-suite.
    worker = conftest.players().get(min_bots = 4, max_bots = 4)
    conftest.place(worker, 0, [(9999, 1.0, 1.0, 0.0), (0, -10.0, -10.0, 0.0)])
    pos = _xz(worker)
    assert pos.shape[0] == 4
    assert pos[0] == pytest.approx([-10.0, -10.0], abs = 0.05), \
        "the valid pose in the same message must still apply"
