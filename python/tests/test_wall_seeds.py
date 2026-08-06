import numpy as np
import torch

from belief import (belief_init, belief_predict, belief_update, belief_read,
                    ARENA_HALF, WALL_SIZE, WALL_AXIS, WALL_VAL,
                    IR_RANGE_NORM, MEAS_SIGMA, _spread, _wall_along_log_w)
from kilobot_gnn import SEED_SIZE, MESSAGE_SIZE
import actor_io
from config import Config


def test_wall_observation_pulls_one_axis_leaves_other_unconstrained():
    g = torch.Generator().manual_seed(3)
    n = 8
    particles = belief_init(n, g)
    zero = torch.zeros(n)
    d_true = 0.05
    strength = 1.0 / (1.0 + d_true * ARENA_HALF)
    seed_obs = torch.zeros(n, SEED_SIZE)
    for _ in range(3):
        particles = belief_predict(particles, zero, zero, zero, g)
        wall_obs = torch.zeros(n, WALL_SIZE)
        wall_obs[:, 0] = strength
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    my = feats[:, 1]
    assert float((my - (WALL_VAL[0] - d_true)).abs().median()) < 0.05
    var_x_direct = ((particles[:, :, 0] - particles[:, :, 0].mean(dim = 1, keepdim = True)) ** 2).mean(dim = 1)
    assert float(var_x_direct.median()) > 0.05


def test_long_confident_axis_does_not_block_the_other_axis_converging():
    # the direct, positive statement of the phase-12 fix: convergence on one
    # axis, however long-since confident, must never prevent the other axis
    # from still receiving fresh injection when it is finally measured, and
    # must not itself be disrupted while that catch-up happens. Before phase
    # 12 this was never guaranteed -- the joint gate only cared about the
    # combined spread, so a long-confident Y could coincidentally still leave
    # the door open for X (if X's own looseness kept the combined number high
    # enough) or, just as easily, an already-tightened combined spread could
    # block X's injection even though X itself had never once been measured
    g = torch.Generator().manual_seed(9)
    n = 20
    particles = belief_init(n, g)
    zero = torch.zeros(n)
    d = 0.03
    strength = 1.0 / (1.0 + d * ARENA_HALF)
    seed_obs = torch.zeros(n, SEED_SIZE)

    for _ in range(10):
        particles = belief_predict(particles, zero, zero, zero, g)
        wall_obs = torch.zeros(n, WALL_SIZE)
        wall_obs[:, 0] = strength
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    assert float(feats[:, 9].median()) < 0.1    # conf_x: never measured, correctly still near zero
    assert float(feats[:, 10].median()) > 0.9   # conf_y: ten north-wall ticks, correctly confident

    for _ in range(10):
        particles = belief_predict(particles, zero, zero, zero, g)
        wall_obs = torch.zeros(n, WALL_SIZE)
        wall_obs[:, 1] = strength
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    assert float(feats[:, 9].median()) > 0.9    # conf_x: now converges despite Y's long-standing confidence
    assert float(feats[:, 10].median()) > 0.9   # conf_y: preserved, not disrupted by X's catch-up phase


def test_wall_observation_alone_does_not_reach_full_position_confidence():
    g = torch.Generator().manual_seed(4)
    n = 8
    particles = belief_init(n, g)
    zero = torch.zeros(n)
    strength = 1.0 / (1.0 + 0.05 * ARENA_HALF)
    seed_obs = torch.zeros(n, SEED_SIZE)
    for _ in range(4):
        particles = belief_predict(particles, zero, zero, zero, g)
        wall_obs = torch.zeros(n, WALL_SIZE)
        wall_obs[:, 0] = strength
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    conf_y = feats[:, 10]
    conf_pos = feats[:, 4]
    assert float(conf_y.median()) > float(conf_pos.median())
    assert float(conf_pos.median()) < 0.5


def test_three_seed_cluster_still_fully_collapses_with_wall_obs_present_but_zero():
    from belief import set_layout
    set_layout("cluster")
    try:
        g = torch.Generator().manual_seed(5)
        n = 8
        particles = belief_init(n, g)
        zero = torch.zeros(n)
        wall_obs = torch.zeros(n, WALL_SIZE)
        for seed_idx in [0, 1, 2]:
            d = 0.03
            strength = 1.0 / (1.0 + d * ARENA_HALF)
            seed_obs = torch.zeros(n, SEED_SIZE)
            seed_obs[:, seed_idx] = strength
            for _ in range(3):
                particles = belief_predict(particles, zero, zero, zero, g)
                particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
        feats = belief_read(particles)
        assert float(feats[:, 4].median()) > 0.5
    finally:
        set_layout("corners")


def test_belief_update_still_fuses_two_simultaneous_wall_axes_if_given_them():
    # belief_update's own fusion math is unchanged by the phase-10 fix (see
    # docs/tuning.md) and still correctly handles a wall_obs with two
    # simultaneously-nonzero sides if handed one directly, as this test does --
    # useful generality to keep, and worth a regression test on its own. But
    # the real pipeline can no longer produce that input: sample_split_event
    # now narrows to exactly one wall side per tick the same way it already
    # narrowed neighbor messages, so a corner takes two separate ticks to
    # constrain both axes, not one. See
    # test_corner_needs_two_ticks_to_constrain_both_axes_via_real_pipeline
    # for what the actual pipeline does now.
    g = torch.Generator().manual_seed(6)
    n = 8
    particles = belief_init(n, g)
    zero = torch.zeros(n)
    d_true = 0.04
    strength = 1.0 / (1.0 + d_true * ARENA_HALF)
    seed_obs = torch.zeros(n, SEED_SIZE)
    for _ in range(6):
        particles = belief_predict(particles, zero, zero, zero, g)
        wall_obs = torch.zeros(n, WALL_SIZE)
        wall_obs[:, 0] = strength
        wall_obs[:, 1] = strength
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    conf_x = feats[:, 9]
    conf_y = feats[:, 10]
    assert float(conf_x.median()) > 0.5
    assert float(conf_y.median()) > 0.5


def test_wall_obs_none_matches_prior_behavior():
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    n = 4
    p1 = belief_init(n, g1)
    p2 = belief_init(n, g2)
    zero = torch.zeros(n)
    seed_obs = torch.zeros(n, SEED_SIZE)
    seed_obs[:, 0] = 0.3
    p1 = belief_predict(p1, zero, zero, zero, g1)
    p2 = belief_predict(p2, zero, zero, zero, g2)
    p1 = belief_update(p1, seed_obs, g1)
    p2 = belief_update(p2, seed_obs, g2, wall_obs = None)
    assert torch.allclose(p1, p2)


def test_wall_axis_and_val_orientation():
    assert WALL_AXIS == [1, 0, 1, 0]
    assert WALL_VAL == [0.95, 0.95, -0.95, -0.95]


def test_point_seed_ring_injection_wins_when_seed_and_wall_both_fire_cold():
    d_seed = 0.03
    seed_strength = 1.0 / (1.0 + d_seed * ARENA_HALF)
    d_wall = 0.05
    wall_strength = 1.0 / (1.0 + d_wall * ARENA_HALF)
    n = 8

    g = torch.Generator().manual_seed(11)
    particles = belief_init(n, g)
    zero = torch.zeros(n)
    seed_obs = torch.zeros(n, SEED_SIZE)
    seed_obs[:, 1] = seed_strength
    wall_obs = torch.zeros(n, WALL_SIZE)
    wall_obs[:, 0] = wall_strength
    for _ in range(20):
        particles = belief_predict(particles, zero, zero, zero, g)
        particles = belief_update(particles, seed_obs, g, wall_obs = wall_obs)
    feats = belief_read(particles)
    # if the wall's band injection had wrongly taken over instead of the seed's
    # ring injection, x would stay pinned near-uniform over the full spawn
    # range every round (as in the pure wall-only case) and this would never
    # tighten; a seed-informed cloud should converge close to the true range
    assert float((feats[:, 8] - d_seed).abs().median()) < 0.1


def _make_wall_seeds_like_unity(half_extent, spacing):
    xs = np.arange(-half_extent, half_extent + 1.0, spacing)
    ys = np.arange(-half_extent, half_extent + 1.0, spacing)
    north = np.stack([xs, np.full_like(xs, half_extent)], axis = 1)
    south = np.stack([xs, np.full_like(xs, -half_extent)], axis = 1)
    east = np.stack([np.full_like(ys, half_extent), ys], axis = 1)
    west = np.stack([np.full_like(ys, -half_extent), ys], axis = 1)
    return [north, east, south, west]


def _all_wall_seed_lines():
    """All four walls' seed positions, from the shared constants."""
    from belief import ARENA_HALF, WALL_SPACING, WALL_SEED_INSET
    inset = ARENA_HALF - WALL_SEED_INSET
    xs = np.arange(-ARENA_HALF, ARENA_HALF + 1.0, WALL_SPACING)
    north = np.stack([xs, np.full_like(xs, inset)], axis = 1)
    south = np.stack([xs, np.full_like(xs, -inset)], axis = 1)
    east = np.stack([np.full_like(xs, inset), xs], axis = 1)
    west = np.stack([np.full_like(xs, -inset), xs], axis = 1)
    return [north, east, south, west]   # SwarmManager's own side order


def test_true_corners_are_within_range_of_both_adjacent_walls():
    from belief import ARENA_HALF, IR_RANGE
    half_extent = ARENA_HALF
    ir = IR_RANGE
    groups = _all_wall_seed_lines()
    corners = [(half_extent, half_extent), (half_extent, -half_extent),
              (-half_extent, half_extent), (-half_extent, -half_extent)]
    # north/south constrain x, east/west constrain y: every true corner should
    # sit within IR_RANGE of its own nearest point in BOTH adjacent groups
    adjacency = {
        (half_extent, half_extent): [0, 1],    # north, east
        (half_extent, -half_extent): [2, 1],   # south, east
        (-half_extent, half_extent): [0, 3],   # north, west
        (-half_extent, -half_extent): [2, 3],  # south, west
    }
    for cx, cz in corners:
        for side in adjacency[(cx, cz)]:
            pts = groups[side]
            d = np.sqrt(((pts - np.array([cx, cz])) ** 2).sum(axis = 1))
            assert float(d.min()) <= ir


def _wall_seed_line(half_extent, spacing, inset, axis_val):
    """One wall's seed positions, the way SwarmManager lays them out.

    Derived from the shared constants rather than read out of a table: the
    replica used to own that table, and Unity generates its own from these same
    numbers, so the constants are what the invariant actually rests on.
    """
    xs = np.arange(-half_extent, half_extent + 1.0, spacing)
    return np.stack([xs, np.full_like(xs, axis_val)], axis = 1)


def test_wall_spacing_guarantees_no_gap_along_any_wall():
    # the design invariant: no point on a wall is farther than IR_RANGE from
    # the nearest wall seed, given WALL_SPACING and WALL_SEED_INSET
    from belief import ARENA_HALF, WALL_SPACING, IR_RANGE, WALL_SEED_INSET
    inset_val = ARENA_HALF - WALL_SEED_INSET
    xs_probe = np.linspace(-ARENA_HALF, ARENA_HALF, 401)
    for axis_val in (inset_val, -inset_val):
        pts = _wall_seed_line(ARENA_HALF, WALL_SPACING, WALL_SEED_INSET, axis_val)
        probe = np.stack([xs_probe, np.full_like(xs_probe, axis_val)], axis = 1)
        d = np.sqrt(((probe[:, None, :] - pts[None, :, :]) ** 2).sum(axis = 2))
        assert float(d.min(axis = 1).max()) <= IR_RANGE


def test_belief_predict_uses_true_heading_not_uncorrectable_estimate():
    # regression test: confirmed directly on a
    # real trajectory -- with no true_heading, particle heading starts
    # uniformly random at belief_init and only ever accumulates relative
    # dtheta on top of that arbitrary start (no measurement in this module
    # constrains heading directly), so an initially-wrong heading can never
    # self-correct -- and since position updates rotate each tick's local
    # displacement by the particle's OWN heading, a wrong heading silently
    # corrupts every subsequent position prediction. Traced on one real
    # robot: belief heading was off by up to 175 degrees essentially the
    # entire trajectory, collapsing median belief-position error from ~80
    # units to ~6 once fixed. This test sets up a particle cloud with a
    # deliberately wrong prior heading (0 rad) and confirms that supplying
    # true_heading makes belief_predict (a) set the resulting particle
    # heading to exactly that true value, not the old wrong one plus
    # dtheta, and (b) rotate the position update using the correct
    # start-of-interval heading (true_heading - dtheta), not the old, wrong
    # prior -- both confirmed by matching a hand-computed expected position.
    import math
    from belief import belief_predict, ARENA_HALF

    n, k = 1, 8
    p = torch.zeros(n, k, 3)
    p[:, :, 0] = 0.0
    p[:, :, 1] = 0.0
    p[:, :, 2] = 0.0   # deliberately WRONG prior heading

    x_local = torch.tensor([20.0])   # raw units, local frame
    y_local = torch.tensor([0.0])
    dtheta = torch.tensor([0.0])     # no turn this interval, to isolate the heading-source effect
    true_heading = torch.tensor([math.pi / 2])   # true heading is 90 degrees, nothing like the wrong prior

    gen = torch.Generator().manual_seed(0)
    out = belief_predict(p, x_local, y_local, dtheta, gen, true_heading = true_heading)

    # heading must be set to the true value exactly, not 0 + dtheta(=0)
    assert torch.allclose(out[:, :, 2], true_heading.unsqueeze(1).expand(-1, k), atol = 1e-5), \
        "particle heading must be set directly to true_heading, not accumulated from the old, wrong prior"

    # position must be rotated by the TRUE start-of-interval heading (here,
    # true_heading - dtheta = 90 degrees, since dtheta=0), not the wrong
    # prior (0 degrees) -- a 20-unit local-x move rotated 90 degrees lands
    # almost entirely on global y, not global x
    mean_x = float(out[:, :, 0].mean()) * ARENA_HALF
    mean_y = float(out[:, :, 1].mean()) * ARENA_HALF
    assert abs(mean_x) < 1.0, "rotating by the true heading should leave global x near zero, not the ~20 a wrong-prior rotation would give"
    assert abs(mean_y - 20.0) < 1.0, "rotating a 20-unit local-x move by a true 90 degree heading should land almost entirely on global y"

    # backward compatibility: omitting true_heading falls back to the
    # original behavior (accumulating on the particle's own prior heading)
    out_no_th = belief_predict(p, x_local, y_local, dtheta, gen)
    mean_x_no_th = float(out_no_th[:, :, 0].mean()) * ARENA_HALF
    assert abs(mean_x_no_th - 20.0) < 1.0, "without true_heading, the original prior-heading-based rotation must still apply"
    # regression test: confirmed directly from
    # real Unity logs and isolated testing -- a genuinely correct single-axis
    # (wall) reading, fused against a prior belief that's drifted far
    # from the true position (which happens over a long committed journey),
    # produced a large, essentially random jump in the axis that measurement
    # says nothing about. This is Monte Carlo resampling noise from too few
    # particles representing a wide prior, not a logic bug in the fusion math
    # itself: at the original BELIEF_PARTICLES=32 this measured ~41 units of
    # spurious shift on this exact scenario, shrinking to single digits by
    # 256+. This is the confirmed, direct mechanism behind committed robots
    # stopping at wildly wrong locations -- the spurious jump happens to land
    # the belief near the assigned target by chance, while the robot's real
    # position (often still visibly at the wall that just fired) has
    # nothing to do with where the target actually is.
    from belief import belief_update, BELIEF_PARTICLES, ARENA_HALF

    true_x, true_y = 48.67, -88.84
    dist_to_wall = 95.0 + true_y
    strength_val = 1.0 / (1.0 + dist_to_wall)
    wall_obs = torch.zeros(1, WALL_SIZE)
    wall_obs[0, 2] = strength_val   # south, genuinely correct for true_y

    stale_x, stale_y, spread_raw = 20.0, -40.0, 30.0   # a drifted, wide-spread prior
    max_shift = 0.0
    for trial_seed in range(10):
        g = torch.Generator().manual_seed(trial_seed)
        p = torch.zeros(1, BELIEF_PARTICLES, 3)
        p[0, :, 0] = (stale_x + torch.randn(BELIEF_PARTICLES, generator = g) * spread_raw) / ARENA_HALF
        p[0, :, 1] = (stale_y + torch.randn(BELIEF_PARTICLES, generator = g) * spread_raw) / ARENA_HALF
        p[0, :, 2] = torch.randn(BELIEF_PARTICLES, generator = g) * 0.3
        mean_x0 = float(p[0, :, 0].mean()) * ARENA_HALF
        p2 = belief_update(p, torch.zeros(1, SEED_SIZE), g, wall_obs = wall_obs)
        mean_x2 = float(p2[0, :, 0].mean()) * ARENA_HALF
        max_shift = max(max_shift, abs(mean_x2 - mean_x0))
    assert max_shift < 20.0, "a wall reading must not produce a large spurious jump in the axis it doesn't measure (got %.2f)" % max_shift


def test_wall_seed_xy_gated_by_reception_competition():
    # wall_seed_xy_part from sample_split_event
    # must be nonzero if and only if the wall channel genuinely won this
    # tick's single-receiver draw -- never leaked when a different
    # channel won, matching wall_part's own gating exactly
    cfg = Config()
    n = 200
    seeds = torch.rand(n, SEED_SIZE) * 0.1
    walls = torch.zeros(n, WALL_SIZE)
    walls[:, 0] = 0.9
    rows = torch.zeros(n, 8, MESSAGE_SIZE + 2)
    valid = torch.zeros(n, 8, dtype = torch.bool)

    wall_seed_xy = torch.zeros(n, WALL_SIZE, 2)
    wall_seed_xy[:, 0] = torch.tensor([-60.0, 95.0])

    rng = torch.Generator().manual_seed(0)
    tc, seed_p, wall_p, wall_xy_p = actor_io.sample_split_event(
        seeds, walls, rows, valid, cfg, rng, wall_seed_xy = wall_seed_xy)

    wall_won = wall_p.sum(dim = 1) > 0
    xy_nonzero = wall_xy_p.abs().sum(dim = 1) > 0
    assert bool((wall_won == xy_nonzero).all()), "wall_seed_xy_part must be nonzero exactly when the wall channel won"
    assert bool(wall_won.any()), "test setup: wall must win at least some draws for this to be a meaningful check"
    assert torch.allclose(wall_xy_p[wall_won], torch.tensor([-60.0, 95.0]).expand(int(wall_won.sum()), 2)), \
        "whenever wall wins, the exposed position must be the real, known seed position"
    assert bool((wall_xy_p[~wall_won] == 0).all()), "whenever wall does not win, the position must be exactly zero, not leaked"


def test_along_wall_likelihood_penalizes_divergent_particle():
    # this is the actual claim the whole
    # mechanism rests on -- a particle whose dead-reckoned along-wall
    # position has diverged from a real wall-seed reading must be
    # penalized relative to one that hasn't, and the penalty must grow
    # with the divergence (this is what lets a wrong heading hypothesis
    # finally get discriminated against, unlike a plain wall reading
    # which never constrained this axis at all)
    ARENA_HALF_V = ARENA_HALF
    v = 1.55 * 0.02
    ratios = []
    for T in [40, 100, 200, 400]:
        start_x, start_y = -60.0, 90.0
        correct_x = start_x + T * v * np.cos(0.0)
        correct_y = start_y + T * v * np.sin(0.0)
        wrong_x = start_x + T * v * np.cos(np.radians(90))
        wrong_y = start_y + T * v * np.sin(np.radians(90))

        k = 8
        p = torch.zeros(2, k, 3)
        p[0, :, 0] = correct_x / ARENA_HALF_V
        p[0, :, 1] = correct_y / ARENA_HALF_V
        p[1, :, 0] = wrong_x / ARENA_HALF_V
        p[1, :, 1] = wrong_y / ARENA_HALF_V

        true_seed_x, true_seed_y = correct_x + 2.0, 90.0
        strength = torch.tensor([0.3, 0.3])
        d_meas = (1.0 / strength.clamp(min = 1e-6) - 1.0) / ARENA_HALF_V
        along_val = torch.tensor([true_seed_y, true_seed_y]) / ARENA_HALF_V
        spread = _spread(p)
        along_bound_sq = (IR_RANGE_NORM ** 2 - d_meas ** 2).clamp(min = 0.0)
        along_sigma = torch.sqrt(along_bound_sq).clamp(min = MEAS_SIGMA) + 0.15 * spread
        vis = torch.tensor([True, True])
        log_w = _wall_along_log_w(p, 1, along_val, along_sigma, vis)
        ratios.append(float(torch.exp(log_w[0].mean() - log_w[1].mean())))

    assert ratios[0] > 1.0, "the correct-heading particle must always be favored, even after a short time"
    assert ratios[-1] > ratios[0] * 3.0, \
        "the discrimination power must grow substantially as the wrong-heading particle's divergence accumulates, not stay flat"


def test_unity_wall_seed_rows_extracted_and_removed_from_competition():
    # real-Unity wall-seed position rows are
    # marked with a negative senderId (SwarmManager.cs); this must be
    # correctly extracted into the same per-band format the replica's own
    # _wall_seed_xy uses, and removed from rows entirely afterward so a
    # wall-seed row never competes as if it were a genuine neighbor
    # message
    n = 3
    max_rows = 5
    rows = torch.zeros(n, max_rows, MESSAGE_SIZE + 2)
    rows[0, 0, 5] = 1.0
    rows[0, 0, MESSAGE_SIZE] = 3.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.4
    rows[0, 1, 0] = 12.0
    rows[0, 1, 1] = 95.0
    rows[0, 1, MESSAGE_SIZE] = -2.0   # band = 2 - 1 = 1 (east)
    rows[0, 1, MESSAGE_SIZE + 1] = 0.3
    rows[1, 0, 0] = -60.0
    rows[1, 0, 1] = -95.0
    rows[1, 0, MESSAGE_SIZE] = -4.0   # band = 4 - 1 = 3 (west)
    rows[1, 0, MESSAGE_SIZE + 1] = 0.25
    rows[2, 0, 3] = 0.7
    rows[2, 0, MESSAGE_SIZE] = 5.0
    rows[2, 0, MESSAGE_SIZE + 1] = 0.5

    sender_col = rows[:, :, MESSAGE_SIZE]
    is_wall_seed_row = sender_col < 0
    wall_seed_xy_unity = torch.zeros(rows.shape[0], WALL_SIZE, 2)
    idxs = is_wall_seed_row.nonzero(as_tuple = False)
    for i_idx, r_idx in idxs.tolist():
        band = int(round(-float(sender_col[i_idx, r_idx]))) - 1
        if 0 <= band < WALL_SIZE:
            wall_seed_xy_unity[i_idx, band, 0] = rows[i_idx, r_idx, 0]
            wall_seed_xy_unity[i_idx, band, 1] = rows[i_idx, r_idx, 1]
    rows_filtered = torch.where(is_wall_seed_row.unsqueeze(-1), torch.zeros_like(rows), rows)

    assert torch.allclose(wall_seed_xy_unity[0, 1], torch.tensor([12.0, 95.0]))
    assert torch.allclose(wall_seed_xy_unity[1, 3], torch.tensor([-60.0, -95.0]))
    assert torch.allclose(wall_seed_xy_unity[2], torch.zeros(WALL_SIZE, 2))
    assert rows_filtered[0, 0, 5] == 1.0, "a genuine neighbor message row must survive filtering untouched"
    assert rows_filtered[0, 1].abs().sum() == 0.0, "the wall-seed row must be fully zeroed out of rows"
    assert rows_filtered[1, 0].abs().sum() == 0.0, "the wall-seed row must be fully zeroed out of rows"
    assert rows_filtered[2, 0, 3] == 0.7, "a genuine neighbor message row must survive filtering untouched"
