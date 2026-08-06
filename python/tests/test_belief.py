import math
import torch

from belief import (belief_init, belief_predict, belief_update, belief_read,
                    SEED_POS, ARENA_HALF, IR_RANGE_NORM, BELIEF_FEATURES)
from kilobot_gnn import SPLIT_ODOM_SIZE, SEED_SIZE, WALL_SIZE
from kinematics import split_tick_motion

MAX_SPEED = 1.55
WHEELBASE = 1.307
DT = 0.05


def simulate(n_steps, seed, start, heading0, motor_plan, sight_every = 1, true_pose_out = False, speed = None):
    g = torch.Generator().manual_seed(seed)
    pos = start.clone()
    heading = heading0.clone()
    n = pos.shape[0]
    particles = belief_init(n, g)
    err_log = []
    for step in range(n_steps):
        motor = motor_plan(step, n)
        ones = torch.ones(n)
        sp = MAX_SPEED if speed is None else speed
        xl, yl, dth, t = split_tick_motion(motor, ones, sp, WHEELBASE, DT)
        c = torch.cos(heading)
        s = torch.sin(heading)
        pos = pos + torch.stack([(xl * c - yl * s) / ARENA_HALF, (xl * s + yl * c) / ARENA_HALF], dim = 1)
        heading = heading + dth
        particles = belief_predict(particles, xl, yl, dth, g)
        seed_obs = torch.zeros(n, SEED_SIZE)
        if step % sight_every == 0:
            for si in range(SEED_POS.shape[0]):
                d = (pos - SEED_POS[si]).norm(dim = 1)
                vis = d < IR_RANGE_NORM
                seed_obs[vis, si] = 1.0 / (1.0 + d[vis] * ARENA_HALF)
        particles = belief_update(particles, seed_obs, g)
        feats = belief_read(particles)
        pos_err = (feats[:, 0:2] - pos).norm(dim = 1)
        head_est = torch.atan2(feats[:, 2], feats[:, 3])
        head_err = torch.atan2(torch.sin(head_est - heading), torch.cos(head_est - heading)).abs()
        if true_pose_out:
            err_log.append((pos_err, head_err, feats, pos.clone(), heading.clone()))
        else:
            err_log.append((pos_err, head_err, feats))
    return err_log


def circle_plan(step, n):
    motor = torch.zeros(n, 2)
    motor[:, 0] = 0.9
    motor[:, 1] = 0.6
    return motor


def wander_plan(step, n):
    motor = torch.ones(n, 2) * 0.9
    if (step // 6) % 3 == 0:
        motor[:, 1] = 0.4
    return motor


def fast_wander_plan(step, n):
    motor = torch.ones(n, 2) * 0.95
    phase = (step // 7) % 4
    if phase == 1:
        motor[:, 1] = 0.3
    if phase == 3:
        motor[:, 0] = 0.35
    return motor


def test_single_seed_exposes_anchor_but_stays_humble():
    torch.manual_seed(0)
    n = 16
    ang = torch.rand(n) * 2 * math.pi
    # start at half the communication range, same proportion the original test
    # used (it predates the IR_RANGE correction to the real Kilobot's 7cm range)
    start = SEED_POS[0] + torch.stack([torch.cos(ang), torch.sin(ang)], dim = 1) * (IR_RANGE_NORM * 0.5)
    heading0 = torch.rand(n) * 2 * math.pi - math.pi
    log = simulate(240, 1, start, heading0, fast_wander_plan, true_pose_out = True)
    pos_err, head_err, feats, pos, heading = log[-1]
    to_seed = SEED_POS[0].unsqueeze(0) - pos
    true_b = torch.atan2(to_seed[:, 1], to_seed[:, 0]) - heading
    est_b = torch.atan2(feats[:, 6], feats[:, 7])
    berr = torch.atan2(torch.sin(est_b - true_b), torch.cos(est_b - true_b)).abs()
    true_d = to_seed.norm(dim = 1)
    derr = (feats[:, 8] - true_d).abs()
    resultant = torch.sqrt(feats[:, 6] ** 2 + feats[:, 7] ** 2)
    assert float(derr.median()) < 0.07
    assert float(feats[:, 4].median()) < 0.9
    confident = resultant > 0.7
    if bool(confident.any()):
        assert float(berr[confident].median()) < float(berr.median()) + 0.05


def _follow_path(points, g, particles):
    for i in range(1, len(points)):
        p0 = torch.tensor(points[i - 1][0:2])
        p1 = torch.tensor(points[i][0:2])
        th0 = points[i - 1][2]
        th1 = points[i][2]
        d_world = (p1 - p0) * ARENA_HALF
        c = math.cos(-th0)
        s = math.sin(-th0)
        xl = torch.tensor([d_world[0] * c - d_world[1] * s])
        yl = torch.tensor([d_world[0] * s + d_world[1] * c])
        dth = torch.tensor([math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))])
        particles = belief_predict(particles, xl, yl, dth, g)
        seed_obs = torch.zeros(1, SEED_SIZE)
        for si in range(SEED_POS.shape[0]):
            d = (torch.tensor(points[i][0:2]) - SEED_POS[si]).norm()
            if d < IR_RANGE_NORM:
                seed_obs[0, si] = 1.0 / (1.0 + d * ARENA_HALF)
        particles = belief_update(particles, seed_obs, g)
    return particles


def _arc(center, r0, r1, a0, a1, steps):
    pts = []
    for i in range(steps + 1):
        f = i / steps
        r = r0 + (r1 - r0) * f
        a = a0 + (a1 - a0) * f
        pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a), a + math.pi / 2))
    return pts


def _line(p0, p1, steps):
    heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = []
    for i in range(steps + 1):
        f = i / steps
        pts.append((p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f, heading))
    return pts


def test_three_beacons_collapse_absolute_pose():
    # arc radii and near-seed offsets scale with IR_RANGE_NORM (this test predates
    # the correction to the real Kilobot's 7cm range); waypoints n1/n2 stay anchored
    # to the true corner seed positions, only their offset from those seeds shrinks,
    # and the dead-reckoning-only line segments between them are unchanged
    scale = IR_RANGE_NORM / 0.30
    path = _arc((0.0, 0.0), 0.10 * scale, 0.24 * scale, 0.0, 2.4, 40)
    n1 = (-0.9 + 0.18 * scale, 0.9 - 0.12 * scale)
    path = path + _line(path[-1][0:2], n1, 90)[1:]
    path = path + _arc((-0.9, 0.9), 0.21 * scale, 0.12 * scale, math.atan2(n1[1] - 0.9, n1[0] + 0.9), math.atan2(n1[1] - 0.9, n1[0] + 0.9) + 1.6, 35)[1:]
    n2 = (0.9 - 0.2 * scale, 0.9 - 0.1 * scale)
    path = path + _line(path[-1][0:2], n2, 120)[1:]
    path = path + _arc((0.9, 0.9), 0.22 * scale, 0.12 * scale, math.atan2(n2[1] - 0.9, n2[0] - 0.9), math.atan2(n2[1] - 0.9, n2[0] - 0.9) + 1.6, 35)[1:]
    true = torch.tensor(path[-1][0:2])
    errs = []
    confs = []
    for seed in [4, 5, 9]:
        g = torch.Generator().manual_seed(seed)
        p = belief_init(1, g)
        p = _follow_path(path, g, p)
        f = belief_read(p)
        errs.append(float((f[0, 0:2] - true).norm()))
        confs.append(float(f[0, 4]))
    errs_s = sorted(errs)
    assert errs_s[1] < 0.13
    assert errs_s[0] < 0.06
    assert max(confs) < 0.9


def test_peer_ranging_cannot_mint_anchor_confidence():
    g = torch.Generator().manual_seed(7)
    n = 4
    true_pos = torch.tensor([0.1, 0.1])
    particles = belief_init(n, g)
    anchors = [torch.tensor([0.1, 0.3]), torch.tensor([-0.05, -0.05]), torch.tensor([0.3, 0.05])]
    seed_zero = torch.zeros(n, SEED_SIZE)
    for step in range(120):
        which = anchors[step % 3]
        d = (true_pos - which).norm()
        peer_pos = which.view(1, 1, 2).expand(n, 1, 2)
        peer_conf = torch.full((n, 1), 0.99)
        peer_strength = torch.full((n, 1), float(1.0 / (1.0 + d * ARENA_HALF)))
        particles = belief_update(particles, seed_zero, g, peer_pos = peer_pos,
                                  peer_conf = peer_conf, peer_strength = peer_strength)
        particles = belief_predict(particles, torch.zeros(n), torch.zeros(n), torch.zeros(n), g)
    feats = belief_read(particles)
    assert float(feats[:, 4].max()) < 0.85

def test_belief_no_false_certainty_without_sightings():
    torch.manual_seed(0)
    n = 8
    g = torch.Generator().manual_seed(3)
    particles = belief_init(n, g)
    for step in range(120):
        motor = torch.ones(n, 2) * 0.8
        xl, yl, dth, t = split_tick_motion(motor, torch.ones(n), MAX_SPEED, WHEELBASE, DT)
        particles = belief_predict(particles, xl, yl, dth, g)
    feats = belief_read(particles)
    assert float(feats[:, 4].max()) < 0.2
    assert float(feats[:, 5].max()) < 0.5


def test_belief_update_pulls_to_ring():
    g = torch.Generator().manual_seed(5)
    particles = belief_init(4, g)
    true_pos = SEED_POS[2] + torch.tensor([-0.1, -0.1])
    d = (true_pos - SEED_POS[2]).norm()
    seed_obs = torch.zeros(4, SEED_SIZE)
    seed_obs[:, 2] = 1.0 / (1.0 + d * ARENA_HALF)
    for _ in range(3):
        particles = belief_update(particles, seed_obs, g)
    ring_err = ((particles[:, :, 0:2] - SEED_POS[2]).norm(dim = 2) - d).abs()
    assert float(ring_err.mean()) < 0.06
    assert float(ring_err.median()) < 0.03


def test_belief_deterministic_under_seed():
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    p1 = belief_init(3, g1)
    p2 = belief_init(3, g2)
    assert torch.allclose(p1, p2)
    xl = torch.ones(3) * 0.05
    yl = torch.zeros(3)
    dth = torch.zeros(3)
    a = belief_predict(p1, xl, yl, dth, g1)
    b = belief_predict(p2, xl, yl, dth, g2)
    assert torch.allclose(a, b)


def test_belief_read_shape_and_bounds():
    g = torch.Generator().manual_seed(2)
    p = belief_init(5, g)
    f = belief_read(p)
    assert f.shape == (5, BELIEF_FEATURES)
    assert float(f[:, 4].min()) >= 0.0 and float(f[:, 4].max()) <= 1.0
    assert float(f[:, 5].min()) >= 0.0 and float(f[:, 5].max()) <= 1.0
    assert torch.allclose(f[:, 2] ** 2 + f[:, 3] ** 2, torch.ones(5), atol = 1e-4)


def test_trainer_prop_width_includes_belief():
    from types import SimpleNamespace
    import trainer as T
    import actor_io
    from config import Config
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    worker = SimpleNamespace(hidden = {0: {}}, last_motor = {0: {}}, last_dec_step = {0: {}},
                             step_count = {0: 3}, track_neighbor = {0: {}}, track_seed = {0: {}},
                             belief = {0: {}})
    arena_ids = torch.zeros(4, dtype = torch.long)
    locals_ = torch.arange(4)
    seeds_b = torch.zeros(4, SEED_SIZE)
    seeds_b[1, 0] = 0.2
    walls_b = torch.zeros(4, WALL_SIZE)
    h, prop = actor_io.gather_split_state(worker, arena_ids, locals_, seeds_b, walls_b, tr.cfg, tr.sample_rng)
    assert prop.shape == (4, SPLIT_ODOM_SIZE)
    assert len(worker.belief[0]) == 4
    h2, prop2 = actor_io.gather_split_state(worker, arena_ids, locals_, seeds_b, walls_b, tr.cfg, tr.sample_rng)
    assert prop2.shape == (4, SPLIT_ODOM_SIZE)


def test_set_layout_switches_and_restores():
    # uses "cluster" only as a convenient, different-from-default layout to
    # exercise set_layout's switch/restore mechanism -- cluster itself is
    # DEPRECATED, DO NOT USE as an actual layout choice elsewhere
    import belief as B
    try:
        B.set_layout("cluster")
        assert float((B.SEED_POS[0] - torch.tensor([0.22, 0.0])).norm()) < 1e-6
    finally:
        B.set_layout("corners")
    assert float((B.SEED_POS[0] - torch.tensor([-0.9, 0.9])).norm()) < 1e-6


def _make_lost_cloud(true_pos, drift, spread_scale, seed=0):
    torch.manual_seed(seed)
    p = torch.zeros(1, 256, 3)
    center = (true_pos + drift) / ARENA_HALF
    p[0, :, 0] = center[0] + torch.randn(256) * spread_scale
    p[0, :, 1] = center[1] + torch.randn(256) * spread_scale
    p[0, :, 2] = torch.rand(256) * 2 * math.pi - math.pi
    return p


def test_arrived_claim_injection_rescues_a_genuinely_lost_robot():
    # this is the actual claim the whole
    # mechanism rests on -- a genuinely diffuse (spread > COLD_SPREAD),
    # lost robot receiving a CORRECT arrived-neighbor claim must end up
    # meaningfully closer to the true position, not just differently wrong
    import belief as B
    true_pos = torch.tensor([40.0, -30.0])
    p = _make_lost_cloud(true_pos, torch.tensor([25.0, -18.0]), 0.30)
    assert float(B._spread(p)[0]) > B.COLD_SPREAD, "test setup: must be genuinely cold to exercise injection"
    err_before = float((p[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())

    ac_pos = (true_pos / ARENA_HALF).unsqueeze(0)
    ac_conf = torch.tensor([0.9])
    ac_valid = torch.tensor([True])
    ac_strength = torch.tensor([1.0 / (1.0 + 3.0)])
    gen = torch.Generator().manual_seed(2)
    out = B.belief_update(p, torch.zeros(1, SEED_SIZE), gen,
                           arrived_claim_pos=ac_pos, arrived_claim_conf=ac_conf,
                           arrived_claim_valid=ac_valid, arrived_claim_strength=ac_strength)
    err_after = float((out[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())
    assert err_after < err_before * 0.5, \
        f"a correct arrived claim must meaningfully rescue a lost robot: {err_before:.1f} -> {err_after:.1f}"


def test_arrived_claim_injection_is_gated_not_unconditionally_safe():
    # the risk this mechanism carries, confirmed
    # directly and kept as a permanent regression check -- a WRONG claim
    # that reaches injection makes the receiving robot's error WORSE than
    # doing nothing, not just neutral. This is exactly why the
    # sending-side gate (only arrived, confidence-gated robots may
    # broadcast at all) is load-bearing, not optional -- this test exists
    # to make sure that property is never silently "fixed" by some future
    # change without it being a deliberate decision.
    import belief as B
    true_pos = torch.tensor([40.0, -30.0])
    p = _make_lost_cloud(true_pos, torch.tensor([25.0, -18.0]), 0.30)
    err_before = float((p[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())

    wrong_claim = (true_pos + torch.tensor([70.0, 50.0])) / ARENA_HALF
    ac_conf = torch.tensor([0.9])
    ac_valid = torch.tensor([True])
    ac_strength = torch.tensor([1.0 / (1.0 + 3.0)])
    gen = torch.Generator().manual_seed(2)
    out = B.belief_update(p, torch.zeros(1, SEED_SIZE), gen,
                           arrived_claim_pos=wrong_claim.unsqueeze(0), arrived_claim_conf=ac_conf,
                           arrived_claim_valid=ac_valid, arrived_claim_strength=ac_strength)
    err_after = float((out[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())
    assert err_after > err_before, \
        f"a wrong claim must make error worse, not neutral -- this is why sending-side gating is load-bearing: {err_before:.1f} -> {err_after:.1f}"


def test_arrived_claim_injection_never_fires_below_min_confidence():
    # a hard confidence floor gives complete protection against a
    # low-confidence (honestly uncertain) claim -- confirmed directly as
    # the one thing that fully blocks harm, unlike proportional scaling
    #
    import belief as B
    true_pos = torch.tensor([40.0, -30.0])
    p = _make_lost_cloud(true_pos, torch.tensor([25.0, -18.0]), 0.30)
    err_before = float((p[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())

    wrong_claim = (true_pos + torch.tensor([70.0, 50.0])) / ARENA_HALF
    low_conf = torch.tensor([0.05])
    ac_valid = torch.tensor([True])
    ac_strength = torch.tensor([1.0 / (1.0 + 3.0)])
    gen = torch.Generator().manual_seed(2)
    out = B.belief_update(p, torch.zeros(1, SEED_SIZE), gen,
                           arrived_claim_pos=wrong_claim.unsqueeze(0), arrived_claim_conf=low_conf,
                           arrived_claim_valid=ac_valid, arrived_claim_strength=ac_strength)
    err_after = float((out[0, :, :2].mean(dim=0) * ARENA_HALF - true_pos).norm())
    assert abs(err_after - err_before) < 1e-4, \
        f"a below-threshold confidence claim must have exactly zero effect: {err_before:.2f} -> {err_after:.2f}"


def test_seed_injection_takes_priority_over_arrived_claim_injection():
    # when both a corner-seed reading and an
    # arrived-peer claim are available the same tick, direct map data
    # must win -- checked via the expected blend (50% preserved old
    # particles + 50% new-half centered on whichever source won), since
    # the wrong peer claim's location is far enough from the seed's own
    # position that the two are easy to distinguish
    import belief as B
    true_pos = torch.tensor([40.0, -30.0])
    p = _make_lost_cloud(true_pos, torch.tensor([25.0, -18.0]), 0.30)
    old_mean = p[0, :, :2].mean(dim=0) * ARENA_HALF

    seed_obs = torch.zeros(1, SEED_SIZE)
    seed_obs[0, 0] = 1.0 / (1.0 + 3.0)
    seed0_pos = SEED_POS[0] * ARENA_HALF
    wrong_claim = (true_pos + torch.tensor([70.0, 50.0])) / ARENA_HALF
    wrong_claim_raw = true_pos + torch.tensor([70.0, 50.0])
    ac_conf = torch.tensor([0.9])
    ac_valid = torch.tensor([True])
    ac_strength = torch.tensor([1.0 / (1.0 + 3.0)])

    gen = torch.Generator().manual_seed(2)
    out = B.belief_update(p, seed_obs, gen,
                           arrived_claim_pos=wrong_claim.unsqueeze(0), arrived_claim_conf=ac_conf,
                           arrived_claim_valid=ac_valid, arrived_claim_strength=ac_strength)
    result_mean = out[0, :, :2].mean(dim=0) * ARENA_HALF

    expected_if_seed_wins = 0.5 * old_mean + 0.5 * seed0_pos
    expected_if_peer_wins = 0.5 * old_mean + 0.5 * wrong_claim_raw
    dist_to_seed_expectation = float((result_mean - expected_if_seed_wins).norm())
    dist_to_peer_expectation = float((result_mean - expected_if_peer_wins).norm())
    assert dist_to_seed_expectation < dist_to_peer_expectation, \
        "seed injection must take priority when both are available the same tick"


def test_known_start_heading_reproduces_true_heading_exactness():
    # the core claim the whole mechanism rests on
    # -- a particle starting at the known, correct heading and applying
    # zero-noise dtheta tracking must reproduce true_heading's own
    # exactness bit-for-bit (to floating-point precision), not
    # approximately, since dtheta itself is exact by construction
    import belief as B
    n_particles = B.BELIEF_PARTICLES
    p = torch.zeros(1, n_particles, 3)
    true_heading_actual = 0.0
    speed = 1.55 * 0.02
    orig_mn, orig_nf = B.MOTION_NOISE, B.NOISE_FLOOR
    B.MOTION_NOISE = 0.0
    B.NOISE_FLOOR = 0.0
    gen = torch.Generator().manual_seed(2)
    for t in range(500):
        true_dtheta = 0.02 * math.sin(t * 0.05)
        true_heading_actual += true_dtheta
        x_local = torch.tensor([speed * math.cos(true_heading_actual)])
        y_local = torch.tensor([speed * math.sin(true_heading_actual)])
        dtheta = torch.tensor([true_dtheta])
        p = B.belief_predict(p, x_local, y_local, dtheta, gen)
    B.MOTION_NOISE, B.NOISE_FLOOR = orig_mn, orig_nf

    final_headings = p[0, :, 2]
    spread = float(final_headings.max() - final_headings.min())
    max_dev = float((final_headings - true_heading_actual).abs().max())
    assert spread < 1e-6, f"all particles must stay in exact lockstep: spread={spread}"
    assert max_dev < 1e-5, f"heading must match true heading to floating-point precision: {max_dev}"


def test_known_start_heading_zero_noise_hides_a_systematic_error():
    # the honest risk this design carries -- at
    # exactly zero noise, a small, unmodeled dtheta error produces a
    # confidently WRONG belief (spread stays at zero even though the mean
    # is measurably off), meaning COLD_SPREAD's own rescue mechanism could
    # never fire. This is why HEADING_NOISE_SCALE is not exactly zero.
    import belief as B
    n_particles = B.BELIEF_PARTICLES
    p = torch.zeros(1, n_particles, 3)
    true_heading_actual = 0.0
    speed = 1.55 * 0.02
    orig_mn, orig_nf = B.MOTION_NOISE, B.NOISE_FLOOR
    B.MOTION_NOISE = 0.0
    B.NOISE_FLOOR = 0.0
    gen = torch.Generator().manual_seed(2)
    for t in range(2000):
        true_dtheta = 0.015 * math.sin(t * 0.03)
        true_heading_actual += true_dtheta
        filter_dtheta = true_dtheta * 1.02   # 2% unmodeled error
        x_local = torch.tensor([speed * math.cos(true_heading_actual)])
        y_local = torch.tensor([speed * math.sin(true_heading_actual)])
        dtheta = torch.tensor([filter_dtheta])
        p = B.belief_predict(p, x_local, y_local, dtheta, gen)
    B.MOTION_NOISE, B.NOISE_FLOOR = orig_mn, orig_nf

    final_headings = p[0, :, 2]
    spread = float(final_headings.max() - final_headings.min())
    err = float((final_headings.mean() - true_heading_actual).abs())
    assert err > 0.01, "test setup: the systematic error must actually produce a real deviation"
    assert spread < 1e-6, \
        "documents the risk directly: at exactly zero noise, spread stays zero even while wrong"


def test_heading_noise_scale_zero_keeps_all_particles_synchronized_through_resample():
    # the actual guarantee the new, zero-noise
    # design rests on -- not just belief_predict staying exact (already
    # covered above), but that resampling (which independently jitters
    # heading when heading_noise_scale is nonzero) also adds exactly zero
    # heading spread, so particles stay perfectly synchronized through a
    # realistic mix of predict and update calls, not just predict alone.
    # This is what keeps belief_read's circular-mean concentration (r) at
    # 1.0 rather than ever entering the noise-amplifying regime a nonzero
    # value created (confirmed directly against a real Unity run to be the
    # actual cause of the oscillation and confidence-collapse this change
    # fixes).
    import belief as B
    n_particles = B.BELIEF_PARTICLES
    p = torch.zeros(1, n_particles, 3)
    gen = torch.Generator().manual_seed(3)
    true_heading_actual = 0.0
    speed = 1.55 * 0.02
    for t in range(500):
        true_dtheta = 0.02 * math.sin(t * 0.05)
        true_heading_actual += true_dtheta
        x_local = torch.tensor([speed * math.cos(true_heading_actual)])
        y_local = torch.tensor([speed * math.sin(true_heading_actual)])
        dtheta = torch.tensor([true_dtheta])
        p = B.belief_predict(p, x_local, y_local, dtheta, gen, heading_noise_scale = B.HEADING_NOISE_SCALE)
        if t % 7 == 0:
            # force resampling every few ticks by giving every particle an
            # identical, arbitrary seed observation -- what matters here is
            # only whether resampling itself introduces heading spread
            seed_obs = torch.zeros(1, B.SEED_SIZE)
            seed_obs[0, 0] = 1.0 / (1.0 + 3.0)
            p = B.belief_update(p, seed_obs, gen, heading_noise_scale = B.HEADING_NOISE_SCALE)

    spread = float(p[0, :, 2].max() - p[0, :, 2].min())
    assert spread < 1e-6, \
        f"particles must stay perfectly synchronized on heading through both predict and resample at zero noise: spread={spread}"


def test_heading_noise_scale_zero_makes_position_spread_reflect_only_genuine_position_noise():
    # the actual, direct claim behind this fix --
    # confirmed here rather than only reasoned about. When particles
    # DISAGREE on heading, belief_predict rotates each one's own position
    # update by its own (different) heading, so identical physical motion
    # lands particles in different resulting positions -- inflating
    # position spread for a reason unrelated to genuine position noise.
    # At heading_noise_scale=0, every particle shares the same heading, so
    # this extra, heading-driven contribution to position spread should
    # be entirely absent -- checked directly by comparing position spread
    # after identical motion with particles seeded at genuinely different
    # headings (the pre-fix scenario) against particles synchronized from
    # the start (the fix), holding position noise magnitude fixed in both.
    import belief as B
    n_particles = B.BELIEF_PARTICLES
    gen = torch.Generator().manual_seed(4)

    # scenario A: particles start with DIFFERENT headings (simulating the
    # kind of divergence small per-particle noise would cause over time)
    p_diverged = torch.zeros(1, n_particles, 3)
    p_diverged[0, :, 2] = torch.randn(n_particles, generator=gen) * 0.05   # ~3 degrees of spread
    # scenario B: particles perfectly synchronized (this fix's guarantee)
    p_synced = torch.zeros(1, n_particles, 3)

    speed = 1.55 * 0.02
    x_local = torch.full((1,), speed)
    y_local = torch.zeros(1)
    dtheta = torch.zeros(1)
    orig_mn, orig_nf = B.MOTION_NOISE, B.NOISE_FLOOR
    B.MOTION_NOISE = 0.0   # isolate the heading-divergence effect from ordinary position noise entirely
    B.NOISE_FLOOR = 0.0
    for _ in range(50):
        p_diverged = B.belief_predict(p_diverged, x_local, y_local, dtheta, gen, heading_noise_scale = B.HEADING_NOISE_SCALE)
        p_synced = B.belief_predict(p_synced, x_local, y_local, dtheta, gen, heading_noise_scale = B.HEADING_NOISE_SCALE)
    B.MOTION_NOISE, B.NOISE_FLOOR = orig_mn, orig_nf

    spread_diverged = float((p_diverged[0, :, :2].max(dim=0).values - p_diverged[0, :, :2].min(dim=0).values).norm())
    spread_synced = float((p_synced[0, :, :2].max(dim=0).values - p_synced[0, :, :2].min(dim=0).values).norm())
    assert spread_synced < 1e-6, \
        f"with zero position noise and synchronized heading, position spread must stay exactly zero: {spread_synced}"
    assert spread_diverged > spread_synced, \
        f"with zero position noise but diverged heading, position spread must still grow from rotation mismatch alone: diverged={spread_diverged}, synced={spread_synced}"


def test_belief_update_injection_preserves_tracked_heading_when_known():
    # regression test for a real, direct-pipeline
    # bug found this phase -- three separate injection sites (seed ring,
    # arrived-claim ring, wall band) were each independently
    # overwriting an accurate, tracked heading with a fresh random draw,
    # completely undoing known_start_heading's own benefit the moment any
    # of them fired. Isolated belief_predict testing alone could never have
    # caught this, since none of these sites are inside that function.
    import belief as B
    n_particles = B.BELIEF_PARTICLES
    true_pos = torch.tensor([40.0, -30.0])
    known_heading = 0.837  # arbitrary, nonzero, to make an accidental match implausible
    p = torch.zeros(1, n_particles, 3)
    p[0, :, 0] = (true_pos[0] + torch.randn(n_particles) * 30.0) / B.ARENA_HALF   # wide position spread, to trigger cold injection
    p[0, :, 1] = (true_pos[1] + torch.randn(n_particles) * 30.0) / B.ARENA_HALF
    p[0, :, 2] = known_heading   # every particle already agrees on heading
    assert float(B._spread(p)[0]) > B.COLD_SPREAD, "test setup: must be genuinely cold on position to trigger injection"

    seed_obs = torch.zeros(1, SEED_SIZE)
    seed_obs[0, 0] = 1.0 / (1.0 + 3.0)
    gen = torch.Generator().manual_seed(5)
    out = B.belief_update(p, seed_obs, gen, heading_noise_scale = B.HEADING_NOISE_SCALE)

    result_spread = float(out[0, :, 2].max() - out[0, :, 2].min())
    result_mean = float(out[0, :, 2].mean())
    assert result_spread < 0.01, \
        f"heading must stay tight through injection when already known, not get scrambled: spread={result_spread}"
    assert abs(result_mean - known_heading) < 0.01, \
        f"heading must stay at its already-correct value through injection: {result_mean} vs {known_heading}"
