"""renewal_oracle.py tests -- the amended-task (noisy-odometry) baseline.

Everything here runs against fake workers with hand-built states and
beliefs, the same pattern tests/test_fixes.py's simple_oracle tests use:
renewal_oracle_motors is called directly with SimpleNamespace cfgs and
torch.Generator rngs, no player, no trainer.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

import belief as B
import renewal_oracle as RO
from config import Config


def _mk_cfg(**over):
    # The machinery tests below exercise the seed-pass controller with the
    # private filter (its original configuration); the default tangent
    # configuration has its own tests.
    cfg = SimpleNamespace(prop_max_speed=Config.prop_max_speed,
                          prop_wheelbase=Config.prop_wheelbase,
                          dt_fixed=Config.dt_fixed,
                          tau_v=getattr(Config, "tau_v", 0.05),
                          renewal_controller="seed_pass", renewal_private_filter=True,
                          renewal_wall_exit_conf=0.97, renewal_anchor_heading_max=0.0)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _mk_worker():
    return SimpleNamespace(
        image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 0})


class _Form:
    def sample_points(self, k):
        return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))


def _call(worker, walls, cfg=None, wall_seed_xy=None):
    n = walls.shape[0]
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    if wall_seed_xy is None:
        wall_seed_xy = torch.zeros(n, B.WALL_SIZE, 2)
    cfg = cfg or _mk_cfg()
    rng = torch.Generator().manual_seed(0)
    return RO.renewal_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy,
                                    cfg, rng, [_Form()])


def test_seek_wall_exits_on_wall_event_and_records_the_side():
    worker = _mk_worker()
    walls = torch.zeros(2, B.WALL_SIZE)
    walls[0, 0] = 0.5                       # robot 0 hears the north wall
    wall_seed_xy = torch.zeros(2, B.WALL_SIZE, 2)
    wall_seed_xy[0, 0] = torch.tensor([12.0, 95.0])
    motor = _call(worker, walls, wall_seed_xy=wall_seed_xy)
    assert motor.shape == (2, 2)
    assert worker.simple_state[0][0] == "turn"
    assert worker.simple_wall_name[0][0] == "north"
    # no reception: still seeking (steering toward the believed nearest wall)
    assert worker.simple_state[0][1] == "seek_wall"


def _edge_worker(wall="north", heading=0.0, aux_over=None, particles=None):
    worker = _mk_worker()
    worker.simple_heading = {0: {0: heading}}
    worker.simple_state = {0: {0: "edge_follow"}}
    worker.simple_turn_accum = {0: {0: 10.0}}
    worker.simple_wall_name = {0: {0: wall}}
    aux = RO._fresh_aux(0)
    aux["path_since_fix"] = 0.0
    aux["fix_valid"] = False
    aux.update(aux_over or {})
    worker.renewal_aux = {0: {0: aux}}
    if particles is None:
        particles = B.belief_init(1, torch.Generator().manual_seed(1),
                                  known_start_heading=True).squeeze(0)
        particles[:, 2] = heading
    worker.belief = {0: {0: particles}}
    worker.simple_target = {0: {0: np.array([0.0, 0.0])}}
    return worker


def test_arc_motor_sign_matches_the_verified_kinematics():
    # TURN_MOTOR (left faster) is the verified clockwise turn; an arc
    # "toward the left" must turn the other way under split_tick_motion.
    from kinematics import split_tick_motion
    m = torch.tensor([RO.arc_motor(True, 0.6)], dtype=torch.float32)
    _x, _y, dth, _t = split_tick_motion(m, torch.tensor([10.0]), Config.prop_max_speed,
                                        Config.prop_wheelbase, Config.dt_fixed)
    assert float(dth[0]) > 0.0                    # counter-clockwise
    m = torch.tensor([RO.arc_motor(False, 0.6)], dtype=torch.float32)
    _x, _y, dth, _t = split_tick_motion(m, torch.tensor([10.0]), Config.prop_max_speed,
                                        Config.prop_wheelbase, Config.dt_fixed)
    assert float(dth[0]) < 0.0


def test_resolve_corner_reads_direction_off_the_next_wall():
    assert RO.resolve_corner("north", "east") == RO.SIGMA_LEFT     # clockwise
    assert RO.resolve_corner("north", "west") == RO.SIGMA_RIGHT    # counter-clockwise
    assert RO.resolve_corner("east", "south") == RO.SIGMA_LEFT
    assert RO.resolve_corner("west", "south") == RO.SIGMA_RIGHT
    assert RO.resolve_corner("north", "south") is None             # opposite wall: no corner


def test_tangent_heading_both_directions():
    assert abs(RO.tangent_heading("north", RO.SIGMA_LEFT) - 0.0) < 1e-9          # east
    assert abs(abs(RO.tangent_heading("north", RO.SIGMA_RIGHT)) - math.pi) < 1e-9  # west
    assert abs(RO.tangent_heading("east", RO.SIGMA_LEFT) + math.pi / 2) < 1e-9    # south


def test_corner_handoff_confirms_sigma_and_does_not_hand_back():
    worker = _edge_worker(wall="north")
    # robot near the NE corner: the east wall out-hears north
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("north")] = 0.2
    _call(worker, walls)
    _hear_rising(worker, "east")
    aux = worker.renewal_aux[0][0]
    assert worker.simple_wall_name[0][0] == "east"
    assert aux["sigma"] == RO.SIGMA_LEFT and aux["sigma_confirmed"]
    # north is loud again right after: it must NOT hand back (and flip sigma)
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("north")] = 0.9
    _call(worker, walls)
    assert worker.simple_wall_name[0][0] == "east"
    assert aux["sigma"] == RO.SIGMA_LEFT


def _hear_rising(worker, side):
    """Three receptions of `side`, getting louder: an approach."""
    for v in (0.3, 0.4, 0.5):
        walls = torch.zeros(1, B.WALL_SIZE)
        walls[0, B.WALL_NAMES.index(side)] = v
        _call(worker, walls)


def test_a_loud_but_receding_neighbour_cannot_trigger_a_corner():
    worker = _edge_worker(wall="north")
    for v in (0.5, 0.4, 0.35):
        walls = torch.zeros(1, B.WALL_SIZE)
        walls[0, B.WALL_NAMES.index("east")] = v
        _call(worker, walls)
    assert worker.simple_wall_name[0][0] == "north"


def test_a_stale_neighbour_cannot_trigger_a_corner():
    worker = _edge_worker(wall="south")
    aux = worker.renewal_aux[0][0]
    aux["s_ema"][B.WALL_NAMES.index("east")] = 0.3      # heard long ago
    aux["s_last"][B.WALL_NAMES.index("east")] = -500
    worker.step_count = {0: 100}
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("south")] = 0.15
    _call(worker, walls)
    assert worker.simple_wall_name[0][0] == "south"


def test_corner_handoff_counter_clockwise_sets_sigma_right():
    worker = _edge_worker(wall="north")
    _hear_rising(worker, "west")
    aux = worker.renewal_aux[0][0]
    assert worker.simple_wall_name[0][0] == "west"
    assert aux["sigma"] == RO.SIGMA_RIGHT and aux["sigma_confirmed"]


def test_approach_motor_is_trend_gated():
    L, R = RO.SIGMA_LEFT, RO.SIGMA_RIGHT
    assert RO.approach_motor(-1, L, 0.6) == ((1.0, 1.0), "far_approach")
    # not approaching: arc toward the wall side (left = CCW = right wheel faster)
    assert RO.approach_motor(0, L, 0.6) == ((0.6, 1.0), "far_arc")
    assert RO.approach_motor(1, R, 0.6) == ((1.0, 0.6), "far_arc")


def test_pivots_turn_the_way_their_names_say():
    from kinematics import split_tick_motion
    for motor, sign in ((RO.PIVOT_CCW, 1.0), (RO.PIVOT_CW, -1.0)):
        m = torch.tensor([motor], dtype=torch.float32)
        _x, _y, dth, _t = split_tick_motion(m, torch.tensor([10.0]), Config.prop_max_speed,
                                            Config.prop_wheelbase, Config.dt_fixed)
        assert float(dth[0]) * sign > 0.0


def test_peak_step_finds_one_closest_approach_per_seed():
    # a robot parallel to the seed line at offset 4, seeds every 8u at path 4, 12, 20
    det = {"min": None, "last_peak_path": None}
    peaks = []
    for t in range(0, 240):
        path = t * 0.1
        dx = (path - 4.0) % 8.0
        dx = min(dx, 8.0 - dx)
        d = math.hypot(4.0, dx)
        pk = RO.peak_step(det, path, d, 0.0, 0.5, 4.0)
        if pk is not None:
            peaks.append(pk)
    assert [round(p[0]) for p in peaks] == [4, 12, 20]
    assert all(abs(p[1] - 4.0) < 0.02 for p in peaks)


def test_segment_angle_from_geometry_and_missed_peaks():
    a, dk = RO.segment_angle((0.0, 4.0, 0.0), (8.0, 4.0 + 8.0 * math.tan(0.1), 0.0))
    assert dk == 1 and abs(a - 0.1) < 1e-9
    # a missed peak: two seeds passed, the along-wall advance is 16
    a, dk = RO.segment_angle((0.0, 4.0, 0.0), (16.3, 4.0 - 16.0 * math.tan(0.05), 0.0))
    assert dk == 2 and abs(a + 0.05) < 1e-9


def test_pivot_command_signs():
    L, R = RO.SIGMA_LEFT, RO.SIGMA_RIGHT
    # at the setpoint, drifting away (alpha > 0) with the wall on the left:
    # rotate counter-clockwise, back toward the wall
    rot = RO.pivot_command(0.1, 4.0, 4.0, 16.0, 0.35, 0.785, L)
    assert abs(rot - 0.1) < 1e-9
    # same with the wall on the right: clockwise
    assert abs(RO.pivot_command(0.1, 4.0, 4.0, 16.0, 0.35, 0.785, R) + 0.1) < 1e-9
    # parallel but too close (offset 2 < 4): aim away -> clockwise for wall-on-left
    assert RO.pivot_command(0.0, 2.0, 4.0, 16.0, 0.35, 0.785, L) < 0.0
    # clamps
    assert RO.pivot_command(-2.0, 4.0, 4.0, 16.0, 0.35, 0.785, L) == -0.785


def test_approach_angle_reads_the_closing_slope():
    # closing on the seed line at 30 degrees: d falls by sin(30) per unit path
    ctrl = [(p, 7.0 - 0.5 * p) for p in np.linspace(0.0, 4.0, 9)]
    assert abs(RO.approach_angle(ctrl) - math.radians(-30.0)) < 1e-6
    assert RO.approach_angle(ctrl[:2]) is None


def test_pressed_against_wall_signature():
    assert RO.pressed_against_wall([4.1, 2.86, 2.91], 2.7, 3.05)       # slide, slip jitter ok
    assert not RO.pressed_against_wall([2.86, 3.40], 2.7, 3.05)        # left the gap
    assert not RO.pressed_against_wall([2.5, 2.4], 2.7, 3.05)          # inner side, below the gap
    assert not RO.pressed_against_wall([2.75, 3.0], 2.7, 3.05)         # in the window but not flat
    assert not RO.pressed_against_wall([4.01, 4.02, 4.00], 2.7, 3.05)  # tracking at the setpoint
    assert not RO.pressed_against_wall([2.86], 2.7, 3.05)              # too few


def test_consistent_correction_agrees_or_refuses():
    assert abs(RO.consistent_correction([-0.4, -0.41, -0.39], 0.17) + 0.4) < 1e-9
    assert RO.consistent_correction([-0.4, 0.0, -0.4], 0.17) is None
    # across the wrap
    c = RO.consistent_correction([math.pi - 0.01, -math.pi + 0.01], 0.17)
    assert abs(abs(c) - math.pi) < 1e-6


def _track_worker_at_peak(heading, sigma_confirmed, corrs):
    """A tracking robot on north (sigma left) whose running min is a seed
    pass at path 16 / offset y16, previous peak at path 8 / offset 3.5, so
    the segment's alpha is exactly 0.1; the next sample completes the peak."""
    alpha = 0.1
    y16 = 3.5 + 8.0 * math.tan(alpha)
    aux = {"mode": "track", "path": 16.5, "track_path": 8.0, "sigma_confirmed": sigma_confirmed,
           "prev_peak": (8.0, 3.5, heading), "corrs": list(corrs),
           "det": {"min": (16.0, y16, heading), "last_peak_path": 8.0},
           "d_ema": y16, "d_prev": y16, "ctrl": [(16.0, y16)]}
    worker = _edge_worker(wall="north", heading=heading, aux_over=aux)
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("north")] = 1.0 / (1.0 + y16 + 1.0)   # d climbed 1.0 past the min
    return worker, walls


def test_heading_fix_waits_for_confirmation():
    worker, walls = _track_worker_at_peak(0.3, False, [-0.4, -0.4, -0.4])
    _call(worker, walls)
    assert abs(worker.simple_heading[0][0] - 0.3) < 1e-9
    assert not worker.renewal_aux[0][0]["fix_valid"]


def test_heading_fix_corrects_heading_and_particles_then_pivots():
    # true segment heading on north, wall on the left, drifting away by 0.1:
    # 0 - 0.1 = -0.1; dead-reckoned 0.3 -> correction -0.4, matching the three
    # stored corrections. The prior is wide (heading budget spent: 0.01 rad/u
    # over 100u), so the Kalman gain is ~1 and the fix applies in full.
    worker, walls = _track_worker_at_peak(0.3, True, [-0.4, -0.4, -0.4])
    worker.renewal_aux[0][0]["path_since_fix"] = 100.0
    _call(worker, walls, cfg=_mk_cfg(renewal_heading_drift=0.01, renewal_fix_blend=True, renewal_fix_peaks=4))
    aux = worker.renewal_aux[0][0]
    assert abs(worker.simple_heading[0][0] - (-0.1)) < 1e-3
    assert aux["fix_valid"] and aux["path_since_fix"] == 0.0
    assert torch.allclose(worker.belief[0][0][:, 2], torch.full((B.BELIEF_PARTICLES,), -0.1), atol=1e-3)
    # and the controller pivots back toward the setpoint: counter-clockwise
    assert aux["mode"] == "pivot" and aux["pivot_target"] > 0.0


def test_heading_fix_barely_moves_a_tight_prior():
    # no drift, fresh heading: the 0.03 rad prior and the fixes' floor share
    # the weight -- the blend moves the heading only PART of the way
    worker, walls = _track_worker_at_peak(0.3, True, [-0.4, -0.4, -0.4])
    _call(worker, walls, cfg=_mk_cfg(renewal_fix_blend=True, renewal_fix_peaks=4))
    h = worker.simple_heading[0][0]
    assert -0.099 > h > -0.1 - 0.4 or -0.1 < h < 0.3
    assert abs(h - (-0.1)) > 0.01


def test_blend_gain():
    g, post = RO.blend_gain(1.0, [0.1, 0.1, 0.1, 0.1], 0.02)
    assert g > 0.999 and post < 0.011
    g, post = RO.blend_gain(0.01, [0.1, 0.1, 0.1, 0.1], 0.02)
    assert abs(g - 0.5) < 1e-6 and post < 0.01


def test_no_heading_fix_from_a_slide_along_the_wall():
    # both peaks inside the wall gap: a slide, not tracking. The stored
    # corrections match this segment's own (-0.01 - 0.3), so ONLY the gap
    # gate can block the fix.
    worker, walls = _track_worker_at_peak(0.3, True, [-0.31, -0.31, -0.31])
    aux = worker.renewal_aux[0][0]
    aux["prev_peak"] = (8.0, 2.85, 0.3)
    y16 = 2.85 + 8.0 * math.tan(0.01)
    aux["det"]["min"] = (16.0, y16, 0.3)
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("north")] = 1.0 / (1.0 + y16 + 1.0)
    _call(worker, walls)
    assert abs(worker.simple_heading[0][0] - 0.3) < 1e-9
    assert not aux["fix_valid"]


def test_no_peak_recovery_pivots_away_from_the_wall():
    aux = {"mode": "track", "path": 30.0, "track_path": 5.0, "d_ema": 3.0, "d_prev": 3.0,
           "ctrl": [(29.0, 3.0)], "det": {"min": (29.0, 3.0, 0.0), "last_peak_path": None}}
    worker = _edge_worker(wall="north", aux_over=aux)
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, B.WALL_NAMES.index("north")] = 1.0 / (1.0 + 3.0)
    _call(worker, walls)
    aux = worker.renewal_aux[0][0]
    assert aux["mode"] == "pivot" and aux["pivot_target"] < 0.0     # clockwise = away, wall on left


def test_corner_pivots_a_quarter_turn_away_from_the_new_wall():
    worker = _edge_worker(wall="north")
    _hear_rising(worker, "east")
    aux = worker.renewal_aux[0][0]
    assert aux["mode"] == "pivot" and abs(aux["pivot_target"] + math.pi / 2) < 1e-9


def test_edge_follow_exit_requires_confirmed_sigma_and_anchored_heading():
    p = torch.zeros(B.BELIEF_PARTICLES, 3)
    p[:, 0] = 10.0 / B.ARENA_HALF
    p[:, 1] = 90.0 / B.ARENA_HALF               # collapsed: conf = 1.0
    worker = _edge_worker(particles=p.clone())
    _call(worker, torch.zeros(1, B.WALL_SIZE))
    assert worker.simple_state[0][0] == "edge_follow"      # sigma unconfirmed
    # heading budget spent (drift * path >> anchor_heading_max): a fix is required
    spent = _mk_cfg(renewal_heading_drift=0.01)
    worker = _edge_worker(particles=p.clone(), aux_over={"sigma_confirmed": True, "path_since_fix": 100.0})
    _call(worker, torch.zeros(1, B.WALL_SIZE), cfg=spent)
    assert worker.simple_state[0][0] == "edge_follow"      # no heading fix yet
    worker = _edge_worker(particles=p.clone(), aux_over={"sigma_confirmed": True, "fix_valid": True})
    _call(worker, torch.zeros(1, B.WALL_SIZE))
    assert worker.simple_state[0][0] == "navigate"


def test_navigate_renews_when_the_heading_budget_is_spent():
    p = torch.zeros(B.BELIEF_PARTICLES, 3)
    p[:, 0] = 10.0 / B.ARENA_HALF
    p[:, 1] = 10.0 / B.ARENA_HALF
    worker = _edge_worker(particles=p, aux_over={"path_since_fix": 200.0})
    worker.simple_state[0][0] = "navigate"
    worker.simple_target = {0: {0: np.array([60.0, 60.0])}}
    # drift 0.003 rad/u * 200u + 0.03 = 0.63 > 0.35
    _call(worker, torch.zeros(1, B.WALL_SIZE), cfg=_mk_cfg(renewal_heading_drift=0.003))
    assert worker.simple_state[0][0] == "seek_wall"


def test_navigate_renews_when_conf_decays_below_the_bar():
    worker = _mk_worker()
    worker.simple_heading = {0: {0: 0.0}}
    worker.simple_state = {0: {0: "navigate"}}
    worker.simple_turn_accum = {0: {0: 10.0}}
    worker.simple_wall_name = {0: {0: "north"}}
    worker.renewal_aux = {0: {}}
    # a fresh uniform cloud: conf ~ exp(-1/3 / 0.02) ~ 0 -- far below the bar
    worker.belief = {0: {0: B.belief_init(1, torch.Generator().manual_seed(2),
                                          known_start_heading=True).squeeze(0)}}
    worker.simple_target = {0: {0: np.array([50.0, 50.0])}}
    _call(worker, torch.zeros(1, B.WALL_SIZE))
    assert worker.simple_state[0][0] == "seek_wall"


def test_navigate_arrives_on_collapsed_belief_and_stops():
    worker = _mk_worker()
    worker.simple_heading = {0: {0: 0.0}}
    worker.simple_state = {0: {0: "navigate"}}
    worker.simple_turn_accum = {0: {0: 10.0}}
    worker.simple_wall_name = {0: {0: "north"}}
    worker.renewal_aux = {0: {}}
    p = torch.zeros(B.BELIEF_PARTICLES, 3)
    p[:, 0] = 10.0 / B.ARENA_HALF               # the filter's normalized frame
    p[:, 1] = 10.0 / B.ARENA_HALF
    p[:, 2] = 0.0
    worker.belief = {0: {0: p}}                 # collapsed: conf = 1.0
    worker.simple_target = {0: {0: np.array([10.0, 10.0])}}
    motor = _call(worker, torch.zeros(1, B.WALL_SIZE))
    assert worker.simple_state[0][0] == "arrived"
    assert float(motor[0].abs().sum()) == 0.0
    # the label convention: navigating's transition labels the tick "arrived"
    assert worker.simple_motor_state[0][0] == "arrived"


def test_heading_noise_scale_reaches_the_filter(monkeypatch):
    captured = {}
    orig = RO.belief_predict

    def cap(*a, **kw):
        captured["hns"] = kw.get("heading_noise_scale")
        return orig(*a, **kw)

    monkeypatch.setattr(RO, "belief_predict", cap)
    cfg = _mk_cfg(renewal_heading_noise=0.02)
    _call(_mk_worker(), torch.zeros(1, B.WALL_SIZE), cfg=cfg)
    assert captured["hns"] == 0.02


def test_launch_parses_slip_and_renewal_env_vars(monkeypatch):
    import launch
    monkeypatch.setenv("KILOBOT_SLIP_SIGMA_BIAS", "0.02")
    monkeypatch.setenv("KILOBOT_SLIP_SIGMA_TICK", "0.01")
    monkeypatch.setenv("KILOBOT_SLIP_SEED", "7")
    monkeypatch.setenv("KILOBOT_MOTION_NOISE", "0.05")
    monkeypatch.setenv("KILOBOT_RENEWAL_WALL_EXIT_CONF", "0.55")
    monkeypatch.setenv("KILOBOT_RENEWAL_CONF_MIN", "0.25")
    monkeypatch.setenv("KILOBOT_RENEWAL_HEADING_DRIFT", "0.004")
    cfg = Config()
    cfg.slip_sigma_bias = launch._env_float("KILOBOT_SLIP_SIGMA_BIAS", cfg.slip_sigma_bias)
    cfg.slip_sigma_tick = launch._env_float("KILOBOT_SLIP_SIGMA_TICK", cfg.slip_sigma_tick)
    cfg.slip_seed = launch._env_int("KILOBOT_SLIP_SEED", cfg.slip_seed)
    cfg.motion_noise = launch._env_float("KILOBOT_MOTION_NOISE", cfg.motion_noise)
    cfg.renewal_wall_exit_conf = launch._env_float("KILOBOT_RENEWAL_WALL_EXIT_CONF",
                                                   cfg.renewal_wall_exit_conf)
    cfg.renewal_conf_min = launch._env_float("KILOBOT_RENEWAL_CONF_MIN", cfg.renewal_conf_min)
    cfg.renewal_heading_drift = launch._env_float("KILOBOT_RENEWAL_HEADING_DRIFT",
                                                  cfg.renewal_heading_drift)
    assert abs(cfg.slip_sigma_bias - 0.02) < 1e-9
    assert abs(cfg.slip_sigma_tick - 0.01) < 1e-9
    assert cfg.slip_seed == 7
    assert abs(cfg.motion_noise - 0.05) < 1e-9
    assert abs(cfg.renewal_wall_exit_conf - 0.55) < 1e-9
    assert abs(cfg.renewal_conf_min - 0.25) < 1e-9
    assert abs(cfg.renewal_heading_drift - 0.004) < 1e-9


def test_wall_point_log_w_models_the_nearest_seed_not_the_line():
    # a robot 4u inside the north seed line, halfway between two seeds
    # (seeds at x = -100 + 8k, so x = 0 sits between -4 and 4): its true
    # range is sqrt(16 + 16)
    true = torch.tensor([0.0, 95.0 - 4.0]) / B.ARENA_HALF
    d_meas = torch.tensor([math.hypot(4.0, 4.0) / B.ARENA_HALF])
    p = torch.zeros(1, 3, 3)
    p[0, 0, :2] = true
    p[0, 1, :2] = torch.tensor([0.0, 95.0 - math.hypot(4.0, 4.0)]) / B.ARENA_HALF   # the line model's answer
    p[0, 2, :2] = torch.tensor([0.0, 95.0 - 6.0]) / B.ARENA_HALF
    lw = RO.wall_point_log_w(p, B.WALL_NAMES.index("north"), d_meas, None,
                             torch.tensor([True]), 0.01)
    assert int(lw[0].argmax()) == 0
    assert float(lw[0, 0]) > -1e-6


def test_wall_point_log_w_uses_the_broadcast_seed_position():
    seed_xy = torch.tensor([[-92.0, 95.0]])
    d = 3.0
    p = torch.zeros(1, 2, 3)
    p[0, 0, :2] = torch.tensor([-92.0, 95.0 - d]) / B.ARENA_HALF        # 3u from that seed
    p[0, 1, :2] = torch.tensor([-60.0, 95.0 - d]) / B.ARENA_HALF        # 3u from ANOTHER seed (x=-60)
    lw = RO.wall_point_log_w(p, B.WALL_NAMES.index("north"), torch.tensor([d / B.ARENA_HALF]),
                             seed_xy, torch.tensor([True]), 0.01)
    assert float(lw[0, 0]) > float(lw[0, 1]) + 10.0


def test_wheel_bias_estimator_recovers_the_biases():
    # true biases; segments of straight, clockwise and counter-clockwise
    # pivots, each giving the heading error the slip model predicts
    bl, br = 0.004, -0.003
    est = RO.WheelBiasEstimator(sigma_c=0.01, sigma_b=0.05)
    rng = np.random.RandomState(0)
    for k in range(30):
        cmd = [(1.0, 1.0), (0.9, 0.15), (0.15, 0.9)][k % 3]
        t = 2.0 + rng.rand()
        il, ir = cmd[0] * t, cmd[1] * t
        d_c = math.pi * (bl * il - br * ir) + rng.randn() * 0.002
        est.add(il, ir, d_c)
    assert abs(est.b[0] - bl) < 0.001 and abs(est.b[1] - br) < 0.001
    # and the compensation it predicts for a straight command
    assert abs(est.heading_rate(1.0, 1.0) - math.pi * (bl - br)) < 0.005


def test_the_oracle_never_reads_true_pose_outside_debug(monkeypatch):
    """Same inputs as the RL policy: Unity's true pose is diagnostic-only.
    Any non-debug read fails this test, in every state."""
    def boom(*a, **k):
        raise AssertionError("_true_pose read outside debug")
    monkeypatch.setattr(RO, "_true_pose", boom)
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, 0] = 0.3
    for state in ("seek_wall", "turn", "edge_follow", "navigate", "arrived"):
        for controller in ("seed_pass", "tangent"):
          for private_filter in (True, False):
            # calibration ON too: the wheel-bias estimator reads only the
            # robot's own commands and its segment measurements
            worker = _edge_worker(wall="north")
            worker.simple_state[0][0] = state
            worker.simple_turn_accum[0][0] = 10.0
            worker.simple_target = {0: {0: np.array([10.0, 10.0])}}
            _call(worker, walls, cfg=_mk_cfg(renewal_controller=controller,
                                             renewal_private_filter=private_filter,
                                             renewal_calib_sigma_b=0.003))
            _hear_rising(worker, "east")    # exercises the corner path too



def test_default_config_is_the_tangent_controller():
    cfg = Config()
    assert cfg.renewal_controller == "tangent"
    assert cfg.renewal_private_filter is False


def test_tangent_mode_steers_inward_before_feedback():
    # on north, wall on the left (eastbound): before any seed-pass feedback
    # the goal is the tangent rotated slightly toward the wall (+y): a left turn
    worker = _edge_worker(wall="north", heading=0.0, aux_over={"mode": "track"})
    walls = torch.zeros(1, B.WALL_SIZE)
    walls[0, 0] = 0.2
    motor = _call(worker, walls, cfg=_mk_cfg(renewal_controller="tangent", renewal_tangent_inward=0.07))
    assert float(motor[0, 0]) < float(motor[0, 1])     # right wheel faster: turning left, toward the wall


def test_peer_ring_update_fuses_a_consistent_claim_and_ignores_a_contradiction():
    g = torch.Generator().manual_seed(0)
    # a cloud spread +-10u around (0, 0); a peer at (20, 0) hears it at 20u
    p = torch.zeros(1, B.BELIEF_PARTICLES, 3)
    p[0, :, 0] = (torch.rand(B.BELIEF_PARTICLES, generator=g) - 0.5) * 0.2
    p[0, :, 1] = (torch.rand(B.BELIEF_PARTICLES, generator=g) - 0.5) * 0.2
    peer = torch.tensor([[0.2, 0.0]])
    d = torch.tensor([0.2])
    use = torch.tensor([True])
    out = RO.peer_ring_update(p, peer, torch.tensor([0.99]), d, use, g, 0.1, 0.0)
    # the surviving particles sit on the 20u ring (within the ring's sigma)
    r = torch.sqrt((out[0, :, 0] - 0.2) ** 2 + out[0, :, 1] ** 2)
    assert float((r - 0.2).abs().median()) < 0.03
    # a claim no particle agrees with (peer 80u away, heard at 5u) changes nothing
    far = torch.tensor([[0.8, 0.0]])
    out2 = RO.peer_ring_update(p, far, torch.tensor([0.99]), torch.tensor([0.05]), use, g, 0.1, 0.0)
    assert torch.equal(out2, p)


def test_choose_detour_picks_the_landmark_that_cuts_the_blind_leg():
    lms = np.array([[-45.0, 45.0], [45.0, 45.0], [-45.0, -45.0], [45.0, -45.0]])
    # from (60, 90) to (40, -40), 131u direct: (45, -45) is 11u extra and
    # leaves a 7u final leg -- the one to take; (45, 45) is on the way but
    # leaves 85u; the west landmarks are 100u+ out of the way
    est, tgt = np.array([60.0, 90.0]), np.array([40.0, -40.0])
    assert RO.choose_detour(est, tgt, lms, 40.0, 20.0) == 3
    # a short leg gains nothing worth a detour
    assert RO.choose_detour(np.array([40.0, -30.0]), tgt, lms, 40.0, 20.0) is None


def test_landmark_heading_fix_recovers_the_heading_error():
    # truly drove from (0, 0) to the landmark at (50, 0); dead reckoning,
    # 6 degrees off, thinks it went to (50 cos 6, 50 sin 6)
    th = math.radians(6.0)
    phi = RO.landmark_heading_fix(np.array([0.0, 0.0]), np.array([50 * math.cos(th), 50 * math.sin(th)]),
                                  np.array([50.0, 0.0]))
    assert abs(phi + th) < 1e-9
    assert RO.landmark_heading_fix(np.zeros(2), np.array([5.0, 0.0]), np.array([5.0, 0.0])) is None
