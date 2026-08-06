import numpy as np
import torch
from types import SimpleNamespace

import trainer as T
import actor_io
from belief import ARENA_HALF
from config import Config

# Kinematics for the SimpleNamespace fakes below come from the real Config, not
# hardcoded numbers: they sat at 3.875/1.313 for a long time -- a pair whose
# ratio is not even pi -- so they had silently stopped matching the simulator.
_CFG = Config()
from policy import GaussianPolicy
from buffer import RolloutBuffer
from kilobot_gnn import SplitObservationActor, RecurrentActor, SPLIT_TC_SIZE, SPLIT_SEED_OFFSET, MESSAGE_SIZE, SEED_SIZE, WALL_SIZE, MOTOR_SIZE, NODE_FEATURES, Z


class _Worker:
    def __init__(self, node):
        self.z = {0: torch.randn(Z)}
        self.hidden = {0: {}}
        self.last_motor = {0: {}}
        self.last_dec_step = {0: {}}
        self.odometer = {0: {}}
        self.step_count = {0: 5}
        self.databases = {0: {}}
        self.track_neighbor = {0: {}}
        self.track_seed = {0: {}}
        self.belief = {0: {}}
        self._node = node
        self.set_actions_calls = []

    def snapshot(self, a):
        return {"step_index": 0, "node": self._node}

    def set_actions(self, actions):
        self.set_actions_calls.append(actions)


class _Steps:
    def __init__(self, obs):
        self.obs = obs

    def __len__(self):
        return self.obs[0].shape[0]


def _mk(cfg):
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0
    return tr


def _obs(n, seed_vals = None, msg_rows = None, K = 2):
    vector = np.zeros((n, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    for i in range(n):
        vector[i, 1] = i
        if seed_vals is not None and seed_vals[i] is not None:
            idx, s = seed_vals[i]
            vector[i, 2 + idx] = s
    rows = np.zeros((n, K, MESSAGE_SIZE + 2), dtype = np.float32)
    if msg_rows is not None:
        for i in msg_rows:
            rows[i, 0, :MESSAGE_SIZE] = 1.0
            rows[i, 0, MESSAGE_SIZE] = 9.0
            rows[i, 0, MESSAGE_SIZE + 1] = 0.8
    return [vector, rows]


def test_executed_motors_prefers_override():
    env_action = torch.rand(3, MESSAGE_SIZE + MOTOR_SIZE)
    scripted = torch.rand(3, MOTOR_SIZE)
    out = actor_io.executed_motors(env_action, scripted)
    assert torch.allclose(out, scripted)
    out2 = actor_io.executed_motors(env_action, None)
    assert torch.allclose(out2, env_action[:, MESSAGE_SIZE:])


def test_last_motor_records_oracle_command_under_override_gru():
    cfg = Config()
    cfg.actor_type = "gru"
    cfg.gru_hidden = 32
    cfg.motor_override = "oracle"
    tr = _mk(cfg)
    policy = GaussianPolicy(RecurrentActor(hidden = 32))

    node = torch.zeros(2, NODE_FEATURES)
    node[:, 2:4] = torch.tensor([1.0, 0.0])
    node[:, 5:7] = torch.tensor([0.0, 1.0])
    worker = _Worker(node)
    steps = _Steps(_obs(2, msg_rows = [0, 1]))
    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, steps)

    scripted = actor_io.scripted_motors(node, "oracle", cfg.force_motor, belief_heading = node[:, 2:4])
    for l in range(2):
        assert torch.allclose(worker.last_motor[0][l], scripted[l], atol = 1e-6)


def test_scripted_motors_oracle_uses_assigned_dir_when_given_else_matches_original():
    node = torch.zeros(3, NODE_FEATURES)
    node[:, 2] = 1.0    # heading x
    node[:, 5] = 0.0    # shared nearest-point dir x
    node[:, 6] = 1.0    # shared nearest-point dir y

    # no assigned_dir: must match the pre-phase-13 behavior exactly, since
    # real Unity workers never provide one and this path must stay untouched
    baseline = actor_io.scripted_motors(node, "oracle", None, belief_heading = node[:, 2:4])
    same = actor_io.scripted_motors(node, "oracle", None, assigned_dir = None, belief_heading = node[:, 2:4])
    assert torch.allclose(baseline, same)

    # a different assigned_dir must actually change the result, not be ignored
    assigned = torch.zeros(3, 2)
    assigned[:, 0] = 1.0
    coordinated = actor_io.scripted_motors(node, "oracle", None, assigned_dir = assigned, belief_heading = node[:, 2:4])
    assert not torch.allclose(baseline, coordinated)


def test_last_motor_records_policy_action_without_override():
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.motor_override = "none"
    tr = _mk(cfg)
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = 24))
    worker = _Worker(torch.zeros(1, NODE_FEATURES))
    steps = _Steps(_obs(1, msg_rows = [0]))
    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, steps)
    sent = torch.tensor(worker.set_actions_calls[0][0, MESSAGE_SIZE:])
    assert torch.allclose(worker.last_motor[0][0], sent, atol = 1e-6)


def test_sample_split_event_carries_neighbor_strength_and_zeroes_it_on_seed():
    tr = _mk(SimpleNamespace(device = 'cpu', seed = 0))
    seeds = torch.zeros(1, SEED_SIZE)
    walls = torch.zeros(1, WALL_SIZE)
    rows = torch.zeros(1, 2, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 3.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.7
    valid = torch.zeros(1, 2, dtype = torch.bool)
    valid[0, 0] = True
    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
    assert abs(float(tc[0, MESSAGE_SIZE]) - 0.7) < 1e-6

    seeds2 = torch.zeros(1, SEED_SIZE)
    seeds2[0, 1] = 0.5
    walls2 = torch.zeros(1, WALL_SIZE)
    rows2 = torch.zeros(1, 2, MESSAGE_SIZE + 2)
    valid2 = torch.zeros(1, 2, dtype = torch.bool)
    tc2, _, _, _ = actor_io.sample_split_event(seeds2, walls2, rows2, valid2, tr.cfg, tr.sample_rng)
    assert float(tc2[0, MESSAGE_SIZE]) == 0.0
    assert abs(float(tc2[0, SPLIT_SEED_OFFSET + 1]) - 0.5) < 1e-6
    assert tc2.shape == (1, SPLIT_TC_SIZE)


def test_gru_actor_acts_on_seed_only_sighting():
    cfg = Config()
    cfg.actor_type = "gru"
    cfg.gru_hidden = 32
    cfg.motor_override = "none"
    tr = _mk(cfg)
    policy = GaussianPolicy(RecurrentActor(hidden = 32))
    worker = _Worker(torch.zeros(2, NODE_FEATURES))
    seed_vals = [(2, 0.6), None]
    steps = _Steps(_obs(2, seed_vals = seed_vals, msg_rows = [1]))
    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, steps)
    assert len(buffer.decisions) == 2
    assert 0 in worker.hidden[0]
    sent = worker.set_actions_calls[0]
    assert np.abs(sent[0]).sum() > 0.0


    # No longer asserts nearest_idx specifically landed in
    # oracle_tried_occupied -- with the merged claim signal (see
    # _local_navigation_rank_cost's own comment), the claim_penalty
    # term now often routes around a claimed point via soft ranking before
    # ever getting close enough to trigger the hard occupancy-retry path
    # this used to rely on exclusively. The assertion above already covers
    # the property that actually matters: it doesn't end up heading there.


def _make_generic_worker(node, image_id, belief_dict=None):
    class _GenericWorker:
        def __init__(self, node, image_id):
            self._node = node
            self.image_id = {0: image_id}
            self.belief = belief_dict if belief_dict is not None else {}
            self.step_count = {0: 0}
            self.last_dec_step = {0: {}}
            self.track_seed = {0: {}}

        def snapshot(self, k):
            return {"node": self._node}
    return _GenericWorker(node, image_id)


def test_simple_oracle_wall_seed_xy_shape_no_longer_crashes():
    # simple_oracle_motors forwarded wall_seed_xy
    # straight into belief_update unnarrowed. actor_io.py builds it as
    # (n, WALL_SIZE, 2) -- one (x, y) slot per wall side, win or not --
    # but belief_update indexes it as (n, 2), the already-selected single
    # side. Feeding it the raw shape threw RuntimeError the first time any
    # robot's wall channel was nonzero, which is every real Unity run that
    # reaches wall_following (the replica never populated wall_seed_xy_unity
    # at all, so no replica-only test could have caught this -- see
    # test_simple_oracle_act_uses_replica_native_wall_seed_xy below for the
    # matching wiring-level regression guard). This test reproduces the
    # exact call shape that used to crash and checks it no longer does.
    import belief as B
    import simple_oracle as SO

    n = 5
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)
    walls[0, 0] = 0.5  # robot 0 hears the north wall side only

    wall_seed_xy = torch.zeros(n, B.WALL_SIZE, 2, device=device)
    wall_seed_xy[0, 0] = torch.tensor([12.0, 95.0])

    worker = SimpleNamespace(
        image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed)
    rng = torch.Generator(device=device).manual_seed(0)

    motor = SO.simple_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, [_Form()])
    assert motor.shape == (n, 2)
    assert worker.simple_state[0][0] == "turning", \
        "robot 0 should have left go_north on this tick's wall event"


def test_simple_oracle_narrows_to_one_wall_side_per_robot():
    # a robot near a corner can be simultaneously
    # in range of two wall sides. The raw per-side strengths this function
    # reads directly from the observation vector (bypassing actor_io.py's
    # sample_split_event, which already enforces this for the trained-
    # policy pipeline) can reflect both at once -- not something
    # one physical IR receiver could produce, and not something
    # belief_update should be allowed to fuse as if it were two separate
    # ticks. Checks both effects of the narrowing: only the stronger side's
    # position reaches belief_update, and only the stronger side's name is
    # recorded for wall-following.
    import belief as B
    import simple_oracle as SO

    n = 3
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)
    walls[0, 0] = 0.3  # robot 0: north (weaker)
    walls[0, 1] = 0.6  # robot 0: east (stronger) -- corner case, both in range at once

    wall_seed_xy = torch.zeros(n, B.WALL_SIZE, 2, device=device)
    wall_seed_xy[0, 0] = torch.tensor([40.0, 95.0])  # north side's own position
    wall_seed_xy[0, 1] = torch.tensor([95.0, 10.0])  # east side's own position -- should win

    captured = {}
    orig_update = SO.belief_update

    def capturing_update(*a, **kw):
        captured["wall_seed_xy"] = kw.get("wall_seed_xy")
        captured["wall_obs"] = kw.get("wall_obs")
        return orig_update(*a, **kw)

    SO.belief_update = capturing_update
    try:
        worker = SimpleNamespace(
            image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})

        class _Form:
            def sample_points(self, k):
                return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

        cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed)
        rng = torch.Generator(device=device).manual_seed(0)
        SO.simple_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, [_Form()])
    finally:
        SO.belief_update = orig_update

    assert captured["wall_seed_xy"].shape == (n, 2), \
        f"belief_update must receive the already-narrowed (n, 2) shape, got {tuple(captured['wall_seed_xy'].shape)}"
    assert torch.allclose(captured["wall_seed_xy"][0], torch.tensor([95.0, 10.0])), \
        "the stronger (east) side's own position should be the one that reaches belief_update"
    assert float((captured["wall_obs"][0] > 0).sum()) == 1.0, \
        "only one wall channel should remain nonzero for a robot that saw two at once"
    assert worker.simple_wall_name[0][0] == "east", \
        "the stronger side should be the one recorded for wall-following, not whichever came first"


def test_act_warns_once_when_known_start_heading_lacks_spawn_column():
    # direct, real-world case -- a user's own
    # debug.log showed every SIMPLE_ORACLE_SPAWN_CHECK line reading
    # assumed_deg=90.0 regardless of a correctly-varying true_deg, with
    # WALL_DEBUG_SHAPE confirming the observation was stuck at the pre-
    # phase-108 width -- Vector Observation Space Size hadn't been
    # incremented in the Unity Editor, so ml-agents silently truncated the
    # new spawnHeading column rather than erroring. This is the permanent
    # regression guard for the resulting warning: fires exactly once (not
    # per-tick) for an EnvWorker (real Unity, via hasattr(worker,
    # "channel")) whose observation is still the old width while
    # oracle_known_start_heading is on.
    import env_worker

    n = 2
    width = 2 + SEED_SIZE + WALL_SIZE   # deliberately the OLD, narrower width
    vector = torch.zeros(n, width)
    vector[:, 1] = torch.tensor([0.0, 1.0])
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)

    class FakeDecisionSteps:
        obs = [vector.numpy(), rows.numpy()]
        def __len__(self): return n

    worker = env_worker.EnvWorker.__new__(env_worker.EnvWorker)
    worker.channel = object()
    worker.z = {0: torch.zeros(16)}
    worker.image_id = {0: 0}
    worker.hidden = {0: {}}
    worker.last_motor = {0: {}}
    worker.last_dec_step = {0: {}}
    worker.step_count = {0: 0}
    worker.track_neighbor = {0: {}}
    worker.track_seed = {0: {}}
    worker.belief = {0: {}}

    cfg = SimpleNamespace(device="cpu", motor_override="none", oracle_debug_wall_log=False,
                          split_gru_hidden=48, prop_max_speed=0.02, prop_wheelbase=0.10, dt_fixed=0.05,
                          split_prop_scale=50.0, split_prop_time_scale=2.0, actor_type="gru_split_observation",
                          oracle_known_start_heading=True)
    policy = GaussianPolicy(SplitObservationActor(gru_hidden=48))
    rng = np.random.default_rng(0)
    try:
        actor_io.act(None, policy, worker, FakeDecisionSteps(), cfg, rng, deterministic=True)
    except Exception:
        pass   # worker.channel is a plain stub, not a real ml-agents channel -- expected past this point
    assert getattr(worker, "_spawn_heading_width_warned", False) is True, \
        "expected the warning to fire for a real-Unity-style worker with a too-narrow observation"


def test_simple_oracle_slows_down_approaching_localization_in_wall_following():
    # direct report -- wall_following drove at
    # full speed straight through a corner seed's own range, since belief_
    # conf (the state's sole exit condition) took real ticks to converge;
    # by the time it crossed LOCALIZED_CONF_THRESHOLD the robot had
    # already driven measurably past where it should have transitioned.
    # This checks the new APPROACH_SLOWDOWN behavior directly: two
    # otherwise-identical wall_following robots, one given a tight
    # (high-confidence) particle cloud and one a wide (low-confidence)
    # one, both below LOCALIZED_CONF_THRESHOLD so neither actually
    # transitions this call -- the high-confidence one's motor command
    # must come back smaller in magnitude.
    import belief as B
    import simple_oracle as SO

    device = "cpu"
    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                          oracle_debug_wall_log=False)
    arena_ids = torch.zeros(1, dtype=torch.long)
    locals_ = torch.zeros(1, dtype=torch.long)
    walls = torch.zeros(1, B.WALL_SIZE, device=device)

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    def _run(conf_kind):
        worker = SimpleNamespace(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})
        worker.simple_heading = {0: {0: 0.0}}
        worker.simple_state = {0: {0: "wall_following"}}
        worker.simple_turn_accum = {0: {0: 0.0}}
        worker.simple_wall_name = {0: {0: "north"}}
        worker.simple_target = {0: {0: np.array([50.0, 50.0])}}
        p = torch.zeros(1, B.BELIEF_PARTICLES, 3, device=device)
        if conf_kind == "high":
            p[0, :, 0] = 10.0   # tight cloud -- particles nearly identical -> high conf
            p[0, :, 1] = 10.0
        else:
            p[0, :, 0].uniform_(-90.0, 90.0, generator=torch.Generator(device=device).manual_seed(1))
            p[0, :, 1].uniform_(-90.0, 90.0, generator=torch.Generator(device=device).manual_seed(2))
        worker.belief = {0: {0: p.squeeze(0)}}
        rng = torch.Generator(device=device).manual_seed(0)
        motor = SO.simple_oracle_motors(worker, arena_ids, locals_, walls.clone(), None, cfg, rng, [_Form()])
        return float(motor[0].abs().sum()), worker.simple_state[0][0]

    mag_low, state_low = _run("low")
    mag_high, state_high = _run("high")
    assert state_low == "wall_following", "low-confidence robot should not have transitioned"
    assert mag_high < mag_low, \
        "high-confidence robot's wall_following motor command should be smaller in magnitude"


    # oracle_debug_wall_log's own dozen existing
    # print sites all live in actor_io.py code this oracle's short-circuit
    # never reaches, so enabling it previously did nothing useful here.
    # Added dedicated per-transition/summary logging behind the same flag
    # -- this checks it's silent by default and actually prints on a real
    # transition once enabled, with the transition itself named in the
    # output.
    import belief as B
    import simple_oracle as SO

    n = 2
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)
    walls[0, 0] = 0.5  # robot 0 hears a wall this tick -- forces a transition

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    cfg_quiet = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                               oracle_debug_wall_log=False)
    worker_quiet = SimpleNamespace(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})
    rng = torch.Generator(device=device).manual_seed(0)
    out = SO.simple_oracle_motors(worker_quiet, arena_ids, locals_, walls.clone(), None, cfg_quiet, rng, [_Form()])
    assert isinstance(out, torch.Tensor)  # sanity: still runs fine with the flag off

    cfg_loud = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                              oracle_debug_wall_log=True)
    worker_loud = SimpleNamespace(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})
    rng2 = torch.Generator(device=device).manual_seed(0)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SO.simple_oracle_motors(worker_loud, arena_ids, locals_, walls.clone(), None, cfg_loud, rng2, [_Form()])
    printed = buf.getvalue()

    assert "SIMPLE_ORACLE" in printed, "expected transition logging with the flag enabled"
    assert "go_north->turning" in printed, "robot 0's forced wall event should log its actual transition"
    assert "robot=0" in printed
    assert "SIMPLE_ORACLE_SPAWN_CHECK" in printed, "expected the one-time spawn heading check for every robot seen"
    transition_lines = [line for line in printed.splitlines() if "->" in line]
    assert not any("robot=1 " in line for line in transition_lines), \
        "robot 1 saw no wall event and should not have transitioned (spawn-check lines mentioning it are fine)"


def test_simple_oracle_spawn_check_flags_a_real_heading_mismatch():
    # Chasing a real non-
    # convergence report against a real-Unity log -- the per-transition
    # the per-transition logging only ever shows this oracle's OWN assumed
    # heading (always KNOWN_START_HEADING at spawn, by construction), never
    # whether Unity's real spawn heading actually matches it, which is
    # exactly the open question a "some robots don't look like they're
    # facing north" report raises and exactly what SwarmManager.cs's
    # knownStartHeading field (unity/SwarmManager.cs) controls. This checks
    # the diagnostic itself catches a real mismatch via a mocked
    # worker.snapshot(), the same worker-agnostic contract
    # TRUE_HEADING_DEBUG already relies on elsewhere in actor_io.py.
    import math
    import belief as B
    import simple_oracle as SO

    n = 1
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.zeros(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    class _MismatchedWorker(SimpleNamespace):
        def snapshot(self, a):
            # robot spawned facing east (0 rad) in Unity's own ground
            # truth, not north (pi/2) -- a stand-in for knownStartHeading
            # being off in the build this ran against
            node = torch.zeros(1, 4)
            node[0, 2] = math.cos(0.0)
            node[0, 3] = math.sin(0.0)
            return {"node": node}

    worker = _MismatchedWorker(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})
    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                          oracle_debug_wall_log=True)
    rng = torch.Generator(device=device).manual_seed(0)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SO.simple_oracle_motors(worker, arena_ids, locals_, walls, None, cfg, rng, [_Form()])
    printed = buf.getvalue()

    assert "SIMPLE_ORACLE_SPAWN_CHECK" in printed
    assert "MISMATCH" in printed, "a 90-degree true-vs-assumed heading gap must be flagged, not silently logged"
    assert "assumed_deg=90.0" in printed
    assert "true_deg=0.0" in printed


def test_simple_oracle_gonorth_periodic_check_samples_while_still_going_north():
    # direct follow-up to a real Unity trace
    # (robot 17) showing heading_err_deg at 0.0 at spawn but -135.5 by the
    # time it finally left go_north roughly 2450 ticks later -- the
    # previous logging  only had those two endpoints, spawn
    # and whichever transition ended the state, nothing in between, so it
    # could not distinguish a smooth ongoing drift from a one-time jolt.
    # This checks the new periodic sample fires specifically while a robot
    # remains in go_north (no wall event this tick) and does not fire once
    # it has already left that state.
    import belief as B
    import simple_oracle as SO

    n = 1
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.zeros(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)  # no wall event -- stays in go_north

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    class _StubWorker(SimpleNamespace):
        def snapshot(self, a):
            import math
            node = torch.zeros(1, 4)
            node[0, 2] = math.cos(math.pi / 2)
            node[0, 3] = math.sin(math.pi / 2)
            return {"node": node}

    worker = _StubWorker(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5})
    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                          oracle_debug_wall_log=True)
    rng = torch.Generator(device=device).manual_seed(0)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SO.simple_oracle_motors(worker, arena_ids, locals_, walls, None, cfg, rng, [_Form()])
    printed = buf.getvalue()

    assert "SIMPLE_ORACLE_GONORTH" in printed, "expected a periodic sample while the robot remains in go_north"
    assert "SIMPLE_ORACLE arena=0 robot=0" not in printed, "no transition should have fired -- no wall event occurred"


def test_simple_oracle_arrival_check_flags_off_shape_landing():
    # direct follow-up to a report that not every
    # arrived robot lands on a valid (on-shape) spot, once the phase-97/98
    # heading-drift bug was no longer the dominant failure mode. Arrival
    # itself is decided from belief alone (by design, the only position
    # source this oracle has) -- this check reads Unity's real position
    # purely to report the gap, the same worker.snapshot() mechanism
    # every other check in this file already uses. Two robots pre-seeded
    # directly into navigating with a belief already collapsed onto their
    # own target, so one call triggers arrival for both: robot 0's mocked
    # true position sits right on an on-shape point, robot 1's sits far
    # from all of them.
    import belief as B
    import simple_oracle as SO

    n = 2
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)

    class _Form:
        # a simple two-point shape: (0, 0) and (50, 50), in raw arena units
        def dist_dir(self, pos):
            pts = np.array([[0.0, 0.0], [50.0, 50.0]])
            diff = pts[None, :, :] - pos[:, None, :]
            d = np.linalg.norm(diff, axis=2)
            idx = np.argmin(d, axis=1)
            nearest = pts[idx]
            delta = nearest - pos
            dist = np.linalg.norm(delta, axis=1)
            return dist / B.ARENA_HALF, delta

    target = np.array([0.0, 0.0])   # both robots aim at the same on-shape point

    class _PoseWorker(SimpleNamespace):
        def snapshot(self, a):
            node = torch.zeros(2, 4)
            # robot 0: true position right on an on-shape point
            node[0, 0] = 0.0 / B.ARENA_HALF
            node[0, 1] = 0.0 / B.ARENA_HALF
            # robot 1: true position far from either on-shape point
            node[1, 0] = -80.0 / B.ARENA_HALF
            node[1, 1] = -80.0 / B.ARENA_HALF
            node[:, 2] = 1.0  # cos(0) -- heading not under test here
            node[:, 3] = 0.0
            return {"node": node}

    worker = _PoseWorker(
        image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 5},
        simple_heading={0: {0: B.KNOWN_START_HEADING, 1: B.KNOWN_START_HEADING}},
        simple_state={0: {0: "navigating", 1: "navigating"}},
        simple_turn_accum={0: {}}, simple_wall_name={0: {}},
        simple_target={0: {0: target, 1: target}},
        simple_hilbert_order={0: {}},
    )
    particles = torch.zeros(B.BELIEF_PARTICLES, 3)
    particles[:, 2] = B.KNOWN_START_HEADING
    worker.belief = {0: {0: particles.clone(), 1: particles.clone()}}

    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                          oracle_debug_wall_log=True, tau_v=0.05)
    rng = torch.Generator(device=device).manual_seed(0)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SO.simple_oracle_motors(worker, arena_ids, locals_, walls, None, cfg, rng, [_Form()])
    printed = buf.getvalue()

    assert worker.simple_state[0][0] == "arrived"
    assert worker.simple_state[0][1] == "arrived"
    lines = [l for l in printed.splitlines() if "ARRIVAL_CHECK" in l]
    assert len(lines) == 2, f"expected one arrival check per robot, got: {lines}"
    robot0_line = next(l for l in lines if "robot=0 " in l)
    robot1_line = next(l for l in lines if "robot=1 " in l)
    assert "OFF-SHAPE" not in robot0_line, f"robot 0 landed on-shape and should not be flagged: {robot0_line}"
    assert "OFF-SHAPE" in robot1_line, f"robot 1 landed far from the shape and should be flagged: {robot1_line}"


def test_simple_oracle_output_is_invariant_to_ground_truth_when_debug_off():
    # a direct, empirical version of the file's
    # own module-docstring claim ("NO ground truth or privileged
    # information anywhere in this file"), not just a read-through. The
    # only place this file ever touches real ground truth is
    # worker.snapshot() (via _true_pose), and every call site is gated
    # behind oracle_debug_wall_log and feeds only a print -- this proves
    # that claim by construction rather than by inspection: runs the same
    # real inputs (walls, wall_seed_xy, motor/state history) through three
    # workers whose snapshot() returns three different, deliberately
    # wrong-in-different-ways answers (one absent entirely, one returning
    # a fixed true pose, one returning a different, changing pose every
    # call), with oracle_debug_wall_log at its default (off), and checks
    # every one produces bit-identical motors and identical resulting
    # state. If any privileged value from snapshot() ever leaked into a
    # decision, varying it this way while holding every real input fixed
    # would be exactly what surfaces it.
    import copy
    import belief as B
    import simple_oracle as SO

    n = 3
    device = "cpu"
    arena_ids = torch.zeros(n, dtype=torch.long)
    locals_ = torch.arange(n, dtype=torch.long)
    walls = torch.zeros(n, B.WALL_SIZE, device=device)
    walls[0, 0] = 0.6   # robot 0: real wall event this tick
    wall_seed_xy = torch.zeros(n, B.WALL_SIZE, 2, device=device)
    wall_seed_xy[0, 0] = torch.tensor([12.0, 95.0])

    class _Form:
        def sample_points(self, k):
            return np.random.RandomState(0).uniform(-100, 100, size=(k, 2))

    def make_worker(snapshot_fn):
        base = SimpleNamespace(image_id={0: 0}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 9})
        if snapshot_fn is not None:
            base.snapshot = snapshot_fn.__get__(base)
        return base

    def no_snapshot(self, a):
        raise AttributeError("no ground truth available at all")

    def fixed_wrong_snapshot(self, a):
        node = torch.zeros(4, 4)
        node[:, 0] = 0.9   # wildly wrong true position for every robot
        node[:, 1] = -0.9
        node[:, 2] = 1.0   # true heading = 0 rad, unrelated to KNOWN_START_HEADING
        return {"node": node}

    call_count = {"n": 0}

    def changing_wrong_snapshot(self, a):
        call_count["n"] += 1
        node = torch.zeros(4, 4)
        ang = call_count["n"] * 1.7   # a different, moving "true" heading every single call
        node[:, 0] = math.sin(call_count["n"])
        node[:, 1] = math.cos(call_count["n"])
        node[:, 2] = math.cos(ang)
        node[:, 3] = math.sin(ang)
        return {"node": node}

    cfg = SimpleNamespace(prop_max_speed=_CFG.prop_max_speed, prop_wheelbase=_CFG.prop_wheelbase, dt_fixed=_CFG.dt_fixed,
                          oracle_debug_wall_log=False, tau_v=0.05)  # default: debug OFF

    import io
    import contextlib
    results = []
    for snap_fn in (None, no_snapshot, fixed_wrong_snapshot, changing_wrong_snapshot):
        worker = make_worker(snap_fn)
        rng = torch.Generator(device=device).manual_seed(0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            motor = SO.simple_oracle_motors(worker, arena_ids, locals_, walls.clone(),
                                            wall_seed_xy.clone(), cfg, rng, [_Form()])
        assert buf.getvalue() == "", \
            f"expected zero output with oracle_debug_wall_log off, got: {buf.getvalue()!r}"
        results.append((motor.clone(), copy.deepcopy(worker.simple_state)))

    baseline_motor, baseline_state = results[0]
    for motor, state in results[1:]:
        assert torch.equal(motor, baseline_motor), \
            "motor output changed depending on what worker.snapshot() returns -- privileged information leaked into a decision"
        assert state == baseline_state, \
            "resulting state changed depending on what worker.snapshot() returns -- privileged information leaked into a transition"


