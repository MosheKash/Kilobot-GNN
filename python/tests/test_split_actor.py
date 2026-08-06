import math
import torch
from types import SimpleNamespace

import actor_io
from kilobot_gnn import SplitObservationActor, split_forward_batch, SPLIT_TC_SIZE, SPLIT_SEED_OFFSET, SPLIT_ODOM_SIZE, MESSAGE_SIZE, MOTOR_SIZE, SEED_SIZE, WALL_SIZE, Z, NODE_FEATURES, SPLIT_GRU_HIDDEN
from kinematics import split_tick_motion, split_track_update, split_track_read
from policy import GaussianPolicy, squash_action
from config import Config
from buffer import RolloutBuffer
import ppo


def test_split_actor_param_count_under_budget():
    # SplitObservationActor() with no arguments
    # (as this test previously constructed it) uses kilobot_gnn.py's own
    # module-level defaults, not config.py's -- the two are kept in sync
    # (see SPLIT_GRU_HIDDEN's own comment) but build_actor(cfg) is what
    # actually gets built in practice, with use_arrived_head and
    # use_turn_anchor both genuinely enabled together going forward, not
    # the bare default. Checking that real, practical configuration here
    # too, not only the bare one, so this budget assertion cannot again
    # silently stop reflecting what's actually shipped.
    actor = SplitObservationActor()
    total = sum(p.numel() for p in actor.parameters())
    assert total < 24 * 1024

    actor_real = SplitObservationActor(use_arrived_head=True, use_turn_anchor=True)
    total_real = sum(p.numel() for p in actor_real.parameters())
    assert total_real < 24 * 1024


def test_split_actor_single_matches_batched():
    torch.manual_seed(0)
    actor = SplitObservationActor()
    actor.eval()
    tc = torch.randn(SPLIT_TC_SIZE)
    prop = torch.randn(SPLIT_ODOM_SIZE)
    h = torch.randn(actor.hidden_size)

    with torch.no_grad():
        out_msg, motor_pre, h_new = actor(tc, prop, h)
        mean_b, h_new_b = split_forward_batch(actor, tc.unsqueeze(0), prop.unsqueeze(0), h.unsqueeze(0))

    assert torch.allclose(out_msg, mean_b[0, :MESSAGE_SIZE])
    assert torch.allclose(motor_pre, mean_b[0, MESSAGE_SIZE:])
    assert torch.allclose(h_new, h_new_b[0])


def test_split_actor_cold_start_is_finite():
    torch.manual_seed(1)
    actor = SplitObservationActor()
    n = 5
    h0 = actor.initial_hidden(n)
    assert h0.shape == (n, actor.hidden_size)
    assert torch.count_nonzero(h0) == 0
    tc = torch.zeros(n, SPLIT_TC_SIZE)
    prop = torch.zeros(n, SPLIT_ODOM_SIZE)
    mean, h_new = split_forward_batch(actor, tc, prop, h0)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(h_new).all()


def test_split_actor_hidden_state_evolves():
    torch.manual_seed(0)
    policy = GaussianPolicy(SplitObservationActor(), -0.5)
    n = 4
    tc = torch.randn(n, SPLIT_TC_SIZE)
    prop = torch.randn(n, SPLIT_ODOM_SIZE)
    h0 = torch.zeros(n, policy.actor.hidden_size)
    _, _, _, h1 = policy.act_batch_split(tc, prop, h0, deterministic=True)
    _, _, _, h2 = policy.act_batch_split(tc, prop, h1, deterministic=True)
    assert not torch.allclose(h0, h1)
    assert not torch.allclose(h1, h2)


def test_split_actor_replay_consistent():
    torch.manual_seed(0)
    policy = GaussianPolicy(SplitObservationActor(), -0.5)
    n = 8
    tc = torch.randn(n, SPLIT_TC_SIZE)
    prop = torch.randn(n, SPLIT_ODOM_SIZE)
    h = torch.zeros(n, policy.actor.hidden_size)
    torch.manual_seed(1)
    u, env_action, lp, h_new = policy.act_batch_split(tc, prop, h, deterministic=False)
    assert torch.allclose(env_action, squash_action(u), atol=1e-6)
    assert env_action[:, MESSAGE_SIZE:].min() >= -1e-6
    assert env_action[:, MESSAGE_SIZE:].max() <= 1.0 + 1e-6
    lp2, _ = policy.evaluate_batch_split(tc, prop, h, u)
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_split_actor_deterministic_is_squashed_mean():
    torch.manual_seed(0)
    policy = GaussianPolicy(SplitObservationActor(), -0.5)
    policy.eval()
    n = 5
    tc = torch.randn(n, SPLIT_TC_SIZE)
    prop = torch.randn(n, SPLIT_ODOM_SIZE)
    h = torch.zeros(n, policy.actor.hidden_size)
    with torch.no_grad():
        mean, env_action, lp, _ = policy.act_batch_split(tc, prop, h, deterministic=True)
    assert torch.allclose(env_action, squash_action(mean))
    assert torch.allclose(lp, torch.zeros(n))


def test_split_tick_motion_straight_line_x_equals_euclid_y_zero():
    straight = torch.tensor([[0.5, 0.5]])
    x, y, dtheta, t = split_tick_motion(straight, torch.tensor([10.0]), 0.02, 0.10, 0.05)
    assert abs(float(dtheta)) < 1e-6
    assert abs(float(y)) < 1e-6
    assert float(x) > 0.0
    assert abs(float(t) - 0.5) < 1e-6


def test_split_tick_motion_turning_gives_nonzero_dtheta_and_y():
    turn = torch.tensor([[0.9, 0.1]])
    x, y, dtheta, t = split_tick_motion(turn, torch.tensor([20.0]), 0.02, 0.10, 0.05)
    assert abs(float(dtheta)) > 1e-3
    assert abs(float(y)) > 1e-6


def test_split_tick_motion_zero_steps_gives_zero_everything():
    still = torch.tensor([[0.7, 0.3]])
    x, y, dtheta, t = split_tick_motion(still, torch.tensor([0.0]), 0.02, 0.10, 0.05)
    assert abs(float(x)) < 1e-6
    assert abs(float(y)) < 1e-6
    assert abs(float(dtheta)) < 1e-6
    assert abs(float(t)) < 1e-6


def test_split_track_update_accumulates_regardless_of_step_split():
    motors = torch.tensor([[0.3, 0.9]])
    one_shot = torch.zeros(1, 4)
    x, y, dtheta, t = split_tick_motion(motors, torch.tensor([20.0]), 0.02, 0.10, 0.05)
    one_shot = split_track_update(one_shot, x, y, dtheta, t)

    split = torch.zeros(1, 4)
    for _ in range(2):
        x, y, dtheta, t = split_tick_motion(motors, torch.tensor([10.0]), 0.02, 0.10, 0.05)
        split = split_track_update(split, x, y, dtheta, t)

    assert torch.allclose(one_shot, split, atol=1e-4)


def test_split_track_read_straight_drive_then_stop_reads_directly_behind():
    track = torch.zeros(1, 4)
    x, y, dtheta, t = split_tick_motion(torch.tensor([[0.5, 0.5]]), torch.tensor([10.0]), 0.02, 0.10, 0.05)
    track = split_track_update(track, x, y, dtheta, t)
    out = split_track_read(track, scale=1.0, time_scale=1.0)
    assert out.shape == (1, 4)
    assert float(out[0, 0]) > 0.0
    assert abs(float(out[0, 1])) < 1e-5
    assert abs(float(out[0, 2]) - (-1.0)) < 1e-5
    assert abs(float(out[0, 3]) - 0.5) < 1e-5


def test_split_track_read_quarter_turn_after_straight_drive():
    dx, dy, heading, elapsed = 5.0, 0.0, math.pi / 2, 1.0
    track = torch.tensor([[dx, dy, heading, elapsed]])
    out = split_track_read(track, scale=1.0, time_scale=1.0)
    assert abs(float(out[0, 0]) - 5.0) < 1e-5
    assert abs(float(out[0, 1]) - 1.0) < 1e-4
    assert abs(float(out[0, 2])) < 1e-4


def test_split_track_read_zero_distance_defaults_to_zero_sin_one_cos():
    track = torch.zeros(1, 4)
    out = split_track_read(track, scale=1.0, time_scale=1.0)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_split_track_read_scale_and_time_scale_apply():
    track = torch.tensor([[3.0, 4.0, 0.0, 2.0]])
    out = split_track_read(track, scale=10.0, time_scale=5.0)
    assert abs(float(out[0, 0]) - 50.0) < 1e-4
    assert abs(float(out[0, 3]) - 10.0) < 1e-4


def _mk_trainer(cfg):
    import trainer as T
    if not hasattr(cfg, "device"):
        cfg.device = "cpu"
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    return tr


def test_sample_split_event_neighbor_only_populates_actor_half():
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 3, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = torch.arange(1, MESSAGE_SIZE + 1).float()
    rows[0, 0, MESSAGE_SIZE] = 7.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.9
    valid = torch.zeros(n, 3, dtype=torch.bool)
    valid[0, 0] = True

    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
    assert torch.allclose(tc[0, :MESSAGE_SIZE], rows[0, 0, :MESSAGE_SIZE])
    assert torch.allclose(tc[0, MESSAGE_SIZE], rows[0, 0, MESSAGE_SIZE + 1])
    assert torch.allclose(tc[0, SPLIT_SEED_OFFSET:], torch.zeros(SEED_SIZE + WALL_SIZE))


def test_sample_split_event_seed_only_populates_seed_half():
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    seeds[0, 2] = 0.6
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 3, MESSAGE_SIZE + 2)
    valid = torch.zeros(n, 3, dtype=torch.bool)

    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
    assert torch.allclose(tc[0, :SPLIT_SEED_OFFSET], torch.zeros(SPLIT_SEED_OFFSET))
    assert torch.allclose(tc[0, SPLIT_SEED_OFFSET:SPLIT_SEED_OFFSET + SEED_SIZE], seeds[0])
    assert torch.allclose(tc[0, SPLIT_SEED_OFFSET + SEED_SIZE:], torch.zeros(WALL_SIZE))


def test_sample_split_event_nothing_available_is_all_zero():
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer(cfg)
    n = 4
    seeds = torch.zeros(n, SEED_SIZE)
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 5, MESSAGE_SIZE + 2)
    valid = torch.zeros(n, 5, dtype=torch.bool)

    tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
    assert torch.allclose(tc, torch.zeros(n, SPLIT_TC_SIZE))


def test_sample_split_event_weighted_toward_stronger_signal():
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    seeds[0, 0] = 0.01
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 2, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 1.0
    rows[0, 0, MESSAGE_SIZE + 1] = 20.0
    rows[0, 1, :MESSAGE_SIZE] = 2.0
    rows[0, 1, MESSAGE_SIZE + 1] = 0.01
    valid = torch.ones(n, 2, dtype=torch.bool)

    counts = {"actor": 0, "seed": 0}
    for _ in range(500):
        tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
        if bool((tc[0, :MESSAGE_SIZE] == 1.0).all()):
            counts["actor"] = counts["actor"] + 1
        elif bool((tc[0, :MESSAGE_SIZE] == 2.0).all()):
            pass
        else:
            counts["seed"] = counts["seed"] + 1
    assert counts["actor"] > 450


def test_sample_split_event_no_ties_between_seed_and_neighbor():
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    seeds[0, 1] = 0.5
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 9.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.5
    valid = torch.ones(n, 1, dtype=torch.bool)

    seen_seed = False
    seen_actor = False
    for _ in range(200):
        tc, _, _, _ = actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
        is_seed = bool((tc[0, SPLIT_SEED_OFFSET:] != 0).any())
        is_actor = bool((tc[0, :MESSAGE_SIZE] != 0).any())
        assert is_seed != is_actor
        seen_seed = seen_seed or is_seed
        seen_actor = seen_actor or is_actor
    assert seen_seed and seen_actor


def test_gather_split_state_reads_worker_state():
    cfg = SimpleNamespace(seed=0, split_gru_hidden=48, prop_max_speed=0.02,
                          prop_wheelbase=0.10, dt_fixed=0.05, split_prop_scale=50.0,
                          split_prop_time_scale=2.0)
    tr = _mk_trainer(cfg)
    worker = SimpleNamespace(hidden={0: {0: torch.ones(48)}},
                             last_motor={0: {0: torch.tensor([0.5, 0.5])}},
                             last_dec_step={0: {0: 3}}, step_count={0: 7},
                             track_neighbor={0: {}}, track_seed={0: {}}, belief={0: {}})
    arena_ids = torch.tensor([0])
    locals_ = torch.tensor([0])
    walls_b = torch.zeros(arena_ids.shape[0], WALL_SIZE)
    h_prev, prop = actor_io.gather_split_state(worker, arena_ids, locals_, torch.zeros(arena_ids.shape[0], SEED_SIZE), walls_b, tr.cfg, tr.sample_rng)
    assert torch.allclose(h_prev[0], torch.ones(48))
    assert prop.shape == (1, SPLIT_ODOM_SIZE)
    assert torch.isfinite(prop).all()


def test_gather_split_state_defaults_when_no_prior_decision():
    cfg = SimpleNamespace(seed=0, split_gru_hidden=48, prop_max_speed=0.02,
                          prop_wheelbase=0.10, dt_fixed=0.05, split_prop_scale=50.0,
                          split_prop_time_scale=2.0)
    tr = _mk_trainer(cfg)
    worker = SimpleNamespace(hidden={0: {}}, last_motor={0: {}}, last_dec_step={0: {}}, step_count={0: 0},
                             track_neighbor={0: {}}, track_seed={0: {}}, belief={0: {}})
    walls_b = torch.zeros(1, WALL_SIZE)
    h_prev, prop = actor_io.gather_split_state(worker, torch.tensor([0]), torch.tensor([0]), torch.zeros(1, SEED_SIZE), walls_b, tr.cfg, tr.sample_rng)
    assert torch.allclose(h_prev, torch.zeros(1, 48))
    assert prop.shape == (1, SPLIT_ODOM_SIZE)
    assert torch.allclose(prop[:, :8], torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]), atol=1e-6)
    assert float(prop[0, 12]) < 0.2


def test_split_state_neighbor_tracker_accumulates_across_two_seed_events():
    cfg = SimpleNamespace(seed = 0, device = "cpu", split_gru_hidden = 8, prop_max_speed = 0.02,
                          prop_wheelbase = 0.10, dt_fixed = 0.05, split_prop_scale = 1.0,
                          split_prop_time_scale = 1.0)
    tr = _mk_trainer(cfg)
    worker = SimpleNamespace(hidden = {0: {}}, last_motor = {0: {}}, last_dec_step = {0: {}},
                             step_count = {0: 0}, track_neighbor = {0: {}}, track_seed = {0: {}}, belief = {0: {}})
    arena_ids = torch.tensor([0])
    locals_ = torch.tensor([0])
    walls_b = torch.zeros(arena_ids.shape[0], WALL_SIZE)

    worker.last_motor[0][0] = torch.tensor([0.6, 0.6])
    worker.last_dec_step[0][0] = 0
    worker.step_count[0] = 5
    actor_io.gather_split_state(worker, arena_ids, locals_, torch.zeros(arena_ids.shape[0], SEED_SIZE), walls_b, tr.cfg, tr.sample_rng)
    worker.track_seed[0][0] = torch.zeros(4)
    neighbor_after_1 = worker.track_neighbor[0][0].clone()

    worker.last_motor[0][0] = torch.tensor([0.6, 0.6])
    worker.last_dec_step[0][0] = 5
    worker.step_count[0] = 10
    actor_io.gather_split_state(worker, arena_ids, locals_, torch.zeros(arena_ids.shape[0], SEED_SIZE), walls_b, tr.cfg, tr.sample_rng)
    worker.track_seed[0][0] = torch.zeros(4)
    neighbor_after_2 = worker.track_neighbor[0][0].clone()
    seed_after_2 = worker.track_seed[0][0].clone()

    assert float(neighbor_after_1[0]) > 0.0
    assert float(neighbor_after_2[0]) > float(neighbor_after_1[0]) * 1.8
    assert torch.allclose(seed_after_2, torch.zeros(4), atol=1e-6)


def _split_decision(z_dim=Z, hidden=SPLIT_GRU_HIDDEN):
    return {"z": torch.randn(z_dim), "seed": torch.randn(SPLIT_TC_SIZE),
            "transmission": torch.randn(SPLIT_TC_SIZE), "prev_db": None,
            "action": torch.randn(MESSAGE_SIZE + MOTOR_SIZE), "old_log_prob": torch.tensor(0.1),
            "bc_target": None, "prev_hidden": torch.randn(hidden), "prop": torch.randn(SPLIT_ODOM_SIZE)}


def test_ppo_actor_loss_routes_to_split_evaluate():
    torch.manual_seed(0)
    policy = GaussianPolicy(SplitObservationActor())
    decisions = [_split_decision() for _ in range(6)]
    data = ppo._stack_decisions(decisions)
    advantages = torch.randn(6)
    chunk = torch.arange(6)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.clip = 0.2
    cfg.entropy_coef = 0.01
    loss, entropy, kl, clipped = ppo._actor_loss(policy, data, advantages, chunk, cfg)
    assert torch.isfinite(loss)
    assert torch.isfinite(entropy)


def test_ppo_update_runs_for_split_actor():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.minibatch = 4
    cfg.ppo_epochs = 2
    policy = GaussianPolicy(SplitObservationActor())
    from kilobot_gnn import Critic
    critic = Critic()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    before_motor = policy.actor.head_motor.weight.detach().clone()
    before_msg = policy.actor.head_msg.weight.detach().clone()

    buffer = RolloutBuffer(cfg)
    from kilobot_gnn import NODE_FEATURES
    from reward import compute_rewards
    z = torch.randn(Z)
    m = 3
    for step in range(4):
        node = torch.randn(m, NODE_FEATURES)
        edge_index = torch.randint(0, m, (2, m * 2))
        edge_attr = torch.rand(m * 2, 1)
        reward = compute_rewards(node, cfg)
        term = torch.zeros(m)
        cut = torch.zeros(m)
        if step == 3:
            cut[:] = 1.0
        traj = torch.arange(m, dtype=torch.long)
        is_decision = torch.ones(m, dtype=torch.bool)
        si = buffer.add_step(0, step, node, edge_index, edge_attr, z, traj, reward, term, cut, is_decision)
        for local in range(m):
            buffer.add_decision(si, local, z, torch.randn(SPLIT_TC_SIZE), torch.randn(SPLIT_TC_SIZE),
                                None, torch.randn(MESSAGE_SIZE + MOTOR_SIZE), torch.tensor(0.0),
                                prev_hidden=torch.randn(SPLIT_GRU_HIDDEN), prop=torch.randn(SPLIT_ODOM_SIZE))

    logs = ppo.ppo_update(policy, critic, actor_opt, critic_opt, buffer, cfg)
    assert "actor_loss" in logs and "critic_loss" in logs
    assert not torch.allclose(before_motor, policy.actor.head_motor.weight.detach())
    assert not torch.allclose(before_msg, policy.actor.head_msg.weight.detach())


def test_motor_diagnostics_work_for_split_actor():
    torch.manual_seed(0)
    policy = GaussianPolicy(SplitObservationActor())
    n = 4
    tc = torch.randn(n, SPLIT_TC_SIZE)
    prop = torch.randn(n, SPLIT_ODOM_SIZE)
    h = torch.zeros(n, policy.actor.hidden_size)
    mean, h_new = split_forward_batch(policy.actor, tc, prop, h)
    loss = mean.sum()
    policy.zero_grad()
    loss.backward()
    mgn, sgn, sat, absmean = ppo._motor_diagnostics(policy)
    assert mgn > 0.0
    assert sgn > 0.0

    motor_vec, msg_vec = ppo._motor_param_vec(policy)
    expected = policy.actor.head_motor.weight.numel() + policy.actor.head_motor.bias.numel()
    assert motor_vec.numel() == expected


def test_pack_unpack_roundtrip_split_actor_decision():
    from parallel import pack_buffer, unpack_buffer
    cfg = SimpleNamespace()
    buffer = RolloutBuffer(cfg)
    buffer.steps.append({"arena_id": 0, "env_step": 0, "node": torch.randn(1, 19),
                         "edge_index": torch.zeros(2, 0, dtype=torch.long), "edge_attr": torch.zeros(0, 1),
                         "z": torch.randn(Z), "traj_id": torch.tensor([1]), "reward": torch.randn(1),
                         "term": torch.zeros(1), "cut": torch.zeros(1), "is_decision": torch.tensor([True])})
    tc = torch.randn(SPLIT_TC_SIZE)
    buffer.add_decision(0, 0, torch.randn(Z), tc, tc, None, torch.randn(MESSAGE_SIZE + MOTOR_SIZE),
                        torch.tensor(0.2), prev_hidden=torch.randn(48), prop=torch.randn(SPLIT_ODOM_SIZE))

    packed = pack_buffer(buffer)
    out = unpack_buffer(packed, cfg)
    assert len(out.decisions) == 1
    assert torch.allclose(out.decisions[0]["transmission"], tc)
    assert torch.allclose(out.decisions[0]["seed"], tc)
    assert torch.allclose(out.decisions[0]["prev_hidden"], buffer.decisions[0]["prev_hidden"])
    assert torch.allclose(out.decisions[0]["prop"], buffer.decisions[0]["prop"])


def test_build_actor_constructs_split_observation_actor():
    import sys
    import types
    if "mlagents_envs" not in sys.modules:
        pkg = types.ModuleType("mlagents_envs")
        pkg.__kilobot_stub__ = True
        sys.modules["mlagents_envs"] = pkg
        env_mod = types.ModuleType("mlagents_envs.environment")
        env_mod.UnityEnvironment = object
        sys.modules["mlagents_envs.environment"] = env_mod
        base_env_mod = types.ModuleType("mlagents_envs.base_env")
        base_env_mod.ActionTuple = object
        sys.modules["mlagents_envs.base_env"] = base_env_mod
        sys.modules["mlagents_envs.side_channel"] = types.ModuleType("mlagents_envs.side_channel")
        sc_mod = types.ModuleType("mlagents_envs.side_channel.side_channel")
        sc_mod.SideChannel = object
        sys.modules["mlagents_envs.side_channel.side_channel"] = sc_mod
        im_mod = types.ModuleType("mlagents_envs.side_channel.incoming_message")
        im_mod.IncomingMessage = object
        sys.modules["mlagents_envs.side_channel.incoming_message"] = im_mod
        om_mod = types.ModuleType("mlagents_envs.side_channel.outgoing_message")
        om_mod.OutgoingMessage = object
        sys.modules["mlagents_envs.side_channel.outgoing_message"] = om_mod
        epc_mod = types.ModuleType("mlagents_envs.side_channel.environment_parameters_channel")
        epc_mod.EnvironmentParametersChannel = object
        sys.modules["mlagents_envs.side_channel.environment_parameters_channel"] = epc_mod
        ecc_mod = types.ModuleType("mlagents_envs.side_channel.engine_configuration_channel")
        ecc_mod.EngineConfigurationChannel = object
        sys.modules["mlagents_envs.side_channel.engine_configuration_channel"] = ecc_mod
    import launch

    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_upscale_hidden = 20
    cfg.split_gru_hidden = 24
    cfg.split_head_hidden = 20
    actor = launch.build_actor(cfg)
    assert isinstance(actor, SplitObservationActor)
    assert actor.hidden_size == 24
    assert actor.up1.out_features == 20


class _FakeDecisionSteps:
    def __init__(self, obs):
        self.obs = obs

    def __len__(self):
        return self.obs[0].shape[0]


class _FakeActWorker:
    def __init__(self, node):
        self.z = {0: torch.randn(Z)}
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


def test_act_registers_seed_only_and_neighbor_only_decisions_for_split_actor():
    import numpy as np
    import trainer as T
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "none"

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))

    K = 2
    vector = np.zeros((3, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # robot 0: local=0, seed idx 2 nonzero (seed-only)
    vector[1] = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]   # robot 1: local=1, seed idx 3 nonzero (seed-only)
    vector[2] = [0, 2, 0, 0, 0, 0, 0, 0, 0, 0]   # robot 2: local=2, no seed
    rows = np.zeros((3, K, MESSAGE_SIZE + 2), dtype = np.float32)
    rows[2, 0, :MESSAGE_SIZE] = 1.0
    rows[2, 0, MESSAGE_SIZE] = 9.0
    rows[2, 0, MESSAGE_SIZE + 1] = 0.8   # robot 2: one valid neighbor message, no seed

    decision_steps = _FakeDecisionSteps([vector, rows])
    from kilobot_gnn import NODE_FEATURES
    node = torch.zeros(3, NODE_FEATURES)
    worker = _FakeActWorker(node)

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 3
    assert 0 in worker.hidden[0] and 1 in worker.hidden[0] and 2 in worker.hidden[0]
    assert 0 in worker.last_dec_step[0] and 1 in worker.last_dec_step[0] and 2 in worker.last_dec_step[0]

    tc0 = buffer.decisions[0]["transmission"]
    tc1 = buffer.decisions[1]["transmission"]
    tc2 = buffer.decisions[2]["transmission"]
    assert torch.allclose(tc0[:SPLIT_SEED_OFFSET], torch.zeros(SPLIT_SEED_OFFSET))
    assert bool((tc0[SPLIT_SEED_OFFSET:] != 0).any())
    assert torch.allclose(tc1[:SPLIT_SEED_OFFSET], torch.zeros(SPLIT_SEED_OFFSET))
    assert bool((tc1[SPLIT_SEED_OFFSET:] != 0).any())
    assert torch.allclose(tc2[SPLIT_SEED_OFFSET:], torch.zeros(SEED_SIZE + WALL_SIZE))
    assert bool((tc2[:SPLIT_SEED_OFFSET] != 0).any())

    assert tr._roll_split_total_events == 3
    assert tr._roll_split_seed_events == 2


def test_rollout_stats_reports_split_seed_fraction():
    from metrics import rollout_stats
    payload = {"reward_sum": 0.0, "reward_count": 0, "cov_sum": 0.0, "cov_count": 0,
              "disp_sum": 0.0, "disp_count": 0, "split_seed_events": 3, "split_total_events": 12,
              "decisions": 12, "agent_steps": 12, "ep_records": []}
    stats = rollout_stats(payload)
    assert abs(stats["rollout/split_seed_fraction"] - 0.25) < 1e-9


def test_rollout_stats_omits_split_seed_fraction_for_non_split_actors():
    from metrics import rollout_stats
    payload = {"reward_sum": 0.0, "reward_count": 0, "cov_sum": 0.0, "cov_count": 0,
              "disp_sum": 0.0, "disp_count": 0, "decisions": 5, "agent_steps": 5, "ep_records": []}
    stats = rollout_stats(payload)
    assert "rollout/split_seed_fraction" not in stats


def test_aggregate_payloads_sums_split_seed_counters_across_workers():
    from metrics import aggregate_payloads
    empty_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0, "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
    p1 = {"reward_sum": 0.0, "reward_count": 0, "cov_sum": 0.0, "cov_count": 0,
         "disp_sum": 0.0, "disp_count": 0, "split_seed_events": 2, "split_total_events": 10,
         "decisions": 10, "agent_steps": 10, "ep_records": [], "comp": empty_comp}
    p2 = {"reward_sum": 0.0, "reward_count": 0, "cov_sum": 0.0, "cov_count": 0,
         "disp_sum": 0.0, "disp_count": 0, "split_seed_events": 5, "split_total_events": 10,
         "decisions": 10, "agent_steps": 10, "ep_records": [], "comp": empty_comp}
    agg = aggregate_payloads([p1, p2])
    assert agg["split_seed_events"] == 7
    assert agg["split_total_events"] == 20


def test_sample_split_event_boost_at_default_matches_unboosted():
    cfg = SimpleNamespace(seed=0, split_seed_weight_boost=1.0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    seeds[0, 0] = 0.5
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 9.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.5
    valid = torch.ones(n, 1, dtype=torch.bool)

    torch.manual_seed(0)
    count_default = sum(bool((actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)[0][0, MESSAGE_SIZE:] != 0).any())
                        for _ in range(400))

    cfg_missing = SimpleNamespace(seed=0)
    tr2 = _mk_trainer(cfg_missing)
    torch.manual_seed(0)
    count_missing = sum(bool((actor_io.sample_split_event(seeds, walls, rows, valid, tr2.cfg, tr2.sample_rng)[0][0, MESSAGE_SIZE:] != 0).any())
                        for _ in range(400))

    assert count_default == count_missing


def test_sample_split_event_boost_increases_seed_selection_rate():
    cfg = SimpleNamespace(seed=0, split_seed_weight_boost=8.0)
    tr = _mk_trainer(cfg)
    n = 1
    seeds = torch.zeros(n, SEED_SIZE)
    seeds[0, 0] = 0.5
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)
    rows[0, 0, :MESSAGE_SIZE] = 9.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.5
    valid = torch.ones(n, 1, dtype=torch.bool)

    torch.manual_seed(0)
    seed_count = sum(bool((actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)[0][0, MESSAGE_SIZE:] != 0).any())
                     for _ in range(400))
    assert seed_count > 300


def test_act_split_observation_captures_bc_target_when_scripted():
    import numpy as np
    import trainer as T
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "oracle"

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0
    tr._bc_capture = True

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))

    K = 1
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    rows = np.zeros((1, K, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    node = torch.zeros(1, NODE_FEATURES)
    node[0, 2:4] = torch.tensor([1.0, 0.0])
    node[0, 5:7] = torch.tensor([1.0, 0.0])
    worker = _FakeActWorker(node)

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 1
    assert buffer.decisions[0]["bc_target"] is not None
    assert buffer.decisions[0]["bc_target"].shape == (MOTOR_SIZE,)


def test_act_split_observation_no_bc_target_without_capture():
    import numpy as np
    import trainer as T
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "oracle"

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0
    tr._bc_capture = False

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))

    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    node = torch.zeros(1, NODE_FEATURES)
    worker = _FakeActWorker(node)

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 1
    assert buffer.decisions[0]["bc_target"] is None


def test_init_globals_sample_rng_constructed_with_cfg_device():
    original_generator = torch.Generator
    seen_devices = []

    def spy_generator(*args, **kwargs):
        seen_devices.append(kwargs.get("device"))
        return original_generator(*args, **kwargs)

    cfg = SimpleNamespace(seed = 0, device = "cpu")
    torch.Generator = spy_generator
    try:
        _ = _mk_trainer(cfg)
    finally:
        torch.Generator = original_generator
    # _init_globals now wraps cfg.device in an
    # explicit torch.device(...) before constructing the generator (a real
    # user's own GPU run mismatched despite this same cfg.device value
    # reaching torch.Generator as a bare string) -- so the captured kwarg
    # is now a torch.device object, not the original bare string this
    # test checked for. Same intent, updated form: still confirms
    # cfg.device is what reaches the generator, just checked correctly
    # against what this file now actually constructs.
    assert torch.device("cpu") in seen_devices


def test_split_obs_threads_cfg_device():
    import numpy as np
    vector = np.zeros((3, 7), dtype = np.float32)
    rows = np.zeros((3, 2, MESSAGE_SIZE + 2), dtype = np.float32)
    v, r = actor_io.split_obs([vector, rows], "meta")
    assert v.device.type == "meta"
    assert r.device.type == "meta"


def test_gather_nodes_threads_cfg_device():
    worker = SimpleNamespace(snapshot = _snapshot_stub)
    out = actor_io.gather_nodes(worker, torch.tensor([0, 0]), torch.tensor([0, 1]), 2, "meta")
    assert out.device.type == "meta"


def test_gather_databases_threads_cfg_device():
    worker = SimpleNamespace(databases = {0: {}})
    rows, valid = actor_io.gather_databases(worker, torch.tensor([0]), torch.tensor([0]), "meta")
    assert rows.device.type == "meta"
    assert valid.device.type == "meta"


def test_gather_gru_state_threads_cfg_device():
    cfg = SimpleNamespace(seed = 0, device = "meta", gru_hidden = 64, prop_max_speed = 0.02,
                          prop_wheelbase = 0.10, dt_fixed = 0.05, prop_scale = 50.0,
                          prop_time_scale = 2.0, prop_cum_scale = 5.0)
    worker = SimpleNamespace(hidden = {0: {}}, last_motor = {0: {}}, last_dec_step = {0: {}},
                             step_count = {0: 0}, odometer = {0: {}})
    h_prev, prop, raw_path = actor_io.gather_gru_state(worker, torch.tensor([0]), torch.tensor([0]), cfg)
    assert h_prev.device.type == "meta"
    assert prop.device.type == "meta"
    assert raw_path.device.type == "meta"


def test_gather_split_state_threads_cfg_device():
    cfg = SimpleNamespace(seed = 0, device = "cpu", split_gru_hidden = 48, prop_max_speed = 0.02,
                          prop_wheelbase = 0.10, dt_fixed = 0.05, split_prop_scale = 50.0,
                          split_prop_time_scale = 2.0)
    rng = torch.Generator(device = "cpu")
    rng.manual_seed(0)
    worker = SimpleNamespace(hidden = {0: {}}, last_motor = {0: {}}, last_dec_step = {0: {}}, step_count = {0: 0},
                             track_neighbor = {0: {}}, track_seed = {0: {}}, belief = {0: {}})
    h_prev, prop = actor_io.gather_split_state(worker, torch.tensor([0]), torch.tensor([0]), torch.zeros(1, SEED_SIZE),
                                               torch.zeros(1, WALL_SIZE), cfg, rng)
    assert h_prev.device.type == "cpu"
    assert prop.device.type == "cpu"


def _snapshot_stub(a):
    return {"node": torch.zeros(2, NODE_FEATURES)}


def T_module():
    import trainer as T
    return T.Trainer.__new__(T.Trainer)


def test_sample_split_event_arange_matches_rows_device():
    original_arange = torch.arange
    seen_devices = []

    def spy_arange(*args, **kwargs):
        seen_devices.append(kwargs.get("device"))
        return original_arange(*args, **kwargs)

    cfg = SimpleNamespace(seed = 0, device = "cpu")
    tr = _mk_trainer(cfg)
    n = 2
    seeds = torch.zeros(n, SEED_SIZE)
    walls = torch.zeros(n, WALL_SIZE)
    rows = torch.zeros(n, 1, MESSAGE_SIZE + 2)
    valid = torch.ones(n, 1, dtype = torch.bool)
    torch.arange = spy_arange
    try:
        actor_io.sample_split_event(seeds, walls, rows, valid, tr.cfg, tr.sample_rng)
    finally:
        torch.arange = original_arange
    assert rows.device in seen_devices


def test_act_chosen_arange_matches_rows_device():
    import numpy as np
    original_arange = torch.arange
    seen_devices = []

    def spy_arange(*args, **kwargs):
        seen_devices.append(kwargs.get("device"))
        return original_arange(*args, **kwargs)

    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.device = "cpu"
    tr = T_module()
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    rows[0, 0, :MESSAGE_SIZE] = 1.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.5
    decision_steps = _FakeDecisionSteps([vector, rows])
    worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))

    buffer = RolloutBuffer(cfg)
    torch.arange = spy_arange
    try:
        tr._act(buffer, policy, worker, decision_steps)
    finally:
        torch.arange = original_arange
    assert len(seen_devices) > 0
    assert all(d == torch.device("cpu") for d in seen_devices if d is not None)


def test_act_inactive_robot_still_accumulates_but_records_no_decision():
    import numpy as np
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "none"

    tr = T_module()
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))

    worker.track_neighbor[0][0] = torch.tensor([1.0, 2.0, 0.3, 4.0])
    worker.track_seed[0][0] = torch.tensor([5.0, 6.0, 0.1, 8.0])
    worker.last_motor[0][0] = torch.tensor([0.6, 0.6])
    worker.last_dec_step[0][0] = 0
    worker.step_count[0] = 5

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 0
    assert float(worker.track_neighbor[0][0][0]) > 1.0
    assert float(worker.track_seed[0][0][0]) > 5.0
    assert abs(float(worker.track_neighbor[0][0][2]) - 0.3) < 1e-6
    assert abs(float(worker.track_seed[0][0][2]) - 0.1) < 1e-6


def test_act_neighbor_event_resets_only_neighbor_tracker():
    import numpy as np
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "none"

    tr = T_module()
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    rows[0, 0, :MESSAGE_SIZE] = 1.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.8
    decision_steps = _FakeDecisionSteps([vector, rows])
    worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))

    worker.track_neighbor[0][0] = torch.tensor([1.0, 2.0, 0.3, 4.0])
    worker.track_seed[0][0] = torch.tensor([5.0, 6.0, 0.1, 8.0])
    worker.last_motor[0][0] = torch.tensor([0.6, 0.6])
    worker.last_dec_step[0][0] = 0
    worker.step_count[0] = 5

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 1
    assert torch.allclose(worker.track_neighbor[0][0], torch.zeros(4), atol=1e-6)
    assert float(worker.track_seed[0][0][0]) > 5.0


def test_act_seed_event_resets_only_seed_tracker():
    import numpy as np
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.split_gru_hidden = 24
    cfg.actor_priv_mode = "none"
    cfg.motor_override = "none"

    tr = T_module()
    tr.cfg = cfg
    tr._init_globals()
    tr._roll_split_seed_events = 0
    tr._roll_split_total_events = 0
    tr._roll_split_heartbeat_events = 0

    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))

    worker.track_neighbor[0][0] = torch.tensor([1.0, 2.0, 0.3, 4.0])
    worker.track_seed[0][0] = torch.tensor([5.0, 6.0, 0.1, 8.0])
    worker.last_motor[0][0] = torch.tensor([0.6, 0.6])
    worker.last_dec_step[0][0] = 0
    worker.step_count[0] = 5

    buffer = RolloutBuffer(cfg)
    tr._act(buffer, policy, worker, decision_steps)

    assert len(buffer.decisions) == 1
    assert torch.allclose(worker.track_seed[0][0], torch.zeros(4), atol=1e-6)
    assert float(worker.track_neighbor[0][0][0]) > 1.0

