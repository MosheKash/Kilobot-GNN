import math
import numpy as np
import torch

from belief import belief_conf, belief_read, belief_init
from config import Config
from trainer import Trainer, conf_bonus_schedule
from buffer import RolloutBuffer
import conftest
from conftest import FakeSteps as _Steps, MAX_ROWS
from kilobot_gnn import MESSAGE_SIZE, SEED_SIZE


def _tight_particles(x, y, spread = 0.001):
    g = torch.Generator().manual_seed(0)
    p = torch.zeros(32, 3)
    p[:, 0] = x + torch.randn(32, generator = g) * spread
    p[:, 1] = y + torch.randn(32, generator = g) * spread
    p[:, 2] = torch.rand(32, generator = g) * 2.0 * math.pi - math.pi
    return p


def test_belief_conf_matches_belief_read_column():
    g = torch.Generator().manual_seed(3)
    p = belief_init(5, g)
    conf = belief_conf(p)
    read = belief_read(p)
    assert torch.allclose(conf, read[:, 4], atol = 1e-6)


def test_belief_conf_high_for_collapsed_low_for_uniform():
    g = torch.Generator().manual_seed(4)
    tight = _tight_particles(0.1, 0.2).unsqueeze(0)
    spread = belief_init(1, g)
    assert float(belief_conf(tight)[0]) > 0.95
    assert float(belief_conf(spread)[0]) < 0.05


def test_conf_bonus_schedule_anneals_linearly_and_clamps():
    assert conf_bonus_schedule(0.05, 10, 0) == 0.05
    assert abs(conf_bonus_schedule(0.05, 10, 5) - 0.025) < 1e-9
    assert conf_bonus_schedule(0.05, 10, 10) == 0.0
    assert conf_bonus_schedule(0.05, 10, 25) == 0.0
    assert conf_bonus_schedule(0.05, 0, 7) == 0.05
    assert conf_bonus_schedule(0.0, 10, 3) == 0.0


def _empty_steps():
    vec = np.zeros((0, 2 + SEED_SIZE), dtype = np.float32)
    rows = np.zeros((0, MAX_ROWS, MESSAGE_SIZE + 2), dtype = np.float32)
    return _Steps(vec, rows)


def _mk_split_trainer(bonus):
    cfg = conftest.unity_cfg()
    cfg.actor_type = "gru_split_observation"
    cfg.device = "cpu"
    cfg.num_arenas = 1
    cfg.reward_shaping = 0.0
    cfg.belief_conf_bonus = bonus
    tr, worker = conftest.unity_trainer(cfg, min_bots = 4, max_bots = 4)
    tr.setup()
    return tr, worker, cfg


def _reward_of_first_step(tr, worker):
    buf = RolloutBuffer(tr.cfg)
    tr._ep_records = []
    tr._roll_reward_sum = 0.0
    tr._roll_reward_count = 0
    tr._roll_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                     "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
    tr._roll_cov_sum = 0.0
    tr._roll_cov_count = 0
    tr._roll_disp_sum = 0.0
    tr._roll_disp_count = 0
    tr._roll_decisions = 0
    tr._roll_agent_steps = 0
    tr._roll_seed_find_bonus_sum = 0.0
    tr._roll_wall_find_penalty_sum = 0.0
    tr._roll_belief_conf_bonus_sum = 0.0
    tr._roll_shaping_sum = 0.0
    tr._roll_conf_pos_sum = 0.0
    tr._roll_conf_x_sum = 0.0
    tr._roll_conf_y_sum = 0.0
    tr._roll_localized_count = 0
    tr._roll_belief_count = 0
    tr._record_snapshots(buf, worker, _empty_steps())
    return buf.steps[0]["reward"].clone()


def _plant_on_robot_zero(worker):
    # Read the position from THIS worker, immediately before planting. Both
    # trainers here come from the session-scoped factory and so are usually the
    # same cached player, which keeps ticking in between -- reusing the first
    # trainer's coordinates for the second plants the cloud wherever robot 0
    # used to be, and the bonus under test only applies to a robot that is
    # actually localized.
    node = worker.snapshot(0)["node"]
    worker.belief[0][0] = _tight_particles(float(node[0, 0]), float(node[0, 1]))


def test_record_snapshots_adds_conf_bonus_for_localized_split_robot():
    tr, worker, cfg = _mk_split_trainer(bonus = 0.05)
    _plant_on_robot_zero(worker)
    with_bonus = _reward_of_first_step(tr, worker)

    tr2, worker2, _ = _mk_split_trainer(bonus = 0.0)
    _plant_on_robot_zero(worker2)
    without = _reward_of_first_step(tr2, worker2)

    delta = with_bonus - without
    assert abs(float(delta[0]) - 0.05) < 2e-3
    assert torch.allclose(delta[1:], torch.zeros_like(delta[1:]), atol = 1e-6)


def test_record_snapshots_conf_bonus_ignored_for_non_split_actor():
    tr, worker, cfg = _mk_split_trainer(bonus = 0.05)
    cfg.actor_type = "gru"
    worker.belief[0][0] = _tight_particles(0.0, 0.0)
    r1 = _reward_of_first_step(tr, worker)

    tr2, worker2, cfg2 = _mk_split_trainer(bonus = 0.0)
    cfg2.actor_type = "gru"
    worker2.belief[0][0] = _tight_particles(0.0, 0.0)
    r2 = _reward_of_first_step(tr2, worker2)
    assert torch.allclose(r1, r2, atol = 1e-6)


def test_launch_parses_belief_env_vars(monkeypatch):
    import launch
    # "cluster" used only as a convenient, non-default value to confirm the
    # env var actually gets parsed into cfg.seed_layout -- cluster itself is
    # DEPRECATED, DO NOT USE as an actual layout choice elsewhere
    monkeypatch.setenv("KILOBOT_SEED_LAYOUT", "cluster")
    monkeypatch.setenv("KILOBOT_BELIEF_COMMS", "1")
    monkeypatch.setenv("KILOBOT_BELIEF_CONF_BONUS", "0.07")
    monkeypatch.setenv("KILOBOT_BELIEF_CONF_BONUS_ITERS", "42")
    cfg = Config()
    cfg.seed_layout = launch._env("KILOBOT_SEED_LAYOUT", cfg.seed_layout)
    cfg.belief_comms = launch._env_bool("KILOBOT_BELIEF_COMMS", cfg.belief_comms)
    cfg.belief_conf_bonus = launch._env_float("KILOBOT_BELIEF_CONF_BONUS", cfg.belief_conf_bonus)
    cfg.belief_conf_bonus_iters = launch._env_int("KILOBOT_BELIEF_CONF_BONUS_ITERS", cfg.belief_conf_bonus_iters)
    assert cfg.seed_layout == "cluster"
    assert cfg.belief_comms is True
    assert abs(cfg.belief_conf_bonus - 0.07) < 1e-9
    assert cfg.belief_conf_bonus_iters == 42
