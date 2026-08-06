import numpy as np
import torch

import trainer as T
from config import Config
from kilobot_gnn import SplitObservationActor, MESSAGE_SIZE, SEED_SIZE, WALL_SIZE, NODE_FEATURES
from policy import GaussianPolicy
from buffer import RolloutBuffer
import conftest

from tests.test_split_actor import _FakeDecisionSteps, _FakeActWorker


def _mk_act_trainer():
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
    return tr, cfg


def test_act_sets_positive_pending_reward_for_landmark_event():
    tr, cfg = _mk_act_trainer()
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0, 2 + 2] = 1.0   # landmark seed idx 2 nonzero, no wall
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    node = torch.zeros(1, NODE_FEATURES)
    worker = _FakeActWorker(node)
    buffer = RolloutBuffer(cfg)

    tr._act(buffer, policy, worker, decision_steps)

    assert worker.pending_find_reward[0].get(0) == 1.0


def test_act_sets_negative_pending_reward_for_wall_event():
    tr, cfg = _mk_act_trainer()
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    vector[0, 2 + SEED_SIZE + 1] = 0.7   # wall channel 1 (east) nonzero, no landmark
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    decision_steps = _FakeDecisionSteps([vector, rows])
    node = torch.zeros(1, NODE_FEATURES)
    worker = _FakeActWorker(node)
    buffer = RolloutBuffer(cfg)

    tr._act(buffer, policy, worker, decision_steps)

    assert worker.pending_find_reward[0].get(0) == -1.0


def test_act_does_not_set_pending_reward_for_neighbor_only_event():
    tr, cfg = _mk_act_trainer()
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype = np.float32)
    rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype = np.float32)
    rows[0, 0, :MESSAGE_SIZE] = 1.0
    rows[0, 0, MESSAGE_SIZE] = 9.0
    rows[0, 0, MESSAGE_SIZE + 1] = 0.8
    decision_steps = _FakeDecisionSteps([vector, rows])
    node = torch.zeros(1, NODE_FEATURES)
    worker = _FakeActWorker(node)
    buffer = RolloutBuffer(cfg)

    tr._act(buffer, policy, worker, decision_steps)

    assert 0 not in worker.pending_find_reward[0]


def _mk_snapshot_trainer(seed_bonus, wall_penalty):
    cfg = conftest.unity_cfg(actor_type = "gru_split_observation", rollout = 8, arenas = 1)
    cfg.reward_shaping = 0.0
    cfg.belief_conf_bonus = 0.0
    cfg.seed_find_bonus = seed_bonus
    cfg.wall_find_penalty = wall_penalty
    torch.manual_seed(0)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 3, max_bots = 3)
    for attr in ["_roll_reward_sum", "_roll_cov_sum", "_roll_seed_find_bonus_sum", "_roll_wall_find_penalty_sum",
                 "_roll_conf_pos_sum", "_roll_conf_x_sum", "_roll_conf_y_sum", "_roll_disp_sum"]:
        setattr(tr, attr, 0.0)
    for attr in ["_roll_reward_count", "_roll_cov_count", "_roll_decisions", "_roll_agent_steps",
                 "_roll_localized_count", "_roll_belief_count", "_roll_disp_count"]:
        setattr(tr, attr, 0)
    tr._roll_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                     "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
    buf = RolloutBuffer(cfg)
    empty = _FakeDecisionSteps([np.zeros((0, 2), dtype = np.float32), np.zeros((0, 1, MESSAGE_SIZE + 2), dtype = np.float32)])
    return tr, worker, buf, empty


def test_record_snapshots_applies_bonus_penalty_and_clears_pending():
    tr, worker, buf, empty = _mk_snapshot_trainer(seed_bonus = 0.02, wall_penalty = 0.03)
    tr2, worker2, buf2, empty2 = _mk_snapshot_trainer(seed_bonus = 0.02, wall_penalty = 0.03)
    worker.pending_find_reward[0] = {0: 1.0, 1: -1.0}   # robot 0 saw a landmark, robot 1 saw a wall

    tr._record_snapshots(buf, worker, empty)
    tr2._record_snapshots(buf2, worker2, empty2)

    reward = buf.steps[-1]["reward"]
    baseline = buf2.steps[-1]["reward"]
    assert abs(float(reward[0] - baseline[0]) - 0.02) < 1e-6
    assert abs(float(reward[1] - baseline[1]) - (-0.03)) < 1e-6
    assert abs(float(reward[2] - baseline[2])) < 1e-6
    assert worker.pending_find_reward[0] == {}


def test_seed_find_reward_disabled_when_both_set_to_zero():
    tr, worker, buf, empty = _mk_snapshot_trainer(seed_bonus = 0.0, wall_penalty = 0.0)
    tr2, worker2, buf2, empty2 = _mk_snapshot_trainer(seed_bonus = 0.0, wall_penalty = 0.0)
    worker.pending_find_reward[0] = {0: 1.0}

    tr._record_snapshots(buf, worker, empty)
    tr2._record_snapshots(buf2, worker2, empty2)

    reward = buf.steps[-1]["reward"]
    baseline = buf2.steps[-1]["reward"]
    assert torch.allclose(reward, baseline)
    # left untouched when the feature is off, since there is nothing to consume
    assert worker.pending_find_reward[0] == {0: 1.0}


def test_seed_find_reward_only_applies_to_split_observation_actor():
    cfg = conftest.unity_cfg(actor_type = "gru", rollout = 8, arenas = 1)
    cfg.reward_shaping = 0.0
    cfg.belief_conf_bonus = 0.0
    cfg.seed_find_bonus = 0.02
    cfg.wall_find_penalty = 0.03

    def mk():
        torch.manual_seed(0)
        tr, worker = conftest.unity_trainer(cfg, min_bots = 3, max_bots = 3)
        for attr in ["_roll_reward_sum", "_roll_cov_sum", "_roll_seed_find_bonus_sum",
                     "_roll_wall_find_penalty_sum", "_roll_disp_sum"]:
            setattr(tr, attr, 0.0)
        for attr in ["_roll_reward_count", "_roll_cov_count", "_roll_decisions",
                     "_roll_agent_steps", "_roll_disp_count"]:
            setattr(tr, attr, 0)
        tr._roll_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                         "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
        buf = RolloutBuffer(cfg)
        empty = _FakeDecisionSteps([np.zeros((0, 2), dtype = np.float32), np.zeros((0, 1, MESSAGE_SIZE + 2), dtype = np.float32)])
        return tr, worker, buf, empty

    tr, worker, buf, empty = mk()
    tr2, worker2, buf2, empty2 = mk()
    worker.pending_find_reward[0] = {0: 1.0}

    tr._record_snapshots(buf, worker, empty)
    tr2._record_snapshots(buf2, worker2, empty2)

    reward = buf.steps[-1]["reward"]
    baseline = buf2.steps[-1]["reward"]
    assert torch.allclose(reward, baseline)
    assert worker.pending_find_reward[0] == {0: 1.0}
