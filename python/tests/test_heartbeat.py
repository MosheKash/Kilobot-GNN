import math

import numpy as np
import torch

from config import Config
from trainer import Trainer
from policy import GaussianPolicy
import conftest
from kilobot_gnn import build_actor
from kilobot_gnn import MESSAGE_SIZE


def _cfg(actor, heartbeat):
    cfg = conftest.unity_cfg()
    cfg.actor_type = actor
    cfg.device = "cpu"
    cfg.num_arenas = 1
    cfg.rollout_steps = 8
    cfg.max_episode_steps = 10000
    cfg.heartbeat_ticks = heartbeat
    return cfg


def _isolate(worker):
    """Plant two robots in opposite corners, so neither can ever hear the other.

    Uses SwarmManager's pose command (conftest.place). It matters because
    the heartbeat is defined by the ABSENCE of events: a spawn that happens to
    drop the two robots within IR_RANGE of each other gives them neighbour
    messages, and the thing under test never fires. Before the pose command
    existed this could only be spawned-and-hoped-for, and skipped when the spawn
    did not cooperate.

    Not the literal corners: at +/-60 each robot is ~35 units from the nearest
    wall seed (those sit at +/-95) and ~42 from the nearest corner seed (+/-90),
    so it can see no landmark either. A wall or seed sighting counts as an event
    in SwarmManager.RequestEligibleDecisions exactly like a neighbour message
    does, so parking them against a wall would suppress the heartbeat just as
    effectively as putting them in range of each other. Headings point them
    away from one another; over a rollout this short they barely move anyway.
    """
    from belief import IR_RANGE, ARENA_HALF
    conftest.place(worker, 0, [(0, -60.0, -60.0, math.pi),   # facing -x
                               (1, 60.0, 60.0, 0.0)])        # facing +x
    snap = worker.snapshot(0)
    pos = snap["node"][:, 0:2].cpu().numpy() * ARENA_HALF
    sep = float(np.linalg.norm(pos[0] - pos[1]))
    assert sep > IR_RANGE, \
        "planting failed: robots are %.1f units apart, inside IR_RANGE (%.1f)" % (sep, IR_RANGE)
    return worker


def _collect_once(actor, heartbeat):
    cfg = _cfg(actor, heartbeat)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 2, max_bots = 2)
    _isolate(worker)
    torch.manual_seed(0)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    buf = tr.collect(policy, None)
    return tr, worker, buf


def test_trainer_commands_and_records_heartbeat_deciders_split():
    tr, worker, buf = _collect_once("gru_split_observation", heartbeat = 3)
    assert len(buf.decisions) > 0
    assert tr._roll_split_heartbeat_events > 0
    moved = any(float(m.abs().sum()) > 0 for a in worker.last_motor.values() for m in a.values())
    assert moved
    for d in buf.decisions:
        tc = d["transmission"]
        assert torch.allclose(tc, torch.zeros_like(tc))


def test_heartbeat_decision_resets_no_tracker():
    tr, worker, buf = _collect_once("gru_split_observation", heartbeat = 3)
    for book in worker.track_neighbor.values():
        for track in book.values():
            assert float(track[3]) > 0.0
    for book in worker.track_seed.values():
        for track in book.values():
            assert float(track[3]) > 0.0


def test_trainer_commands_heartbeat_deciders_gru():
    tr, worker, buf = _collect_once("gru", heartbeat = 3)
    assert len(buf.decisions) > 0
    moved = any(float(m.abs().sum()) > 0 for a in worker.last_motor.values() for m in a.values())
    assert moved


def test_rollout_stats_report_heartbeat_fraction():
    tr, worker, buf = _collect_once("gru_split_observation", heartbeat = 3)
    from metrics import rollout_stats
    stats = rollout_stats(tr.rollout_payload())
    assert stats["rollout/split_heartbeat_fraction"] > 0.0


def test_launch_guard_rejects_heartbeat_with_deepset(monkeypatch):
    import launch
    monkeypatch.setenv("KILOBOT_HEARTBEAT_TICKS", "8")
    cfg = Config()
    cfg.actor_type = "deepset"
    cfg.heartbeat_ticks = launch._env_int("KILOBOT_HEARTBEAT_TICKS", cfg.heartbeat_ticks)
    assert cfg.heartbeat_ticks == 8
