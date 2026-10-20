"""Tests for the RL fine-tune reward (docs/tuning.md phase 164): per-robot
assigned-target shaping, the shared arena potential, and the stop bonus.

Pure CPU, no Unity player needed -- the reward math is faked-worker level,
following test_seed_find_reward.py's structure but without conftest's
session-scoped players.
"""

import numpy as np
import torch
from types import SimpleNamespace

import trainer as T
from config import Config
from belief import ARENA_HALF
from kilobot_gnn import NODE_FEATURES
from reward import (assigned_target_dists, accuracy_income, settled_potential,
                    separation_reward, belief_errors, arena_potential)
from buffer import RolloutBuffer


class _FakeRewardWorker:
    def __init__(self, node, targets, image_id=0):
        self.image_id = {0: image_id}
        self._node = node
        self.simple_target = {0: {l: t for l, t in enumerate(targets)}}
        self.simple_hilbert_order = {}
        self.pending_stop_bonus = {0: {}}
        self.step_count = {0: 5}
        self.ep_reward = {}
        self.traj_id = {}
        self.z = {0: torch.zeros(64)}

    def snapshot(self, a):
        return {"step_index": 0, "env_step": 0, "node": self._node,
                "edge_index": None, "edge_attr": None}


class _FakeEmptySteps:
    def __init__(self):
        self.obs = [np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 1, 11), dtype=np.float32)]

    def __len__(self):
        return 0


def _mk_trainer(**overrides):
    cfg = Config()
    cfg.num_arenas = 1
    cfg.target_shaping = overrides.get("target_shaping", 0.0)
    cfg.stop_bonus = overrides.get("stop_bonus", 0.0)
    cfg.stop_radius = overrides.get("stop_radius", 0.05)
    cfg.global_shaping = overrides.get("global_shaping", 0.0)
    cfg.reward_mode = overrides.get("reward_mode", "normal")
    cfg.target_acc_weight = overrides.get("target_acc_weight", 1.0)
    cfg.target_acc_sigma = overrides.get("target_acc_sigma", 0.05)
    cfg.target_time_cost = overrides.get("target_time_cost", 0.0)
    cfg.success_metric = overrides.get("success_metric", "coverage")
    cfg.belief_shaping = overrides.get("belief_shaping", 0.0)
    cfg.belief_potential_sigma = overrides.get("belief_potential_sigma", 0.20)
    cfg.belief_potential_weight = overrides.get("belief_potential_weight", 1.0)
    cfg.reward_shaping = 0.0
    cfg.belief_conf_bonus = 0.0
    cfg.seed_find_bonus = 0.0
    cfg.wall_find_penalty = 0.0
    cfg.motor_override = "none"
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    tr._init_globals()
    for attr in ["_roll_reward_sum", "_roll_cov_sum", "_roll_target_shaping_sum",
                 "_roll_global_shaping_sum", "_roll_seed_find_bonus_sum",
                 "_roll_wall_find_penalty_sum", "_roll_belief_conf_bonus_sum",
                 "_roll_shaping_sum", "_roll_conf_pos_sum", "_roll_conf_x_sum",
                 "_roll_conf_y_sum", "_roll_disp_sum", "_roll_acc_income_sum",
                 "_roll_time_cost_sum", "_roll_belief_shaping_sum",
                 "_roll_belief_err_sum", "_roll_overclaim_sum"]:
        setattr(tr, attr, 0.0)
    for attr in ["_roll_reward_count", "_roll_cov_count", "_roll_decisions",
                 "_roll_agent_steps", "_roll_localized_count", "_roll_belief_count",
                 "_roll_disp_count", "_roll_split_seed_events",
                 "_roll_split_total_events", "_roll_split_heartbeat_events",
                 "_roll_target_shaping_count", "_roll_global_shaping_count",
                 "_roll_stop_bonus_count", "_roll_stop_bonus_missed",
                 "_roll_belief_err_count"]:
        setattr(tr, attr, 0)
    tr._roll_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                     "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
    tr._ep_records = []
    return tr, cfg


def test_assigned_target_dists_computes_per_robot_distance():
    # targets in raw arena units; positions normalized (x ARENA_HALF)
    targets = [torch.tensor([10.0, 0.0]), torch.tensor([0.0, -50.0])]
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.05   # raw (5, 0): distance to (10, 0) is 5 raw -> 0.05
    node[1, 1] = -0.25  # raw (0, -25): distance to (0, -50) is 25 raw -> 0.25
    worker = _FakeRewardWorker(node, targets)
    cfg = Config()
    cfg._oracle_formation_pool = [object()]
    d = assigned_target_dists(worker, 0, node, cfg)
    assert abs(float(d[0]) - 0.05) < 1e-6
    assert abs(float(d[1]) - 0.25) < 1e-6


def test_assigned_target_dists_nan_without_pool():
    node = torch.zeros(2, NODE_FEATURES)
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])])
    d = assigned_target_dists(worker, 0, node, Config())
    assert bool(torch.isnan(d).all())


def test_target_shaping_positive_when_approaching_zero_when_first_step():
    tr, cfg = _mk_trainer(target_shaping=5.0)
    targets = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.20
    node[1, 0] = 0.20
    worker = _FakeRewardWorker(node, targets)
    cfg._oracle_formation_pool = [object()]

    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    # first step after reset: no prev, so no shaping at all
    r1 = buf.steps[-1]["reward"].clone()
    assert tr._roll_target_shaping_count == 0

    # second tick, robot 0 moved closer (0.20 -> 0.10), robot 1 stayed
    node[0, 0] = 0.10
    buf2 = RolloutBuffer(cfg)
    tr._record_snapshots(buf2, worker, empty)
    r2 = buf2.steps[-1]["reward"]
    delta0 = float(r2[0] - r1[0])
    delta1 = float(r2[1] - r1[1])
    # robot 0: F = 5.0 * (0.20 - 0.99 * 0.10) = 0.505 minus the base-reward
    # change from its own on/off-shape status change -- compare against the
    # same-position baseline instead: recompute exactly
    expected = 5.0 * (0.20 - cfg.gamma * 0.10)
    assert expected > 0.4
    # robot 1 did not move: its shaping is 5.0 * (0.20 - 0.99 * 0.20) = 0.01,
    # the constant-potential offset (policy-invariant), not zero
    assert abs(delta1 - 5.0 * (1.0 - cfg.gamma) * 0.20) < 1e-5
    assert delta0 > delta1
    assert tr._roll_target_shaping_count == 2


def test_target_shaping_absent_for_unresolvable_target():
    tr, cfg = _mk_trainer(target_shaping=5.0)
    node = torch.zeros(1, NODE_FEATURES)
    node[:, 4] = 0.5
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])])
    # no pool: dists all NaN -> valid mask false -> zero shaping, never NaN
    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    r1 = buf.steps[-1]["reward"].clone()
    worker2 = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])])
    worker2.prev_target_dist = worker.prev_target_dist
    tr._record_snapshots(buf, worker2, empty)
    r2 = buf.steps[-1]["reward"]
    assert not bool(torch.isnan(r2).any())
    assert torch.allclose(r1, r2)
    assert tr._roll_target_shaping_count == 0


def test_global_shaping_added_equally_to_every_robot():
    tr, cfg = _mk_trainer(global_shaping=2.0, target_shaping=0.0)
    targets = [torch.tensor([0.0, 0.0])] * 3
    node = torch.zeros(3, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.20   # kernel exp(-4)
    node[1, 0] = 0.01   # kernel exp(-0.01)
    node[2, 0] = 0.02   # kernel exp(-0.04)
    worker = _FakeRewardWorker(node, targets)
    cfg._oracle_formation_pool = [object()]

    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    r1 = buf.steps[-1]["reward"].clone()
    assert tr._roll_global_shaping_count == 3  # count even when prev is None

    # next tick: robot 2 jumps far out, others unchanged
    node[2, 0] = 0.30
    buf2 = RolloutBuffer(cfg)
    tr._record_snapshots(buf2, worker, empty)
    r2 = buf2.steps[-1]["reward"]
    k = lambda x: float(torch.exp(torch.tensor(-(x / 0.10) ** 2.0)))
    prev_pot, pot = (k(0.20) + k(0.01) + k(0.02)) / 3.0, (k(0.20) + k(0.01) + k(0.30)) / 3.0
    # SIGN CORRECTED (phase 164/165 had this backwards): pot is a
    # higher-is-better kernel, so F = k*(gamma*pot - prev_pot) -- the arena
    # getting WORSE here (robot 2 leaving) pays NEGATIVE, improvement pays
    # positive. The old (prev_pot - gamma*pot) form rewarded degradation.
    expected_g = 2.0 * (cfg.gamma * pot - prev_pot)
    assert expected_g < 0
    # robots 0 and 1 keep identical positions across the two ticks, so their
    # two-tick reward deltas differ ONLY by the shared global term: equal
    assert abs(float(r2[0] - r1[0]) - float(r2[1] - r1[1])) < 1e-6
    # and the accumulated sum matches the shared term times robot count
    assert abs(tr._roll_global_shaping_sum - expected_g * 3) < 1e-6
    assert tr._roll_global_shaping_count == 6


def test_global_shaping_not_added_on_first_step_after_reset():
    tr, cfg = _mk_trainer(global_shaping=2.0)
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    cfg._oracle_formation_pool = [object()]
    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    # no prev_global_pot yet: the global term contributes nothing on tick 1
    assert tr._roll_global_shaping_sum == 0.0


def test_stop_bonus_graded_by_true_distance():
    tr, cfg = _mk_trainer(stop_bonus=1.0, stop_radius=0.05)
    targets = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.03   # graded bonus exp(-(0.03/0.05)^2) = 0.698
    node[1, 0] = 0.40   # effectively nothing (exp(-64))
    worker = _FakeRewardWorker(node, targets)
    cfg._oracle_formation_pool = [object()]
    worker.pending_stop_bonus[0] = {0: True, 1: True}

    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    reward = buf.steps[-1]["reward"]

    worker.pending_stop_bonus[0] = {}
    tr2, _ = _mk_trainer(stop_bonus=1.0, stop_radius=0.05)
    worker2 = _FakeRewardWorker(node.clone(), targets)
    worker2.image_id = worker.image_id
    worker2.simple_target = worker.simple_target
    worker2.simple_hilbert_order = worker.simple_hilbert_order
    tr2.cfg._oracle_formation_pool = [object()]
    buf2 = RolloutBuffer(tr2.cfg)
    tr2._record_snapshots(buf2, worker2, _FakeEmptySteps())

    expected0 = float(torch.exp(torch.tensor(-(0.03 / 0.05) ** 2.0)))
    assert abs(float(reward[0] - buf2.steps[-1]["reward"][0]) - expected0) < 1e-6
    # the far latch pays below the 1e-6 floor: counted missed, not paid
    assert abs(float(reward[1] - buf2.steps[-1]["reward"][1])) < 1e-6
    assert tr._roll_stop_bonus_count == 1
    assert tr._roll_stop_bonus_missed == 1
    assert worker.pending_stop_bonus[0] == {}


def test_accuracy_income_peaks_at_pixel_and_fades():
    dt = 0.02
    d = torch.tensor([0.0, 0.05, 0.10, 0.20])
    valid = torch.ones(4, dtype=torch.bool)
    inc = accuracy_income(d, valid, weight=1.0, sigma=0.05, dt=dt)
    assert abs(float(inc[0]) - dt) < 1e-9                 # full income at the pixel
    assert abs(float(inc[1]) - dt * float(torch.exp(torch.tensor(-1.0)))) < 1e-9   # one sigma out
    assert float(inc[3]) < 1e-5                          # gone beyond ~3 sigma
    # unresolvable target: no income and never NaN
    inc_nan = accuracy_income(d, torch.zeros(4, dtype=torch.bool), 1.0, 0.05, dt)
    assert not bool(torch.isnan(inc_nan).any())
    assert float(inc_nan.abs().sum()) == 0.0


def test_time_cost_charged_only_while_unsettled():
    # the carrying cost is conditional: settled robots (true distance within
    # stop_radius) stop paying, unsettled robots keep paying -- that
    # conditionality IS the speed gradient. An unconditional constant would
    # cancel in every advantage (same total on every trajectory) and be a
    # no-op for the policy.
    tr, cfg = _mk_trainer(reward_mode="target", target_acc_weight=0.0,
                          target_time_cost=0.1, stop_radius=0.05)
    targets = [torch.tensor([0.0, 0.0])] * 3
    node = torch.zeros(3, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.01   # settled
    node[1, 0] = 0.03   # settled
    node[2, 0] = 0.30   # unsettled
    worker = _FakeRewardWorker(node, targets)
    cfg._oracle_formation_pool = [object()]
    tr._record_snapshots(RolloutBuffer(cfg), worker, _FakeEmptySteps())
    # exactly one robot (the unsettled one) pays w_time * dt this tick
    assert abs(tr._roll_time_cost_sum - 0.1 * 0.02) < 1e-9


def test_time_cost_none_without_resolvable_target():
    tr, cfg = _mk_trainer(reward_mode="target", target_acc_weight=0.0, target_time_cost=0.1)
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    # no pool: no resolvable targets, so no carrying cost anywhere
    tr._record_snapshots(RolloutBuffer(cfg), worker, _FakeEmptySteps())
    assert tr._roll_time_cost_sum == 0.0


def test_target_mode_drops_any_pixel_income():
    # same node, target mode vs normal mode: the base r_on/off/pack income is
    # replaced by separation only, so the difference between modes is exactly
    # the on-shape/off-shape term's value
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5   # far off-shape
    node[:, 7] = 0.2   # nearest robot at 0.2 > tau_sep: no crowding
    cfg_n = _mk_trainer(reward_mode="normal")[0].cfg
    cfg_t = _mk_trainer(reward_mode="target", target_acc_weight=0.0)[0].cfg
    r_n = T.compute_rewards(node, cfg_n, None)
    r_t = separation_reward(node, cfg_t)
    # off_penalty at d=0.5: -1.0 * clamp((0.5-0.05)/1.0, 0, 1) = -0.45, x dt
    assert abs(float(r_n[0] - r_t[0]) - (-0.45 * 0.02)) < 1e-9


def test_settled_success_metric_uses_settled_fraction():
    tr, cfg = _mk_trainer(success_metric="settled", stop_radius=0.05, target_shaping=0.1)
    targets = [torch.tensor([0.0, 0.0])] * 3
    node = torch.zeros(3, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.01   # settled
    node[1, 0] = 0.03   # settled
    node[2, 0] = 0.30   # not
    worker = _FakeRewardWorker(node, targets)
    cfg._oracle_formation_pool = [object()]
    tr._record_snapshots(RolloutBuffer(cfg), worker, _FakeEmptySteps())
    # coverage accumulator now holds the settled fraction, 2/3 -- not
    # coverage()'s ground-truth any-pixel 0.0 (all robots at dist 0.5)
    assert abs(tr._roll_cov_sum - 2.0 / 3.0) < 1e-6
    assert tr._roll_cov_count == 1


def test_global_potential_smooth_kernel():
    d = torch.tensor([0.20, 0.01, 0.02])
    valid = torch.ones(3, dtype=torch.bool)
    pot = settled_potential(d, valid, sigma=0.10)
    expected = float(torch.exp(torch.tensor(-4.0)) + torch.exp(torch.tensor(-0.01)) + torch.exp(torch.tensor(-0.04))) / 3.0
    assert abs(pot - expected) < 1e-6
    # a robot moving 12 -> 6 units (0.12 -> 0.06) MOVES the smooth potential,
    # where a threshold at 0.05 would not:
    d2 = torch.tensor([0.20, 0.01, 0.06])
    assert settled_potential(d2, valid, 0.10) != pot
    # no resolvable targets: zero, not NaN
    assert settled_potential(d, torch.zeros(3, dtype=torch.bool), 0.10) == 0.0


def test_stop_bonus_paid_once_even_after_release_and_re_latch():
    # with arrived_release_threshold > 0 a latched robot can release and
    # re-latch; the bonus must be paid on the FIRST latch only, or an arrived
    # logit oscillating across the hysteresis band would mine it every cycle
    tr, cfg = _mk_trainer(stop_bonus=1.0, stop_radius=0.05)
    node = torch.zeros(1, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 0] = 0.01   # at the pixel: full graded bonus
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])])
    cfg._oracle_formation_pool = [object()]
    empty = _FakeEmptySteps()

    worker.pending_stop_bonus[0] = {0: True}
    tr._record_snapshots(RolloutBuffer(cfg), worker, empty)
    assert tr._roll_stop_bonus_count == 1

    # release, re-latch: pending fires again, but the guard pays nothing
    worker.pending_stop_bonus[0] = {0: True}
    tr._record_snapshots(RolloutBuffer(cfg), worker, empty)
    assert tr._roll_stop_bonus_count == 1
    assert worker.pending_stop_bonus[0] == {}


def test_message_std_cap():
    from policy import GaussianPolicy, MESSAGE_SIZE, MOTOR_SIZE
    from kilobot_gnn import SplitObservationActor
    import policy as policy_mod
    old_cap = policy_mod.MSG_LOG_STD_MAX
    policy_mod.MSG_LOG_STD_MAX = -2.3   # message std capped at ~0.1
    try:
        p = GaussianPolicy(SplitObservationActor(gru_hidden=24))
        with torch.no_grad():
            p.log_std.fill_(2.0)   # far above every ceiling
        std = p._std()
        assert abs(float(std[:MESSAGE_SIZE].max()) - float(torch.exp(torch.tensor(-2.3)))) < 1e-6
        # motor dims unaffected by the message cap: they sit at the global
        # LOG_STD_MAX (std 1.0), not at the message ceiling
        assert abs(float(std[MESSAGE_SIZE:].min()) - 1.0) < 1e-6
    finally:
        policy_mod.MSG_LOG_STD_MAX = old_cap


def test_gate_transition_sets_pending_stop_bonus_once():

    from actor_io import _arrived_head_gate
    cfg = Config()
    cfg.motor_override = "none"
    cfg.arrived_confidence_threshold = 0.95
    policy = SimpleNamespace(actor=SimpleNamespace(_arrived_logit=torch.tensor([[20.0]]),
                                                   head_arrived=torch.nn.Linear(1, 1)))
    worker = SimpleNamespace(arrived_switched_off={0: {}}, pending_stop_bonus={0: {}})
    assert _arrived_head_gate(worker, policy, cfg, 0, 0, 0) is True
    assert worker.pending_stop_bonus[0].get(0) is True
    # second call: already off, no new pending entry churn (still True once)
    worker.pending_stop_bonus[0] = {}
    assert _arrived_head_gate(worker, policy, cfg, 0, 0, 0) is True
    assert worker.pending_stop_bonus[0] == {}


# ── belief-error reward terms (the transmission phase) ─────────────────────

def _particles_at(mean_xy, spread=0.01):
    # a (k, 3) particle cloud with the given mean position; heading column
    # is irrelevant to belief_errors
    k = 16
    p = torch.zeros(k, 3)
    p[:, 0] = mean_xy[0] + spread * torch.linspace(-1, 1, k)
    p[:, 1] = mean_xy[1] + spread * torch.linspace(1, -1, k)
    p[:, 2] = 0.0
    return p


def test_belief_errors_measured_against_true_position():
    # robot 0's filter mean sits 0.1 from truth, robot 1's sits on truth,
    # robot 2 has no belief entry yet -> NaN, never zero
    node = torch.zeros(3, NODE_FEATURES)
    node[0, 0:2] = torch.tensor([0.30, 0.00])
    node[1, 0:2] = torch.tensor([-0.20, 0.40])
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 3)
    worker.belief = {0: {0: _particles_at([0.20, 0.00]),   # err 0.1
                         1: _particles_at([-0.20, 0.40])}}  # err 0.0
    e = belief_errors(worker, 0, node)
    assert abs(float(e[0]) - 0.10) < 1e-5
    assert float(e[1]) < 1e-5
    assert bool(torch.isnan(e[2]))


def test_belief_shaping_pays_for_converging_filter():
    # correctness-grounded, not confidence-grounded: a CLOUD COLLAPSING IN
    # THE WRONG PLACE pays nothing (error unchanged or worse); only the
    # filter's estimate MOVING TOWARD TRUTH pays.
    tr, cfg = _mk_trainer(belief_shaping=2.0)
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5   # off-shape: keep the settled/coverage success gate quiet
    node[0, 0:2] = torch.tensor([0.30, 0.00])
    node[1, 0:2] = torch.tensor([-0.20, 0.40])
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    # both filters start confidently converged WRONG: means far from truth
    worker.belief = {0: {0: _particles_at([0.05, 0.05]),
                         1: _particles_at([0.60, 0.60])}}
    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    r1 = buf.steps[-1]["reward"].clone()

    # robot 0's filter moves toward truth (0.05 -> 0.15 along the line);
    # robot 1's collapses FURTHER from truth (confidently wronger)
    worker.belief[0][0] = _particles_at([0.15, 0.05])
    worker.belief[0][1] = _particles_at([0.65, 0.65])
    buf2 = RolloutBuffer(cfg)
    tr._record_snapshots(buf2, worker, empty)
    r2 = buf2.steps[-1]["reward"]
    # robot 0: F = 2 * (e_prev - gamma * e) > 0; robot 1's e grew, paying
    # negative -- wrong convergence buys nothing, correctness pays
    d0 = float(r2[0] - r1[0])
    d1 = float(r2[1] - r1[1])
    assert d0 > 0.0
    assert d1 < 0.0
    assert d0 > d1
    assert tr._roll_belief_shaping_sum != 0.0


def test_belief_shaping_absent_without_belief_entries():
    tr, cfg = _mk_trainer(belief_shaping=2.0)
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5   # off-shape: keep the settled/coverage success gate quiet
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    # no worker.belief attribute at all: all NaN, no term, never a crash
    buf = RolloutBuffer(cfg)
    tr._record_snapshots(buf, worker, _FakeEmptySteps())
    assert tr._roll_belief_shaping_sum == 0.0
    assert not bool(torch.isnan(buf.steps[-1]["reward"]).any())


def test_arena_potential_belief_band_moves_where_settled_band_is_flat():
    # THE structural fix this phase exists for: at d = 0.5 from its pixel the
    # settled kernel (sigma 0.10) is exp(-25) ~ 1e-11 -- a receiver whose
    # localization a broadcast improves from terrible to good mid-arena moves
    # the settled band by NOTHING. The belief band (sigma 0.20) moves.
    cfg = Config()
    dists = torch.tensor([0.5])
    valid = torch.ones(1, dtype=torch.bool)
    flat = settled_potential(dists, valid, 0.10)
    assert flat < 1e-9
    e_far = torch.tensor([0.60])
    e_near = torch.tensor([0.05])
    cfg.belief_potential_sigma = 0.20
    cfg.belief_potential_weight = 1.0
    pot_far = arena_potential(dists, valid, e_far, None, cfg)
    pot_near = arena_potential(dists, valid, e_near, None, cfg)
    assert pot_near - pot_far > 0.5   # a localization gain mid-approach PAYS
    # dists None (no pool): the belief band alone still defines the potential
    assert arena_potential(None, None, e_near, None, cfg) > arena_potential(None, None, e_far, None, cfg)
    # weight 0: pure phase-165 potential, bit-for-bit
    cfg.belief_potential_weight = 0.0
    assert abs(arena_potential(dists, valid, e_near, None, cfg) - flat) < 1e-9


def test_global_shaping_belief_band_credits_without_target_resolution():
    # sender credit must not depend on the formation pool: an arena where no
    # target resolves (dists all NaN) still pays the shared term when the
    # swarm's belief errors shrink -- this is exactly the regime the first
    # half of an episode lives in.
    tr, cfg = _mk_trainer(global_shaping=2.0, belief_shaping=0.0)
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5   # off-shape: keep the settled/coverage success gate quiet
    node[0, 0:2] = torch.tensor([0.30, 0.00])
    node[1, 0:2] = torch.tensor([-0.20, 0.40])
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    # NO formation pool: dists all NaN
    worker.belief = {0: {0: _particles_at([0.60, 0.60]),
                         1: _particles_at([0.60, -0.60])}}
    buf = RolloutBuffer(cfg)
    empty = _FakeEmptySteps()
    tr._record_snapshots(buf, worker, empty)
    r1 = buf.steps[-1]["reward"].clone()
    assert tr._roll_global_shaping_count == 2

    # both filters converge to truth: the shared potential rises, paid EQUALLY
    worker.belief[0][0] = _particles_at([0.30, 0.00])
    worker.belief[0][1] = _particles_at([-0.20, 0.40])
    buf2 = RolloutBuffer(cfg)
    tr._record_snapshots(buf2, worker, empty)
    r2 = buf2.steps[-1]["reward"]
    d0 = float(r2[0] - r1[0])
    d1 = float(r2[1] - r1[1])
    assert d0 > 0.0 and d1 > 0.0
    # equal payment to every robot is the sender-credit mechanism itself
    assert abs(d0 - d1) < 1e-6


def test_motor_log_std_quiet_start_touches_only_motor_dims():
    # the warm-start collapse fix: the BC checkpoint's saved log_std is an
    # untrained placeholder (std 0.61); deterministic eval never executes it,
    # but the first stochastic rollout does. The override must set the MOTOR
    # dims without touching the message dims.
    import launch
    from kilobot_gnn import SplitObservationActor
    from policy import GaussianPolicy, MESSAGE_SIZE, MOTOR_SIZE
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.quiet_msg_head = False
    cfg.msg_log_std = 0.0
    cfg.motor_log_std = -2.5
    policy = GaussianPolicy(SplitObservationActor(gru_hidden=24))
    with torch.no_grad():
        policy.log_std.fill_(-0.5)   # the placeholder every bc_offline export carries
    launch.apply_warm_start_message_policy(policy, cfg)
    assert torch.allclose(policy.log_std[MESSAGE_SIZE:], torch.full((MOTOR_SIZE,), -2.5))
    assert torch.allclose(policy.log_std[:MESSAGE_SIZE], torch.full((MESSAGE_SIZE,), -0.5))


def test_belief_comms_learned_leaves_policy_broadcast_untouched():
    # with belief_comms_learned the honest-beacon override of slots 0-2 is
    # skipped: the receiver-side filter still fuses whatever the policy
    # broadcasts (observation.gather_split_state is unchanged), but the
    # CONTENT is the policy's own output -- that is the learned protocol.
    # With it off (default), the override writes the filter's own mean
    # position and conf into slots 0-2 exactly as it always has.
    import trainer as T_mod
    from kilobot_gnn import SplitObservationActor, MESSAGE_SIZE, SEED_SIZE, WALL_SIZE
    from policy import GaussianPolicy
    from tests.test_split_actor import _FakeDecisionSteps, _FakeActWorker

    def _run(belief_comms, belief_comms_learned):
        cfg = Config()
        cfg.actor_type = "gru_split_observation"
        cfg.split_gru_hidden = 24
        cfg.actor_priv_mode = "none"
        cfg.motor_override = "none"
        cfg.belief_comms = belief_comms
        cfg.belief_comms_learned = belief_comms_learned
        tr = T_mod.Trainer.__new__(T_mod.Trainer)
        tr.cfg = cfg
        tr._init_globals()
        tr._roll_split_seed_events = 0
        tr._roll_split_total_events = 0
        tr._roll_split_heartbeat_events = 0
        torch.manual_seed(7)
        policy = GaussianPolicy(SplitObservationActor(gru_hidden=cfg.split_gru_hidden))
        vector = np.zeros((1, 2 + SEED_SIZE + WALL_SIZE), dtype=np.float32)
        vector[0, 2 + 2] = 1.0   # one landmark-seed event so the robot is active
        rows = np.zeros((1, 1, MESSAGE_SIZE + 2), dtype=np.float32)
        decision_steps = _FakeDecisionSteps([vector, rows])
        worker = _FakeActWorker(torch.zeros(1, NODE_FEATURES))
        torch.manual_seed(7)
        T_mod.Trainer._act(tr, RolloutBuffer(cfg), policy, worker, decision_steps)
        return worker.set_actions_calls[-1][0].copy()

    learned = _run(True, True)
    no_comms = _run(False, False)
    honest = _run(True, False)
    # learned beacon: the broadcast is EXACTLY what a no-messaging run sends
    # (the override is the only belief_comms-dependent write in the action
    # path, and the all-zero rows make peer fusion a no-op)
    assert np.allclose(learned, no_comms, atol=1e-6)
    # honest beacon: slots 0-2 are overwritten with the filter's own values
    # and therefore differ from the policy's output there, while every other
    # slot (claims, motor) is untouched
    assert not np.allclose(honest[:3], no_comms[:3], atol=1e-6)
    assert np.allclose(honest[3:], no_comms[3:], atol=1e-6)


def test_msg_supervision_pulls_mean_toward_honest_beacon():
    # the cold-start ladder: slots 0-2 of the message mean are pulled toward
    # atanh of the robot's own (x_m, y_m, conf_pos*r) -- the exact content
    # actor_io's honest-beacon override force-writes in action space. One
    # gradient step on the auxiliary loss must move the mean toward the
    # target; with the head-only evaluation path the gradient must reach
    # head_msg ONLY -- not the motor head, and not the shared trunk (the
    # trunk IS the warm-started motor policy; a trunk-connected supervision
    # was measured to wreck it within five iterations).
    from kilobot_gnn import SplitObservationActor, MESSAGE_SIZE
    from policy import GaussianPolicy
    from ppo import _actor_loss

    torch.manual_seed(3)
    policy = GaussianPolicy(SplitObservationActor(gru_hidden=16, msg_head_prop_input=True))
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.msg_supervision = 1.0
    cfg.entropy_coef = 0.0
    from kilobot_gnn import SEED_SIZE, WALL_SIZE
    n = 8
    tc_w = MESSAGE_SIZE + 1 + SEED_SIZE + WALL_SIZE   # msg + strength + seeds + walls
    prop_w = 8 + 11 + 3                                # odom + belief + target triple
    data = {"z": torch.zeros(n, 8),
            "seed": torch.zeros(n, 3),
            "tx": torch.zeros(n, tc_w),
            "action": torch.zeros(n, MESSAGE_SIZE + 2),
            "old_lp": torch.zeros(n),
            "prev_hidden": torch.zeros(n, 16),
            "prop": torch.zeros(n, prop_w)}
    # targets: belief mean (x_m, y_m) at prop[:, 8:10], conf_pos*r at
    # prop[:, 12]*prop[:, 13] -- one localized robot, one lost
    data["prop"][0, 8], data["prop"][0, 9] = 0.4, -0.2
    data["prop"][0, 12], data["prop"][0, 13] = 0.95, 0.9   # conf 0.855 -> beacon ON
    data["prop"][1, 12], data["prop"][1, 13] = 0.1, 0.1    # conf 0.01 -> beacon OFF
    adv = torch.zeros(n)

    mean_before = policy.act_batch_split(data["tx"], data["prop"], data["prev_hidden"],
                                         deterministic=True)[0].detach().clone()
    loss, _, _, _ = _actor_loss(policy, data, adv, torch.arange(n), cfg)
    loss.backward()
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    opt.step()
    mean_after = policy.act_batch_split(data["tx"], data["prop"], data["prev_hidden"],
                                        deterministic=True)[0].detach()

    # the lost robot's target is ~0 everywhere; the localized robot's is
    # (0.4, -0.2, 0.855) in ACTION space -- after one big step the mean
    # must have moved toward it, in slots 0-2 specifically
    delta = mean_after[0] - mean_before[0]
    assert float(delta[:3].abs().sum()) > 0.05
    assert abs(float(mean_after[0, 2]) - float(mean_before[0, 2])) > 1e-3
    # the gradient reaches head_msg's first three rows and NOTHING else in the
    # actor: the trunk and the motor head are the warm start, and this loss
    # must not touch them (zero advantage and entropy-off make the surrogate's
    # gradient exactly zero at ratio 1)
    wg = policy.actor.head_msg.weight.grad
    assert float(wg[3:].abs().sum()) == 0.0
    assert float(wg[:3, 16:].abs().sum()) > 0.0      # the prop-side columns: the readout
    # THE property: the trunk and the motor head -- the warm start -- get no
    # gradient from this loss (the head's own g-side columns may adapt; the
    # g VALUES enter detached, so nothing flows back through the GRU)
    assert policy.actor.head_motor.weight.grad is None or float(policy.actor.head_motor.weight.grad.abs().sum()) == 0.0
    assert policy.actor.up1.weight.grad is None or float(policy.actor.up1.weight.grad.abs().sum()) == 0.0
    assert policy.actor.gru.weight_hh.grad is None or float(policy.actor.gru.weight_hh.grad.abs().sum()) == 0.0


def test_beacon_readout_init_matches_honest_protocol():
    # the warm start installs the honest beacon as the message head's
    # INITIALIZATION: a localized robot broadcasts its own believed position
    # with conf past the receiver gate; an unlocalized one broadcasts low
    # conf and is ignored by the receiver. This is the cold start the
    # supervision then refines.
    import launch
    from kilobot_gnn import SplitObservationActor, split_forward_batch
    from policy import GaussianPolicy, MESSAGE_SIZE

    torch.manual_seed(11)
    policy = GaussianPolicy(SplitObservationActor(gru_hidden=16, msg_head_prop_input=True))
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.quiet_msg_head = True
    launch.apply_warm_start_message_policy(policy, cfg)

    n = 4
    prop_w = policy.actor.head_msg.in_features - policy.actor.head1.out_features
    prop = torch.zeros(n, prop_w)
    prop[0, 8], prop[0, 9], prop[0, 12] = 0.40, -0.20, 0.95   # localized
    prop[1, 8], prop[1, 9], prop[1, 12] = 0.60, 0.60, 0.10   # lost
    from kilobot_gnn import SPLIT_TC_SIZE
    with torch.no_grad():
        mean, _ = split_forward_batch(policy.actor, torch.zeros(n, SPLIT_TC_SIZE), prop,
                                      torch.zeros(n, 16))
    beacon = torch.tanh(mean[:, :3])
    # localized: conf past the 0.85 receiver gate, position close to the belief
    assert float(beacon[0, 2]) > 0.85
    assert abs(float(beacon[0, 0]) - 0.40) < 0.12
    assert abs(float(beacon[0, 1]) + 0.20) < 0.12
    # lost: conf well below the gate -- the receiver ignores the row
    assert float(beacon[1, 2]) < 0.5
    # claims (slots 3+) stay zero
    assert float(mean[:, 3:MESSAGE_SIZE].abs().sum()) == 0.0


def test_warm_start_load_skips_only_head_msg():
    # a pre-flag checkpoint loads into the wider-head actor with exactly the
    # head_msg keys skipped, and refuses anything wider than that
    import launch
    from kilobot_gnn import SplitObservationActor
    from policy import GaussianPolicy
    import tempfile, os

    torch.manual_seed(5)
    old_policy = GaussianPolicy(SplitObservationActor(gru_hidden=16))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "actor_best.pt")
        torch.save({"actor": old_policy.actor.state_dict(),
                    "log_std": torch.full((11,), -0.5)}, path)
        new_policy = GaussianPolicy(SplitObservationActor(gru_hidden=16, msg_head_prop_input=True))
        launch.warm_start_load(new_policy, path, "cpu")
        # every non-head_msg weight transferred bit-for-bit
        for k, v in old_policy.actor.state_dict().items():
            if not k.startswith("head_msg."):
                assert torch.allclose(new_policy.actor.state_dict()[k], v)
        # and a non-head module with a wrong shape is refused outright --
        # a size mismatch raises even under strict=False, which is the point
        blob = torch.load(path, weights_only=False)
        blob["actor"]["up1.weight"] = torch.zeros(1, 1)
        bad = os.path.join(td, "bad.pt")
        torch.save(blob, bad)
        try:
            launch.warm_start_load(new_policy, bad, "cpu")
            raised = False
        except RuntimeError:
            raised = True
        assert raised


def test_reset_arena_clears_late_per_robot_caches():
    # the episode-2 collapse: _reset_arena reused local indices for new
    # robots while _spawn_heading_cache/turn_anchor/belief_fit/belief_anchor
    # kept the OLD robots' entries -- with oracle_known_start_heading the
    # new robot's filter was told the previous robot's spawn heading and
    # could never localize (berr above spawn level from episode 2 on)
    from types import SimpleNamespace
    worker = SimpleNamespace(
        z={}, image_id={}, databases={}, hidden={}, last_motor={},
        last_dec_step={}, odometer={}, track_neighbor={}, track_seed={},
        belief={}, pending_find_reward={}, step_count={}, ep_reward={},
        simple_heading={}, simple_state={}, simple_turn_accum={},
        simple_wall_name={}, simple_target={}, arrived_switched_off={},
        simple_hilbert_order={},
        _spawn_heading_cache={0: {2: 1.57}},
        turn_anchor={0: {2: (0.3, 0.9)}},
        turn_anchor_set_tick={0: {2: 123.0}},
        belief_fit={0: {2: torch.ones(2)}},
        belief_anchor={0: {2: torch.ones(8)}},
        _stable_belief_heading={0: {2: (0.1, 0.99)}},
        sent=[])
    worker.send_reset = lambda k, i: worker.sent.append(("reset", k, i))
    worker.send_image = lambda k, i: worker.sent.append(("image", k, i))
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = Config()
    tr.image_pool = [torch.zeros(1, 28, 28)]
    tr.encoder = lambda img: torch.zeros(64)
    tr.image_names = None
    tr.formations_dir = None
    tr._reset_arena(worker, 0, send_reset=False)
    assert worker._spawn_heading_cache == {}
    assert worker.turn_anchor == {}
    assert worker.turn_anchor_set_tick == {}
    assert worker.belief_fit == {}
    assert worker.belief_anchor == {}
    assert worker._stable_belief_heading == {}


def test_beacon_overclaim_penalty_charged_only_above_honest():
    # the honesty constraint: a robot broadcasting conf above its own filter
    # confidence pays per tick for the excess; one at or below pays nothing;
    # one with no belief yet pays nothing (NaN -> no term, never zero-penalty
    # and never a crash)
    from reward import beacon_overclaim_penalty, honest_confidences
    node = torch.zeros(3, NODE_FEATURES)
    node[0, 12] = 0.97   # claimed conf, slot 2 of the transmission
    node[1, 12] = 0.50
    node[2, 12] = 0.95
    honest = torch.tensor([0.90, 0.85, float("nan")])
    pen = beacon_overclaim_penalty(node, honest, weight=4.0, dt=0.02)
    assert abs(float(pen[0]) - (-4.0 * 0.07 * 0.02)) < 1e-7
    assert float(pen[1]) == 0.0        # at/below honest: free
    assert float(pen[2]) == 0.0        # no belief: free
    # honest_confidences reproduces the honest override's conf_pos * r
    worker = _FakeRewardWorker(torch.zeros(2, NODE_FEATURES),
                               [torch.tensor([0.0, 0.0])] * 2)
    worker.belief = {0: {0: _particles_at([0.3, -0.2], spread=0.05),
                         1: _particles_at([0.0, 0.0], spread=0.30)}}
    node = torch.zeros(3, NODE_FEATURES)
    hc = honest_confidences(worker, 0, node)
    assert 0.7 < float(hc[0]) <= 1.0   # tight cloud: high honest conf
    assert float(hc[1]) < 0.2          # wide cloud: low honest conf
    assert bool(torch.isnan(hc[2]))    # no belief entry: NaN, never zero


def test_record_snapshots_applies_overclaim_penalty():
    tr, cfg = _mk_trainer()
    cfg.beacon_honesty = 4.0
    node = torch.zeros(2, NODE_FEATURES)
    node[:, 4] = 0.5
    node[0, 12] = 0.97
    node[1, 12] = 0.40
    worker = _FakeRewardWorker(node, [torch.tensor([0.0, 0.0])] * 2)
    worker.belief = {0: {0: _particles_at([0.0, 0.0], spread=0.01),
                         1: _particles_at([0.0, 0.0], spread=0.01)}}
    # both filters are tight: honest conf ~ 1.0; robot 0 overclaims 0.97 vs
    # ~1.0 -> free (claimed below honest); force the honest value down by
    # giving robot 0 a wide cloud instead
    worker.belief[0][0] = _particles_at([0.0, 0.0], spread=0.35)
    buf = RolloutBuffer(cfg)
    tr._record_snapshots(buf, worker, _FakeEmptySteps())
    # honest for a 0.35-spread cloud is ~exp(-var/0.02) << 0.4, so robot 0
    # pays ~ -4*(0.97 - honest)*dt and robot 1 ~ -4*(0.40 - honest)*dt
    assert tr._roll_overclaim_sum < 0.0
    assert float(tr._roll_overclaim_sum) < -4.0 * 0.5 * 0.02
