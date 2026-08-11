"""The oracle-form motor head: budget, path equivalence, and that it IS the teacher.

The head composes the motor command from simple_oracle's own five commands, each
computed in closed form from quantities already in the observation, mixed by the
state head's posterior. Three things have to hold, and each has burned this
project before in a different guise:

  * it must cost no parameters and no inputs (the 24KB budget, and the standing
    constraint that inputs and outputs do not change);
  * the training path (bc_offline.forward_chunk) and the deployed path
    (split_forward_batch, which actor_io.act and val_tape.replay_tape both use)
    must produce the same numbers -- a second copy of a forward pass is exactly
    how the two drift apart;
  * fed confident state and wall labels, it must reproduce simple_oracle's own
    command, not merely something shaped like it.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from kilobot_gnn import (SPLIT_TC_SIZE, SPLIT_ODOM_SIZE, TURN_ANCHOR_SIZE, build_actor,
                         oracle_form_motor, split_forward_batch, PROP_SIN_H, PROP_COS_H,
                         PROP_SIN_T, PROP_COS_T, PROP_CONF_POS, PROP_DIST_T, closed_form_arrived)
from belief import LOCALIZED_CONF_THRESHOLD
from policy import squash_action
import simple_oracle as SO


def _cfg(oracle_head = True):
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.use_arrived_head = True
    cfg.use_turn_anchor = True
    cfg.use_state_head = True
    cfg.use_wall_head = True
    cfg.use_oracle_head = oracle_head
    cfg.split_activation = "elu"
    cfg.device = "cpu"
    return cfg


def _obs(n, seed = 0):
    g = torch.Generator().manual_seed(seed)
    tc = torch.randn(n, SPLIT_TC_SIZE, generator = g)
    prop = torch.randn(n, SPLIT_ODOM_SIZE + TURN_ANCHOR_SIZE, generator = g)
    return tc, prop


def test_costs_no_parameters_and_no_inputs():
    on = build_actor(_cfg(True))
    off = build_actor(_cfg(False))
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
    assert on.up1.in_features == off.up1.in_features
    assert on.head_motor.out_features == off.head_motor.out_features == 2
    # the standing 24KB-as-int8 budget (docs/tuning.md phase 147)
    assert sum(p.numel() for p in on.parameters()) <= 24 * 1024


def test_training_path_matches_deployed_path():
    """forward_chunk and split_forward_batch must agree step for step."""
    from bc_offline import forward_chunk
    actor = build_actor(_cfg(True))
    actor.eval()
    K, B = 7, 5
    tc = torch.stack([_obs(B, seed = t)[0] for t in range(K)])
    prop = torch.stack([_obs(B, seed = t)[1] for t in range(K)])
    valid = torch.ones(K, B, dtype = torch.bool)
    valid[3, 2] = False  # a robot that did not decide keeps its hidden state
    h0 = actor.initial_hidden(B)
    with torch.no_grad():
        motors, _, _, _, _, _ = forward_chunk(actor, tc, prop, valid, h0)
        h = h0
        for t in range(K):
            mean, h_new = split_forward_batch(actor, tc[t], prop[t], h)
            h = torch.where(valid[t].unsqueeze(1), h_new, h)
            step = squash_action(mean)[:, -2:]
            # Valid rows only. On a PADDED row the two differ by construction and
            # always have, for both heads: forward_chunk reads its output off the
            # masked (carried) hidden state while split_forward_batch reads it off
            # the freshly stepped one. A padded row is a robot that did not decide,
            # so it is never scored and never commanded -- measured at 1.5e-2 for
            # the plain head and 3.3e-3 for this one, against 1e-7 on valid rows.
            d = (step - motors[t]).abs()[valid[t]]
            assert float(d.max()) < 1e-5, "paths diverge at t=%d by %.2e" % (t, float(d.max()))


def test_reproduces_simple_oracle_exactly():
    """With confident labels the head IS the teacher, wheel for wheel.

    simple_oracle is called through its own _steer, not a reimplementation of it,
    so this fails if either side is ever retuned without the other.
    """
    class Stub:
        use_oracle_head = True
        oracle_residual = 0.0
        oracle_residual_turn = 0.0

        def __init__(self, ps, pw):
            self._ps, self._pw = ps, pw

        def head_state(self, g):
            return self._ps

        def head_wall(self, g):
            return self._pw

    torch.manual_seed(0)
    n = 400
    prop = torch.zeros(n, SPLIT_ODOM_SIZE + TURN_ANCHOR_SIZE)
    heading = torch.rand(n) * 2 * math.pi - math.pi
    prop[:, PROP_SIN_H] = torch.sin(heading)
    prop[:, PROP_COS_H] = torch.cos(heading)
    prop[:, PROP_CONF_POS] = 0.0                      # below the slowdown band: scale 1.0
    bearing = torch.rand(n) * 2 * math.pi - math.pi   # bearing to own target, rel. heading
    prop[:, PROP_SIN_T] = torch.sin(bearing)
    prop[:, PROP_COS_T] = torch.cos(bearing)

    BIG = 40.0
    walls = torch.randint(0, 4, (n,))
    pw = torch.nn.functional.one_hot(walls, 4).float() * BIG
    for si, name in enumerate(["go_north", "turning", "wall_following", "navigating", "arrived"]):
        ps = torch.nn.functional.one_hot(torch.full((n,), si), 5).float() * BIG
        pre = oracle_form_motor(Stub(ps, pw), None, prop)
        got = 0.5 * (torch.tanh(pre) + 1.0)
        want = torch.zeros(n, 2)
        for i in range(n):
            h_vec = torch.tensor([math.cos(heading[i]), math.sin(heading[i])]).numpy()
            if name == "go_north":
                want[i] = torch.tensor([1.0, 1.0])
            elif name == "turning":
                want[i] = torch.tensor(SO.TURN_MOTOR)
            elif name == "wall_following":
                want[i] = torch.tensor(SO._steer(h_vec, SO.WALL_TANGENT[SO.WALL_NAMES[walls[i]]]))
            elif name == "navigating":
                # the oracle steers toward g in world frame; the observation
                # gives the same angle already relative to the heading
                ang = heading[i] + bearing[i]
                g_vec = torch.tensor([math.cos(ang), math.sin(ang)]).numpy()
                want[i] = torch.tensor(SO._steer(h_vec, g_vec))
            else:
                want[i] = torch.tensor([0.0, 0.0])
        # 1e-3 is the squash bound: 0 and 1 are reachable only at -+inf, so a
        # saturated command lands at 0.001 / 0.999 by construction
        assert (got - want).abs().max() < 2e-3, \
            "%s: max %.5f" % (name, float((got - want).abs().max()))


def test_requires_the_two_discrete_heads():
    cfg = _cfg(True)
    cfg.use_wall_head = False
    with pytest.raises(ValueError):
        build_actor(cfg)


def test_residual_is_bounded():
    """The closed form is a prior, not a starting point the fit can regress away."""
    actor = build_actor(_cfg(True))
    actor.oracle_residual_turn = 0.003   # the swept ablation value, not the default
    with torch.no_grad():
        actor.head_motor.weight.fill_(50.0)
        actor.head_motor.bias.fill_(50.0)
    tc, prop = _obs(64)
    with torch.no_grad():
        mean, _ = split_forward_batch(actor, tc, prop, actor.initial_hidden(64))
        motors = squash_action(mean)[:, -2:]
        actor.oracle_residual = 0.0
        actor.oracle_residual_turn = 0.0
        mean0, _ = split_forward_batch(actor, tc, prop, actor.initial_hidden(64))
        base = squash_action(mean0)[:, -2:]
    # bounded by the two scales together: the common mode moves both wheels,
    # the differential moves them apart
    assert (motors - base).abs().max() <= 0.05 + 0.003 + 1e-4
    # The invariant that matters, and the one that is exactly true: the common
    # half moves both wheels together and so cannot change the DIFFERENTIAL at
    # all, and the differential half is bounded by twice its own scale. (`turn`
    # itself is a ratio, so a common-mode change does rescale it slightly
    # whenever the base turn is nonzero -- at the wall_following operating
    # point, where the teacher's own turn has sd 0.0093, that term is negligible and
    # the differential bound is what binds: 2*0.003*1.8/(0.7*1.8) = 0.0086.)
    d_res = ((motors[:, 1] - motors[:, 0]) - (base[:, 1] - base[:, 0])).abs().max()
    assert float(d_res) <= 2 * 0.003 + 1e-4, float(d_res)
    with torch.no_grad():
        actor.oracle_residual_turn = 0.0
        actor.oracle_residual = 0.05
        mean_c, _ = split_forward_batch(actor, tc, prop, actor.initial_hidden(64))
        common_only = squash_action(mean_c)[:, -2:]
    d_common = ((common_only[:, 1] - common_only[:, 0]) - (base[:, 1] - base[:, 0])).abs().max()
    assert float(d_common) < 1e-6, "the common half must not touch the differential: %g" % d_common


def test_closed_form_arrived_is_the_oracle_rule():
    """closed_form_arrived must equal simple_oracle's arrival condition, read
    from the actor's own observation: filter distance to the robot's own target
    strictly below cfg.tau_v, after it has localized (conf >= the same
    LOCALIZED_CONF_THRESHOLD simple_oracle uses to enter navigating), and only
    for robots that have an assigned target at all."""
    cfg = _cfg()
    n = 6
    prop = torch.zeros(n, 4 + 4 + 11 + 3 + 2)
    prop[:, PROP_CONF_POS] = 0.9
    prop[:, PROP_DIST_T] = 0.03
    prop[:, PROP_SIN_T] = 0.0
    prop[:, PROP_COS_T] = 1.0
    # distance AT tau_v is not arrival (the oracle's check is strict <)
    prop[1, PROP_DIST_T] = cfg.tau_v
    # confidence below the localization floor is not arrival
    prop[2, PROP_CONF_POS] = LOCALIZED_CONF_THRESHOLD - 0.01
    # no assigned target: the distance column reads 0 and would false-fire
    prop[3, PROP_SIN_T] = 0.0
    prop[3, PROP_COS_T] = 0.0
    got = closed_form_arrived(prop, cfg.tau_v).tolist()
    assert got == [True, False, False, False, True, True], got
    # and it is strict on the distance the same way simple_oracle reads it:
    # d_target == tau_v must not fire, however confident
    prop = torch.zeros(1, 4 + 4 + 11 + 3 + 2)
    prop[:, PROP_CONF_POS] = 1.0
    prop[:, PROP_DIST_T] = cfg.tau_v - 1e-4
    prop[:, PROP_SIN_T] = 0.0
    prop[:, PROP_COS_T] = 1.0
    assert closed_form_arrived(prop, cfg.tau_v).item() is True
    prop[:, PROP_DIST_T] = cfg.tau_v
    assert closed_form_arrived(prop, cfg.tau_v).item() is False


def test_closed_form_gate_latches_terminal():
    """With use_closed_form_arrived the _arrived_head_gate stops a robot exactly
    when the closed form fires, and -- like the oracle's arrived state -- that
    switch-off is terminal for the episode: it does not re-read the form."""
    from actor_io import _arrived_head_gate
    from types import SimpleNamespace
    cfg = _cfg()
    cfg.use_closed_form_arrived = True
    cfg.motor_override = "none"
    worker = SimpleNamespace()
    policy = SimpleNamespace(actor = SimpleNamespace(head_arrived = None))
    flags = torch.tensor([True, False, True, False])
    stopped = [_arrived_head_gate(worker, policy, cfg, 0, l, l, closed_form = flags)
               for l in range(4)]
    assert stopped == [True, False, True, False], stopped
    # a robot that was off stays off even when the form stops firing; a robot
    # that was on switches on when the form fires
    flags2 = torch.tensor([False, True, False, True])
    stopped2 = [_arrived_head_gate(worker, policy, cfg, 0, l, l, closed_form = flags2)
                for l in range(4)]
    assert stopped2 == [True, True, True, True], stopped2


def test_closed_form_hybrid_or_learned_head():
    """With closed_form_hybrid the gate stops on EITHER the closed form or the
    learned head, terminal either way -- the two miss different robots."""
    from actor_io import _arrived_head_gate
    from types import SimpleNamespace
    cfg = _cfg()
    cfg.use_closed_form_arrived = True
    cfg.closed_form_hybrid = True
    cfg.motor_override = "none"
    cfg.arrived_confidence_threshold = 0.9
    # learned logits below show up as how the deployed path reads them: sigmoid
    worker = SimpleNamespace()
    logs = torch.tensor([3.0, -4.0, -4.0, -4.0])   # sigmoid -> 0.95, 0.018, ...
    policy = SimpleNamespace(actor = SimpleNamespace(head_arrived = object()))
    policy.actor._arrived_logit = logs
    flags = torch.tensor([False, True, False, False])
    # robot0: learned head fires (0.95 > 0.9); robot1: closed form fires
    stopped = [_arrived_head_gate(worker, policy, cfg, 0, l, l, closed_form = flags)
               for l in range(4)]
    assert stopped == [True, True, False, False], stopped
    # terminal: robots 0 (learned-latched) and 1 (closed-form-latched) stay off
    # even with both signals now absent; robots 2,3 (never stopped) stay on
    policy.actor._arrived_logit = torch.full((4,), -4.0)
    flags2 = torch.tensor([False, False, False, False])
    stopped2 = [_arrived_head_gate(worker, policy, cfg, 0, l, l, closed_form = flags2)
                for l in range(4)]
    assert stopped2 == [True, True, False, False], stopped2
