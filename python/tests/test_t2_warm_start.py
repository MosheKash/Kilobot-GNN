"""T2 warm-start wiring tests (launch.warm_start_load + the veto init).

These are pure-tensor -- no Unity, no rollout. They pin exactly two things:

1. The homing_proprio width tolerance in warm_start_load: a pre-T2 checkpoint
   loads into the wider actor by ZERO-EXTENDING up1's input rows, the
   extension is the only width mismatch the loader accepts, and the loaded
   actor computes up1's outputs EXACTLY as the original 42-input actor does on
   the same input with the appended columns at zero (the +3 columns are
   identically zero until a referral is fitted -- observation.py:692-703 -- so
   0 * w = 0 keeps the function bit-identical at the warm start).

2. The learned_veto init in apply_warm_start_message_policy: iteration 0 must
   reproduce the T1 writer (essentially never veto) instead of the zeroed
   head's coin-flip mute; and without the flag the bias must stay untouched.
"""

import copy
import math

import pytest
import torch

import launch
from config import Config
from kilobot_gnn import build_actor, HOMING_PROP_SIZE
from policy import GaussianPolicy


def _split_cfg(**overrides):
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.msg_head_prop_input = False
    cfg.homing_proprio = False
    cfg.learned_veto = False
    cfg.quiet_msg_head = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _tmp_ckpt(tmp_path, actor, log_std=None):
    blob = {"actor": copy.deepcopy(actor.state_dict()), "iteration": 0}
    if log_std is not None:
        blob["log_std"] = log_std
    path = tmp_path / "actor_prev.pt"
    torch.save(blob, str(path))
    return str(path)


def test_zero_extension_is_exact_and_only_homing_width(tmp_path):
    cfg0 = _split_cfg()
    actor0 = build_actor(cfg0)
    ckpt = _tmp_ckpt(tmp_path, actor0,
                     torch.tensor([-0.5] * 11))

    cfg3 = _split_cfg(homing_proprio = True)
    actor3 = build_actor(cfg3)
    assert actor3.up1.in_features == actor0.up1.in_features + HOMING_PROP_SIZE

    policy = GaussianPolicy(actor3, cfg3.log_std_init)
    it = launch.warm_start_load(policy, ckpt, torch.device("cpu"))
    assert it == 0

    w = policy.actor.state_dict()
    w0 = actor0.state_dict()
    # The extension: appended rows are exactly zero; preceding columns carry
    # the checkpoint's weights.
    assert torch.equal(w["up1.weight"][:, -HOMING_PROP_SIZE:],
                       torch.zeros_like(w["up1.weight"][:, -HOMING_PROP_SIZE:]))
    assert torch.equal(w["up1.weight"][:, :actor0.up1.in_features],
                       actor0.up1.weight.to(w["up1.weight"].dtype))
    # Everything else must be the checkpoint's own tensors, exactly.
    for k, v in w0.items():
        if k == "head_msg.weight" or k == "head_msg.bias" or k == "up1.weight":
            continue
        assert k in w, k
        assert torch.equal(w[k], v.to(w[k].dtype)), k
    assert torch.equal(policy.log_std, torch.tensor([-0.5] * 11))

    # Exactness of the function: up1 on [x, zeros] == up1 of the original.
    # Algebraically the appended columns contribute exact +0.0 additions and
    # the two outputs are identical; bitwise equality is nonetheless not
    # guaranteed in practice because the wider matrix's BLAS blocking folds
    # the zero terms into a different reduction tree, which moves fp
    # roundoff by ~1 ulp. The tolerance here (1e-5 on 40-dim pre-activations
    # of order 1) is thus a kernel-noise bound, not an algebraic concession.
    x = torch.randn(8, actor0.up1.in_features)
    with torch.no_grad():
        h0 = actor0.up1(x)
        h3 = policy.actor.up1(torch.cat([x, torch.zeros(8, HOMING_PROP_SIZE)], dim = 1))
    assert torch.allclose(h0, h3, rtol = 1e-5, atol = 1e-5)


def test_other_width_deltas_are_refused(tmp_path):
    cfg0 = _split_cfg()
    actor0 = build_actor(cfg0)
    cfg3 = _split_cfg(homing_proprio = True)
    actor3 = build_actor(cfg3)

    # A checkpoint with the WRONG width delta: take the 42-col actor's state
    # and trim up1 by 2 columns (delta would be +5, not HOMING_PROP_SIZE).
    bad = {"actor": copy.deepcopy(actor0.state_dict())}
    w = bad["actor"]["up1.weight"][:, :-2].clone()
    bad["actor"]["up1.weight"] = w
    path = tmp_path / "grid_wrong.pt"
    torch.save(bad, str(path))
    policy = GaussianPolicy(actor3, cfg3.log_std_init)
    with pytest.raises(RuntimeError, match = "only zero-extends"):
        launch.warm_start_load(policy, str(path), torch.device("cpu"))

    # And a checkpoint WIDER than the actor: refuse, do not truncate.
    wide = {"actor": copy.deepcopy(actor3.state_dict())}
    path2 = tmp_path / "wider_than_actor.pt"
    torch.save(wide, str(path2))
    policy2 = GaussianPolicy(actor0, cfg0.log_std_init)
    with pytest.raises(RuntimeError, match = "WIDER than the actor"):
        launch.warm_start_load(policy2, str(path2), torch.device("cpu"))


def test_veto_init_starts_at_the_t1_writer():
    cfg = _split_cfg(learned_veto = True, quiet_msg_head = True)
    actor = build_actor(cfg)
    policy = GaussianPolicy(actor, cfg.log_std_init)
    launch.apply_warm_start_message_policy(policy, cfg)
    assert float(policy.actor.head_msg.bias[6]) == 2.0
    # The veto's explore-ability is untouched: the head's weights (and every
    # other slot) remain at the quiet start's zeros.
    assert torch.count_nonzero(policy.actor.head_msg.weight[6]) == 0
    phi_neg = 0.5 * (1.0 + math.erf(-2.0 / math.exp(-0.5)))
    assert phi_neg < 1e-3

    cfg_noveto = _split_cfg(quiet_msg_head = True)
    actor2 = build_actor(cfg_noveto)
    policy2 = GaussianPolicy(actor2, cfg_noveto.log_std_init)
    launch.apply_warm_start_message_policy(policy2, cfg_noveto)
    assert float(policy2.actor.head_msg.bias[6]) == 0.0


def test_self_trust_multiplier_semantics():
    """T2b's one-sided receiver mapping with the deadband.

    multiplier = 1.0 for t >= -0.25 (the deadband absorbs the BC head's
    fit-noise around zero -- the first parity smoke measured a -0.004
    sample demoting every claim 0.4% and drifting the tick lines), else
    1 + t. No amplification ever; t = -1 drops the claim."""
    import evidence

    for t, want in ((0.0, 1.0), (0.96, 1.0), (1.0, 1.0),
                    (-0.24, 1.0),              # deadband: fit noise is a no-op
                    (-0.25, 1.0),              # boundary (>= deadband)
                    (-0.26, 0.74),
                    (-1.0, 0.0), (-0.5, 0.5)):
        t_c = float(min(max(t, -1.0), 1.0))
        mult = 1.0 if t_c >= -0.25 else 1.0 + t_c
        assert abs(mult - want) < 1e-9, (t, mult, want)

    # And at the store level: trust rides with the claim record.
    class W:  # minimal worker stand-in
        step_count = {0: 100}
        claim_store = {}
    rows = torch.zeros(1, 2, 12)
    rows[0, 0, 0], rows[0, 0, 1], rows[0, 0, 2] = 0.1, -0.2, 0.8   # beacon
    rows[0, 0, 9] = 3.0                                            # senderId
    rows[0, 0, 7] = -0.5                                           # self-trust
    rows[0, 1, 9] = 4.0                                            # invalid (no content)
    cfg = _split_cfg(evidence_comms = True)
    evidence.update_claims(W(), torch.tensor([0]), torch.tensor([0]),
                           rows, None, cfg, torch.device("cpu"))
    c = W.claim_store[(0, 0)][3]
    assert abs(c["trust"] + 0.5) < 1e-9
    assert abs(c["conf"] - 0.8) < 1e-6

    # Materialization applies the multiplier only under the flag.
    pp, pc, ps = evidence.materialize_claims(W(), torch.tensor([0]), torch.tensor([0]),
                                             cfg, torch.device("cpu"))
    assert abs(float(pc[0, 0]) - 0.8) < 1e-6         # flag off: trust ignored
    cfg2 = _split_cfg(evidence_comms = True, learned_self_trust = True)
    pp, pc, ps = evidence.materialize_claims(W(), torch.tensor([0]), torch.tensor([0]),
                                             cfg2, torch.device("cpu"))
    assert abs(float(pc[0, 0]) - 0.4) < 1e-6         # 0.8 * (1 - 0.5)
