import os
import tempfile
import torch

from kilobot_gnn import Actor, Critic, Z
from policy import GaussianPolicy
from checkpoint import save_checkpoint, load_checkpoint, export_actor


def _build():
    actor = Actor()
    policy = GaussianPolicy(actor, -0.5)
    critic = Critic()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    return policy, critic, actor_opt, critic_opt


def _populate_opt(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.randn_like(p)
    opt.step()


def _same_params(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.allclose(sa[k], sb[k]), k


def test_checkpoint_creates_missing_dir():
    policy, critic, actor_opt, critic_opt = _build()
    with tempfile.TemporaryDirectory() as d:
        nested = os.path.join(d, "run_20260101_000000", "sub")
        path = os.path.join(nested, "ckpt.pt")
        # parent does not exist yet; save_checkpoint should create it
        save_checkpoint(path, 1, policy, critic, actor_opt, critic_opt)
        assert os.path.exists(path)
        actor_path = os.path.join(nested, "deeper", "actor_final.pt")
        export_actor(actor_path, policy)
        assert os.path.exists(actor_path)


def test_checkpoint_roundtrip():
    policy, critic, actor_opt, critic_opt = _build()
    _populate_opt(actor_opt)
    _populate_opt(critic_opt)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(path, 42, policy, critic, actor_opt, critic_opt)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

        p2, c2, ao2, co2 = _build()
        it = load_checkpoint(path, p2, c2, ao2, co2, "cpu")

        assert it == 42
        _same_params(policy, p2)
        _same_params(critic, c2)

        # restored optimizer momentum
        s1 = actor_opt.state_dict()["state"]
        s2 = ao2.state_dict()["state"]
        assert len(s2) == len(s1) and len(s2) > 0
        k = next(iter(s1))
        assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"])


def test_export_actor():
    policy, critic, actor_opt, critic_opt = _build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "actor_final.pt")
        export_actor(path, policy)

        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob["meta"]["z"] == Z
        assert "log_std" in blob

        fresh = Actor()
        fresh.load_state_dict(blob["actor"])
        ref = policy.actor.state_dict()
        got = fresh.state_dict()
        for key in ref:
            assert torch.allclose(ref[key], got[key]), key


def test_load_for_eval_both_formats():
    from checkpoint import save_checkpoint, export_actor, load_for_eval
    policy, critic, ao, co = _build()
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ckpt.pt")
        ex = os.path.join(d, "actor_final.pt")
        save_checkpoint(ck, 9, policy, critic, ao, co)
        export_actor(ex, policy)

        p_ck = GaussianPolicy(Actor(), 0.0)
        it = load_for_eval(ck, p_ck, "cpu")
        assert it == 9
        _same_params(policy.actor, p_ck.actor)

        p_ex = GaussianPolicy(Actor(), 0.0)
        load_for_eval(ex, p_ex, "cpu")
        _same_params(policy.actor, p_ex.actor)
        assert torch.allclose(policy.log_std, p_ex.log_std)
