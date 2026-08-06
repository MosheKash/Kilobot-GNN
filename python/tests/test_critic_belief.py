import torch

from kilobot_gnn import build_actor
import conftest
from trainer import make_critic, critic_extra_features
from policy import GaussianPolicy
from ppo import ppo_update
from kilobot_gnn import NODE_FEATURES
from belief import BELIEF_FEATURES


def _collect(critic_belief, rollout = 8, swarm_rng = None):
    cfg = conftest.unity_cfg(actor_type = "gru_split_observation", rollout = rollout, arenas = 1)
    cfg.seed_layout = "cluster"
    cfg.critic_belief_features = critic_belief
    cfg.belief_conf_bonus = 0.5
    cfg.seed = 3
    torch.manual_seed(3)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    critic = make_critic(cfg)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 10, max_bots = 12,
                                        swarm_rng = swarm_rng)
    torch.manual_seed(cfg.seed + 1000)
    buf = tr.collect(policy, critic)
    return cfg, policy, critic, buf, worker


def test_flag_off_keeps_node_width():
    cfg, policy, critic, buf, worker = _collect(False)
    assert critic_extra_features(cfg) == 0
    assert buf.steps[0]["node"].shape[1] == NODE_FEATURES


def test_flag_on_appends_belief_columns():
    cfg, policy, critic, buf, worker = _collect(True)
    assert critic_extra_features(cfg) == BELIEF_FEATURES
    node = buf.steps[-1]["node"]
    assert node.shape[1] == NODE_FEATURES + BELIEF_FEATURES
    extra = node[:, NODE_FEATURES:]
    assert float(extra.abs().sum()) > 0.0
    conf = extra[:, 4]
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0
    assert float(extra[:, 8].max()) <= 1.5 + 1e-6


def test_flag_on_end_to_end_ppo_update_runs():
    cfg, policy, critic, buf, worker = _collect(True)
    a_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr = cfg.critic_lr)
    logs = ppo_update(policy, critic, a_opt, c_opt, buf, cfg)
    assert torch.isfinite(torch.tensor(logs["actor_loss"]))
    assert torch.isfinite(torch.tensor(logs["critic_loss"]))


def test_flag_does_not_change_rewards_or_base_features():
    # The strong form of this test: two full rollouts
    # differing only in the flag, compared step for step. It needs both runs to
    # see the same arena, which on a real player takes SwarmManager's swarm-RNG
    # pin (EnvWorker.set_swarm_rng) -- the same seed replays the same
    # population, positions and headings, so the only difference left between
    # the two collects is the flag itself.
    #
    # The two collects also have to make identical decisions for the comparison
    # to mean anything. They do: the flag only widens the CRITIC's node input,
    # the actor is built and seeded identically in both, and _collect reseeds
    # torch immediately before collect() so the differently-shaped critic's own
    # initialization cannot desynchronize the action stream.
    SEED = 4242
    cfg_off, _p, _c, buf_off, _w = _collect(False, swarm_rng = SEED)
    cfg_on, _p2, _c2, buf_on, _w2 = _collect(True, swarm_rng = SEED)
    assert len(buf_off.steps) > 0 and len(buf_on.steps) > 0
    assert len(buf_off.steps) == len(buf_on.steps), \
        "same seed and same actor should produce the same number of steps"

    off_widths = {s["node"].shape[1] for s in buf_off.steps}
    on_widths = {s["node"].shape[1] for s in buf_on.steps}
    assert off_widths == {NODE_FEATURES}, \
        "with the flag off the node must be exactly the base features, got %s" % off_widths
    assert on_widths == {NODE_FEATURES + BELIEF_FEATURES}, \
        "with the flag on exactly BELIEF_FEATURES columns should be appended, got %s" % on_widths

    for i, (s_off, s_on) in enumerate(zip(buf_off.steps, buf_on.steps)):
        base_off = s_off["node"][:, :NODE_FEATURES]
        base_on = s_on["node"][:, :NODE_FEATURES]
        assert base_off.shape == base_on.shape, \
            "step %d: same seed should give the same robot count" % i
        assert torch.allclose(base_off, base_on, atol = 1e-5), \
            "step %d: appending belief columns changed the base features" % i
        assert torch.allclose(s_off["reward"], s_on["reward"], atol = 1e-5), \
            "step %d: appending belief columns changed the reward" % i
