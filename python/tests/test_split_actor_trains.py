import torch

from kilobot_gnn import SplitObservationActor, Critic, NODE_FEATURES, SPLIT_TC_SIZE, SPLIT_SEED_OFFSET, SPLIT_ODOM_SIZE, MESSAGE_SIZE, Z
from policy import GaussianPolicy
from buffer import RolloutBuffer
from config import Config
from ppo import ppo_update


def _make_config():
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.minibatch = 32
    cfg.ppo_epochs = 4
    cfg.entropy_coef = 0.01
    return cfg


def _target_motor(is_seed):
    left_right = torch.where(is_seed.unsqueeze(1), torch.tensor([1.0, 0.0]).expand(is_seed.shape[0], 2),
                             torch.tensor([0.0, 1.0]).expand(is_seed.shape[0], 2))
    return left_right


def _build_batch(cfg, batch_size):
    is_seed = torch.rand(batch_size) > 0.5
    tc = torch.zeros(batch_size, SPLIT_TC_SIZE)
    for i in range(batch_size):
        if is_seed[i]:
            tc[i, SPLIT_SEED_OFFSET + 1] = 1.0
        else:
            tc[i, 0] = 1.0
    z = 0.05 * torch.randn(batch_size, Z)
    prop = 0.05 * torch.randn(batch_size, SPLIT_ODOM_SIZE)
    h0 = torch.zeros(batch_size, cfg.split_gru_hidden)
    target = _target_motor(is_seed)
    return is_seed, tc, z, prop, h0, target


def _run_iterations(seed, n_iters, batch_size = 64):
    torch.manual_seed(seed)
    cfg = _make_config()
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden), cfg.log_std_init)
    critic = Critic()
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr = cfg.critic_lr)

    mse_hist = []
    logs = None
    for it in range(n_iters):
        is_seed, tc, z, prop, h0, target = _build_batch(cfg, batch_size)
        with torch.no_grad():
            u, env_action, log_prob, h_new = policy.act_batch_split(tc, prop, h0, deterministic = False)
        motor = env_action[:, MESSAGE_SIZE:]
        mse = ((motor - target) ** 2).sum(dim = 1)
        reward = -mse

        buffer = RolloutBuffer(cfg)
        node = torch.zeros(batch_size, NODE_FEATURES)
        node[:, 0] = is_seed.float()
        edge_index = torch.zeros(2, 0, dtype = torch.long)
        edge_attr = torch.zeros(0, 1)
        traj = torch.arange(batch_size, dtype = torch.long)
        term = torch.ones(batch_size)
        cut = torch.ones(batch_size)
        is_decision = torch.ones(batch_size, dtype = torch.bool)
        si = buffer.add_step(0, it, node, edge_index, edge_attr, z[0], traj, reward, term, cut, is_decision)
        for idx in range(batch_size):
            buffer.add_decision(si, idx, z[idx], tc[idx], tc[idx], None, u[idx], log_prob[idx],
                                prev_hidden = h0[idx], prop = prop[idx])

        logs = ppo_update(policy, critic, actor_opt, critic_opt, buffer, cfg)
        mse_hist.append(float(mse.mean()))

    return mse_hist, logs, policy


def test_split_actor_trains_toward_target_motor():
    # each decision is one event (seed or neighbor), one-hot encoded in tc; the
    # target motor command depends only on which kind of event it was. If tc is
    # not actually reaching head_motor through the upscale MLP and GRU, mean
    # squared error to the target will not fall.
    mse_hist, logs, policy = _run_iterations(seed = 0, n_iters = 40)
    first5 = sum(mse_hist[:5]) / 5
    last5 = sum(mse_hist[-5:]) / 5
    assert last5 < 0.15 * first5
    assert last5 < 0.1
    assert logs["motor_param_drift"] > 0.0
    assert torch.isfinite(torch.tensor(logs["entropy"]))


def test_split_actor_trains_reduces_mse_monotonically_in_aggregate():
    # not every single iteration has to improve on the last (on-policy sampling
    # noise), but the running mean over the run should trend down substantially
    mse_hist, _, _ = _run_iterations(seed = 1, n_iters = 40)
    first_half = sum(mse_hist[:20]) / 20
    second_half = sum(mse_hist[20:]) / 20
    assert second_half < 0.3 * first_half


def test_split_actor_deterministic_policy_reaches_low_error_after_training():
    _, _, policy = _run_iterations(seed = 2, n_iters = 40)
    policy.eval()
    cfg = _make_config()
    batch_size = 64
    is_seed, tc, z, prop, h0, target = _build_batch(cfg, batch_size)
    with torch.no_grad():
        mean, env_action, log_prob, h_new = policy.act_batch_split(tc, prop, h0, deterministic = True)
    motor = env_action[:, MESSAGE_SIZE:]
    mse = ((motor - target) ** 2).sum(dim = 1).mean()
    assert float(mse) < 0.1
