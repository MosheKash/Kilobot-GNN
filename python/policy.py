"""The stochastic policy wrapper: an actor network -> sampled actions.

GaussianPolicy adds a learned per-dimension log-std to whatever actor it is
given, and squash_action/squash_log_det are the tanh squashing and its
log-determinant correction. Nothing here knows what the actions mean; the
network architectures live in kilobot_gnn.py.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kilobot_gnn import MESSAGE_SIZE, MOTOR_SIZE, actor_forward_batch, recurrent_forward_batch, split_forward_batch

ACTION_SIZE = MESSAGE_SIZE + MOTOR_SIZE


def squash_action(u):
    # Map an unbounded pre-squash sample u into the action range with tanh: message
    # dims into [-1,1], motor dims into [0,1]. The executed action is always in range,
    # so no clamping is needed and the executed action is exactly the one whose
    # log-probability the policy gradient optimizes (after the Jacobian correction).
    t = torch.tanh(u)
    msg = t[..., :MESSAGE_SIZE]
    motor = 0.5 * (t[..., MESSAGE_SIZE:] + 1.0)
    return torch.cat([msg, motor], dim=-1)


def squash_log_det(u):
    # Sum of log|d action / d u| over dims, subtracted from the base Gaussian log-prob so
    # the result is the true density of the squashed (executed) action. For tanh dims
    # da/du = 1 - tanh^2 (stable SAC form below); motor dims carry an extra 0.5 scale
    # (tanh -> [0,1]), a constant that cancels in PPO ratios but is kept for correctness.
    corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))  # = log(1 - tanh(u)^2) per dim
    return corr.sum(dim=-1) + MOTOR_SIZE * math.log(0.5)

# Bounds on the policy's log standard deviation. The lower bound is a floor on
# exploration; the upper bound stops the spread running away. Both are env-
# configurable so the exploration noise can be pinned to a chosen level: setting
# KILOBOT_LOG_STD_MIN == KILOBOT_LOG_STD_MAX fixes std exactly (e.g. -3.91 -> 0.02).
LOG_STD_MIN = float(os.environ.get("KILOBOT_LOG_STD_MIN", -2.0))
LOG_STD_MAX = float(os.environ.get("KILOBOT_LOG_STD_MAX", 0.0))


class GaussianPolicy(nn.Module):
    def __init__(self, actor, log_std_init=-0.5):
        super().__init__()
        self.actor = actor
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), float(log_std_init)))

    def _std(self):
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def clamp_log_std(self):
        with torch.no_grad():
            self.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

    def _mean(self, z, seed_robots, transmission, database):
        output_transmission, motor_commands, database = self.actor(z, seed_robots, transmission, database)
        mean = torch.cat([output_transmission, motor_commands], dim=0)
        return mean, database

    def act(self, z, seed_robots, transmission, database):
        mean, database = self._mean(z, seed_robots, transmission, database)
        std = self._std()
        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        env_action = squash_action(u)
        log_prob = dist.log_prob(u).sum() - squash_log_det(u)
        return u, env_action, log_prob, database

    def evaluate(self, z, seed_robots, transmission, database, action):
        mean, _ = self._mean(z, seed_robots, transmission, database)
        std = self._std()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum() - squash_log_det(action)
        entropy = dist.entropy().sum()
        return log_prob, entropy

    def act_batch(self, z_b, seed_b, msg_b, sender_b, db_rows, db_valid, deterministic=False):
        mean, db_rows, db_valid = actor_forward_batch(self.actor, z_b, seed_b, msg_b, sender_b, db_rows, db_valid)
        self._last_mean = mean.detach()
        std = self._std()
        if deterministic:
            env_action = squash_action(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            return mean, env_action, log_prob, db_rows, db_valid
        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        env_action = squash_action(u)
        log_prob = dist.log_prob(u).sum(dim=1) - squash_log_det(u)
        # Stored action is the pre-squash sample u; the env executes squash(u). The
        # Jacobian-corrected log_prob is the density of the executed action, so reward
        # and gradient refer to the same action.
        return u, env_action, log_prob, db_rows, db_valid

    def evaluate_batch(self, z_b, seed_b, msg_b, sender_b, db_rows, db_valid, action_b):
        mean, _, _ = actor_forward_batch(self.actor, z_b, seed_b, msg_b, sender_b, db_rows, db_valid)
        std = self._std()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_b).sum(dim=1) - squash_log_det(action_b)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, entropy

    def act_batch_gru(self, z_b, seed_b, msg_b, prop_b, h_prev, deterministic=False):
        mean, h_new = recurrent_forward_batch(self.actor, z_b, seed_b, msg_b, prop_b, h_prev)
        self._last_mean = mean.detach()
        std = self._std()
        if deterministic:
            env_action = squash_action(mean)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
            return mean, env_action, log_prob, h_new
        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        env_action = squash_action(u)
        log_prob = dist.log_prob(u).sum(dim=1) - squash_log_det(u)
        return u, env_action, log_prob, h_new

    def evaluate_batch_gru(self, z_b, seed_b, msg_b, prop_b, h_prev, action_b):
        mean, _ = recurrent_forward_batch(self.actor, z_b, seed_b, msg_b, prop_b, h_prev)
        std = self._std()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_b).sum(dim=1) - squash_log_det(action_b)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, entropy

    def act_batch_split(self, tc_b, prop_b, h_prev, deterministic = False):
        mean, h_new = split_forward_batch(self.actor, tc_b, prop_b, h_prev)
        self._last_mean = mean.detach()
        std = self._std()
        if deterministic:
            env_action = squash_action(mean)
            log_prob = torch.zeros(mean.shape[0], device = mean.device)
            return mean, env_action, log_prob, h_new
        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        env_action = squash_action(u)
        log_prob = dist.log_prob(u).sum(dim = 1) - squash_log_det(u)
        return u, env_action, log_prob, h_new

    def evaluate_batch_split(self, tc_b, prop_b, h_prev, action_b):
        mean, _ = split_forward_batch(self.actor, tc_b, prop_b, h_prev)
        std = self._std()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_b).sum(dim = 1) - squash_log_det(action_b)
        entropy = dist.entropy().sum(dim = 1)
        return log_prob, entropy
