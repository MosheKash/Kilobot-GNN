"""Generalised advantage estimation.

One function, deliberately alone: it is pure math over a reward/value/done
sequence, shared by the PPO path and the tests that check it in isolation.
"""

import torch


def compute_gae(rewards, values, term, cut, boot, gamma, lam):
    n = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in range(n - 1, -1, -1):
        if cut[t] > 0 or t == n - 1:
            v_next = (1.0 - term[t]) * boot[t]
            carried = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        else:
            v_next = values[t + 1]
            carried = running

        delta = rewards[t] + gamma * v_next - values[t]
        running = delta + gamma * lam * carried
        adv[t] = running

    returns = adv + values
    return adv, returns
