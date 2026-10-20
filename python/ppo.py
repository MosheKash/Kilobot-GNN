"""The PPO update: a filled RolloutBuffer in, gradients applied, stats out.

Clipped surrogate objective, value regression and entropy bonus, plus the
diagnostics an update can report about itself (gradient norms, explained
variance, motor saturation). Advantage estimation is gae.py; the loop that
calls this is trainer.py.
"""

import torch
import torch.nn as nn
from kilobot_gnn import MESSAGE_SIZE, DB_ROW_SIZE, DB_CAPACITY


def _stack_decisions(decisions):
    n = len(decisions)
    db_rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE)
    db_valid = torch.zeros(n, DB_CAPACITY, dtype=torch.bool)
    for i, d in enumerate(decisions):
        prev = d["prev_db"]
        if prev is not None:
            m = prev.shape[0]
            if m > 0:
                db_rows[i, :m] = prev
                db_valid[i, :m] = True
    data = {
        "z": torch.stack([d["z"] for d in decisions]),
        "seed": torch.stack([d["seed"] for d in decisions]),
        "tx": torch.stack([d["transmission"] for d in decisions]),
        "action": torch.stack([d["action"] for d in decisions]),
        "old_lp": torch.stack([d["old_log_prob"].view(()) for d in decisions]),
        "db_rows": db_rows,
        "db_valid": db_valid
    }
    if decisions[0].get("prev_hidden") is not None:
        data["prev_hidden"] = torch.stack([d["prev_hidden"] for d in decisions])
        data["prop"] = torch.stack([d["prop"] for d in decisions])
    return data


def _grad_norm(*tensors):
    s = 0.0
    for t in tensors:
        if t is not None:
            s += float((t.detach() ** 2).sum())
    return s ** 0.5


def _motor_diagnostics(policy):
    # Compare the gradient reaching the motor output against the message output, and the
    # saturation of the motor pre-activations. Works for both the DeepSet actor (shared
    # rho_2 rows) and the GRU actor (separate motor/message heads).
    actor = policy.actor
    if hasattr(actor, "deepset"):
        rho2 = actor.deepset.rho_2
        wg = rho2.weight.grad
        bg = rho2.bias.grad
        msg_norm = _grad_norm(wg[:MESSAGE_SIZE] if wg is not None else None,
                              bg[:MESSAGE_SIZE] if bg is not None else None)
        if getattr(actor, "direct_motor", False):
            dh = actor.direct_head
            motor_norm = _grad_norm(dh.weight.grad, dh.bias.grad)
        else:
            motor_norm = _grad_norm(wg[MESSAGE_SIZE:] if wg is not None else None,
                                    bg[MESSAGE_SIZE:] if bg is not None else None)
    else:
        motor_norm = _grad_norm(actor.head_motor.weight.grad, actor.head_motor.bias.grad)
        msg_norm = _grad_norm(actor.head_msg.weight.grad, actor.head_msg.bias.grad)
    pre = getattr(actor, "_motor_preact", None)
    if pre is not None and pre.numel() > 0:
        sat = float((pre.abs() >= 3.0).float().mean())   # Hardsigmoid is flat for |x|>=3
        absmean = float(pre.abs().mean())
    else:
        sat = 0.0
        absmean = 0.0
    return motor_norm, msg_norm, sat, absmean


def _motor_param_vec(policy):
    # Flat snapshot of the motor-producing and message-producing parameters, for drift.
    actor = policy.actor
    if hasattr(actor, "deepset"):
        rho2 = actor.deepset.rho_2
        msg = torch.cat([rho2.weight[:MESSAGE_SIZE].reshape(-1), rho2.bias[:MESSAGE_SIZE].reshape(-1)])
        if getattr(actor, "direct_motor", False):
            dh = actor.direct_head
            motor = torch.cat([dh.weight.reshape(-1), dh.bias.reshape(-1)])
        else:
            motor = torch.cat([rho2.weight[MESSAGE_SIZE:].reshape(-1), rho2.bias[MESSAGE_SIZE:].reshape(-1)])
    else:
        motor = torch.cat([actor.head_motor.weight.reshape(-1), actor.head_motor.bias.reshape(-1)])
        msg = torch.cat([actor.head_msg.weight.reshape(-1), actor.head_msg.bias.reshape(-1)])
    return motor.detach().clone(), msg.detach().clone()


def ppo_update(policy, critic, actor_opt, critic_opt, buffer, cfg):
    buffer.compute_returns(critic)
    device = next(policy.parameters()).device

    raw_advantages = buffer.decision_advantages()
    adv_mean = float(raw_advantages.mean()) if raw_advantages.numel() > 0 else 0.0
    adv_std = float(raw_advantages.std()) if raw_advantages.numel() > 1 else 0.0
    advantages = raw_advantages
    if advantages.numel() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.to(device)

    logs = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0,
            "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
            "motor_grad_norm": 0.0, "msg_grad_norm": 0.0,
            "motor_saturation": 0.0, "motor_preact_absmean": 0.0,
            "motor_out_mean": 0.0, "motor_preact_mean": 0.0,
            "motor_param_drift": 0.0, "msg_param_drift": 0.0, "log_std_motor": 0.0,
            "explained_variance": 0.0, "adv_mean": adv_mean, "adv_std": adv_std}

    motor_before, msg_before = _motor_param_vec(policy)

    n_dec = len(buffer.decisions)
    indices = torch.arange(n_dec, device=device)
    data = _stack_decisions(buffer.decisions) if n_dec > 0 else None
    if data is not None:
        data = {k: v.to(device) for k, v in data.items()}

    kl_sum = 0.0
    clip_sum = 0.0
    item_count = 0

    for _ in range(cfg.ppo_epochs):
        perm = indices[torch.randperm(n_dec, device=device)] if n_dec > 0 else indices
        for start in range(0, n_dec, cfg.minibatch):
            chunk = perm[start:start + cfg.minibatch]
            actor_loss, entropy, kl, clipped = _actor_loss(policy, data, advantages, chunk, cfg)

            actor_opt.zero_grad()
            actor_loss.backward()
            mgn, sgn, sat, pam = _motor_diagnostics(policy)
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            actor_opt.step()
            policy.clamp_log_std()

            logs["actor_loss"] = float(actor_loss.detach())
            logs["entropy"] = float(entropy.detach())
            logs["actor_grad_norm"] = float(grad_norm)
            logs["motor_grad_norm"] = mgn
            logs["msg_grad_norm"] = sgn
            logs["motor_saturation"] = sat
            logs["motor_preact_absmean"] = pam
            kl_sum = kl_sum + kl
            clip_sum = clip_sum + clipped
            item_count = item_count + len(chunk)

        critic_loss, critic_grad_norm = _critic_update(critic, critic_opt, buffer, cfg)
        logs["critic_loss"] = critic_loss
        logs["critic_grad_norm"] = critic_grad_norm

    if item_count > 0:
        logs["approx_kl"] = kl_sum / item_count
        logs["clip_fraction"] = clip_sum / item_count

    logs["explained_variance"] = _explained_variance(critic, buffer)
    logs["log_std_mean"] = float(policy.log_std.detach().mean())
    logs["std_mean"] = float(policy.log_std.detach().exp().mean())
    logs["mean_value"] = _mean_cat(buffer.values)
    logs["mean_return"] = _mean_cat(buffer.returns)
    logs["return_std"] = _std_cat(buffer.returns)
    # EFFECTIVE per-kind std (after the clamp stack, including any
    # KILOBOT_MSG_LOG_STD_MAX message cap) -- what actually shaped the
    # samples this iteration, unlike the raw parameter means above.
    with torch.no_grad():
        std_eff = policy._std()
        logs["std_msg"] = float(std_eff[:MESSAGE_SIZE].mean())
        logs["std_motor_eff"] = float(std_eff[MESSAGE_SIZE:].mean())

    # Parameter drift over this iteration: how far the motor-producing params actually
    # moved under the gradient, vs the message params of the same layer. A healthy motor
    # gradient with near-zero motor drift means the optimizer is not integrating it.
    motor_after, msg_after = _motor_param_vec(policy)
    logs["motor_param_drift"] = float((motor_after - motor_before).norm())
    logs["msg_param_drift"] = float((msg_after - msg_before).norm())
    logs["log_std_motor"] = float(policy.log_std.detach()[MESSAGE_SIZE:].exp().mean())
    pre = getattr(policy.actor, "_motor_preact", None)
    if pre is not None and pre.numel() > 0:
        logs["motor_preact_mean"] = float(pre.mean())
        logs["motor_out_mean"] = float((0.5 * (torch.tanh(pre) + 1.0)).mean())
    return logs


def _actor_loss(policy, data, advantages, chunk, cfg):
    z = data["z"][chunk]
    seed = data["seed"][chunk]
    tx = data["tx"][chunk]
    action = data["action"][chunk]
    old_lp = data["old_lp"][chunk]
    adv = advantages[chunk]

    mean = None
    if getattr(cfg, "actor_type", None) == "gru_split_observation":
        h_prev = data["prev_hidden"][chunk]
        prop = data["prop"][chunk]
        log_prob, entropy, mean = policy.evaluate_batch_split(tx, prop, h_prev, action)
    elif "prev_hidden" in data:
        msg = tx[:, :MESSAGE_SIZE]
        h_prev = data["prev_hidden"][chunk]
        prop = data["prop"][chunk]
        log_prob, entropy = policy.evaluate_batch_gru(z, seed, msg, prop, h_prev, action)
    else:
        msg = tx[:, :MESSAGE_SIZE]
        sender = tx[:, MESSAGE_SIZE]
        db_rows = data["db_rows"][chunk]
        db_valid = data["db_valid"][chunk]
        log_prob, entropy = policy.evaluate_batch(z, seed, msg, sender, db_rows, db_valid, action)
    log_ratio = log_prob - old_lp
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv

    entropy_mean = entropy.mean()
    loss = -torch.min(unclipped, clipped).mean() - cfg.entropy_coef * entropy_mean

    # Auxiliary supervision of the learned beacon (the transmission phase).
    # The message head must broadcast its own believed position and
    # confidence -- slots 0-2, exactly what actor_io's honest-beacon override
    # force-writes when belief_comms_learned is off. Without a prior the
    # beacon can never switch on: the receiver-side fusion gate needs the
    # squashed conf past PEER_CONF_MIN=0.85, i.e. a pre-tanh mean above
    # ~1.73, unreachable from a quiet start (entropy explores amplitude, not
    # offset). The target comes from the decision's OWN stored prop (x_m,
    # y_m at [8]/[9], conf_pos*r at [12]*[13] -- the same columns the
    # override reads). The head is evaluated through
    # kilobot_gnn.message_mean_headonly, so the prior's gradient reaches
    # head_msg and STOPS: a trunk-connected supervision was measured to
    # wreck the warm start within five iterations (kl 7/iter, belief error
    # 0.62 -> 0.82) because the trunk is the motor policy. The weight is a
    # prior, not a cage (the oracle_residual pattern): the PPO surrogate's
    # own, trust-region-clipped gradient still shapes the protocol through
    # everything.
    sup_w = float(getattr(cfg, "msg_supervision", 0.0))
    if (sup_w > 0.0 and getattr(cfg, "actor_type", None) == "gru_split_observation"
            and "prop" in data):
        from kilobot_gnn import message_mean_headonly
        prop = data["prop"][chunk]
        sup_mean = message_mean_headonly(policy.actor, tx, prop, h_prev)
        # Supervised in ACTION space, deliberately NOT pre-tanh: the readout
        # initialization already sits near the honest beacon, and an atanh
        # target (up to 3.8 for conf ~ 0.999) would systematically grind the
        # conf mean ~1.8 units uphill through a std-0.05 Gaussian -- measured
        # as kl 1->10 per iteration before this switch, the ratio exploding
        # on exactly the slot that was already right. In action space the
        # gradient decays with tanh saturation, so the prior corrects real
        # errors and leaves saturated slots alone.
        tgt = torch.stack([
            prop[:, 8].clamp(-0.999, 0.999),
            prop[:, 9].clamp(-0.999, 0.999),
            (prop[:, 12] * prop[:, 13]).clamp(0.0, 0.999)], dim=1)
        msg_loss = ((torch.tanh(sup_mean[:, :3]) - tgt) ** 2).mean()
        loss = loss + sup_w * msg_loss

    kl_total = float((ratio - 1.0 - log_ratio).detach().sum())
    out = (ratio > 1.0 + cfg.clip) | (ratio < 1.0 - cfg.clip)
    clip_total = float(out.float().detach().sum())
    return loss, entropy_mean, kl_total, clip_total


def _critic_update(critic, critic_opt, buffer, cfg):
    # Chunked critic step. The critic runs over every step's graph, which can be
    # too large for one GPU forward, so we process cfg.critic_chunk_steps graphs
    # at a time and back-propagate per chunk. Each chunk's loss is the chunk's
    # summed squared error divided by the total node count, so the accumulated
    # gradient equals the gradient of the full mean squared error. With LayerNorm
    # and per-graph pooling there is no cross-chunk coupling, so this matches the
    # single-pass result exactly.
    device = next(critic.parameters()).device
    total_n = sum(int(r.shape[0]) for r in buffer.returns)
    if total_n == 0:
        return 0.0, 0.0
    chunk = getattr(cfg, "critic_chunk_steps", 0)
    critic_opt.zero_grad()
    total_loss = 0.0
    for x, edge_attr, edge_index, z, batch, targets in buffer.critic_chunks(chunk):
        values = critic(x.to(device), edge_attr.to(device), edge_index.to(device),
                        z.to(device), batch.to(device)).squeeze(-1)
        loss = ((values - targets.to(device)) ** 2).sum() / total_n
        loss.backward()
        total_loss = total_loss + float(loss.detach())
    grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
    critic_opt.step()
    return total_loss, float(grad_norm)


def _explained_variance(critic, buffer):
    if len(buffer.returns) == 0:
        return 0.0
    device = next(critic.parameters()).device
    chunk = getattr(buffer.cfg, "critic_chunk_steps", 0)
    vals = []
    with torch.no_grad():
        for x, edge_attr, edge_index, z, batch, _t in buffer.critic_chunks(chunk):
            v = critic(x.to(device), edge_attr.to(device), edge_index.to(device),
                       z.to(device), batch.to(device)).squeeze(-1).cpu()
            vals.append(v)
    values = torch.cat(vals) if vals else torch.zeros(0)
    targets = torch.cat(buffer.returns, dim=0)
    var_y = float(targets.var())
    if var_y <= 0.0:
        return 0.0
    return float(1.0 - (targets - values).var() / var_y)


def _mean_cat(tensors):
    if len(tensors) == 0:
        return 0.0
    return float(torch.cat(tensors).mean())

def _std_cat(tensor_list):
    if len(tensor_list) == 0:
        return 0.0
    cat = torch.cat(tensor_list, dim=0)
    return float(cat.std()) if cat.numel() > 1 else 0.0

