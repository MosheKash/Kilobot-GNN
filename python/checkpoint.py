"""Saving and loading: full training checkpoints and bare actor exports.

save_checkpoint/load_checkpoint round-trip everything needed to resume (policy,
critic, both optimizers, iteration). export_actor/load_for_eval handle the
smaller artifact used to warm-start or evaluate an actor on its own. Writes are
atomic (temp file then os.replace), so an interruption cannot leave a corrupted
file behind.
"""

import os
import torch
from kilobot_gnn import Z
from policy import ACTION_SIZE


def save_checkpoint(path, iteration, policy, critic, actor_opt, critic_opt):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"iteration": iteration,
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "meta": {"z": Z, "action_size": ACTION_SIZE}}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, critic, actor_opt, critic_opt, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    if critic is not None and "critic" in ckpt:
        critic.load_state_dict(ckpt["critic"])
    if actor_opt is not None and "actor_opt" in ckpt:
        actor_opt.load_state_dict(ckpt["actor_opt"])
        _opt_to_device(actor_opt, device)
    if critic_opt is not None and "critic_opt" in ckpt:
        critic_opt.load_state_dict(ckpt["critic_opt"])
        _opt_to_device(critic_opt, device)
    return int(ckpt.get("iteration", 0))


def _opt_to_device(opt, device):
    for state in opt.state.values():
        for k in state:
            v = state[k]
            if torch.is_tensor(v):
                state[k] = v.to(device)


def export_actor(path, policy, iteration = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    blob = {"actor": policy.actor.state_dict(),
            "log_std": policy.log_std.detach().cpu(),
            "meta": {"z": Z, "action_size": ACTION_SIZE}}
    if iteration is not None:
        blob["iteration"] = int(iteration)
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_for_eval(path, policy, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    if "policy" in blob:
        policy.load_state_dict(blob["policy"])
    elif "actor" in blob:
        policy.actor.load_state_dict(blob["actor"])
        if "log_std" in blob:
            with torch.no_grad():
                policy.log_std.copy_(blob["log_std"].to(policy.log_std.device))
    else:
        raise RuntimeError("unrecognized weights file (no 'policy' or 'actor'): %s" % path)
    return int(blob.get("iteration", 0))
