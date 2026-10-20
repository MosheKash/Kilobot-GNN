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


def architecture_meta(actor, meta=None):
    """The architecture switches a checkpoint must carry to be loadable and
    to mean the same FUNCTION when it is loaded.

    save_checkpoint used to record only {z, action_size}: for the original
    actor that was enough, but the split-observation actor now carries
    opt-in heads (arrived, state, wall), a turn-anchor input (+2 columns on
    up1) and the oracle-form forward switch -- the last of which is NOT in
    the state_dict at all (it reuses existing heads, so no tensor can tell
    you it was on). A checkpoint saved without this meta loads into a
    differently-built actor and silently computes a different function --
    the exact failure phase 160's save_actor/meta work existed to prevent
    on the BC side, and which the eval tool hit on the first RL checkpoint
    (up1 42 vs 40, unexpected head keys).

    The switches are read off the actor's own attributes -- the same source
    of truth the deployed forward uses. Existing keys in `meta` are kept.
    """
    out = dict(meta) if meta else {}
    # use_turn_anchor is not stored as an actor attribute (unlike
    # use_steer_feature/use_oracle_head); it is fully determined by the
    # up1 input width -- AFTER subtracting every optional appended block.
    # The first version inferred from the raw width (== TC + ODOM + 2), which
    # misreported use_turn_anchor=False for any T2-era actor whose up1 also
    # carries the homing columns: the turn anchor was genuinely there, and
    # every consumer of that meta (the eval's arch auto-detect) then built a
    # narrower actor than the checkpoint. Subtract exactly what the actor
    # itself says is appended (homing_proprio), by attribute.
    from kilobot_gnn import SPLIT_TC_SIZE, SPLIT_ODOM_SIZE, HOMING_PROP_SIZE
    up1_in = int(actor.up1.in_features) if hasattr(actor, "up1") else 0
    base_in = up1_in - (HOMING_PROP_SIZE if getattr(actor, "homing_proprio", False) else 0)
    out.update({
        "z": Z,
        "action_size": ACTION_SIZE,
        "activation": getattr(actor, "activation_name", "relu"),
        "use_arrived_head": getattr(actor, "head_arrived", None) is not None,
        "use_state_head": getattr(actor, "head_state", None) is not None,
        "use_wall_head": getattr(actor, "head_wall", None) is not None,
        "use_turn_anchor": bool(base_in == SPLIT_TC_SIZE + SPLIT_ODOM_SIZE + 2),
        "use_steer_feature": bool(getattr(actor, "use_steer_feature", False)),
        "use_oracle_head": bool(getattr(actor, "use_oracle_head", False)),
        "msg_head_prop_input": bool(getattr(actor, "msg_head_prop_input", False)),
        "homing_proprio": bool(getattr(actor, "homing_proprio", False)),
        "oracle_residual": float(getattr(actor, "oracle_residual", 0.0)),
        "oracle_residual_turn": float(getattr(actor, "oracle_residual_turn", 0.0)),
        "up1_in_features": up1_in,
    })
    return out


def save_checkpoint(path, iteration, policy, critic, actor_opt, critic_opt):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"iteration": iteration,
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "meta": architecture_meta(policy.actor)}, tmp)
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
            "meta": architecture_meta(policy.actor)}
    if iteration is not None:
        blob["iteration"] = int(iteration)
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_for_eval(path, policy, device):
    """Strict eval load, with ONE tolerance: the homing-width zero-extension.

    load_state_dict's default strictness is what caught the turn-anchor
    mislabel in the T2 checkpoints (the meta lied, the tensors didn't match
    the built actor) -- keep that property. The tolerated exception mirrors
    warm_start_load's rule exactly: an actor built with homing_proprio (up1
    +3 columns) may load a checkpoint whose up1 is 3 columns narrower, by
    zero-extending -- behavior-identical at eval because the homing columns
    are identically zero until a referral is fitted (observation.py's
    homing_readout), so 0 * w = 0 and the network computes the same function
    the narrow checkpoint did. Everything else mismatches LOUDLY, as before.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    from kilobot_gnn import HOMING_PROP_SIZE
    if "policy" in blob:
        policy.load_state_dict(blob["policy"])
    elif "actor" in blob:
        state = dict(blob["actor"])
        actor = policy.actor
        if "up1.weight" in state and hasattr(actor, "up1"):
            ckpt_in = int(state["up1.weight"].shape[1])
            expected_in = int(actor.up1.in_features)
            d = expected_in - ckpt_in
            if d == HOMING_PROP_SIZE:
                w = state["up1.weight"]
                ext = torch.zeros(int(w.shape[0]), expected_in, device = device,
                                  dtype = w.dtype)
                ext[:, :ckpt_in] = w.to(device)
                state["up1.weight"] = ext
                print("eval load: up1 zero-extended %d -> %d inputs (homing "
                      "columns; behavior-identical, the readout reads zeros "
                      "until a referral fits)" % (ckpt_in, expected_in))
            elif d != 0:
                raise RuntimeError(
                    "eval load %s: up1 width mismatch (checkpoint %d cols vs "
                    "actor %d cols) -- build the actor with the flags the "
                    "checkpoint was trained with (--homing-proprio), or "
                    "pass them to match (refusing anything beyond the exact "
                    "homing zero-extension)" % (path, ckpt_in, expected_in))
        actor.load_state_dict(state)
        if "log_std" in blob:
            with torch.no_grad():
                policy.log_std.copy_(blob["log_std"].to(policy.log_std.device))
    else:
        raise RuntimeError("unrecognized weights file (no 'policy' or 'actor'): %s" % path)
    return int(blob.get("iteration", 0))
