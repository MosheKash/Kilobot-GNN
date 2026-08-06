"""The reward function: a graph snapshot in, per-robot reward out.

Every term lives here -- on-shape reward, packing, separation, potential-based
shaping, the belief-confidence bonus and the seed/wall find bonuses -- along
with `coverage`, the fraction-on-shape measure used for logging and success.
Weights come from Config; nothing here reads the environment directly.
"""

import torch

from belief import belief_conf

# node feature layout: P(0:2) H(2:4) |D|(4) dir_D(5:7) C(7) M(8:10) T(10:19)
POS_COLS = slice(0, 2)
DIST_COL = 4
NEAREST_COL = 7


def _nearest_on_shape_neighbor(node, edge_index, on_mask, default=1.0):
    # For each robot, the distance to its nearest IR neighbor that is on-shape.
    # Edges are undirected IR links, so we look at both endpoints. Robots with no
    # on-shape neighbor get `default` (large), which yields zero packing bonus.
    n = node.shape[0]
    pos = node[:, POS_COLS]
    out = torch.full((n,), float(default), device=node.device)
    if edge_index is None or edge_index.numel() == 0:
        return out
    ei = edge_index.long()
    s = torch.cat([ei[0], ei[1]])
    t = torch.cat([ei[1], ei[0]])
    keep = on_mask[s]
    s = s[keep]
    t = t[keep]
    if s.numel() == 0:
        return out
    dist = (pos[t] - pos[s]).norm(dim=1)
    acc = torch.full((n,), float("inf"), device=node.device)
    acc = acc.scatter_reduce(0, t, dist, reduce="amin", include_self=False)
    finite = ~torch.isinf(acc)
    out[finite] = acc[finite]
    return out


def _terms(node, cfg, edge_index=None):
    d = node[:, DIST_COL]
    on_image = d < cfg.tau_v

    # Off-shape penalty, bounded: linear near the boundary, saturating at -k_pos.
    off_penalty = -cfg.k_pos * torch.clamp((d - cfg.tau_v) / cfg.l_scale, min=0.0, max=1.0)

    # Privileged tracing bonus: an on-shape robot is rewarded for having an
    # on-shape neighbor at a good spacing. On a thin stroke the on-shape neighbors
    # lie along the curve, so this strings robots out along the stroke instead of
    # clumping them, and the bonus is collectable only by joining the on-shape set,
    # so the off-shape huddle earns nothing. On-shape status of neighbors is read
    # from their true distance feature (privileged at training; the actor never
    # sees it). With no edge_index, falls back to plain nearest-neighbor spacing.
    if edge_index is not None:
        c_on = _nearest_on_shape_neighbor(node, edge_index, on_image)
    else:
        c_on = node[:, NEAREST_COL]
    pack = cfg.r_pack * (1.0 - torch.clamp((c_on - cfg.tau_sep) / cfg.pack_range, min=0.0, max=1.0))

    r_pos = torch.where(on_image, cfg.r_on + pack, off_penalty)

    crowding = torch.clamp(cfg.tau_sep - node[:, NEAREST_COL], min=0.0)
    r_sep = -cfg.k_sep * crowding / cfg.tau_sep
    return on_image, r_pos, r_sep, pack


def compute_rewards(node, cfg, edge_index=None):
    _, r_pos, r_sep, _ = _terms(node, cfg, edge_index)
    return (r_pos + r_sep) * cfg.dt_fixed


def reward_components(node, cfg, edge_index=None):
    on_image, r_pos, r_sep, pack = _terms(node, cfg, edge_index)
    dt = cfg.dt_fixed
    on_base = torch.where(on_image, torch.full_like(pack, cfg.r_on), torch.zeros_like(pack)) * dt
    pack_on = torch.where(on_image, pack, torch.zeros_like(pack)) * dt
    off_pen = torch.where(on_image, torch.zeros_like(r_pos), r_pos) * dt
    return {
        "on_count": float(on_image.float().sum()),
        "on_bonus_sum": float(on_base.sum()),
        "pack_sum": float(pack_on.sum()),
        "off_pen_sum": float(off_pen.sum()),
        "sep_sum": float((r_sep * dt).sum()),
        "count": float(node.shape[0]),
    }


def coverage(node, cfg):
    d = node[:, DIST_COL]
    on_image = d < cfg.tau_v
    return on_image.float().mean()


def belief_confidence_bonus(node, belief, bonus_weight):
    # Localization scaffolding for the split actor: pay bonus_weight * conf_pos per
    # step for holding a collapsed pose belief, so visiting beacons becomes rewarding
    # before navigation itself pays. Meant to be annealed to zero by the run loop,
    # leaving the final objective unchanged. `belief` is the {local: particles} dict
    # for one arena; robots with no entry get zero.
    m = node.shape[0]
    extra = torch.zeros(m, device = node.device)
    if not belief:
        return extra
    idx = [l for l in belief if l < m]
    if not idx:
        return extra
    parts = torch.stack([belief[l] for l in idx])
    conf = belief_conf(parts).detach().to(node.device)
    extra[torch.tensor(idx, dtype = torch.long, device = node.device)] = conf
    return bonus_weight * extra


def seed_wall_find_reward(node, pending, seed_bonus, wall_penalty):
    # One-time nudge toward seeking landmark seeds over wall seeds. `pending`
    # is the {local: +1.0 or -1.0} dict the trainer's event classification set on the
    # previous tick (landmark, or wall -- wall being coarse one-axis contact rather
    # than a full fix); consumed and cleared
    # here so it pays out exactly once per event, not for as long as it happens to
    # sit there. Deliberately much smaller than the on-shape reward (see config.py)
    # -- this is scaffolding to bias exploration, not a reward for the task itself.
    m = node.shape[0]
    extra = torch.zeros(m, device = node.device)
    if not pending:
        return extra
    idx = [l for l in pending if l < m]
    for l in idx:
        extra[l] = seed_bonus if pending[l] > 0.0 else -wall_penalty
    pending.clear()
    return extra
