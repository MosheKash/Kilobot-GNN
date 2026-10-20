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


def separation_reward(node, cfg):
    # The anti-clumping term alone, times dt -- the one piece of the base
    # shape reward that reward_mode="target" keeps. Extracted from _terms so
    # the target-mode path and the coverage-mode path share one definition
    # instead of drifting apart.
    crowding = torch.clamp(cfg.tau_sep - node[:, NEAREST_COL], min=0.0)
    return (-cfg.k_sep * crowding / cfg.tau_sep) * cfg.dt_fixed


def target_kernel(dists, sigma):
    # Gaussian kernel over normalized own-target distance: 1.0 at the pixel,
    # exp(-1) at one sigma, effectively zero beyond ~3 sigma. This is the
    # accuracy shape the task metric itself has: the settle report counts
    # within-5/10/20-unit bands, and phase 159 measured the per-robot error
    # distribution as a spike inside 5 units with an unrelated flat tail --
    # what deserves reward is the spike, not the tail.
    return torch.exp(-(dists / sigma) ** 2)


def accuracy_income(dists, valid, weight, sigma, dt):
    # Per-step income for being at the robot's OWN assigned pixel (mode
    # "target"). Localization is rewarded instrumentally: a robot that has
    # not figured out where it is cannot find the pixel, so no separate
    # confidence bonus is needed (and phase 2/9's measured lesson was that
    # paying for confidence directly makes a confidently-wrong cloud
    # profitable). Zero where the target is unresolvable -- never NaN into
    # the reward.
    inc = torch.where(valid, target_kernel(dists, sigma), torch.zeros_like(dists))
    return weight * inc * dt


def settled_potential(dists, valid, sigma):
    # The shared arena potential behind global_shaping, SMOOTH: the mean
    # kernel over robots with resolvable targets. A threshold (fraction
    # within a radius) is flat almost everywhere -- it only moves at the
    # exact tick a robot crosses the bar, so a sender whose message improves
    # a neighbour from 12 to 6 units would earn nothing. The kernel has
    # gradient at every distance, which is what gives broadcast actions a
    # usable learning signal (docs/tuning.md phase 165).
    if not bool(valid.any()):
        return 0.0
    k = target_kernel(dists[valid], sigma)
    return float(k.mean())


def belief_errors(worker, k, node):
    """Per-robot distance between the belief filter's mean position and truth.

    e_i = |mean(particles_i)[:, 0:2] - node_i[:, 0:2]| in normalized units.
    Privileged like every reward quantity (the actor's observation never
    contains its true position); what it measures is localization
    CORRECTNESS, not self-reported confidence -- a confidently wrong cloud
    keeps e large and earns nothing, which is the property phases 2/9
    demanded after the conf bonus was measured to reward wrong convergence.

    Returns a float tensor of per-robot errors, NaN for robots with no
    belief entry yet (no gradient for them, never treated as zero).
    """
    m = node.shape[0]
    out = torch.full((m,), float("nan"), device=node.device)
    bel = (worker.belief.get(k, {}) if hasattr(worker, "belief") else None)
    if not bel:
        return out
    idx = [l for l in bel if l < m]
    if not idx:
        return out
    parts = torch.stack([bel[l] for l in idx]).to(node.device)
    mean_xy = parts[:, :, 0:2].mean(dim=1)
    true_xy = node[torch.tensor(idx, dtype=torch.long, device=node.device), 0:2]
    out[torch.tensor(idx, dtype=torch.long, device=node.device)] = (mean_xy - true_xy).norm(dim=1)
    return out


def honest_confidences(worker, k, node):
    """Per-robot honest beacon confidence: conf_pos * r over own particles.

    Exactly what actor_io's honest-beacon override would broadcast (and what
    message supervision targets) -- conf_pos from the position spread, r the
    heading concentration (identically ~1 under the zero-heading-noise
    filter, but computed rather than assumed). NaN for robots with no
    belief entry yet; callers must treat NaN as no term, never zero.
    """
    m = node.shape[0]
    out = torch.full((m,), float("nan"), device=node.device)
    bel = (worker.belief.get(k, {}) if hasattr(worker, "belief") else None)
    if not bel:
        return out
    idx = [l for l in bel if l < m]
    if not idx:
        return out
    parts = torch.stack([bel[l] for l in idx]).to(node.device)
    mx = parts[:, :, 0].mean(dim=1)
    my = parts[:, :, 1].mean(dim=1)
    var = ((parts[:, :, 0] - mx.unsqueeze(1)) ** 2).mean(dim=1) + \
          ((parts[:, :, 1] - my.unsqueeze(1)) ** 2).mean(dim=1)
    sin_m = torch.sin(parts[:, :, 2]).mean(dim=1)
    cos_m = torch.cos(parts[:, :, 2]).mean(dim=1)
    r = torch.sqrt(sin_m * sin_m + cos_m * cos_m)
    out[torch.tensor(idx, dtype=torch.long, device=node.device)] = torch.exp(-var / 0.02) * r
    return out


def beacon_overclaim_penalty(node, honest, weight, dt):
    """Charged per tick while a robot's BROADCAST conf exceeds its own honest
    filter confidence: -weight * relu(claimed - honest) * dt.

    Why a reward term rather than supervision: the measured failure mode of
    the first transmission run. PPO pushed the learned conf gate to 2.4x the
    honest rate (24% of decisions broadcasting past PEER_CONF_MIN against an
    honest 10%), and 14% of beacons were OVERCONFIDENT -- sent by robots
    whose own filter had not localized, so their position claims were ~50
    units wrong while claiming conf ~0.9. The receiver's fusion picks the
    MOST confident valid row (argmax), so the liar always wins the slot, and
    the resulting wrong range-rings are what dropped the with-comms eval 0.18
    below the no-comms one. Action-space supervision cannot hold the slot
    down: its gradient vanishes at tanh saturation (tanh'(2.0) ~ 0.02)
    exactly where the conf sits, while PPO's advantage gradient does not --
    so the honesty constraint has to ride the same advantage channel. The
    claimed value is read from node's transmission column (slot 2), the
    honest one from the sender's own particles; both are privileged to the
    reward, invisible to the actor.
    """
    m = node.shape[0]
    out = torch.zeros(m, device=node.device)
    if honest is None or weight <= 0.0:
        return out
    valid = ~torch.isnan(honest)
    if not bool(valid.any()):
        return out
    claimed = node[:, 12]
    over = (claimed - honest).clamp(min=0.0)
    out = -weight * torch.where(valid, over, torch.zeros_like(over)) * dt
    return out


def arena_potential(dists, valid_t, errs, valid_e, cfg):
    """The shared arena potential for sender credit, in two bands.

    pot = mean_i kernel(d_i, sigma_g) + w_e * mean_i kernel(e_i, sigma_e),
    each band over the robots where that quantity is defined. The settled
    band is the phase-165 kernel and alone it is structurally blind to the
    one improvement messaging produces: the accuracy kernel (sigma 0.05) is
    exp(-1) at one sigma and ~0 beyond three, so a robot at 50 units from
    its pixel moves it by nothing, and the whole first ~3000 ticks of the
    settle curve -- the localization phase a helpful beacon actually speeds
    up -- pay a sender exactly zero (phase 163's curve: settled<5u is 0.016
    at tick 2000, 0.63 at 8000). The belief-error band fixes that: e shrinks
    from the first landmark fix onward, including mid-arena, and shrinks
    FURTHER when a receiver fuses a true peer beacon -- while a garbage
    broadcast with high conf grows e and drops the potential, which is the
    gradient that makes honesty the profitable protocol.
    """
    pot = 0.0
    if valid_t is not None and bool(valid_t.any()):
        pot = pot + float(target_kernel(dists[valid_t], cfg.global_potential_sigma).mean())
    w_e = float(getattr(cfg, "belief_potential_weight", 1.0))
    if w_e > 0.0 and errs is not None:
        e_valid = ~torch.isnan(errs) if valid_e is None else valid_e
        if bool(e_valid.any()):
            pot = pot + w_e * float(target_kernel(
                torch.where(e_valid, errs, torch.zeros_like(errs))[e_valid],
                getattr(cfg, "belief_potential_sigma", 0.20)).mean())
    return pot


def assigned_target_dists(worker, k, node, cfg):
    """Per-robot normalized distance to its OWN assigned target point.

    The RL fine-tune's reward geometry (docs/tuning.md phase 164): what the
    settle metric scores is distance to the point observation.ensure_target
    assigned this robot, so that is what the reward should shape. Resolved
    through the same shared ensure_target cache the actor's target-relative
    proprioception uses, so reward and observation can never disagree about
    which point a robot was sent to. Positions come from the privileged
    critic snapshot (node[:, 0:2], normalized) -- legitimate here, rewards
    are allowed privileged information, and the actor never sees this value
    except through the return.

    Returns a float tensor of normalized distances, NaN for robots with no
    resolvable target (no formation pool, or the arena's image_id unknown) --
    callers must treat NaN as "no gradient for this robot", never as zero.
    """
    pool = getattr(cfg, "_oracle_formation_pool", None)
    image_id = worker.image_id.get(k)
    m = node.shape[0]
    out = torch.full((m,), float("nan"))
    if not pool or image_id is None:
        return out
    from observation import ensure_target
    from belief import ARENA_HALF
    pos = node[:, 0:2] * ARENA_HALF
    for l in range(m):
        tgt = ensure_target(worker, k, l, image_id, pool)
        out[l] = float((pos[l] - tgt).norm()) / ARENA_HALF
    return out


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
