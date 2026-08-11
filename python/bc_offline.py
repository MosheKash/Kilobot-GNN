"""bc_offline.py -- behaviour cloning from recorded tapes, as sequences.

The online path (bc.py) interleaves collection and fitting: every iteration
collects a fresh rollout window and fits single decisions against a hidden state
that was cached during collection. That has two structural problems for a
recurrent actor, both independent of any hyperparameter:

  1. The stored h_prev came from an OLDER actor. The network is asked to map
     (observation, someone else's hidden state) -> action, while at deployment
     it must map (observation, its OWN hidden state) -> action. The two agree
     only once the actor has stopped changing.
  2. Nothing in the loss ever asks the GRU to CARRY anything. Each sample is one
     step; the gradient never flows through the recurrence, so state that must
     survive many ticks -- which wall a robot is following, that it is in
     wall_following at all -- is never trained for, only hoped for.

Here the same oracle data is replayed as ordered per-robot sequences from a cold
start (h = 0), with truncated backpropagation through time, which is what makes
the recurrence trainable and what makes training-time hidden states the ones the
actor will actually have. The data is a tape (val_tape.py's format, recorded by
tools/record_tape.py), so an epoch is pure GPU compute with no simulation, and a
run is reproducible from the tape files and a seed alone.

The teacher, the observations, and the targets are exactly the online path's:
tapes are recorded through actor_io.act with simple_oracle driving.

usage:
  python bc_offline.py ../results/bc_v2/run1 \
      --train-tape ../results/bc_v2/tape_train.pt \
      --val-tape ../results/bc_v2/tape_val.pt --epochs 30
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bc_replay import BC_STATES, ARRIVED
from kilobot_gnn import (MESSAGE_SIZE, SPLIT_WALL_OFFSET, WALL_SIZE, build_actor,
                         split_motor_from_head)
from policy import squash_action


# ─── tapes ────────────────────────────────────────────────────────────────────

def load_tape(path, device = "cpu"):
    """A recorded tape, kept in float16 on `device`, cast per batch.

    Padded (T, R) storage wastes a little memory on robots whose sequences are
    shorter than the longest, but keeps every access a plain slice. float16
    halves that; every consumer below casts the slice it takes to float32
    before it reaches the network, so nothing trains in half precision.
    """
    blob = torch.load(path, map_location = "cpu", weights_only = False)
    out = {}
    for k in ("tc", "prop", "tgt", "arrived"):
        out[k] = blob[k].to(device = device, dtype = torch.float16)
    for k in ("state", "valid", "arrived_valid"):
        out[k] = blob[k].to(device)
    return out


def concat_tapes(tapes):
    """Several tapes as one, side by side, padded to the longest sequence.

    Sequences are independent -- each is one robot's episode, replayed from a
    cold start -- so combining tapes is a concatenation along the robot axis and
    nothing else. This is what a DAgger round consumes: the original
    oracle-driven tape plus the actor-driven, oracle-labelled ones, trained on
    together rather than sequentially, so the earlier data cannot be forgotten.
    """
    if len(tapes) == 1:
        return tapes[0]
    T = max(t["valid"].shape[0] for t in tapes)
    out = {}
    for key in ("tc", "prop", "tgt", "arrived", "state", "valid", "arrived_valid"):
        parts = []
        for t in tapes:
            x = t[key]
            pad = T - x.shape[0]
            if pad > 0:
                shape = (pad,) + tuple(x.shape[1:])
                fill = -1 if key == "state" else 0
                x = torch.cat([x, torch.full(shape, fill, dtype = x.dtype, device = x.device)], dim = 0)
            parts.append(x)
        out[key] = torch.cat(parts, dim = 1)
    return out


def wall_labels(tape, wall_offset = SPLIT_WALL_OFFSET, n_walls = WALL_SIZE):
    """Which wall each decision follows, derived from the tape, not re-recorded.

    Tc carries the wall band that fired on this decision and zeros otherwise, so
    the wall a robot is currently following is the last nonzero one -- exactly
    what simple_oracle latches into `simple_wall_name` when go_north sees a wall,
    and never changes afterwards. Forward-filling that along each sequence
    reconstructs the label for free. -1 until a robot has touched anything,
    which the loss drops rather than guessing at.
    """
    wall = tape["tc"][:, :, wall_offset:wall_offset + n_walls].float()
    hit = wall.sum(dim = 2) > 0
    idx = wall.argmax(dim = 2)
    T, R = hit.shape
    out = torch.full((T, R), -1, dtype = torch.long, device = wall.device)
    cur = torch.full((R,), -1, dtype = torch.long, device = wall.device)
    for t in range(T):
        cur = torch.where(hit[t], idx[t], cur)
        out[t] = cur
    return out


def wall_labels_from_targets(tape, min_votes = 3):
    """The wall the ORACLE latched, recovered from its own commands.

    `wall_labels` above reconstructs which wall was seen most recently, which is
    NOT what simple_oracle steers by: it latches `simple_wall_name` once, at the
    go_north -> turning transition, and never updates it. On the teacher's own
    trajectories the two coincide (a robot localizes at its first corner and only
    ever deals with one wall), which is why supervising the recent one looked
    fine off-policy and left `wall_following` error pinned at 0.28 on-policy --
    the clone's looser trajectories touch several walls, and then the two labels
    differ by a 90-degree tangent.

    The latched wall is recoverable with no new recording, because the command
    reveals it. In wall_following the oracle emits
        L = (0.9 - 0.35*turn) * s,  R = (0.9 + 0.35*turn) * s,
        turn = sin(theta_tangent - theta_heading)
    and the heading is in prop. So the implied tangent is
    theta_heading + asin(turn), which -- measured -- lands within 5 degrees of a
    cardinal axis for 100% of oracle-driven decisions and 90% of the clone's own.
    Which cardinal it is IS the latched wall. It is constant for a robot's whole
    episode, so a per-sequence vote recovers one label for every decision that
    robot ever makes, including the ones before it saw any wall at all.

    Returns (T, R) longs in WALL_NAMES order, -1 where too few usable votes.
    """
    import math
    v = tape["valid"]
    st = tape["state"]
    prop = tape["prop"].float()
    tgt = tape["tgt"].float()
    T, R = v.shape
    wf = v & (st == BC_STATES.index("wall_following"))
    L = tgt[..., 0]
    Rw = tgt[..., 1]
    tot = (L + Rw).clamp(min = 1e-6)
    turn = -(L - Rw) * 1.8 / (0.7 * tot)
    s = tot / 1.8
    # unclipped commands only: at the clip the differential no longer carries the
    # steering angle, and the reacquire branch emits a constant +-0.45 that says
    # nothing about the tangent either
    usable = wf & (L > 0.02) & (L < 0.98) & (Rw > 0.02) & (Rw < 0.98) & (s > 0.1) \
             & (turn.abs() < 0.9) & ((turn.abs() - 0.45).abs() > 0.02)
    th_h = torch.atan2(prop[..., 10], prop[..., 11])
    th_g = th_h + torch.asin(turn.clamp(-1, 1))
    k = torch.round(th_g / (math.pi / 2)).long() % 4
    # k counts quarter turns CCW from +x; WALL_TANGENT is north(1,0) east(0,-1)
    # south(-1,0) west(0,1), i.e. angles 0, -pi/2, pi, +pi/2
    to_wall = torch.tensor([0, 3, 2, 1], device = k.device)
    wall = to_wall[k.clamp(0, 3)]
    out = torch.full((T, R), -1, dtype = torch.long, device = v.device)
    for j in range(R):
        u = usable[:, j]
        if int(u.sum()) < min_votes:
            continue
        votes = torch.bincount(wall[u, j], minlength = 4)
        if int(votes.max()) < min_votes:
            continue
        out[:, j] = int(votes.argmax())
    return out


def alignment_weights(tape, n_bins = 5, cap = 6.0):
    """Per-decision weights that stop the aligned majority drowning the rest.

    Measured, and it is the whole difficulty of cloning this teacher: the oracle
    is a stabilising controller, so on ITS trajectories 99.9% of wall_following
    decisions have the robot within 5 degrees of the wall tangent. The clone
    starts misaligned and spends two thirds of its time beyond that, where the
    demonstration set is nearly empty and its error is 2-6x larger. State
    balancing does not touch this -- it is a skew INSIDE wall_following and
    navigating, not between states.

    The misalignment is recoverable from the command itself (the same inversion
    wall_labels_from_targets uses), so this needs no new signal: bin by it,
    weight by inverse frequency, normalise to mean 1 so the loss scale and the
    effective learning rate are unchanged. Rows where the command does not
    reveal an angle -- clipped, or the hard reacquire branch -- get weight 1.
    """
    import math
    v = tape["valid"]
    st = tape["state"]
    tgt = tape["tgt"].float()
    steer = [BC_STATES.index("wall_following"), BC_STATES.index("navigating")]
    L = tgt[..., 0]
    R = tgt[..., 1]
    tot = (L + R).clamp(min = 1e-6)
    turn = (-(L - R) * 1.8 / (0.7 * tot)).clamp(-1, 1)
    usable = v & (L > 0.02) & (L < 0.98) & (R > 0.02) & (R < 0.98) \
             & ((turn.abs() - 0.45).abs() > 0.02)
    usable = usable & sum((st == i) for i in steer).bool()
    delta = torch.asin(turn).abs() * 180.0 / math.pi
    edges = torch.tensor([0.0, 5.0, 10.0, 20.0, 40.0, 181.0], device = v.device)
    w = torch.ones_like(delta)
    idx = torch.bucketize(delta, edges[1:-1])
    counts = torch.zeros(n_bins, device = v.device)
    for b in range(n_bins):
        counts[b] = float((usable & (idx == b)).sum())
    live = counts > 0
    if int(live.sum()) > 1:
        inv = torch.where(live, 1.0 / counts.clamp(min = 1), torch.zeros_like(counts))
        mean_w = float((inv * counts).sum() / counts[live].sum())
        inv = (inv / max(mean_w, 1e-12)).clamp(max = cap)
        for b in range(n_bins):
            w = torch.where(usable & (idx == b), inv[b].expand_as(w), w)
    return w


def tape_state_counts(tape):
    st = tape["state"][tape["valid"]]
    return {name: int((st == i).sum()) for i, name in enumerate(BC_STATES)
            if int((st == i).sum()) > 0}


# ─── the model's forward over a chunk of time ────────────────────────────────

def forward_chunk(actor, tc, prop, valid, h):
    """Roll `actor` over a (K, B, ...) chunk, returning per-step outputs.

    The two input MLP layers and the two output heads are time-independent, so
    they run once over the whole chunk; only the GRU cell is stepped. A padded
    step (valid = False) does not advance the hidden state, matching
    val_tape.replay_tape and the deployed path, where a robot that did not
    decide keeps the hidden state it had.
    """
    K, B = valid.shape
    x = actor.relu(actor.up1(torch.cat([tc, prop], dim = 2)))
    x = actor.tanh(actor.up2(x))
    hs = []
    for t in range(K):
        h_new = actor.gru(x[t], h)
        h = torch.where(valid[t].unsqueeze(1), h_new, h)
        hs.append(h)
    hcat = torch.stack(hs, dim = 0)
    g = actor.relu(actor.head1(hcat))
    # the one definition, shared with the deployed path
    motor_pre = split_motor_from_head(actor, g, prop)
    msg_pre = actor.head_msg(g)
    mean = torch.cat([msg_pre, motor_pre], dim = 2)
    motors = squash_action(mean)[..., MESSAGE_SIZE:]
    logit = actor.head_arrived(g).squeeze(-1) if actor.head_arrived is not None else None
    state_logits = actor.head_state(g) if getattr(actor, "head_state", None) is not None else None
    wall_logits = actor.head_wall(g) if getattr(actor, "head_wall", None) is not None else None
    return motors, logit, g, h, state_logits, wall_logits


# ─── the steering channel ────────────────────────────────────────────────────

def turn_from_motors(L, R):
    """The oracle's own steering variable, recovered from a wheel pair.

    simple_oracle emits L = s*(0.9 - 0.35*turn), R = s*(0.9 + 0.35*turn), so
    turn = (R - L)*1.8 / (0.7*(L + R)) exactly, and the speed scale s cancels.
    This is the quantity that decides where a robot goes; the wheel pair's other
    degree of freedom, the common mode, is nearly constant and is what a plain
    MSE on the pair spends itself reproducing.
    """
    tot = (L + R).clamp(min = 1e-6)
    return (R - L) * 1.8 / (0.7 * tot)


def steer_rows(tgt):
    """Rows whose command actually encodes a steering angle.

    Excluded: a clipped wheel (the differential no longer carries the angle),
    a stopped or near-stopped command, and the hard +-0.45 reacquire branch,
    which is a constant that says nothing about the direction being steered to.
    """
    L, R = tgt[..., 0], tgt[..., 1]
    turn = turn_from_motors(L, R)
    return (L > 0.02) & (L < 0.98) & (R > 0.02) & (R < 0.98) & ((L + R) > 0.18) \
           & (turn.abs() < 0.9) & ((turn.abs() - 0.45).abs() > 0.02)


# ─── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(actor, tape, device, chunk = 256, batch = 256, skip_arrived = True,
             arrived_threshold = 0.95, dead_stats = True, tol = 0.05, wall_label = None):
    """Held-out imitation error, per oracle state, plus the arrived head's P/R/F1.

    Same definition val_tape.replay_tape uses -- roll the CURRENT network
    forward through each recorded sequence from a cold start and score its
    motor output against the oracle's -- with the per-state balanced mean as
    the headline number, since pooling is dominated by whichever state is most
    numerous and a regression in `turning` (well under 1% of decisions) would
    otherwise be invisible.
    """
    actor.eval()
    T, R = tape["valid"].shape
    n_states = len(BC_STATES)
    sq = torch.zeros(n_states, device = device)
    cnt = torch.zeros(n_states, device = device)
    abs_err = torch.zeros(n_states, device = device)
    within = torch.zeros(n_states, device = device)
    tp = fp = fn = tn = 0
    dead_sum = None
    dead_n = 0
    # steering channel: squared error, and the first two moments of the oracle's
    # own turn, so the residual can be reported against the size of the signal
    # rather than on its own -- an rms of 0.09 reads as excellent next to a
    # motor MSE of 0.0027 and as catastrophic next to a signal whose spread is
    # 0.0093, which is what it actually is.
    st_e2 = torch.zeros(n_states, device = device)
    st_o1 = torch.zeros(n_states, device = device)
    st_o2 = torch.zeros(n_states, device = device)
    st_n = torch.zeros(n_states, device = device)
    # The two discrete latents. With use_oracle_head these are no longer
    # auxiliary -- they ARE the motor command's mixture weights -- so their
    # held-out accuracy is the thing to watch when the steering error moves.
    sh_ok = sh_n = wh_ok = wh_n = 0
    # Per-robot mean steering error during wall_following. The controls in
    # docs/tuning.md phase 159 are unambiguous about which kind of error matters:
    # driving the ORACLE with i.i.d. noise of 0.12 leaves it at 0.98 stopped,
    # while a PERSISTENT per-robot bias of 0.10 drops it to 0.40. So a metric
    # that pools every decision hides the distinction it most needs to make.
    R_all = tape["valid"].shape[1]
    bias_sum = torch.zeros(R_all, device = device)
    bias_cnt = torch.zeros(R_all, device = device)
    abs_err_acc = []
    for b0 in range(0, R, batch):
        b1 = min(b0 + batch, R)
        h = actor.initial_hidden(b1 - b0, device = device)
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            # .to(device) on every slice: with --tape-device cpu the tape lives
            # on the host and only the slice in use belongs on the GPU
            v = tape["valid"][t0:t1, b0:b1].to(device)
            if not bool(v.any()):
                continue
            tc = tape["tc"][t0:t1, b0:b1].to(device).float()
            prop = tape["prop"][t0:t1, b0:b1].to(device).float()
            tgt = tape["tgt"][t0:t1, b0:b1].to(device).float()
            motors, logit, g, h, s_log, w_log = forward_chunk(actor, tc, prop, v, h)
            err = ((motors - tgt) ** 2).mean(dim = 2)
            aerr = (motors - tgt).abs().mean(dim = 2)
            # "close enough to be the same command": both wheels within `tol` of
            # the oracle's, in the [0,1] motor scale. A squared error is the
            # right thing to optimise and the wrong thing to read -- it says
            # nothing about how often the actor is actually issuing the
            # teacher's command, which is what "emulates the oracle" means.
            hit = ((motors - tgt).abs().amax(dim = 2) <= tol)
            st = tape["state"][t0:t1, b0:b1].to(device)
            turn_o = turn_from_motors(tgt[..., 0], tgt[..., 1])
            turn_a = turn_from_motors(motors[..., 0], motors[..., 1])
            usable = v & steer_rows(tgt)
            for s in range(n_states):
                m = v & (st == s)
                if bool(m.any()):
                    sq[s] += err[m].sum()
                    abs_err[s] += aerr[m].sum()
                    within[s] += hit[m].sum()
                    cnt[s] += int(m.sum())
                u = usable & (st == s)
                if bool(u.any()):
                    st_e2[s] += ((turn_a - turn_o)[u] ** 2).sum()
                    st_o1[s] += turn_o[u].sum()
                    st_o2[s] += (turn_o[u] ** 2).sum()
                    st_n[s] += int(u.sum())
                    if BC_STATES[s] == "wall_following":
                        cols = torch.arange(b0, b1, device = device).expand(t1 - t0, -1)[u]
                        bias_sum.index_add_(0, cols, (turn_a - turn_o)[u])
                        bias_cnt.index_add_(0, cols, torch.ones_like(cols, dtype = torch.float32))
                        abs_err_acc.append((turn_a - turn_o)[u].abs())
            if logit is not None:
                usable = v & tape["arrived_valid"][t0:t1, b0:b1].to(device)
                pred = torch.sigmoid(logit) > arrived_threshold
                lab = tape["arrived"][t0:t1, b0:b1].to(device).float() > 0.5
                tp += int((usable & pred & lab).sum())
                fp += int((usable & pred & (~lab)).sum())
                fn += int((usable & (~pred) & lab).sum())
                tn += int((usable & (~pred) & (~lab)).sum())
            if s_log is not None:
                stm = tape["state"][t0:t1, b0:b1].to(device)
                k = v & (stm >= 0)
                if bool(k.any()):
                    sh_ok += int((s_log.argmax(dim = 2)[k] == stm[k]).sum()); sh_n += int(k.sum())
            if w_log is not None and wall_label is not None:
                wlm = wall_label[t0:t1, b0:b1].to(device)
                k = v & (wlm >= 0)
                if bool(k.any()):
                    wh_ok += int((w_log.argmax(dim = 2)[k] == wlm[k]).sum()); wh_n += int(k.sum())
            if dead_stats:
                gm = g[v]
                if gm.numel():
                    z = (gm.abs() < 1e-8).float().mean(dim = 0)
                    dead_sum = z if dead_sum is None else dead_sum + z
                    dead_n += 1
    out = {}
    for s, name in enumerate(BC_STATES):
        if float(cnt[s]) > 0:
            out[name] = float(sq[s] / cnt[s])
            out["mae_" + name] = float(abs_err[s] / cnt[s])
            out["within_" + name] = float(within[s] / cnt[s])
            out["n_" + name] = int(cnt[s])
    for s, name in enumerate(BC_STATES):
        n = float(st_n[s])
        if n < 100:
            continue
        mean = float(st_o1[s]) / n
        var = max(float(st_o2[s]) / n - mean * mean, 1e-12)
        mse = float(st_e2[s]) / n
        out["turn_rms_" + name] = math.sqrt(mse)
        out["turn_sd_" + name] = math.sqrt(var)
        out["turn_r2_" + name] = 1.0 - mse / var
        out["turn_n_" + name] = int(n)
    # The headline steering number is wall_following's. It is the one steering
    # state whose command is EXACTLY a function of the observation (measured:
    # sin(latched tangent - belief heading) reproduces it with rms 8e-5 and
    # correlation 1.0000), and it is the state that decides localization,
    # because with oracle_wall_seed_position off a robot's belief only collapses
    # when it reaches a corner -- so how well it follows the wall is how well it
    # knows where it is. `navigating` is deliberately NOT in it: there the
    # teacher steers by its own private particle filter and the observation
    # cannot reproduce the command even in principle.
    keep = bias_cnt >= 20
    if bool(keep.any()):
        per_robot = (bias_sum[keep] / bias_cnt[keep])
        out["turn_bias_wall_following"] = float((per_robot ** 2).mean().sqrt())
        out["turn_bias_med_wall_following"] = float(per_robot.abs().median())
    if abs_err_acc:
        ae = torch.cat(abs_err_acc)
        out["turn_med_wall_following"] = float(ae.median())
        out["turn_p90_wall_following"] = float(ae.quantile(0.9))
    if sh_n:
        out["state_head_acc"] = sh_ok / float(sh_n)
    if wh_n:
        out["wall_head_acc"] = wh_ok / float(wh_n)
    if "turn_rms_wall_following" in out:
        out["steer"] = out["turn_rms_wall_following"]
    total = float(cnt.sum())
    if total > 0:
        out["all"] = float(sq.sum() / total)
        out["within_all"] = float(within.sum() / total)
        out["within_tol"] = tol
        scored = [n for n in BC_STATES if n in out and not (skip_arrived and n == ARRIVED)]
        out["balanced"] = sum(out[n] for n in scored) / len(scored)
        out["scored_states"] = scored
    if tp + fp + fn + tn > 0:
        out["arrived_precision"] = tp / float(tp + fp) if tp + fp else float("nan")
        out["arrived_recall"] = tp / float(tp + fn) if tp + fn else 0.0
        p, r = out["arrived_precision"], out["arrived_recall"]
        out["arrived_f1"] = (2 * p * r / (p + r)) if (p == p and p + r > 0) else 0.0
        out["arrived_tp"], out["arrived_fp"] = tp, fp
        out["arrived_fn"], out["arrived_tn"] = fn, tn
    if dead_sum is not None and dead_n:
        frac = dead_sum / dead_n
        out["dead_units"] = int((frac > 0.999).sum())
        out["dead_frac"] = float((frac > 0.999).float().mean())
    actor.train()
    return out


# ─── training ────────────────────────────────────────────────────────────────

def state_weights(tape, cap = 0.0, skip_arrived = True):
    """Per-state sample weights that equalise how much each state contributes.

    `turning` is roughly 0.4% of an episode's decisions and `arrived` most of
    it. Weighting each decision by the inverse of its state's frequency makes
    the fit care about a rare state as much as a common one, which is what the
    balanced replay reservoir approximates by resampling. Doing it as a weight
    instead is exact, costs nothing, and cannot starve a state of examples the
    way a sampler can.
    """
    st = tape["state"][tape["valid"]]
    counts = torch.zeros(len(BC_STATES))
    for i in range(len(BC_STATES)):
        counts[i] = int((st == i).sum())
    scored = torch.tensor([0.0 if (skip_arrived and BC_STATES[i] == ARRIVED) else 1.0
                           for i in range(len(BC_STATES))])
    live = (counts > 0).float() * scored
    w = torch.where(counts > 0, 1.0 / counts.clamp(min = 1), torch.zeros_like(counts)) * live
    if float(w.sum()) > 0:
        # normalise so the MEAN weight over the training distribution is 1,
        # keeping the loss scale (and so the effective learning rate) the same
        # as the unweighted version
        mean_w = float((w * counts).sum() / counts[live > 0].sum())
        w = w / max(mean_w, 1e-12)
    if cap and cap > 0:
        w = w.clamp(max = float(cap))
    return w


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--train-tape", required = True, nargs = "+",
                    help = "one or more recorded tapes; several are concatenated along the "
                           "robot axis, which is how a DAgger round is trained together with "
                           "the original oracle data rather than after it")
    ap.add_argument("--val-tape", required = True)
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--batch-robots", type = int, default = 128,
                    help = "sequences per gradient chunk; the effective batch is this "
                           "times --chunk decisions")
    ap.add_argument("--chunk", type = int, default = 96,
                    help = "truncated-BPTT window in decisions; the hidden state carries "
                           "across chunks, the gradient does not")
    ap.add_argument("--lr", type = float, default = 1e-3)
    ap.add_argument("--weight-decay", type = float, default = 0.0)
    ap.add_argument("--warmup", type = int, default = 200, help = "warmup steps")
    ap.add_argument("--max-grad-norm", type = float, default = 1.0)
    ap.add_argument("--activation", default = "elu", choices = ["relu", "elu", "silu", "leaky_relu", "tanh"],
                    help = "hidden activation of the split actor's trunk and head")
    ap.add_argument("--state-balance", dest = "state_balance", action = "store_true", default = True)
    ap.add_argument("--no-state-balance", dest = "state_balance", action = "store_false")
    ap.add_argument("--state-weight-cap", type = float, default = 8.0,
                    help = "clamp on any single state's weight, so a state with a handful "
                           "of decisions cannot dominate a whole epoch")
    ap.add_argument("--skip-arrived", dest = "skip_arrived", action = "store_true", default = True,
                    help = "drop arrived decisions from the MOTOR loss (their target is exactly "
                           "[0,0], which tanh reaches only at -inf); the arrived head is what "
                           "stops a robot")
    ap.add_argument("--no-skip-arrived", dest = "skip_arrived", action = "store_false")
    ap.add_argument("--arrived-motor-weight", type = float, default = 0.1,
                    help = "weight given back to arrived rows in the MOTOR loss, against a "
                           "floored target (--arrived-motor-floor) rather than the oracle's "
                           "exact [0,0]. 0 reproduces --skip-arrived exactly. The point is a "
                           "fallback: with the motor head trained to crawl when it recognises "
                           "an arrived state, an arrived-head false negative leaves a robot "
                           "creeping at its target instead of driving off across the arena")
    ap.add_argument("--arrived-motor-floor", type = float, default = 0.02,
                    help = "target the motor head is given on arrived rows. Nonzero on purpose: "
                           "squash_action reaches exactly 0 only as its pre-activation goes to "
                           "-inf, which is what drove the trunk into the phase-154 collapse")
    ap.add_argument("--arrived-loss-weight", type = float, default = 1.0)
    ap.add_argument("--arrived-balance", default = "balanced", choices = ["balanced", "natural"],
                    help = "balanced reweights the BCE so arrived and not-arrived contribute "
                           "equally, which keeps the head from collapsing onto the 86%% prior")
    ap.add_argument("--arrived-threshold", type = float, default = 0.95)
    ap.add_argument("--state-head-weight", type = float, default = 0.0,
                    help = "weight of an auxiliary cross-entropy on which of the oracle's five "
                           "states produced the command. The head is training-only -- no deployed "
                           "path reads it -- and its point is to supervise the recurrent "
                           "representation directly: cloning the motor command alone leaves "
                           "'which state am I in' an unsupervised latent, and the states where "
                           "the closed loop falls apart are exactly the ones that need it")
    ap.add_argument("--wall-head-weight", type = float, default = 0.0,
                    help = "weight of an auxiliary cross-entropy on which wall the robot last "
                           "touched. The label needs no new recording -- it is the last nonzero "
                           "wall slot in the tape's own Tc, forward-filled, which is exactly what "
                           "simple_oracle latches at the go_north->turning transition. Training-"
                           "only, 164 parameters, and aimed at the specific defect measured in "
                           "phase 156: a constant +0.108 steering bias during wall_following "
                           "where the teacher's own command is straight")
    ap.add_argument("--steer-feature", action = "store_true",
                    help = "config.py's own use_steer_feature: hand the motor head the "
                           "wall-gated sin/cos of (tangent - heading). Requires "
                           "--wall-head-weight > 0, since the gate is the wall head's posterior")
    ap.add_argument("--oracle-head", action = "store_true",
                    help = "config.py's own use_oracle_head: build the motor command as a soft "
                           "mixture over the ORACLE's five commands, each in closed form from "
                           "quantities the actor already observes, weighted by the state head's "
                           "posterior. Costs zero parameters and zero inputs -- it reuses "
                           "head_state, head_wall and head_motor. Requires --state-head-weight "
                           "and --wall-head-weight above 0")
    ap.add_argument("--oracle-residual", type = float, default = 0.05,
                    help = "magnitude, in motor units, of the learned residual added to that "
                           "mixture. Small on purpose: the closed form is meant to be a prior "
                           "the network corrects, not a starting point it can regress away")
    ap.add_argument("--oracle-residual-turn", type = float, default = 0.0,
                    help = "the residual's DIFFERENTIAL half, bounded separately. The default "
                           "lets a learned correction move the steering variable by at most the "
                           "spread of the oracle's own steering during wall_following")
    ap.add_argument("--steer-weight", type = float, default = 0.0,
                    help = "extra loss on the DIFFERENTIAL of the wheel pair, on top of the plain "
                           "MSE. The two wheels' common and differential modes are orthogonal, and "
                           "MSE weights them equally -- but during wall_following the differential "
                           "carries 0.1%% of the target variance and 100%% of where the robot ends "
                           "up. This is the un-structured way to make the loss see it; --oracle-head "
                           "is the structured one")
    ap.add_argument("--select", default = "balanced",
                    choices = ["balanced", "steer", "turn_bias_wall_following",
                               "turn_med_wall_following"],
                    help = "which held-out number picks actor_best.pt. `balanced` is the per-state "
                           "mean motor MSE, which is what every run before phase 160 used and which "
                           "is measurably ANTI-correlated with closed-loop outcome across those "
                           "runs. `steer` is the rms error in the oracle's own steering variable "
                           "during wall_following")
    ap.add_argument("--align-balance", action = "store_true",
                    help = "reweight wall_following/navigating decisions by how misaligned the "
                           "robot is with the direction it is being steered toward. The teacher "
                           "holds itself within 5 degrees essentially always, so those states are "
                           "99.9% of its own data and the clone's own failures live outside them")
    ap.add_argument("--wall-label", default = "latched", choices = ["latched", "recent"],
                    help = "which wall the auxiliary head is trained to name. `latched` is the "
                           "one simple_oracle actually steers by, recovered from its own commands "
                           "(wall_labels_from_targets); `recent` is the last one observed, which "
                           "is what the first version of this head used and is wrong wherever a "
                           "robot has touched more than one wall")
    ap.add_argument("--obs-noise", type = float, default = 0.0,
                    help = "standard deviation of Gaussian noise added to the observation "
                           "during training only. A behaviour-cloned policy is brittle exactly "
                           "off the expert's own trajectory, and the cheapest way to widen the "
                           "region it is correct in -- without collecting anything -- is to make "
                           "it fit a neighbourhood of each recorded observation rather than the "
                           "point itself. Most channels here are O(1), so 0.02-0.05 is a small "
                           "perturbation; the targets are left untouched")
    ap.add_argument("--msg-weight", type = float, default = 0.0,
                    help = "weight of an L2 pull on the broadcast-message head toward zero. The "
                           "oracle defines no message target, so this head is otherwise "
                           "unconstrained; a small value keeps peer message content stable "
                           "instead of drifting wherever the shared trunk pushes it")
    ap.add_argument("--use-arrived-head", action = "store_true", default = True)
    ap.add_argument("--no-arrived-head", dest = "use_arrived_head", action = "store_false")
    ap.add_argument("--use-turn-anchor", action = "store_true", default = True)
    ap.add_argument("--no-turn-anchor", dest = "use_turn_anchor", action = "store_false")
    ap.add_argument("--gru-hidden", type = int, default = None,
                    help = "override config.py's split_gru_hidden. The defaults are sized to a "
                           "24KB int8 budget (phase 147); raising them is a diagnostic -- does "
                           "the fit stop improving because of the data or because of the width -- "
                           "not something a deployable checkpoint can use")
    ap.add_argument("--head-hidden", type = int, default = None)
    ap.add_argument("--upscale-hidden", type = int, default = None)
    ap.add_argument("--init", default = None, help = "warm-start from this actor checkpoint")
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--eval-every", type = int, default = 1)
    ap.add_argument("--tape-device", default = None,
                    help = "where to keep the tapes (default: --device if it fits, else cpu)")
    return ap


def make_cfg(args, tc_dim, prop_dim):
    from config import Config
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.use_arrived_head = args.use_arrived_head
    cfg.use_turn_anchor = args.use_turn_anchor
    cfg.use_state_head = args.state_head_weight > 0
    cfg.use_wall_head = args.wall_head_weight > 0
    cfg.use_steer_feature = bool(args.steer_feature and args.wall_head_weight > 0)
    cfg.use_oracle_head = bool(args.oracle_head)
    cfg.oracle_residual = args.oracle_residual
    cfg.oracle_residual_turn = args.oracle_residual_turn
    if cfg.use_oracle_head and not (args.state_head_weight > 0 and args.wall_head_weight > 0):
        raise SystemExit("--oracle-head mixes by the state and wall heads' posteriors, so it "
                         "needs --state-head-weight and --wall-head-weight above 0")
    cfg.split_activation = args.activation
    cfg.device = args.device
    if args.gru_hidden:
        cfg.split_gru_hidden = args.gru_hidden
    if args.head_hidden:
        cfg.split_head_hidden = args.head_hidden
    if args.upscale_hidden:
        cfg.split_upscale_hidden = args.upscale_hidden
    return cfg


def main(argv = None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok = True)
    torch.manual_seed(args.seed)
    device = args.device
    tape_device = args.tape_device or device

    train = concat_tapes([load_tape(p, tape_device) for p in args.train_tape])
    val = load_tape(args.val_tape, tape_device)
    print("train tape %s: %d sequences x %d steps, %d decisions %s"
          % (", ".join(args.train_tape), train["valid"].shape[1], train["valid"].shape[0],
             int(train["valid"].sum()), tape_state_counts(train)), flush = True)
    print("val   tape %s: %d sequences x %d steps, %d decisions %s"
          % (args.val_tape, val["valid"].shape[1], val["valid"].shape[0],
             int(val["valid"].sum()), tape_state_counts(val)), flush = True)

    tc_dim = train["tc"].shape[2]
    prop_dim = train["prop"].shape[2]
    cfg = make_cfg(args, tc_dim, prop_dim)
    actor = build_actor(cfg).to(device)
    expect = actor.up1.in_features
    if tc_dim + prop_dim != expect:
        raise SystemExit("tape observation width %d (tc %d + prop %d) does not match the actor's "
                         "%d -- the tape was recorded with different --use-turn-anchor/"
                         "--use-arrived-head settings than this run asks for"
                         % (tc_dim + prop_dim, tc_dim, prop_dim, expect))
    if args.init:
        blob = torch.load(args.init, map_location = device, weights_only = False)
        actor.load_state_dict(blob["actor"] if "actor" in blob else blob)
        print("warm-started from %s" % args.init, flush = True)
    n_params = sum(p.numel() for p in actor.parameters())
    print("actor: %d parameters (%.1f KB as int8), activation=%s, arrived_head=%s, turn_anchor=%s"
          % (n_params, n_params / 1024.0, args.activation, args.use_arrived_head, args.use_turn_anchor),
          flush = True)

    opt = torch.optim.AdamW(actor.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    w_state = state_weights(train, cap = args.state_weight_cap,
                            skip_arrived = args.skip_arrived).to(device)
    if not args.state_balance:
        w_state = torch.tensor([0.0 if (args.skip_arrived and n == ARRIVED) else 1.0
                                for n in BC_STATES], device = device)
    print("state weights: %s" % {BC_STATES[i]: round(float(w_state[i]), 3)
                                 for i in range(len(BC_STATES))}, flush = True)

    # class balance for the arrived head, measured once on the training tape
    av = train["arrived_valid"] & train["valid"]
    pos = float((train["arrived"].float()[av] > 0.5).float().mean()) if bool(av.any()) else 0.0
    arrived_balance = args.arrived_balance
    if pos < 1e-3 or pos > 1 - 1e-3:
        # A tape too short to contain any arrivals would otherwise get a class
        # weight of 0.5/pos -- thousands -- on whichever handful of positives
        # it does have, and the head would train on essentially nothing at a
        # huge effective learning rate. Refuse to balance instead.
        print("WARNING: arrived is %.4f%% of the training tape -- too degenerate to balance, "
              "falling back to natural weighting" % (100 * pos), flush = True)
        arrived_balance = "natural"
        pos = min(max(pos, 1e-3), 1 - 1e-3)
    print("arrived head: %.1f%% positive in the training tape (%s weighting)"
          % (100 * pos, arrived_balance), flush = True)

    if args.wall_head_weight > 0:
        wall_label = (wall_labels_from_targets(train) if args.wall_label == "latched"
                      else wall_labels(train))
        have = float((wall_label >= 0).float().mean())
        print("wall head: %s label, %.1f%% of decisions carry one" % (args.wall_label, 100 * have),
              flush = True)
    else:
        wall_label = None
    # the same latched-wall label on the VALIDATION tape, for scoring only
    val_wall_label = (wall_labels_from_targets(val)
                      if (args.wall_head_weight > 0 and args.wall_label == "latched")
                      else (wall_labels(val) if args.wall_head_weight > 0 else None))
    align_w = alignment_weights(train) if args.align_balance else None
    if align_w is not None:
        print("alignment balancing on: weights span %.2f..%.2f"
              % (float(align_w.min()), float(align_w.max())), flush = True)
    arrived_idx = BC_STATES.index(ARRIVED)
    T, R = train["valid"].shape
    steps_per_epoch = math.ceil(R / args.batch_robots) * math.ceil(T / args.chunk)
    total_steps = steps_per_epoch * args.epochs
    gen = torch.Generator().manual_seed(args.seed)

    # A second run started against the same --out-dir appends to the same
    # history and overwrites the same checkpoints, and the two interleave
    # silently -- the history ends up with every epoch twice and the checkpoint
    # belongs to whichever process wrote last. Cheap to prevent, and it is a
    # mistake that is invisible until the plots look wrong.
    lock_path = os.path.join(args.out_dir, "run.lock")
    if os.path.exists(lock_path):
        try:
            with open(lock_path) as f:
                other = int(f.read().strip() or 0)
            alive = other > 0 and os.path.exists("/proc/%d" % other)
        except Exception:
            alive = False
        if alive:
            raise SystemExit("another bc_offline.py (pid %d) is already writing to %s -- "
                             "use a different --out-dir" % (other, args.out_dir))
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))
    history_path = os.path.join(args.out_dir, "history.jsonl")
    hist = open(history_path, "a")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent = 2)

    best = [None]
    step = [0]

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / max(args.warmup, 1)
        p = (s - args.warmup) / max(total_steps - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    t0 = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(R, generator = gen)
        ep_loss = ep_motor = ep_arr = 0.0
        ep_n = 0
        for b0 in range(0, R, args.batch_robots):
            idx = order[b0:b0 + args.batch_robots].to(tape_device)
            B = idx.numel()
            h = actor.initial_hidden(B, device = device)
            for t0c in range(0, T, args.chunk):
                t1c = min(t0c + args.chunk, T)
                v = train["valid"][t0c:t1c][:, idx].to(device)
                if not bool(v.any()):
                    continue
                tc = train["tc"][t0c:t1c][:, idx].to(device).float()
                prop = train["prop"][t0c:t1c][:, idx].to(device).float()
                tgt = train["tgt"][t0c:t1c][:, idx].to(device).float()
                st = train["state"][t0c:t1c][:, idx].to(device)
                if args.obs_noise > 0:
                    tc = tc + torch.randn_like(tc) * args.obs_noise
                    prop = prop + torch.randn_like(prop) * args.obs_noise
                motors, logit, g, h_new, state_logits, wall_logits = forward_chunk(actor, tc, prop, v, h)

                w = torch.zeros_like(v, dtype = torch.float32)
                for s in range(len(BC_STATES)):
                    w = torch.where(v & (st == s), w_state[s].expand_as(w), w)
                if args.arrived_motor_weight > 0:
                    is_arr = v & (st == arrived_idx)
                    w = torch.where(is_arr, torch.full_like(w, args.arrived_motor_weight), w)
                    tgt = torch.where(is_arr.unsqueeze(2),
                                      tgt.clamp(min = args.arrived_motor_floor), tgt)
                if align_w is not None:
                    w = w * align_w[t0c:t1c][:, idx].to(device)
                denom = w.sum().clamp(min = 1e-6)
                motor_loss = (((motors - tgt) ** 2).mean(dim = 2) * w).sum() / denom
                loss = motor_loss
                if args.steer_weight > 0:
                    d_a = motors[..., 1] - motors[..., 0]
                    d_t = tgt[..., 1] - tgt[..., 0]
                    loss = loss + args.steer_weight * (((d_a - d_t) ** 2) * w).sum() / denom
                arr_loss = torch.zeros((), device = device)
                if logit is not None:
                    av_c = (v & train["arrived_valid"][t0c:t1c][:, idx].to(device))
                    if bool(av_c.any()):
                        lab = train["arrived"][t0c:t1c][:, idx].to(device).float()
                        raw = F.binary_cross_entropy_with_logits(logit, lab, reduction = "none")
                        if arrived_balance == "balanced":
                            cw = torch.where(lab > 0.5,
                                             torch.full_like(lab, 0.5 / pos),
                                             torch.full_like(lab, 0.5 / (1 - pos)))
                        else:
                            cw = torch.ones_like(lab)
                        cw = cw * av_c.float()
                        arr_loss = (raw * cw).sum() / cw.sum().clamp(min = 1e-6)
                        loss = loss + args.arrived_loss_weight * arr_loss
                if state_logits is not None and args.state_head_weight > 0:
                    # -1 marks a decision the teacher gave no state label, and
                    # padded steps are not decisions at all; both are dropped
                    # rather than given a substitute class.
                    keep = v & (st >= 0)
                    if bool(keep.any()):
                        ce = F.cross_entropy(state_logits[keep], st[keep])
                        loss = loss + args.state_head_weight * ce
                if wall_logits is not None and args.wall_head_weight > 0:
                    wl = wall_label[t0c:t1c][:, idx].to(device)
                    keep = v & (wl >= 0)
                    if bool(keep.any()):
                        loss = loss + args.wall_head_weight * F.cross_entropy(
                            wall_logits[keep], wl[keep])
                if args.msg_weight > 0:
                    msg_pre = actor.head_msg(g)
                    loss = loss + args.msg_weight * ((msg_pre ** 2).mean(dim = 2) * v).sum() / v.sum().clamp(min = 1)

                for gp in opt.param_groups:
                    gp["lr"] = lr_at(step[0])
                opt.zero_grad(set_to_none = True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                opt.step()
                step[0] += 1
                h = h_new.detach()
                ep_loss += float(loss.detach()); ep_motor += float(motor_loss.detach())
                ep_arr += float(arr_loss.detach()); ep_n += 1

        row = {"epoch": epoch, "train_loss": ep_loss / max(ep_n, 1),
               "train_motor_mse": ep_motor / max(ep_n, 1),
               "train_arrived_bce": ep_arr / max(ep_n, 1),
               "lr": lr_at(step[0]), "steps": step[0], "seconds": time.time() - t0}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            sc = evaluate(actor, val, device, skip_arrived = args.skip_arrived,
                          arrived_threshold = args.arrived_threshold, wall_label = val_wall_label)
            tr = evaluate(actor, train, device, skip_arrived = args.skip_arrived,
                          arrived_threshold = args.arrived_threshold, batch = 128,
                          dead_stats = False)
            row["val"] = {k: v for k, v in sc.items() if not isinstance(v, list)}
            row["train_eval_balanced"] = tr.get("balanced")
            per = "  ".join("%s %.5f" % (n[:4], sc[n]) for n in BC_STATES if n in sc)
            ah = ("  arrived P %.3f R %.3f F1 %.3f" % (sc.get("arrived_precision", float("nan")),
                                                       sc.get("arrived_recall", 0.0),
                                                       sc.get("arrived_f1", 0.0))
                  if "arrived_f1" in sc else "")
            print("epoch %3d  train %.5f (motor %.5f arr %.4f)  VAL balanced %.5f  "
                  "STEER wall rms %.4f med %.4f bias %.4f (R2 %+.2f) navi %.4f  within%.2f %.3f  %s%s  dead %d/%d  %.1f min"
                  % (epoch, row["train_loss"], row["train_motor_mse"], row["train_arrived_bce"],
                     sc["balanced"], sc.get("turn_rms_wall_following", float("nan")),
                     sc.get("turn_med_wall_following", float("nan")),
                     sc.get("turn_bias_wall_following", float("nan")),
                     sc.get("turn_r2_wall_following", float("nan")),
                     sc.get("turn_rms_navigating", float("nan")),
                     sc.get("within_tol", 0.05), sc.get("within_all", float("nan")),
                     "heads %.3f/%.3f  " % (sc.get("state_head_acc", float("nan")),
                                            sc.get("wall_head_acc", float("nan"))) + per, ah,
                     sc.get("dead_units", -1), actor.head1.out_features,
                     (time.time() - t0) / 60.0), flush = True)
            score = sc.get(args.select, sc["balanced"])
            if best[0] is None or score < best[0]:
                best[0] = score
                save_actor(os.path.join(args.out_dir, "actor_best.pt"), actor, args, epoch, sc)
                print("            NEW BEST %s %.5f -> actor_best.pt" % (args.select, best[0]), flush = True)
        hist.write(json.dumps(row) + "\n")
        hist.flush()
        save_actor(os.path.join(args.out_dir, "actor_latest.pt"), actor, args, epoch, None)
    hist.close()
    try:
        os.remove(lock_path)
    except OSError:
        pass
    print("done in %.1f min. best held-out balanced imitation error %.5f"
          % ((time.time() - t0) / 60.0, best[0] if best[0] is not None else float("nan")), flush = True)


def save_actor(path, actor, args, epoch, scores):
    """The same artifact checkpoint.export_actor writes, plus what built it.

    log_std is included because every consumer (launch.py's eval path,
    watch_actor.sh, the RL warm start) loads a GaussianPolicy, not a bare
    actor; BC never trains it, so it is the configured constant. `meta` records
    the architecture switches -- activation, arrived head, turn anchor -- so a
    checkpoint cannot be silently loaded into a differently-shaped actor.
    """
    from config import Config
    from policy import ACTION_SIZE
    from kilobot_gnn import Z
    tmp = path + ".tmp"
    torch.save({"actor": actor.state_dict(),
                "log_std": torch.full((ACTION_SIZE,), float(Config().log_std_init)),
                "iteration": int(epoch),
                "meta": {"z": Z, "action_size": ACTION_SIZE,
                         # read off the built actor, not off args: --gru-hidden
                         # and friends default to None and the real widths come
                         # from config.py, so recording the flags would record
                         # nothing for a default-width run
                         "gru_hidden": int(actor.hidden_size),
                         "head_hidden": int(actor.head1.out_features),
                         "upscale_hidden": int(actor.up1.out_features),
                         "activation": args.activation,
                         "use_arrived_head": bool(args.use_arrived_head),
                         "use_turn_anchor": bool(args.use_turn_anchor),
                         "use_state_head": bool(args.state_head_weight > 0),
                         "use_wall_head": bool(args.wall_head_weight > 0),
                         "use_steer_feature": bool(args.steer_feature and args.wall_head_weight > 0),
                         "use_oracle_head": bool(args.oracle_head),
                         "oracle_residual": float(args.oracle_residual),
                         "oracle_residual_turn": float(args.oracle_residual_turn),
                         "trainer": "bc_offline",
                         "val": {k: v for k, v in (scores or {}).items()
                                 if isinstance(v, (int, float))}}}, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
