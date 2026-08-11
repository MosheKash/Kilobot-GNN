"""Recorded validation tapes: held-out BC scoring with no simulation.

A tape is observations plus oracle targets, captured once and replayed against
any later checkpoint to report per-state imitation error -- which is what the BC
phase actually optimises, unlike coverage, which measures the downstream task.
Because it is only recorded data, one tape scores any checkpoint and costs no
environment time.
"""

import os
import torch

from bc_replay import BC_STATES, ARRIVED
from kilobot_gnn import MESSAGE_SIZE, split_forward_batch
from policy import squash_action

# A validation "tape": one real, oracle-driven rollout over held-out
# formations, recorded once as an ordered per-robot sequence of observations
# and oracle motor targets, then replayed through whatever the actor currently
# is at every eval.
#
# Why a tape rather than simulating validation each time. The states this
# project keeps forgetting (go_north, turning) live at the start of an episode
# and the ones that dominate it (navigating, arrived) only appear many
# thousands of ticks in, so any validation short enough to run per-iteration
# only ever sees one phase. Staggering separate val arenas does NOT fix that:
# an arena whose step_count starts high just times out sooner, it does not
# behave as though it were late in an episode, because episode phase is
# produced by actually simulating those ticks.
#
# Replaying a recorded sequence sidesteps the cost entirely -- the tape is
# simulated once and then reused, and every eval afterwards is pure network
# compute with no physics at all.
#
# Crucially the hidden state is NOT stored. It is recomputed by rolling the
# CURRENT network forward through each robot's own observation sequence from a
# genuine cold start, so the tape never goes stale as the actor changes -- the
# exact confound that makes a frozen (observation, hidden) batch degrade for
# reasons unrelated to actor quality.
#
# The observations themselves are actor-independent: the tape is recorded
# under motor_override="simple_oracle", so tc comes from the environment and
# prop's odometry dead-reckons from the ORACLE's executed motor, neither of
# which depends on which actor is being scored.

TAPE_VERSION = 2


def build_tape(trainer, cfg, policy, ticks, max_robots = 0, stride = 1, verbose = True):
    worker = trainer.workers[0] if hasattr(trainer, "workers") else None
    prev_override = cfg.motor_override
    cfg.motor_override = "simple_oracle"
    trainer._bc_capture = True
    rows = []
    collected = 0
    while collected < ticks:
        buf = trainer.collect(policy, None, deterministic = False)
        for d in buf.decisions:
            if d.get("bc_target") is None or d.get("prev_hidden") is None:
                continue
            step = buf.steps[d["step_index"]]
            rows.append((int(step["arena_id"]), int(d["local"]), int(step["env_step"]),
                         d["transmission"].detach().cpu(), d["prop"].detach().cpu(),
                         d["bc_target"].detach().cpu(), d.get("oracle_state"),
                         d.get("arrived_target")))
        collected = collected + int(cfg.rollout_steps)
        if verbose:
            print("  val tape: %d/%d ticks, %d decisions" % (collected, ticks, len(rows)), flush = True)
    cfg.motor_override = prev_override
    trainer._bc_capture = False
    return _pack(rows, max_robots, stride)


def _pack(rows, max_robots, stride):
    by_robot = {}
    for r in rows:
        by_robot.setdefault((r[0], r[1]), []).append(r)
    keys = sorted(by_robot)
    if max_robots > 0 and len(keys) > max_robots:
        keys = keys[:max_robots]
    seqs = []
    for k in keys:
        seq = sorted(by_robot[k], key = lambda_free_sort_key(by_robot[k]))
        seqs.append(seq[::max(1, stride)])
    T = max(len(s) for s in seqs)
    R = len(seqs)
    tc_dim = seqs[0][0][3].shape[0]
    prop_dim = seqs[0][0][4].shape[0]
    tc = torch.zeros(T, R, tc_dim)
    prop = torch.zeros(T, R, prop_dim)
    tgt = torch.zeros(T, R, 2)
    state = torch.full((T, R), -1, dtype = torch.long)
    arrived = torch.zeros(T, R)
    arrived_valid = torch.zeros(T, R, dtype = torch.bool)
    valid = torch.zeros(T, R, dtype = torch.bool)
    idx = {s: i for i, s in enumerate(BC_STATES)}
    for j, seq in enumerate(seqs):
        for t, r in enumerate(seq):
            tc[t, j] = r[3]
            prop[t, j] = r[4]
            tgt[t, j] = r[5]
            state[t, j] = idx.get(r[6], -1)
            if r[7] is not None:
                arrived[t, j] = float(r[7].reshape(1)[0])
                arrived_valid[t, j] = True
            valid[t, j] = True
    return {"version": TAPE_VERSION, "tc": tc, "prop": prop, "tgt": tgt, "state": state,
            "arrived": arrived, "arrived_valid": arrived_valid, "valid": valid}


def lambda_free_sort_key(_seq):
    def key(r):
        return r[2]
    return key


def save_tape(tape, path):
    tmp = path + ".tmp"
    torch.save(tape, tmp)
    os.replace(tmp, path)


def load_tape(path):
    if not os.path.exists(path):
        return None
    tape = torch.load(path, map_location = "cpu", weights_only = False)
    if not isinstance(tape, dict) or tape.get("version") != TAPE_VERSION:
        return None
    return tape


def tape_state_counts(tape):
    out = {}
    st = tape["state"][tape["valid"]]
    for i, name in enumerate(BC_STATES):
        n = int((st == i).sum())
        if n > 0:
            out[name] = n
    return out


def replay_tape(policy, tape, device = "cpu", skip_arrived = False, arrived_threshold = 0.95):
    # Roll the current network forward through every robot's own recorded
    # sequence from h = 0, so the hidden states scored here are the ones this
    # actor would actually produce. Returns per-oracle-state motor MSE, plus
    # the arrived head's own held-out precision/recall.
    #
    # skip_arrived MUST mirror cfg.bc_motor_skip_arrived. Found the hard way in
    # a real 40-iteration run: with the motor head deliberately never trained
    # on arrived, its error there climbs toward 1.0 BECAUSE the fix is working,
    # and folding that into the average made it roughly thirty times larger
    # than the other four states combined. The criterion then rewarded
    # whichever checkpoint had progressed LEAST -- it selected iteration 9
    # while the four states that are actually optimised improved from 0.0579
    # to 0.0321 over the run. A validation metric has to apply the same mask
    # the loss does.
    actor = policy.actor
    T, R = tape["valid"].shape
    h = actor.initial_hidden(R, device = device)
    # .float() because tools/record_tape.py stores the observation tensors as
    # float16 (a training-sized tape is millions of decisions); build_tape's own
    # tapes are already float32 and are unaffected. Without it the first matmul
    # fails on a dtype mismatch against float32 weights.
    tc_all = tape["tc"].to(device).float()
    prop_all = tape["prop"].to(device).float()
    tgt_all = tape["tgt"].to(device).float()
    valid_all = tape["valid"].to(device)
    state_all = tape["state"].to(device)
    arrived_all = tape["arrived"].to(device)
    arrived_valid_all = tape["arrived_valid"].to(device)
    sq_sum = torch.zeros(len(BC_STATES), device = device)
    count = torch.zeros(len(BC_STATES), device = device)
    tp = fp = fn = 0
    with torch.no_grad():
        for t in range(T):
            v = valid_all[t]
            if not bool(v.any()):
                continue
            mean, h_new = split_forward_batch(actor, tc_all[t], prop_all[t], h)
            # A robot with no decision at this index keeps its previous hidden
            # state rather than stepping the GRU on a padded, meaningless row.
            h = torch.where(v.unsqueeze(1), h_new, h)
            motors = squash_action(mean)[:, MESSAGE_SIZE:]
            err = ((motors - tgt_all[t]) ** 2).mean(dim = 1)
            # The arrived head scored on genuinely held-out data, at the same
            # threshold the runtime gate actually uses. Reported separately
            # rather than mixed into the motor number, since the two measure
            # different things and now have different training signals.
            logit = getattr(actor, "_arrived_logit", None)
            if logit is not None:
                pred = torch.sigmoid(logit).squeeze(-1) > arrived_threshold
                lab = arrived_all[t] > 0.5
                usable = v & arrived_valid_all[t]
                tp = tp + int((usable & pred & lab).sum())
                fp = fp + int((usable & pred & (~lab)).sum())
                fn = fn + int((usable & (~pred) & lab).sum())
            st = state_all[t]
            for s in range(len(BC_STATES)):
                m = v & (st == s)
                if bool(m.any()):
                    sq_sum[s] = sq_sum[s] + err[m].sum()
                    count[s] = count[s] + int(m.sum())
    out = {}
    for s, name in enumerate(BC_STATES):
        if float(count[s]) > 0:
            out[name] = float(sq_sum[s] / count[s])
    total = float(count.sum())
    if total > 0:
        out["all"] = float(sq_sum.sum() / total)
        # The selection criterion: the MEAN of the per-state errors, not the
        # pooled average. Pooling is dominated by whichever state happens to
        # be most numerous, which is exactly how a regression in a rare state
        # (turning is 0.4% of collected data) stays invisible.
        scored = [n for n in BC_STATES if n in out and not (skip_arrived and n == ARRIVED)]
        if scored:
            out["balanced"] = sum(out[n] for n in scored) / len(scored)
        out["scored_states"] = scored
    if tp + fp + fn > 0 or tp > 0:
        out["arrived_precision"] = tp / float(tp + fp) if (tp + fp) > 0 else float("nan")
        out["arrived_recall"] = tp / float(tp + fn) if (tp + fn) > 0 else float("nan")
        pr = out["arrived_precision"]
        rc = out["arrived_recall"]
        out["arrived_f1"] = (2 * pr * rc / (pr + rc)) if (pr == pr and rc == rc and pr + rc > 0) else 0.0
    return out
