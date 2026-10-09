"""audit_event_draw.py -- measure the double observation draw, and what
unifying it would change.

Background: during BC capture, TWO independent sample_split_event draws run on
the same tick for the same robots. gather_split_state's draw (the full pool:
seeds + walls + neighbours) produces the tc input stored in the tape, and
simple_oracle's own draw (a wall-only pool) decides which wall event the
oracle's belief fuse and wall steering actually consumed. On hardware there is
one receiver, so there is one ground-truth event per tick; when the two draws
disagree, the (input, label) pair is inconsistent -- the label was computed
from an event the input does not show.

This tool quantifies that, two ways:

  tape   no simulation: decode the event encoded in each tape entry's tc and
         cross-tabulate it against the oracle state label. Entries in
         wall-handling states whose tc shows a non-wall event are the
         upper-bound population of inconsistent pairs (the class also contains
         ticks where genuinely no wall signal existed).

  live   one Unity player, oracle-driven through the ordinary bc_capture path
         (the exact tape-recording setup). Per tick it captures both draws and
         runs a TWIN oracle -- a second, separately-stateful simple_oracle
         instance that consumes the actor's narrowed event instead of its own
         draw (the unified convention), same executed motion, same spawn
         headings, its own particle cloud under a separate belief_attr. The
         per-tick label difference between the two lines is exactly what
         unifying the draw would have changed, measured along the real
         trajectory.

usage (from python/):
  python tools/audit_event_draw.py tape ../results/bc_v2/tape_train.pt
  python tools/audit_event_draw.py live --ticks 1500 --arenas 2 \
      --out ../results/audit_event_draw/live.json
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bc_replay import BC_STATES
from kilobot_gnn import MESSAGE_SIZE, SEED_SIZE, WALL_SIZE, MOTOR_SIZE

STRENGTH_COL = MESSAGE_SIZE  # actor_part = MESSAGE_SIZE content + 1 strength
TC_NEIGHBOR = slice(0, MESSAGE_SIZE + 1)
TC_SEED = slice(MESSAGE_SIZE + 1, MESSAGE_SIZE + 1 + SEED_SIZE)
TC_WALL = slice(MESSAGE_SIZE + 1 + SEED_SIZE, MESSAGE_SIZE + 1 + SEED_SIZE + WALL_SIZE)

STATE_NAMES = list(BC_STATES)  # go_north, turning, wall_following, navigating, arrived
WALL_STATES = [BC_STATES.index("turning"), BC_STATES.index("wall_following")]

# worker attributes simple_oracle keeps its state machine in. The twin line is
# run by pointer-swapping these (its particle cloud lives under its own
# belief_attr and needs no swap).
ORACLE_STATE_ATTRS = ("simple_heading", "simple_state", "simple_turn_accum",
                      "simple_wall_name", "simple_motor_state", "simple_target")


# ────────────────────────────────────────────────────────────────────────────
# tape mode


def _decode_events(tc):
    """(N, SPLIT_TC_SIZE) -> (kind idx, wall side or -1). kinds: 0 none, 1 seed,
    2 wall, 3 neighbor."""
    n = tc.shape[0]
    kind = np.zeros(n, np.int64)
    side = np.full(n, -1, np.int64)
    wall = tc[:, TC_WALL] > 0
    has_wall = wall.any(axis = 1)
    seed = tc[:, TC_SEED] > 0
    has_seed = seed.any(axis = 1)
    msg = tc[:, STRENGTH_COL] > 0
    kind[has_wall] = 2
    side[has_wall] = wall[has_wall].argmax(axis = 1)
    kind[~has_wall & has_seed] = 1
    kind[~has_wall & ~has_seed & msg] = 3
    return kind, side


KIND_NAMES = ["none", "seed", "wall", "neighbor"]


def audit_tape(paths, out_json):
    total = np.zeros((len(STATE_NAMES), len(KIND_NAMES)), np.int64)
    wall_sides = np.zeros(WALL_SIZE, np.int64)
    for path in paths:
        t = torch.load(path, map_location = "cpu", weights_only = False)
        valid = t["valid"].to(torch.bool).numpy()
        state = t["state"].numpy()
        tc = t["tc"].to(torch.float32).numpy()
        m = valid & (state >= 0) & (state < len(STATE_NAMES))
        kind, side = _decode_events(tc[m])
        st = state[m]
        np.add.at(total, (st, kind), 1)
        w = kind == 2
        np.add.at(wall_sides, side[w], 1)
        print("%s: %d valid decisions (%d robots)" % (os.path.basename(path), m.sum(),
                                                      int(t["valid"].any(dim = 0).sum())))

    print("\nper-oracle-state event-kind cross-tabulation (rows = state, cols = "
          + ", ".join(KIND_NAMES) + ")")
    header = "state            " + "".join("%10s" % k for k in KIND_NAMES) + "%10s%10s" % ("total", "P(wall)")
    print(header)
    for si, name in enumerate(STATE_NAMES):
        row = total[si]
        tot = row.sum()
        pw = row[2] / tot if tot else float("nan")
        print("%-16s" % name + "".join("%10d" % c for c in row) + "%10d%10.3f" % (tot, pw))
    tot = total.sum()
    print("\nwall-side distribution among wall events:", wall_sides.tolist())
    wall_rows = total[WALL_STATES, :].sum(axis = 0)
    wall_tot = wall_rows.sum()
    print("wall-handling states (turning+wall_following): %d decisions, "
          "P(tc shows non-wall event) = %.4f  -- under the LEGACY double-draw this was an upper "
          "bound on input/label-inconsistent pairs; under oracle_shared_event there is only one "
          "draw, so every pair is consistent by construction and this is purely the "
          "event-competition rate (a wall-state robot's single reception went to a neighbour/seed)"
          % (wall_tot, 1.0 - wall_rows[2] / wall_tot if wall_tot else float("nan")))
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok = True)
        with open(out_json, "w") as f:
            json.dump({"tapes": paths, "crosstab": total.tolist(),
                       "state_names": STATE_NAMES, "kind_names": KIND_NAMES,
                       "wall_sides": wall_sides.tolist()}, f, indent = 2)


# ────────────────────────────────────────────────────────────────────────────
# live mode


def audit_live(args):
    import record_tape as RT
    import actor_io
    import simple_oracle as SO
    import unity_env
    from encoder import load_encoder
    from formations import build_formation_pool
    from images import build_image_pool, formation_paths
    from kilobot_gnn import Z, build_actor
    from policy import GaussianPolicy
    from run_bc_monitored import preprocess
    from trainer import Trainer

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    argv = [args.out, "--ticks", str(args.ticks), "--instances", str(args.instances),
            "--arenas", str(args.arenas), "--formations", args.formations,
            "--limit", str(args.limit), "--heartbeat", str(args.heartbeat),
            "--rollout", str(args.rollout), "--min-bots", str(args.min_bots),
            "--max-bots", str(args.max_bots), "--swarm-rng", str(args.swarm_rng),
            "--seed", str(args.seed), "--device", args.device,
            "--base-port", str(args.base_port), "--time-scale", str(args.time_scale),
            "--driver", "oracle"]
    rt_args = RT.build_args(argv)
    rt_args.out = args.out
    cfg = RT.make_cfg(rt_args)

    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit, device = args.device)
    formation_pool = build_formation_pool(args.formations, limit = args.limit)
    cfg._oracle_formation_pool = formation_pool

    unity_env.set_player_env(formations = args.formations, heartbeat_ticks = cfg.heartbeat_ticks,
                             seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                             min_bots = args.min_bots, max_bots = args.max_bots,
                             swarm_rng = args.swarm_rng)
    worker, env = unity_env.make_unity_worker(worker_id = 0, num_arenas = args.arenas,
                                              no_graphics = True, base_port = args.base_port,
                                              time_scale = args.time_scale)
    names = None
    if os.path.abspath(args.formations).rstrip("/").endswith("formations"):
        names = [os.path.basename(p) for p in formation_paths(args.formations, ".png", args.limit, None)]
    trainer = Trainer.from_workers([worker], cfg, encoder, image_pool, image_names = names)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init).to(cfg.device)
    trainer.setup()
    cfg.motor_override = "simple_oracle"
    trainer._bc_capture = True

    # ── instrumentation ────────────────────────────────────────────────────
    st = {"ctx": None, "actor": None, "oracle_draw": None,
          "twin": False, "armed": True}

    orig_draw = actor_io.sample_split_event

    def draw(seeds, walls, rows, valid, cfg_, rng, wall_seed_xy = None):
        if st["twin"]:
            # Twin calls receive the ALREADY-narrowed event (walls (n, WALL_SIZE)
            # one-hot-valued, wall_seed_xy (n, 2) valid only where a wall won).
            # Bypass the draw entirely -- re-running it here would both re-narrow
            # an already-narrowed table (index error) and, worse, hand the twin a
            # random wall slot on ticks whose narrowed event was a neighbour.
            n = seeds.shape[0]
            dev = seeds.device
            return (torch.zeros(n, 1, device = dev), torch.zeros(n, SEED_SIZE, device = dev),
                    walls, wall_seed_xy)
        tc, seed_part, wall_part, xy_part = orig_draw(seeds, walls, rows, valid, cfg_, rng,
                                                      wall_seed_xy = wall_seed_xy)
        if not st["twin"]:
            if st["ctx"] == "actor" and st["actor"] is None:
                st["actor"] = {"tc": tc.detach(), "seed": seed_part.detach(),
                               "wall": wall_part.detach(),
                               "xy": xy_part.detach() if xy_part is not None else None}
            elif st["ctx"] == "oracle" and st["oracle_draw"] is None:
                st["oracle_draw"] = {"wall": wall_part.detach(),
                                     "xy": xy_part.detach() if xy_part is not None else None}
        return tc, seed_part, wall_part, xy_part

    orig_motors = SO.simple_oracle_motors

    def motors(worker_, arena_ids, locals_, walls, wall_seed_xy, cfg_, rng, pool,
               belief_attr = "belief", **kw):
        twin = st["twin"]
        if not twin:
            st["ctx"] = "oracle"
        try:
            m = orig_motors(worker_, arena_ids, locals_, walls, wall_seed_xy, cfg_, rng, pool,
                            belief_attr = belief_attr, **kw)
        finally:
            if not twin:
                st["ctx"] = None
        if twin or not st["armed"]:
            return m
        # Everything below runs BEFORE this call returns to act(), i.e. while
        # worker.last_dec_step still holds the PRE-tick values -- the same
        # odometry view the real oracle just used. Running the twins after
        # act() would dead-reckon zero motion (the frozen-pose failure
        # record_tape's ordering comment documents) and void every number.
        oc = {
            "motor": m.detach().cpu(),
            "mstate": [worker_.simple_motor_state.get(int(a), {}).get(int(l))
                       for a, l in zip(arena_ids.tolist(), locals_.tolist())],
            "walls_raw": walls.detach().cpu(),
        }
        try:
            _tick_compare(worker_, arena_ids, locals_, oc)
        except Exception:
            skip_reasons["exception"] = skip_reasons.get("exception", 0) + 1
            if detail_printed[0] < 3:
                detail_printed[0] += 1
                traceback.print_exc()
        return m

    orig_act = actor_io.act

    def act(buffer, policy_, worker_, decision_steps, cfg_, rng, *a, **kw):
        st["actor"] = None
        st["oracle_draw"] = None
        st["ctx"] = "actor"
        try:
            return orig_act(buffer, policy_, worker_, decision_steps, cfg_, rng, *a, **kw)
        finally:
            st["ctx"] = None

    twin_state = {}
    control_state = {}
    twin_belief_attr = "audit_twin_belief"
    control_belief_attr = "audit_control_belief"
    series = []  # per-tick divergence series
    skip_reasons = {}
    detail_printed = [0]

    def _run_twin(worker_, arena_ids, locals_, walls_arg, xy_arg, state_store, belief_attr):
        saved = {k: getattr(worker_, k, None) for k in ORACLE_STATE_ATTRS}
        for k in ORACLE_STATE_ATTRS:
            if k in state_store:
                setattr(worker_, k, state_store[k])
            elif saved[k] is not None:
                setattr(worker_, k, {})
        st["twin"] = True
        try:
            motor = orig_motors(worker_, arena_ids, locals_, walls_arg, xy_arg, cfg,
                                trainer.sample_rng, cfg._oracle_formation_pool,
                                belief_attr = belief_attr)
            mstate = [worker_.simple_motor_state.get(int(a), {}).get(int(l))
                      for a, l in zip(arena_ids.tolist(), locals_.tolist())]
        finally:
            st["twin"] = False
            for k in ORACLE_STATE_ATTRS:
                state_store[k] = getattr(worker_, k, None)
                if saved[k] is None:
                    if hasattr(worker_, k):
                        delattr(worker_, k)
                else:
                    setattr(worker_, k, saved[k])
        return motor, mstate

    def _tick_compare(worker_, arena_ids, locals_, oc):
        ad = st["actor"]
        od = st["oracle_draw"]
        n = int(arena_ids.shape[0])
        reason = None
        if ad is None:
            reason = "no-actor-draw"
        elif od is None:
            reason = "no-oracle-draw"
        elif oc["motor"].shape[0] != n or ad["wall"].shape[0] != n:
            reason = "row-mismatch %d vs %d vs %d" % (n, oc["motor"].shape[0], ad["wall"].shape[0])
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if detail_printed[0] < 3 and reason not in ("empty-tick",):
                detail_printed[0] += 1
                print("audit: skip (%s), n=%d" % (reason, n), flush = True)
            return
        # control twin: consumes the ORACLE's own narrowed draw. Must reproduce
        # label A almost exactly -- if it does not, the twin machinery, not the
        # draw, is the difference, and the treatment numbers are void.
        motor_c, mstate_c = _run_twin(worker_, arena_ids, locals_, od["wall"], od.get("xy"),
                                      control_state, control_belief_attr)
        # treatment twin: consumes the ACTOR's narrowed event (the unified draw)
        motor_b, mstate_b = _run_twin(worker_, arena_ids, locals_, ad["wall"], ad.get("xy"),
                                      twin_state, twin_belief_attr)
        # compare
        a_kind, a_side = _decode_events(ad["tc"].to(torch.float32).cpu().numpy())
        o_kind, o_side = _decode_events_od(od["wall"])
        walls_active = oc["walls_raw"].sum(dim = 1) > 0
        dmax = (oc["motor"] - motor_b.cpu()).abs().max(dim = 1).values.numpy()
        dmax_c = (oc["motor"] - motor_c.cpu()).abs().max(dim = 1).values.numpy()
        for i in range(n):
            si = STATE_NAMES.index(oc["mstate"][i]) if oc["mstate"][i] in STATE_NAMES else -1
            row = acc(si)
            row["n"] += 1
            row["dmax_sum"] += float(dmax[i])
            if dmax[i] > args.tol:
                row["changed"] += 1
            if walls_active[i]:
                row["wallactive"] += 1
            dis = (int(a_kind[i]) != o_kind[i]) or (int(a_kind[i]) == 2 and int(a_side[i]) != int(o_side[i]))
            if dis:
                row["drawdis"] += 1
                if dmax[i] > args.tol:
                    row["dis_changed"] += 1
            if mstate_b[i] != oc["mstate"][i]:
                row["linediv"] += 1
            if dmax_c[i] > args.tol:
                row["ctrl_changed"] += 1
            if mstate_c[i] != oc["mstate"][i]:
                row["ctrl_linediv"] += 1
        series.append((float((dmax > args.tol).mean()),
                       float(np.mean([mstate_b[i] != oc["mstate"][i] for i in range(n)]))))

    def _decode_events_od(wall_part):
        w = wall_part.to(torch.float32).cpu().numpy()
        has = w.sum(axis = 1) > 0
        kind = np.where(has, 2, 0)
        side = np.where(has, w.argmax(axis = 1), -1)
        return kind, side

    acc_table = {si: {"n": 0, "changed": 0, "wallactive": 0, "drawdis": 0,
                      "dis_changed": 0, "linediv": 0, "dmax_sum": 0.0,
                      "ctrl_changed": 0, "ctrl_linediv": 0}
                 for si in range(-1, len(STATE_NAMES))}
    acc = lambda si: acc_table[si]

    actor_io.sample_split_event = draw
    SO.sample_split_event = draw
    SO.simple_oracle_motors = motors
    actor_io.act = act

    t0 = time.time()
    ticks = 0
    try:
        while ticks < args.ticks:
            trainer.collect(policy, None, deterministic = False)
            ticks += int(cfg.rollout_steps)
            el = time.time() - t0
            print("  %d/%d ticks, %.1f min elapsed" % (ticks, args.ticks, el / 60.0), flush = True)
    finally:
        actor_io.sample_split_event = orig_draw
        SO.sample_split_event = orig_draw
        SO.simple_oracle_motors = orig_motors
        actor_io.act = orig_act
        try:
            env.close()
        except Exception as exc:
            print("warning: env close failed (%s)" % exc, flush = True)

    # ── report ─────────────────────────────────────────────────────────────
    print("\n================ live draw-unification audit ================")
    print("ticks %d, tolerance %.3f, skip reasons: %s"
          % (ticks, args.tol, skip_reasons or "none"))
    print("\n%-16s%8s%9s%10s%10s%12s%10s%12s%12s" %
          ("decision state", "n", "P(draw", "P(label", "P(label", "P(state-line", "mean",
           "CONTROL", "CONTROL"))
    print("%-16s%8s%9s%10s%10s%12s%10s%12s%12s" %
          ("", "", "disagree)", "change)", "chg|dis)", "divergence)", "|dA-dB|",
           "chg (validate)", "linediv)"))
    rows = [(STATE_NAMES[si] if si >= 0 else "ALL", acc_table[si]) for si in acc_table]
    for name, r in rows:
        if not r["n"]:
            continue
        print("%-16s%8d%9.4f%10.4f%10.4f%12.4f%10.4f%12.4f%12.4f" %
              (name, r["n"],
               r["drawdis"] / r["n"],
               r["changed"] / r["n"],
               r["dis_changed"] / r["drawdis"] if r["drawdis"] else float("nan"),
               r["linediv"] / r["n"],
               r["dmax_sum"] / r["n"],
               r["ctrl_changed"] / r["n"],
               r["ctrl_linediv"] / r["n"]))
    wall_n = sum(acc_table[si]["n"] for si in WALL_STATES)
    if wall_n:
        wd = sum(acc_table[si]["drawdis"] for si in WALL_STATES)
        wc = sum(acc_table[si]["changed"] for si in WALL_STATES)
        wa = sum(acc_table[si]["wallactive"] for si in WALL_STATES)
        print("\nwall-handling states: draw-disagreement %.4f, label-change %.4f, "
              "P(wall signal present) %.4f" % (wd / wall_n, wc / wall_n, wa / wall_n))
    all_rows = acc_table[-1]
    if all_rows["n"]:
        print("ALL states: label-change %.4f, draw-disagreement %.4f"
              % (all_rows["changed"] / all_rows["n"], all_rows["drawdis"] / all_rows["n"]))
    if series:
        arr = np.array(series)
        q = max(1, len(arr) // 10)
        print("\ndivergence over time (P(label change) / P(state-line divergence), by decile):")
        for i in range(0, len(arr), q):
            chunk = arr[i:i + q]
            print("  ticks %5d-%5d: %.4f / %.4f" % (i, i + len(chunk), chunk[:, 0].mean(),
                                                    chunk[:, 1].mean()))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok = True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "skip_reasons": skip_reasons,
                       "per_state": {("ALL" if si < 0 else STATE_NAMES[si]): v
                                     for si, v in acc_table.items()},
                       "state_names": STATE_NAMES, "series": series[-2000:]}, f, indent = 2)
        print("\nwrote %s" % args.out)


def main(argv = None):
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    sub = ap.add_subparsers(dest = "mode", required = True)

    tp = sub.add_parser("tape", help = "decode events from recorded tapes, no simulation")
    tp.add_argument("tapes", nargs = "+")
    tp.add_argument("--json", default = None)

    lv = sub.add_parser("live", help = "short Unity rollout with a twin unified-draw oracle")
    lv.add_argument("--ticks", type = int, default = 1500)
    lv.add_argument("--arenas", type = int, default = 2)
    lv.add_argument("--instances", type = int, default = 1)
    lv.add_argument("--limit", type = int, default = 2000)
    lv.add_argument("--heartbeat", type = int, default = 48)
    lv.add_argument("--rollout", type = int, default = 192)
    lv.add_argument("--min-bots", type = int, default = 40)
    lv.add_argument("--max-bots", type = int, default = 60)
    lv.add_argument("--swarm-rng", type = int, default = 1234)
    lv.add_argument("--seed", type = int, default = 99)
    lv.add_argument("--device", default = "cuda")
    lv.add_argument("--base-port", type = int, default = 5157)
    lv.add_argument("--time-scale", type = float, default = 20.0)
    lv.add_argument("--tol", type = float, default = 0.01,
                    help = "label delta counted as a change")
    lv.add_argument("--formations", default = "../data/formations")
    lv.add_argument("--encoder", default = "../data/image_encoder.pt")
    lv.add_argument("--out", default = None)

    args = ap.parse_args(argv)
    if args.mode == "tape":
        audit_tape(args.tapes, args.json)
    else:
        audit_live(args)


if __name__ == "__main__":
    main()
