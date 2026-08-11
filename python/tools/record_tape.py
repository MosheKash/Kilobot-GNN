"""record_tape.py -- record an oracle-driven BC tape against real Unity.

A "tape" is what val_tape.py already defines: per-robot ORDERED sequences of
(tc, prop) observations with the oracle's motor target and state label at every
decision. val_tape.build_tape records one for validation, inside a training
run, from whatever trainer that run already had. This is the same thing as a
standalone tool, sized for training data rather than for a few dozen validation
sequences:

  * several Unity players in parallel, so a long tape is recorded in a
    reasonable wall clock;
  * flat numpy accumulation instead of a per-decision python list, so a
    multi-million-decision tape does not blow up RSS before it is packed;
  * float16 on disk (roughly 90 bytes per decision) with the same key layout
    val_tape.load_tape/replay_tape already read.

Robot identity comes from the trainer's own traj_id, which is unique per
(arena, robot, episode) across every worker of one trainer -- arena ids alone
collide between workers, and a robot that respawns must not be spliced onto its
predecessor's sequence.

The recorded observations are exactly what actor_io.act would hand the actor at
that decision, because they ARE that: the tape is captured through the ordinary
bc_capture path, with simple_oracle driving the motors. Nothing here recomputes
an observation.

usage:
  python tools/record_tape.py ../results/tapes/train.pt --ticks 12000 \
      --instances 4 --arenas 4 --formations ../data/formations
"""

import argparse
import atexit
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bc_replay import BC_STATES
from val_tape import TAPE_VERSION


def build_args(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ticks", type = int, default = 12000,
                    help = "environment ticks to record (per player; every player records "
                           "concurrently, so the tape holds instances*arenas*bots sequences "
                           "of roughly ticks/decision-interval steps each)")
    ap.add_argument("--instances", type = int, default = 4)
    ap.add_argument("--arenas", type = int, default = 4)
    ap.add_argument("--formations", default = "../data/formations")
    ap.add_argument("--encoder", default = "../data/image_encoder.pt")
    ap.add_argument("--limit", type = int, default = 5000)
    ap.add_argument("--exclude", default = None,
                    help = "path to a _names.json of held-out formation names to exclude "
                           "from this tape's pool (the file ensure_val_dir writes)")
    ap.add_argument("--min-bots", type = int, default = 40)
    ap.add_argument("--max-bots", type = int, default = 60)
    ap.add_argument("--heartbeat", type = int, default = 48)
    ap.add_argument("--rollout", type = int, default = 192)
    ap.add_argument("--max-episode-steps", type = int, default = 18000)
    ap.add_argument("--use-turn-anchor", action = "store_true")
    ap.add_argument("--use-arrived-head", action = "store_true")
    ap.add_argument("--swarm-rng", type = int, default = 0)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--base-port", type = int, default = 5105)
    ap.add_argument("--time-scale", type = float, default = 20.0)
    ap.add_argument("--driver", choices = ["oracle", "actor"], default = "oracle",
                    help = "who moves the robots. `oracle` is plain behaviour cloning data. "
                           "`actor` runs a trained checkpoint closed-loop and labels every "
                           "decision with what the oracle WOULD have commanded from the same "
                           "state -- DAgger, which is the only way to get training data on the "
                           "states the actor's own mistakes lead it into")
    ap.add_argument("--weights", default = None, help = "checkpoint to drive with (--driver actor)")
    ap.add_argument("--motor-noise", type = float, default = 0.0,
                    help = "with --driver oracle: perturb the EXECUTED motor by Gaussian noise of "
                           "this sigma while labelling every decision with the oracle's own clean "
                           "command. This is DART, and it is aimed at the measured reason cloning "
                           "this teacher stalls: the oracle is a stabilising controller, so 99.9%% "
                           "of its wall_following decisions sit within 5 degrees of the direction "
                           "it is steering toward, and the clone -- which starts misaligned -- "
                           "spends two thirds of its time in a regime the demonstrations barely "
                           "contain. Pushing the teacher off its own manifold makes it demonstrate "
                           "the recoveries. Verified survivable first: at sigma 0.12 the oracle "
                           "still reaches 0.463 coverage against its own 0.638")
    ap.add_argument("--oracle-warmup-ticks", type = int, default = 0,
                    help = "with --driver actor: let the ORACLE drive for this many ticks first, "
                           "then hand over. A DAgger round otherwise only ever sees the first "
                           "phase the actor fails in -- robots that stall against a wall never "
                           "reach navigating, so no amount of rounds produces on-policy data for "
                           "the later states. Warming up puts the swarm in a real mid-episode "
                           "state and then lets the actor's own mistakes take it from there, "
                           "which is the same trick DAgger's expert-mixing coefficient buys, "
                           "without needing to interleave the two controllers tick by tick")
    ap.add_argument("--arrived-threshold", type = float, default = 0.95,
                    help = "the arrived gate's threshold while driving (--driver actor). The "
                           "point of a DAgger tape is the states the actor's own mistakes lead "
                           "to, so this should be whatever deployment uses")
    ap.add_argument("--arrived-release-threshold", type = float, default = 0.0)
    ap.add_argument("--freeze-hidden", dest = "freeze_hidden", action = "store_true", default = False,
                    help = "config.py's own arrived_freeze_hidden; off by default, matching "
                           "tools/eval_closed_loop.py")
    ap.add_argument("--activation", default = None,
                    help = "activation of the checkpoint being driven; read from its meta if unset")
    ap.add_argument("--max-robots", type = int, default = 0,
                    help = "cap the number of robot sequences kept (0 = keep all)")
    ap.add_argument("--min-len", type = int, default = 32,
                    help = "drop sequences shorter than this many decisions")
    return ap.parse_args(argv)


def make_cfg(args):
    """The BC-collection config, borrowed verbatim from the training driver.

    run_bc_monitored.build_train_cfg is the single definition of what a BC
    rollout's config is (including split_prop_time_scale, which the observation
    scaling depends on). Reusing it -- rather than assembling a second Config
    here -- is what keeps a recorded tape identical to what the online path
    would have seen.
    """
    import run_bc_monitored as R
    ns = R.build_parser().parse_args(["<unused>"])
    ns.heartbeat = args.heartbeat
    ns.rollout = args.rollout
    ns.arenas = args.arenas
    ns.device = args.device
    ns.max_episode_steps = args.max_episode_steps
    ns.seed = args.seed
    ns.use_arrived_head = args.use_arrived_head
    ns.use_turn_anchor = args.use_turn_anchor
    ns.out_dir = os.path.dirname(os.path.abspath(args.out))
    cfg = R.build_train_cfg(ns)
    cfg.bc_replay_capacity = 0
    cfg.bc_replay_persist = False
    return cfg


def main(argv = None):
    args = build_args(argv)
    import unity_env
    from encoder import load_encoder
    from formations import build_formation_pool
    from images import build_image_pool
    from kilobot_gnn import Z, build_actor
    from policy import GaussianPolicy
    from run_bc_monitored import preprocess
    from trainer import Trainer

    torch.manual_seed(args.seed)
    # images.formation_paths draws its --limit subset with the `random` module,
    # so without this the pool a tape was recorded against could not be
    # reconstructed from the command line alone.
    import random
    random.seed(args.seed)
    if args.driver == "actor":
        if not args.weights:
            raise SystemExit("--driver actor needs --weights")
        meta = (torch.load(args.weights, map_location = "cpu", weights_only = False)
                .get("meta", {}) or {})
        # The architecture switches are properties of the checkpoint, not of
        # this command line: loading a checkpoint into an actor built with
        # different ones is either a hard state_dict error (the two width
        # flags) or, for the activation, a silently different function.
        args.use_arrived_head = bool(meta.get("use_arrived_head", args.use_arrived_head))
        args.use_turn_anchor = bool(meta.get("use_turn_anchor", args.use_turn_anchor))
        args.activation = args.activation or meta.get("activation", "relu")
        args._use_state_head = bool(meta.get("use_state_head", False))
        args._use_wall_head = bool(meta.get("use_wall_head", False))
        args._use_steer_feature = bool(meta.get("use_steer_feature", False))
        args._use_oracle_head = bool(meta.get("use_oracle_head", False))
        args._oracle_residual = float(meta.get("oracle_residual", 0.05))
        args._oracle_residual_turn = float(meta.get("oracle_residual_turn", 0.0))
        from kilobot_gnn import widths_from_state_dict
        args._widths = widths_from_state_dict(blob["actor"] if "actor" in blob else blob)
    cfg = make_cfg(args)
    cfg.split_activation = args.activation or "relu"
    cfg.arrived_confidence_threshold = args.arrived_threshold
    cfg.arrived_release_threshold = args.arrived_release_threshold
    cfg.arrived_freeze_hidden = args.freeze_hidden
    cfg.use_state_head = bool(getattr(args, "_use_state_head", False))
    cfg.use_wall_head = bool(getattr(args, "_use_wall_head", False))
    cfg.use_steer_feature = bool(getattr(args, "_use_steer_feature", False))
    cfg.use_oracle_head = bool(getattr(args, "_use_oracle_head", False))
    cfg.oracle_residual = float(getattr(args, "_oracle_residual", 0.05))
    cfg.oracle_residual_turn = float(getattr(args, "_oracle_residual_turn", 0.0))
    for _k, _v in getattr(args, "_widths", {}).items():
        setattr(cfg, "split_" + _k, _v)

    exclude = None
    if args.exclude:
        import json
        with open(args.exclude) as f:
            exclude = json.load(f)
        print("excluding %d held-out formations from this tape's pool" % len(exclude), flush = True)

    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit,
                                  device = args.device, exclude = exclude)
    formation_pool = build_formation_pool(args.formations, limit = args.limit, exclude = exclude)
    assert len(image_pool) == len(formation_pool)
    cfg._oracle_formation_pool = formation_pool
    print("pool: %d formations from %s" % (len(formation_pool), args.formations), flush = True)

    envs = []
    workers = []
    for i in range(args.instances):
        unity_env.set_player_env(formations = args.formations, heartbeat_ticks = cfg.heartbeat_ticks,
                                 seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                                 min_bots = args.min_bots, max_bots = args.max_bots,
                                 swarm_rng = args.swarm_rng + i)
        w, env = unity_env.make_unity_worker(worker_id = i, num_arenas = args.arenas,
                                             no_graphics = True, base_port = args.base_port,
                                             time_scale = args.time_scale)
        workers.append(w)
        envs.append(env)

    closed = []

    def close_envs():
        if closed:
            return
        closed.append(True)
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                print("warning: env close failed (%s)" % exc, flush = True)

    atexit.register(close_envs)

    # image_names is what Trainer._absolute_image_index needs to translate a
    # pool index into the index Unity means. With --limit narrower than the
    # folder, python's pool is a random sample, so pool index 0 is not Unity's
    # file 0; the trainer recovers the real one from the %06d filename. Without
    # this the player's floor and its distance field refer to a different
    # drawing than the oracle is steering toward. It does not affect a tape --
    # tc, prop, the oracle's targets and the labels are all python-side -- but
    # it does affect anything reward-based, so pass it.
    #
    # Only correct for a canonically-named full dataset: in a carved-out subset
    # directory (a held-out split) the filenames no longer match their position,
    # and the local index is the right one to send.
    names = None
    if os.path.abspath(args.formations).rstrip("/").endswith("formations"):
        from images import formation_paths
        names = [os.path.basename(p) for p in formation_paths(args.formations, ".png",
                                                              args.limit, exclude)]
    trainer = Trainer.from_workers(workers, cfg, encoder, image_pool, image_names = names)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init).to(cfg.device)
    if args.driver == "actor":
        from checkpoint import load_for_eval
        load_for_eval(args.weights, policy, cfg.device)
        print("driving with %s (oracle labels every decision as a shadow)" % args.weights,
              flush = True)
    trainer.setup()
    warmup = int(args.oracle_warmup_ticks) if args.driver == "actor" else 0
    # During a warm-up the ORACLE drives, which is the ordinary bc_capture path
    # and needs no shadow at all; the hand-over below installs it.
    cfg.motor_override = "none" if (args.driver == "actor" and warmup <= 0) else "simple_oracle"
    trainer._bc_capture = cfg.motor_override == "simple_oracle"
    restore = install_shadow_labeller() if (args.driver == "actor" and warmup <= 0) else None
    if args.driver == "oracle" and args.motor_noise > 0:
        restore = install_expert_noise(args.motor_noise)
        print("expert noise: executing the oracle's command + N(0, %.3f), labelling with the clean one"
              % args.motor_noise, flush = True)
    if warmup > 0:
        print("oracle warm-up: %d ticks before the actor takes over" % warmup, flush = True)

    state_index = {s: i for i, s in enumerate(BC_STATES)}
    blocks = []
    ticks = 0
    t0 = time.time()
    # Ctrl-C (or SIGTERM) stops recording and writes the tape recorded so far,
    # rather than throwing away hours of simulation. Recording gets slower as an
    # episode progresses -- robots cluster, so more of them have a neighbour
    # event on any given tick -- so "how many ticks is enough" is a judgement
    # made while watching it, not one that can be fixed up front.
    import signal
    stop = []

    def _stop(_sig, _frm):
        if not stop:
            print("\ninterrupted -- finishing this collect, then writing what is recorded",
                  flush = True)
        stop.append(True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while ticks < args.ticks and not stop:
            if warmup > 0 and ticks >= warmup and cfg.motor_override != "none":
                # Hand over. The shadow labeller keeps using `simple_belief` --
                # the same particle cloud the warm-up oracle has been tracking
                # this whole time -- so the expert does not restart from a cold,
                # uninformative belief exactly when its labels start to matter.
                cfg.motor_override = "none"
                trainer._bc_capture = False
                restore = install_shadow_labeller()
                print("  handing over to the actor at tick %d" % ticks, flush = True)
            buf = trainer.collect(policy, None,
                                  deterministic = (cfg.motor_override == "none"))
            ticks = ticks + int(cfg.rollout_steps)
            blocks.append(_pack_block(buf, state_index))
            n = sum(b["traj"].shape[0] for b in blocks)
            el = time.time() - t0
            print("  %d/%d ticks, %d decisions, %.1f min elapsed, eta %.1f min"
                  % (ticks, args.ticks, n, el / 60.0,
                     (el / max(ticks, 1)) * max(args.ticks - ticks, 0) / 60.0), flush = True)
    finally:
        close_envs()
        if restore is not None:
            restore()

    tape = pack_tape(blocks, max_robots = args.max_robots, min_len = args.min_len)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok = True)
    tmp = args.out + ".tmp"
    torch.save(tape, tmp)
    os.replace(tmp, args.out)
    T, R = tape["valid"].shape
    counts = {}
    st = tape["state"][tape["valid"]]
    for i, name in enumerate(BC_STATES):
        c = int((st == i).sum())
        if c:
            counts[name] = c
    print("wrote %s: %d sequences x %d steps, %d decisions, states %s, %.0f MB"
          % (args.out, R, T, int(tape["valid"].sum()), counts,
             os.path.getsize(args.out) / 1e6), flush = True)


def install_expert_noise(sigma):
    """Perturb what the oracle EXECUTES while labelling with what it MEANT.

    The distinction is the whole point: the executed motor moves the robot off
    the oracle's stable manifold (and feeds its own dead reckoning, so its
    belief stays honest), while the label stays the command the oracle chose at
    the state it actually found itself in -- which, one tick later, is a
    recovery action. Cloning that teaches the student what to do when it is
    misaligned, which no amount of clean demonstration does.

    Returns a callable that restores both patched functions.
    """
    import actor_io
    import simple_oracle as SO

    orig_motors = SO.simple_oracle_motors
    orig_act = actor_io.act
    clean = {}

    def noisy_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, formation_pool,
                     belief_attr = "belief"):
        motor = orig_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng,
                            formation_pool, belief_attr = belief_attr)
        if motor is None:
            return motor
        clean.clear()
        for i in range(int(arena_ids.shape[0])):
            clean[(int(arena_ids[i]), int(locals_[i]))] = motor[i].detach().cpu().clone()
        return (motor + torch.randn_like(motor) * sigma).clamp(0.0, 1.0)

    def act_clean_label(buffer, policy, worker, decision_steps, cfg, rng, *a, **kw):
        n0 = len(buffer.decisions)
        result = orig_act(buffer, policy, worker, decision_steps, cfg, rng, *a, **kw)
        for d in buffer.decisions[n0:]:
            if d.get("bc_target") is None:
                continue
            key = (int(buffer.steps[d["step_index"]]["arena_id"]), int(d["local"]))
            if key in clean:
                d["bc_target"] = clean[key]
        return result

    SO.simple_oracle_motors = noisy_motors
    actor_io.act = act_clean_label

    def restore():
        SO.simple_oracle_motors = orig_motors
        actor_io.act = orig_act

    return restore


def install_shadow_labeller():
    """Label an ACTOR-driven rollout with what the oracle would have done.

    DAgger's one requirement is that the expert can be queried at states the
    learner reached on its own. simple_oracle can: its decision is a function of
    its own particle filter, its own dead-reckoned heading, and the robot's real
    wall observations -- all of which it maintains itself, from the motor
    commands actually executed, whoever issued them. So it runs here as a pure
    observer: the actor's command still moves the robot (this wraps actor_io.act
    and runs strictly after it), the oracle's returned motor is never sent
    anywhere, and only its label is kept.

    The observation, hidden state and buffer entry are the ones act() already
    produced -- this fills in the three fields act() leaves empty outside BC
    capture (bc_target, oracle_state, arrived_target), matching each decision to
    its row in this tick's batch by (arena, robot).

    Its particle filter lives under `simple_belief`, never `belief`, which
    gather_split_state owns for the actor's own proprioception -- the two track
    different things and would corrupt each other tick by tick. `simple_belief`
    specifically, rather than a name of its own, so that a recording which began
    with an oracle warm-up hands the cloud the warm-up built straight to the
    shadow instead of restarting it from a uniform prior mid-episode.

    Returns a callable that puts actor_io.act back.
    """
    import actor_io
    import simple_oracle as SO
    from kilobot_gnn import SEED_SIZE, WALL_SIZE

    orig_act = actor_io.act

    def act_and_label(buffer, policy, worker, decision_steps, cfg, rng, *a, **kw):
        # The oracle runs BEFORE act(), not after, and the ordering is
        # load-bearing rather than stylistic. simple_oracle_motors derives this
        # tick's motion from `step_count - last_dec_step[a][l]`, and act() sets
        # last_dec_step to step_count for every robot it commands. Called
        # afterwards, the oracle therefore sees steps_since == 0 for every
        # robot, dead-reckons zero motion, never advances its particle filter
        # and never accumulates any rotation -- so its turn never completes, its
        # belief never converges, and every label it produces past the first
        # tick is computed from a frozen pose. actor_io.act's own bc_capture
        # path calls the oracle before the same update for exactly this reason.
        labels = {}
        if cfg.motor_override == "none" and len(decision_steps) > 0:
            device = cfg.device
            vector, rows = actor_io.split_obs(decision_steps.obs, device)
            rows, wall_seed_xy_unity = actor_io._extract_wall_seed_rows(rows, device)
            arena_ids = vector[:, 0].long()
            locals_ = vector[:, 1].long()
            walls = vector[:, 2 + SEED_SIZE:2 + SEED_SIZE + WALL_SIZE]
            wall_seed_xy = actor_io._resolve_wall_seed_xy(worker, arena_ids, locals_, cfg, device,
                                                          wall_seed_xy_unity)
            motors = SO.simple_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng,
                                             getattr(cfg, "_oracle_formation_pool", None),
                                             belief_attr = "simple_belief")
            for i in range(int(arena_ids.shape[0])):
                a_id = int(arena_ids[i]); l = int(locals_[i])
                arrived = getattr(worker, "simple_state", {}).get(a_id, {}).get(l) == "arrived"
                labels[(a_id, l)] = (motors[i].detach().cpu(),
                                     getattr(worker, "simple_motor_state", {}).get(a_id, {}).get(l),
                                     torch.tensor([1.0 if arrived else 0.0]))

        n0 = len(buffer.decisions)
        result = orig_act(buffer, policy, worker, decision_steps, cfg, rng, *a, **kw)
        for d in buffer.decisions[n0:]:
            a_id = int(buffer.steps[d["step_index"]]["arena_id"])
            entry = labels.get((a_id, int(d["local"])))
            if entry is None:
                continue
            d["bc_target"], d["oracle_state"], d["arrived_target"] = entry
        return result

    actor_io.act = act_and_label

    def restore():
        actor_io.act = orig_act

    return restore


def _pack_block(buf, state_index):
    """One collect's decisions as flat numpy arrays."""
    traj, step, tc, prop, tgt, state, arrived, arrived_valid = [], [], [], [], [], [], [], []
    for d in buf.decisions:
        if d.get("bc_target") is None or d.get("prop") is None:
            continue
        s = buf.steps[d["step_index"]]
        traj.append(int(s["traj_id"][d["local"]]))
        step.append(int(s["env_step"]))
        tc.append(d["transmission"].detach().cpu().numpy())
        prop.append(d["prop"].detach().cpu().numpy())
        tgt.append(d["bc_target"].detach().cpu().numpy())
        state.append(state_index.get(d.get("oracle_state"), -1))
        at = d.get("arrived_target")
        arrived.append(0.0 if at is None else float(at.reshape(-1)[0]))
        arrived_valid.append(at is not None)
    if not traj:
        return {"traj": np.zeros(0, np.int64), "step": np.zeros(0, np.int64),
                "tc": np.zeros((0, 1), np.float16), "prop": np.zeros((0, 1), np.float16),
                "tgt": np.zeros((0, 2), np.float16), "state": np.zeros(0, np.int8),
                "arrived": np.zeros(0, np.float16), "arrived_valid": np.zeros(0, bool)}
    return {"traj": np.asarray(traj, np.int64), "step": np.asarray(step, np.int64),
            "tc": np.asarray(tc, np.float16), "prop": np.asarray(prop, np.float16),
            "tgt": np.asarray(tgt, np.float16), "state": np.asarray(state, np.int8),
            "arrived": np.asarray(arrived, np.float16),
            "arrived_valid": np.asarray(arrived_valid, bool)}


def pack_tape(blocks, max_robots = 0, min_len = 32):
    """Flat per-decision blocks -> the padded (T, R, ...) tape layout.

    Same keys and semantics as val_tape._pack, so val_tape.load_tape,
    tape_state_counts and replay_tape all read this unchanged; float16 storage
    for the observation tensors, which is what makes a multi-million-decision
    tape a few hundred MB instead of a few GB.
    """
    blocks = [b for b in blocks if b["traj"].shape[0] > 0]
    if not blocks:
        raise SystemExit("no decisions recorded -- nothing to write")
    traj = np.concatenate([b["traj"] for b in blocks])
    step = np.concatenate([b["step"] for b in blocks])
    tc = np.concatenate([b["tc"] for b in blocks])
    prop = np.concatenate([b["prop"] for b in blocks])
    tgt = np.concatenate([b["tgt"] for b in blocks])
    state = np.concatenate([b["state"] for b in blocks])
    arrived = np.concatenate([b["arrived"] for b in blocks])
    arrived_valid = np.concatenate([b["arrived_valid"] for b in blocks])

    # Sort by (traj, env_step) once: every sequence is then a contiguous slice
    # in the correct order, which is both the grouping and the ordering the
    # tape needs, without a python dict of per-robot lists.
    order = np.lexsort((step, traj))
    traj, step = traj[order], step[order]
    tc, prop, tgt = tc[order], prop[order], tgt[order]
    state, arrived, arrived_valid = state[order], arrived[order], arrived_valid[order]

    uniq, start_idx, lengths = np.unique(traj, return_index = True, return_counts = True)
    keep = lengths >= max(1, int(min_len))
    uniq, start_idx, lengths = uniq[keep], start_idx[keep], lengths[keep]
    if max_robots and len(uniq) > max_robots:
        # Longest sequences first: a tape is worth more per stored element the
        # further into an episode it reaches, and the short ones are robots
        # whose episode ended right after the recording started.
        pick = np.argsort(-lengths)[:max_robots]
        pick = np.sort(pick)
        uniq, start_idx, lengths = uniq[pick], start_idx[pick], lengths[pick]
    R = len(uniq)
    T = int(lengths.max())
    tape = {"version": TAPE_VERSION,
            "tc": torch.zeros(T, R, tc.shape[1], dtype = torch.float16),
            "prop": torch.zeros(T, R, prop.shape[1], dtype = torch.float16),
            "tgt": torch.zeros(T, R, 2, dtype = torch.float16),
            "state": torch.full((T, R), -1, dtype = torch.long),
            "arrived": torch.zeros(T, R, dtype = torch.float16),
            "arrived_valid": torch.zeros(T, R, dtype = torch.bool),
            "valid": torch.zeros(T, R, dtype = torch.bool)}
    for j in range(R):
        s = int(start_idx[j]); n = int(lengths[j])
        sl = slice(s, s + n)
        tape["tc"][:n, j] = torch.from_numpy(tc[sl])
        tape["prop"][:n, j] = torch.from_numpy(prop[sl])
        tape["tgt"][:n, j] = torch.from_numpy(tgt[sl])
        tape["state"][:n, j] = torch.from_numpy(state[sl].astype(np.int64))
        tape["arrived"][:n, j] = torch.from_numpy(arrived[sl])
        tape["arrived_valid"][:n, j] = torch.from_numpy(arrived_valid[sl])
        tape["valid"][:n, j] = True
    return tape


if __name__ == "__main__":
    main()
