"""eval_closed_loop.py -- run one driver (the oracle, or a trained actor) in
real Unity on held-out formations and record how the swarm actually does.

This is the measurement the BC metrics are a proxy for. Imitation error says
how closely the actor reproduces the oracle's motor command on recorded
observations; this says whether the swarm the actor drives ends up in the
shape -- closed loop, its own mistakes compounding, on formations it never
trained on.

Both drivers are measured the SAME way, which the in-training numbers are not:
`oracle_cov` in bc.py is the oracle's own belief-based arrived fraction while
`actor_eval_cov` is ground-truth coverage (trainer.py's _record_snapshots picks
one or the other by who is driving), so the two are not comparable. Here every
run reports:

  coverage      fraction of robots within cfg.tau_v of the target shape,
                ground truth from the critic snapshot -- the task itself
  mean_dist     mean normalised distance to the shape, which keeps moving
                after coverage saturates
  stopped       fraction of robots that have stopped: the oracle's own
                `arrived` state, or the actor's arrived-head gate

Run it once per driver with the SAME --swarm-rng and --seed and the two runs
see the same formations and the same spawns, so the comparison is paired.

usage:
  python tools/eval_closed_loop.py ../results/bc_v2/eval_oracle.json --mode oracle
  python tools/eval_closed_loop.py ../results/bc_v2/eval_actor.json  --mode actor \
      --weights ../results/bc_v2/run1/actor_best.pt
"""

import argparse
import atexit
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices = ["oracle", "actor"], required = True)
    ap.add_argument("--weights", default = None, help = "actor checkpoint (required for --mode actor)")
    ap.add_argument("--ticks", type = int, default = 12000)
    ap.add_argument("--sample-every", type = int, default = 200,
                    help = "environment ticks between metric samples")
    ap.add_argument("--instances", type = int, default = 2)
    ap.add_argument("--arenas", type = int, default = 4)
    ap.add_argument("--formations", default = "../results/bc_v2/val_formations")
    ap.add_argument("--encoder", default = "../data/image_encoder.pt")
    ap.add_argument("--limit", type = int, default = 2000)
    ap.add_argument("--min-bots", type = int, default = 40)
    ap.add_argument("--max-bots", type = int, default = 60)
    ap.add_argument("--heartbeat", type = int, default = 48)
    ap.add_argument("--activation", default = None,
                    help = "override the activation; default reads the checkpoint's meta")
    ap.add_argument("--use-arrived-head", action = "store_true", default = None)
    ap.add_argument("--use-turn-anchor", action = "store_true", default = None)
    ap.add_argument("--arrived-threshold", type = float, default = 0.95)
    ap.add_argument("--arrived-release-threshold", type = float, default = 0.0)
    ap.add_argument("--closed-form-arrived", action = "store_true",
                    help = "config.py's own use_closed_form_arrived: gate the actor's stop on the "
                           "oracle's own closed-form arrival rule (filter distance to the robot's "
                           "own target below cfg.tau_v, with conf_pos past the localization floor) "
"computed from the actor's observation, instead of the learned arrived "
                            "head. Terminal, like the oracle's arrived state")
    ap.add_argument("--closed-form-dist", type = float, default = 0.0,
                    help = "arrival radius for --closed-form-arrived, in normalized units. 0 means "
                           "cfg.tau_v (the oracle's own rule); the actor's own filter under-reports "
                           "closeness, so a larger value parks robots where this filter does "
                           "certify arrival")
    ap.add_argument("--closed-form-hybrid", action = "store_true",
                    help = "with --closed-form-arrived, run the closed-form rule in OR with the "
                           "learned arrived head instead of replacing it (config.py's own "
                           "closed_form_hybrid)")
    ap.add_argument("--freeze-hidden", dest = "freeze_hidden", action = "store_true", default = False,
                    help = "config.py's own arrived_freeze_hidden. Off by default here, unlike the "
                           "Config default: a checkpoint trained by bc_offline.py has seen the "
                           "arrived stretches with the recurrence running, so freezing it at "
                           "deployment would be a recurrence the training never saw")
    ap.add_argument("--swarm-rng", type = int, default = 500)
    ap.add_argument("--seed", type = int, default = 7)
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--base-port", type = int, default = 5500)
    ap.add_argument("--headed", action = "store_true",
                    help = "show a LIVE Unity window (no_graphics=False) and stream the same "
                           "coverage/stopped/settled console numbers as the headless run. Use "
                           "--time-scale 1 (the default here) to watch in real time; a single "
                           "--instances 1 arena is what you want for an eyeball check. This is "
                           "the numeric + visual counterpart to scripts/watch_actor.sh")
    ap.add_argument("--time-scale", type = float, default = None,
                    help = "Unity engine time_scale. Default: 1 when --headed (real-time viewing), "
                           "else 20 for headless speed")
    ap.add_argument("--bake-rotation-steps", type = int, default = 0,
                    help = "KILOBOT_BAKE_ROTATION_STEPS for the player. The default here is 0, "
                           "NOT the player's own default of 1: at 1, Unity's baked distance "
                           "field is formations.py's geometry rotated 90 degrees CCW (phase 31 "
                           "added the rotation in C#, phase 33 removed the matching one in "
                           "Python and the two were never reconciled), so `coverage` scores the "
                           "swarm against a shape 90 degrees from the one the oracle is steering "
                           "it into. Measured, not assumed: with this at 1, Unity's per-robot "
                           "distance correlates 0.998 with the python distance computed at "
                           "90-degree-rotated positions, and 0.0 with it computed as-is")
    ap.add_argument("--wall-seed-position", action = "store_true",
                    help = "config.py's own oracle_wall_seed_position. Off in Config and in every "
                           "tape recorded here, which means wall contact constrains one axis only "
                           "and a robot has to reach a CORNER before its belief collapses -- so "
                           "how accurately it localizes depends on how precisely it follows the "
                           "wall. On, the nearest wall seed's own broadcast position constrains "
                           "both axes, which decouples localization from trajectory precision")
    ap.add_argument("--motor-bias", type = float, default = 0.0,
                    help = "with --mode oracle: give each robot a PERSISTENT differential offset "
                           "drawn once from N(0, this), instead of fresh noise every tick. A "
                           "clone's error is not i.i.d. -- it is a steering error that persists "
                           "while the robot stays in the same state -- and a correlated error of "
                           "a given size is far more damaging than an independent one, which is "
                           "what this measures")
    ap.add_argument("--motor-noise", type = float, default = 0.0,
                    help = "with --mode oracle: perturb the ORACLE's own motor command by "
                           "Gaussian noise of this standard deviation before it is executed. "
                           "This is the control for a behaviour-cloning result -- a clone that "
                           "reproduces its teacher's command to within e per wheel is, "
                           "dynamically, the teacher driving with e of noise. If the oracle "
                           "collapses at the clone's own error level, the clone was never going "
                           "to work at that accuracy and the objective, not the fit, is what "
                           "needs changing")
    ap.add_argument("--no-positions", action = "store_true",
                    help = "do not store per-sample robot positions (they are what the "
                           "demo animation is rendered from)")
    return ap


def main(argv = None):
    args = build_parser().parse_args(argv)
    if args.mode == "actor" and not args.weights:
        raise SystemExit("--mode actor needs --weights")

    import unity_env
    from config import Config
    from encoder import load_encoder
    from formations import build_formation_pool
    from images import build_image_pool
    from kilobot_gnn import Z, build_actor
    from policy import GaussianPolicy
    from reward import coverage
    from run_bc_monitored import preprocess
    from trainer import Trainer

    meta = {}
    blob = None
    if args.weights:
        blob = torch.load(args.weights, map_location = "cpu", weights_only = False)
        meta = blob.get("meta", {}) or {}
    activation = args.activation or meta.get("activation", "relu")
    use_arrived_head = (args.use_arrived_head if args.use_arrived_head is not None
                        else bool(meta.get("use_arrived_head", False)))
    use_turn_anchor = (args.use_turn_anchor if args.use_turn_anchor is not None
                       else bool(meta.get("use_turn_anchor", False)))
    use_state_head = bool(meta.get("use_state_head", False))
    use_wall_head = bool(meta.get("use_wall_head", False))
    use_steer_feature = bool(meta.get("use_steer_feature", False))
    use_oracle_head = bool(meta.get("use_oracle_head", False))
    oracle_residual = float(meta.get("oracle_residual", 0.05))
    oracle_residual_turn = float(meta.get("oracle_residual_turn", 0.0))

    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.seed_layout = "corners"
    cfg.heartbeat_ticks = args.heartbeat
    cfg.rollout_steps = args.sample_every
    cfg.num_arenas = args.arenas
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.oracle_known_start_heading = True
    cfg.motor_override = "simple_oracle" if args.mode == "oracle" else "none"
    # Nothing may end an episode during an evaluation: a reset mid-run would
    # restart the swarm on a new formation and the curve would measure two
    # different problems glued together. Coverage cannot exceed 1.0 and the
    # step limit is set past the run length.
    cfg.success_threshold = 1.1
    cfg.max_episode_steps = args.ticks * 10
    cfg.use_arrived_head = use_arrived_head
    cfg.use_turn_anchor = use_turn_anchor
    # Training-only head, but it is in the state_dict, so the actor has to be
    # built with it or load_state_dict rejects the checkpoint.
    cfg.use_state_head = use_state_head
    cfg.use_wall_head = use_wall_head
    cfg.use_steer_feature = use_steer_feature
    cfg.use_oracle_head = use_oracle_head
    cfg.oracle_residual = oracle_residual
    cfg.oracle_residual_turn = oracle_residual_turn
    cfg.split_activation = activation
    # Widths are recovered from the checkpoint's own tensors, so anything
    # trained with --gru-hidden loads instead of failing on a shape mismatch.
    if blob is not None:
        from kilobot_gnn import widths_from_state_dict
        for _k, _v in widths_from_state_dict(blob["actor"] if "actor" in blob else blob).items():
            setattr(cfg, "split_" + _k, _v)
    cfg.arrived_confidence_threshold = args.arrived_threshold
    cfg.arrived_release_threshold = args.arrived_release_threshold
    cfg.use_closed_form_arrived = args.closed_form_arrived
    cfg.closed_form_arrival_dist = args.closed_form_dist
    cfg.closed_form_hybrid = args.closed_form_hybrid
    cfg.arrived_freeze_hidden = args.freeze_hidden
    cfg.oracle_wall_seed_position = args.wall_seed_position
    # matches run_bc_monitored.build_train_cfg, which is what every tape and
    # every trained checkpoint's observations were scaled by
    cfg.split_prop_time_scale = 0.058

    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit, device = args.device)
    formation_pool = build_formation_pool(args.formations, limit = args.limit)
    cfg._oracle_formation_pool = formation_pool
    # Every architecture switch that changes the deployed forward pass is
    # printed, not just the ones that change the state_dict's shape. use_oracle_head
    # does not change any tensor's shape, so a checkpoint trained with it loads
    # cleanly into an actor built without it and silently runs a different
    # function -- the failure would look like a bad policy, not a bad load.
    print("%s: %d formations from %s, activation=%s arrived_head=%s turn_anchor=%s "
          "oracle_head=%s (residual %.3f / %.4f) steer_feature=%s"
          % (args.mode, len(formation_pool), args.formations, activation,
             use_arrived_head, use_turn_anchor, use_oracle_head, oracle_residual,
             oracle_residual_turn, use_steer_feature), flush = True)

    # Must be set before any player is launched: ImageLibrary reads it while
    # baking, in the environment the player inherited at start-up.
    os.environ["KILOBOT_BAKE_ROTATION_STEPS"] = str(int(args.bake_rotation_steps))
    time_scale = args.time_scale if args.time_scale is not None else (1.0 if args.headed else 20.0)
    if args.headed:
        print("HEADED: showing a live Unity window; time_scale=%.1f (Ctrl+C to stop)" % time_scale,
              flush = True)
    envs, workers = [], []
    for i in range(args.instances):
        unity_env.set_player_env(formations = args.formations, heartbeat_ticks = cfg.heartbeat_ticks,
                                 seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                                 min_bots = args.min_bots, max_bots = args.max_bots,
                                 swarm_rng = args.swarm_rng + i)
        w, env = unity_env.make_unity_worker(worker_id = i, num_arenas = args.arenas,
                                             no_graphics = not args.headed, base_port = args.base_port,
                                             time_scale = time_scale)
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

    if args.motor_bias > 0 and args.mode == "oracle":
        import simple_oracle as _sob
        _orig_b = _sob.simple_oracle_motors
        _bias = {}

        def _biased(worker, arena_ids, locals_, *a, **kw):
            motor = _orig_b(worker, arena_ids, locals_, *a, **kw)
            if motor is None:
                return motor
            out = motor.clone()
            for i in range(int(arena_ids.shape[0])):
                key = (int(arena_ids[i]), int(locals_[i]))
                if key not in _bias:
                    _bias[key] = float(torch.randn(1)) * args.motor_bias
                b = _bias[key]
                out[i, 0] = out[i, 0] + b
                out[i, 1] = out[i, 1] - b
            return out.clamp(0.0, 1.0)

        _sob.simple_oracle_motors = _biased
        print("oracle driving with a PERSISTENT per-robot steering bias, sigma=%.3f" % args.motor_bias,
              flush = True)

    if args.motor_noise > 0 and args.mode == "oracle":
        # Wraps the oracle's own motor function rather than the action written to
        # the worker, so the noise lands in exactly one place: the command that
        # is executed. The oracle's belief and heading tracking dead-reckon from
        # the executed motor, so it sees the perturbed motion the way a real
        # controller would -- this is a noisy driver, not a driver being lied to.
        import simple_oracle as _so
        _orig_motors = _so.simple_oracle_motors

        def _noisy(*a, **kw):
            motor = _orig_motors(*a, **kw)
            if motor is None:
                return motor
            return (motor + torch.randn_like(motor) * args.motor_noise).clamp(0.0, 1.0)

        _so.simple_oracle_motors = _noisy
        print("oracle driving with motor noise sigma=%.3f" % args.motor_noise, flush = True)

    trainer = Trainer.from_workers(workers, cfg, encoder, image_pool, formations_dir = os.path.abspath(args.formations))
    trainer._save_positions = not args.no_positions
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init).to(cfg.device)
    if args.weights:
        from checkpoint import load_for_eval
        load_for_eval(args.weights, policy, cfg.device)
        print("loaded %s" % args.weights, flush = True)
    # Re-seeded here, immediately before the first _pick_image, and not only at
    # the top: building the policy draws from the same global generator, and the
    # oracle run builds a different-shaped actor from the one a checkpoint asks
    # for (no arrived head, no turn anchor), so it consumed a different number of
    # draws and every arena then got a different formation. Two runs meant to be
    # paired were comparing different shapes. Spawns were never affected -- those
    # come from the player's own KILOBOT_SWARM_RNG.
    torch.manual_seed(args.seed)
    trainer.setup()

    samples = []
    ticks = 0
    t0 = time.time()
    try:
        # A sample before anything has moved. Coverage has a large floor -- a
        # robot dropped at random in an arena whose target shape is a QuickDraw
        # stroke is already within tau_v of it a third of the time -- so the
        # number is only readable against where it started.
        first = _sample(trainer, workers, cfg, coverage, 0)
        if first["arenas"]:
            samples.append(first)
            print("  tick %6d  coverage %.4f  mean_dist %.4f  stopped %.4f  (spawn)"
                  % (0, first["coverage"], first["mean_dist"], first["stopped"]), flush = True)
        while ticks < args.ticks:
            with torch.no_grad():
                trainer.collect(policy, None, deterministic = True)
            ticks += args.sample_every
            samples.append(_sample(trainer, workers, cfg, coverage, ticks))
            s = samples[-1]
            print("  tick %6d  coverage %.4f  mean_dist %.4f  stopped %.4f  |  settled<5u %.4f  "
                  "near<5u %.4f  median_err %.1f  (%.1f min)"
                  % (ticks, s["coverage"], s["mean_dist"], s["stopped"], s["settled_5"], s["near_5"],
                     float(np.median([a["target_err_median"] for a in s["per_arena"]
                                      if a.get("target_err_median") is not None] or [float("nan")])),
                     (time.time() - t0) / 60.0), flush = True)
    finally:
        close_envs()

    from images import formation_paths
    names = [os.path.basename(p) for p in formation_paths(args.formations, ".png", args.limit, None)]
    final = _final_state(trainer, workers, cfg, formation_pool, names)
    blob = {"mode": args.mode, "weights": args.weights, "meta": meta,
            "args": vars(args), "samples": samples, "final": final,
            "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump(blob, f)
    last = samples[-1]
    print("wrote %s -- final coverage %.4f, mean_dist %.4f, stopped %.4f over %d arenas"
          % (args.out, last["coverage"], last["mean_dist"], last["stopped"], last["arenas"]),
          flush = True)


def _per_robot_stopped(worker, k, m, mode):
    """Which robots have decided they are done, one flag each."""
    if mode == "oracle":
        st = getattr(worker, "simple_state", {}).get(k, {})
        return [bool(st.get(l) == "arrived") for l in range(m)]
    off = getattr(worker, "arrived_switched_off", {}).get(k, {})
    return [bool(off.get(l, False)) for l in range(m)]


def _assigned_targets(worker, k, m):
    """Each robot's own assigned target point, from the authoritative cache.

    observation.ensure_target writes worker.simple_target, and it is the single
    shared resolution point -- simple_oracle steers by it and gather_split_state
    builds the actor's relative-target observation from it, so this is exactly
    the point the robot was told to go to, not a reconstruction of it.
    """
    cache = getattr(worker, "simple_target", {}).get(k, {})
    out = []
    for l in range(m):
        t = cache.get(l)
        out.append(None if t is None else [float(t[0]), float(t[1])])
    return out


def _stopped_fraction(worker, k, m, mode):
    """Fraction of an arena's robots that have decided they are done.

    Deliberately per-driver, because the two drivers stop for different
    reasons and there is no single mechanism to read: the oracle's own
    terminal `arrived` state, and the actor's arrived head having latched the
    gate in actor_io._arrived_head_gate. Both mean "this robot has stopped
    itself", which is what the number is for; neither is ground truth about
    whether it stopped in the right place -- `coverage` is.
    """
    if m <= 0:
        return 0.0
    if mode == "oracle":
        st = getattr(worker, "simple_state", {}).get(k, {})
        return sum(1 for l in range(m) if st.get(l) == "arrived") / float(m)
    off = getattr(worker, "arrived_switched_off", {}).get(k, {})
    return sum(1 for l in range(m) if off.get(l)) / float(m)


def _sample(trainer, workers, cfg, coverage, ticks):
    mode = "oracle" if cfg.motor_override == "simple_oracle" else "actor"
    covs, dists, stops, per_arena = [], [], [], []
    for wi, worker in enumerate(workers):
        for k in range(cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None or snap.get("node") is None:
                continue
            node = snap["node"]
            m = node.shape[0]
            c = float(coverage(node, cfg))
            d = float(node[:, 4].mean())
            s = _stopped_fraction(worker, k, m, mode)
            covs.append(c); dists.append(d); stops.append(s)
            rec = {"worker": wi, "arena": k, "robots": int(m),
                   "coverage": c, "mean_dist": d, "stopped": s}
            # THE metric: a robot has settled when it has stopped itself AND is
            # actually near the point it was assigned. Distance to the shape
            # (coverage) is a weaker question -- a robot can sit on someone
            # else's part of the drawing and count.
            from belief import ARENA_HALF as _AH
            tg = _assigned_targets(worker, k, m)
            st_flags = _per_robot_stopped(worker, k, m, mode)
            pos_raw = node[:, 0:2].cpu().numpy() * _AH
            errs = [float(np.linalg.norm(pos_raw[l] - np.asarray(tg[l])))
                    if tg[l] is not None else None for l in range(m)]
            have = [e for e in errs if e is not None]
            rec["target_err_median"] = float(np.median(have)) if have else None
            for tol in (5.0, 10.0, 20.0):
                near = [e is not None and e < tol for e in errs]
                rec["near_%d" % int(tol)] = float(np.mean(near)) if errs else 0.0
                rec["settled_%d" % int(tol)] = (float(np.mean([n and s2 for n, s2 in zip(near, st_flags)]))
                                                if errs else 0.0)
            if trainer._save_positions:
                # Every sample's positions, in raw arena units, so the run can
                # be replayed as an animation afterwards. ~50 robots x 2 floats
                # per arena per sample: a few MB for a whole evaluation, which
                # is worth being able to SEE the swarm assemble rather than
                # only reading its coverage curve.
                from belief import ARENA_HALF
                rec["pos"] = np.round(node[:, 0:2].cpu().numpy() * ARENA_HALF, 2).tolist()
            per_arena.append(rec)
    agg = {"tick": ticks, "coverage": float(np.mean(covs)) if covs else 0.0,
           "mean_dist": float(np.mean(dists)) if dists else 0.0,
           "stopped": float(np.mean(stops)) if stops else 0.0,
           "arenas": len(covs), "per_arena": per_arena}
    for key in ("near_5", "near_10", "near_20", "settled_5", "settled_10", "settled_20"):
        vals = [a[key] for a in per_arena if a.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else 0.0
    return agg


def _final_state(trainer, workers, cfg, formation_pool, image_names = None):
    """Robot positions and the formation each arena was solving, for the demo plots.

    Positions come out of the snapshot normalised to [-1, 1]; the formation's
    on-pixels are in raw arena units, so positions are scaled by ARENA_HALF to
    put both in the same frame. The shape is subsampled -- a QuickDraw stroke
    can be tens of thousands of pixels and the plot cannot show that anyway.
    """
    from belief import ARENA_HALF
    out = []
    for wi, worker in enumerate(workers):
        for k in range(cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None or snap.get("node") is None:
                continue
            node = snap["node"].cpu().numpy()
            image_id = worker.image_id.get(k) if hasattr(worker, "image_id") else None
            m = node.shape[0]
            entry = {"worker": wi, "arena": k,
                     "image_id": int(image_id) if image_id is not None else None,
                     "pos": (node[:, 0:2] * ARENA_HALF).tolist(), "dist": node[:, 4].tolist(),
                     "target": _assigned_targets(worker, k, m),
                     "stopped_flags": _per_robot_stopped(
                         worker, k, m, "oracle" if cfg.motor_override == "simple_oracle" else "actor")}
            if image_id is not None and formation_pool:
                idx = int(image_id) % len(formation_pool)
                if image_names is not None and idx < len(image_names):
                    entry["formation_name"] = image_names[idx]
                pts = np.asarray(formation_pool[idx].points, dtype = float)
                if pts.shape[0] > 3000:
                    pts = pts[::int(np.ceil(pts.shape[0] / 3000.0))]
                entry["shape"] = pts.tolist()
            out.append(entry)
    return out


if __name__ == "__main__":
    main()
