"""Probes and reporting: characterise the pipeline without training on it.

Nothing here runs a real update. Each function isolates one question:

  control_probe          does a fixed motor command produce the motion the
                         kinematics predict -- i.e. is the action path wired up
  reward_probe           what does the reward function pay for a known geometry
  audit_run              does a stored rollout replay to the same log-probs
                         (audit_replay_ratio is its per-decision ratio check)
  probe_run              what is a loaded policy actually doing, per decision
  watch_oracle           drive the environment from the oracle for a human to
                         watch; no learning at all
  log_iteration_*        the per-iteration and per-arena logging a long run
                         leaves behind for later reading
  print_eval_log_summary format an eval run's captured logs

Behaviour cloning used to live here; it is real training, and moved to bc.py.
"""
import torch

from kilobot_gnn import MESSAGE_SIZE


def watch_oracle(trainer, policy, cfg):
    # KILOBOT_MODE=watch_oracle: open-ended, no-training visual observation
    # of whichever controller is driving -- policy stays exactly as
    # constructed (randomly initialized; the active controller overrides
    # whatever it would have done anyway, so there is nothing to load or
    # warm-start here). Meant for KILOBOT_NUM_ARENAS=1,
    # KILOBOT_NO_GRAPHICS=false. No natural stopping point by design --
    # Ctrl+C when you've seen enough.
    #
    # Previously forced cfg.motor_override =
    # "oracle" unconditionally, "regardless of what KILOBOT_MOTOR_OVERRIDE
    # Fall back only when the caller has not already chosen a controller.
    # Forcing one unconditionally, as an earlier version did, silently
    # discarded whatever KILOBOT_MOTOR_OVERRIDE the launching script had set.
    # The fallback is simple_oracle, the supported teacher.
    if cfg.motor_override not in ("oracle", "simple_oracle"):
        cfg.motor_override = "simple_oracle"
    print(f"watching {cfg.motor_override} drive, no training -- Ctrl+C to stop")
    trainer.setup()
    with torch.no_grad():
        while True:
            trainer.collect(policy, None, deterministic = True)


def control_probe(trainer, policy, cfg, iterations):
    # Characterize the action->motion mapping through the normal action path: force a
    # fixed motor command and measure how far robots actually move. The scripted "fixed"
    # motors are injected through the SAME actions tensor and set_actions call the policy
    # uses (and that the oracle uses), so this is the policy's action path. Confirms a
    # forward command drives the robots and identifies what command means "stop".
    commands = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    print("\n==== CONTROL AUTHORITY: motor command -> displacement ====")
    print("  the 'fixed' override sends commands through the policy's own set_actions path")
    print("  command (L, R)     mean displacement / step")
    results = []
    for cmd in commands:
        cfg.force_motor = cmd
        cfg.motor_override = "fixed"
        trainer.setup()
        ds = dn = 0.0
        for _ in range(iterations):
            trainer.collect(policy, None, deterministic=True)
            p = trainer.rollout_payload()
            ds += p.get("disp_sum", 0.0)
            dn += p.get("disp_count", 0)
        disp = ds / max(dn, 1.0)
        results.append((cmd, disp))
        print("  (%.2f, %.2f)        %.6f" % (cmd[0], cmd[1], disp))
    d_stop = dict(((c[0], c[1]), v) for c, v in results).get((0.5, 0.5), 0.0)
    d_fwd = dict(((c[0], c[1]), v) for c, v in results).get((1.0, 1.0), 0.0)
    print("\n  READ:")
    if d_fwd > max(2 * d_stop, d_stop + 1e-4) and d_fwd > 1e-4:
        print("  full-forward (1,1) moves robots clearly more than (0.5,0.5): the body responds and")
        print("  the action->physics path works. So the policy's parked ~0.5 motor output is simply not")
        print("  being driven toward the high-throttle commands that DO move robots. The fault is in")
        print("  learning to change the motor output, not in the command reaching the robot.")
    elif d_fwd <= max(2 * d_stop, d_stop + 1e-4):
        print("  full-forward (1,1) barely moves robots more than (0.5,0.5): commands are reaching the")
        print("  env but NOT producing proportional motion. The action->physics mapping itself is the")
        print("  suspect -- every upstream thing can be correct and still not matter. This is where the")
        print("  oracle's different outcome would have to come from a path the policy does not share.")
    print("  (the oracle injects motors through this same path, so identical commands move robots")
    print("   identically by construction; this measures whether that shared path has authority.)")
    print()


def audit_replay_ratio(policy, cfg, buf):
    from ppo import _stack_decisions
    if buf is None or len(buf.decisions) == 0:
        return None
    data = _stack_decisions(buf.decisions)
    dev = next(policy.parameters()).device
    for k in data:
        data[k] = data[k].to(dev)
    tx = data["tx"]
    MSG = MESSAGE_SIZE
    actor_type = getattr(cfg, "actor_type", "deepset")
    with torch.no_grad():
        if actor_type == "gru_split_observation":
            new_lp, _ = policy.evaluate_batch_split(tx, data["prop"], data["prev_hidden"], data["action"])
        elif actor_type == "gru":
            new_lp, _ = policy.evaluate_batch_gru(data["z"], data["seed"], tx[:, :MSG], data["prop"],
                                                   data["prev_hidden"], data["action"])
        else:
            new_lp, _ = policy.evaluate_batch(data["z"], data["seed"], tx[:, :MSG], tx[:, MSG],
                                              data["db_rows"], data["db_valid"], data["action"])
    return (new_lp - data["old_lp"]).exp()


def audit_run(trainer, policy, cfg, iterations):
    # Trace the action pipeline. Two checks, no training:
    #  1) executed (clamped) vs stored (unclamped) action: if the env runs a different
    #     action than the one PPO optimizes the log-prob of, reward and gradient diverge.
    #  2) replay consistency: re-evaluating the stored action/obs should reproduce the
    #     collection log-prob (PPO ratio ~1.0 on the first eval); if not, the mean at
    #     update time differs from the mean at action time.
    import numpy as np
    trainer.setup()
    trainer._audit = True
    trainer._audit_log = []
    cfg.motor_override = "none"
    buf = None
    for _ in range(iterations):
        buf = trainer.collect(policy, None, deterministic=False)
    log = trainer._audit_log
    if not log:
        print("audit: no data recorded")
        return
    action = np.concatenate([a for _, a, _ in log], 0)
    env = np.concatenate([e for _, _, e in log], 0)
    std = policy._std().detach().cpu().numpy()
    std = std.reshape(-1)
    MSG = MESSAGE_SIZE
    nd = action.shape[1]

    def np_squash(u):
        t = np.tanh(u)
        out = t.copy()
        out[:, MSG:] = 0.5 * (t[:, MSG:] + 1.0)
        return out

    sq = np_squash(action)
    pipeline_mismatch = float((np.abs(sq - env) > 1e-5).mean())
    oor_msg = float(((env[:, :MSG] < -1.0 - 1e-6) | (env[:, :MSG] > 1.0 + 1e-6)).mean())
    oor_mot = float(((env[:, MSG:] < -1e-6) | (env[:, MSG:] > 1.0 + 1e-6)).mean())

    print("\n==== ACTION PIPELINE AUDIT (squashed-Gaussian) ====")
    print("samples: %d robot-steps, %d action dims (%d message + %d motor)" % (action.shape[0], nd, MSG, nd - MSG))
    print("policy std (u-space): message ~%.3f  motor ~%.3f" % (std[:MSG].mean(), std[MSG:].mean()))

    print("\n-- executed action is in range without clamping --")
    print("  message out-of-range fraction: %.4f%%" % (100 * oor_msg))
    print("  motor   out-of-range fraction: %.4f%%   (squash should bound these to [0,1])" % (100 * oor_mot))

    print("\n-- env executes squash(stored action) --")
    print("  mismatch between squash(stored u) and executed action: %.4f%%" % (100 * pipeline_mismatch))
    print("  (this must be ~0: the action whose log-prob is optimized maps exactly to what ran)")

    print("\n-- replay consistency (PPO ratio on first eval should be ~1.0) --")
    ratio_ok = True
    ratio = audit_replay_ratio(policy, cfg, buf)
    if ratio is not None:
        rmean, rstd = float(ratio.mean()), float(ratio.std())
        print("  decisions: %d   ratio mean %.4f  std %.4f  min %.4f  max %.4f" % (
            len(buf.decisions), rmean, rstd, float(ratio.min()), float(ratio.max())))
        ratio_ok = abs(rmean - 1.0) <= 0.02 and rstd <= 0.05

    print("\n-- GATE --")
    passed = oor_msg < 1e-4 and oor_mot < 1e-4 and pipeline_mismatch < 1e-4 and ratio_ok
    if passed:
        print("  PASS: actions are in range with no clamping, the executed action is exactly")
        print("  squash(stored), and replay reproduces the collection log-prob. The squashed-Gaussian")
        print("  fix is wired correctly; reward and gradient now refer to the same action.")
    else:
        print("  FAIL: one of the invariants is violated (see above). Do NOT start a long run until")
        print("  out-of-range, squash mismatch, and ratio are all clean.")
    print("AUDIT_GATE=%s" % ("PASS" if passed else "FAIL"))
    print()


def reward_probe(trainer, policy, cfg, iterations):
    # Two measurements of the reward, no training:
    #  (B) does coverage earn reward? Compare the oracle (high coverage) to the loaded
    #      actor (low coverage) and see whether reaching the shape actually pays.
    #  (A) the reward-vs-distance surface for an isolated robot, straight from the real
    #      reward function, to see the slope a single robot climbs.
    import numpy as np
    import torch
    from reward import compute_rewards

    out = {}
    for mode in ["oracle", "none"]:
        trainer.setup()
        cfg.motor_override = mode
        rs = rn = cs = cn = 0.0
        for _ in range(iterations):
            trainer.collect(policy, None, deterministic=True)
            p = trainer.rollout_payload()
            rs += p["reward_sum"]; rn += p["reward_count"]
            cs += p["cov_sum"]; cn += p["cov_count"]
        out[mode] = (rs / max(rn, 1.0), cs / max(cn, 1.0))

    o_rew, o_cov = out["oracle"]
    a_rew, a_cov = out["none"]
    print("\n==== DOES COVERAGE EARN REWARD?  (oracle vs loaded actor) ====")
    print("  ORACLE: coverage %.3f    mean reward/step %.5f" % (o_cov, o_rew))
    print("  ACTOR : coverage %.3f    mean reward/step %.5f" % (a_cov, a_rew))
    cov_gap = o_cov - a_cov
    rew_gap = o_rew - a_rew
    print("  gaps  : coverage %+.3f   reward %+.5f" % (cov_gap, rew_gap))
    if abs(cov_gap) > 0.02:
        slope = rew_gap / cov_gap
        print("  reward per unit coverage (slope): %+.4f" % slope)
        if slope > 0.01:
            print("  READ: more coverage earns more reward (positive slope). The reward tracks the goal;")
            print("        the failure is in LEARNING, not in what the reward rewards. The absolute gap")
            print("        is small only because both rollouts reached similar coverage here; the slope")
            print("        is the real signal, and the reward SURFACE below is the definitive check.")
        elif slope > -0.01:
            print("  READ: reward barely changes with coverage. Possible decoupling, but defer to the")
            print("        reward SURFACE below, which measures it exactly without this confound.")
        else:
            print("  READ: reward DECREASES with coverage, which would be misaligned. Check the surface.")
    else:
        print("  (coverage gap too small to compare rewards reliably here; the reward SURFACE below is")
        print("   the definitive, confound-free test of whether reaching the shape pays.)")

    print("\n==== REWARD SURFACE: reward vs distance, isolated robot ====")
    node = torch.zeros(1, 19)
    node[0, 7] = 0.5  # nearest neighbor far away: no packing or separation terms
    ds = np.linspace(0.0, 0.6, 13)
    vals = []
    for dd in ds:
        node[0, 4] = float(dd)
        vals.append(float(compute_rewards(node, cfg, edge_index=None)[0]))
    print("   dist    reward      gained by moving one bin (0.05) closer")
    for i, dd in enumerate(ds):
        step = "" if i == 0 else ("%+.6f" % (vals[i] - vals[i - 1]))
        tag = "  <- on shape" if dd < cfg.tau_v else ""
        print("   %.3f   %+.5f    %s%s" % (dd, vals[i], step, tag))
    off = [vals[i] for i, dd in enumerate(ds) if dd > cfg.tau_v]
    if off:
        print("  off-shape reward spread across the arena: %.5f" % (max(off) - min(off)))
        print("  step up on crossing onto the shape:        %.5f" % (vals[0] - off[0]))
        print("  READ: a non-trivial off-shape slope means the gradient toward the stroke exists")
        print("        across the open arena; a near-zero slope means it is flat until the boundary.")
    print()


def probe_run(trainer, policy, cfg, iterations):
    # Load an actor and report what it actually does: are motors driving, do robots
    # steer toward the stroke, or is the policy frozen/undirected. No training.
    import numpy as np
    trainer.setup()
    trainer._probe = True
    trainer._probe_log = []
    cfg.motor_override = "none"
    for _ in range(iterations):
        trainer.collect(policy, None, deterministic=True)
    log = trainer._probe_log
    if not log:
        print("probe: no data recorded")
        return
    nodes = np.concatenate([a for a, _ in log], axis=0)
    motors = np.concatenate([m for _, m in log], axis=0)
    heading = nodes[:, 2:4]
    dirv = nodes[:, 5:7]
    dist = nodes[:, 4]
    mL = motors[:, 0]
    mR = motors[:, 1]

    def unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-8] = 1.0
        return v / n

    h = unit(heading)
    d = unit(dirv)
    align = np.sum(h * d, axis=1)
    forward = 0.5 * (mL + mR)
    turn = mR - mL
    cross = h[:, 0] * d[:, 1] - h[:, 1] * d[:, 0]
    steer_score = float(np.mean(np.sign(cross) * turn))
    onshape = dist < cfg.tau_v

    print("\n==== BEHAVIOR PROBE ====")
    print("samples: %d robot-steps" % nodes.shape[0])
    print("-- motors --")
    print("  left  mean %.3f std %.3f    right mean %.3f std %.3f" % (mL.mean(), mL.std(), mR.mean(), mR.std()))
    print("  saturated low <0.05: %.1f%%   high >0.95: %.1f%%   mid [0.4,0.6]: %.1f%%" % (
        100 * np.mean(motors < 0.05), 100 * np.mean(motors > 0.95),
        100 * np.mean((motors > 0.4) & (motors < 0.6))))
    print("  forward drive (mL+mR)/2: mean %.3f std %.3f   [0 frozen .. 1 full ahead]" % (forward.mean(), forward.std()))
    print("-- navigation --")
    print("  heading->stroke alignment: mean %.3f   [+1 facing stroke, 0 random, -1 away]" % align.mean())
    print("  steering-direction correlation: %.4f   [magnitude matters; sign is wheel-convention dependent]" % steer_score)
    print("  mean |turn|: %.3f" % float(np.mean(np.abs(turn))))
    print("-- outcome --")
    print("  on-shape fraction: %.3f    mean dist-to-stroke: %.3f" % (onshape.mean(), dist.mean()))
    print("-- read --")
    midfrac = float(np.mean((motors > 0.4) & (motors < 0.6)))
    frozen = forward.mean() < 0.1 or (midfrac > 0.6 and turn.std() < 0.05)
    # alignment is convention-independent: if robots face the stroke and drive, they navigate
    navigating = align.mean() > 0.2 and forward.mean() > 0.2
    if frozen:
        print("  FROZEN/STALLED: motors barely drive or sit mid-band. The policy is not navigating.")
    elif navigating:
        print("  NAVIGATING: robots face the stroke (alignment > 0.2) and drive. The body-level loop works.")
    else:
        print("  ACTIVE BUT UNDIRECTED: robots move, but heading is not aligned toward the stroke")
        print("  (alignment near 0). They are not steering toward the shape.")
    print()


def log_iteration_diagnostics(trainer, cfg, oracle_pay):
    # Direct follow-up to log_high_arrived_arenas (below): broader,
    # general-purpose per-iteration visibility, not narrowly scoped to the
    # arrived/coverage discrepancy that motivated that one. Everything here
    # reuses data the training loop already computes every iteration --
    # oracle_pay (trainer.rollout_payload(), captured by the caller right
    # after the oracle-driven collect()) and trainer.collect_timing() --
    # not a second, separate computation pass. A genuinely new arena-level
    # coverage histogram is the one new computation.
    # Everything oracle-related uses arrival%, not
    # coverage() -- same computation trainer.py's own success/reset check
    # and cov_sum/cov_count reporting already switched to: the oracle's
    # own "arrived" state label (worker.simple_state), genuinely valid
    # here since this runs right after the oracle-driven collect() above,
    # not coverage()'s ground-truth position check.
    buckets = {"<10%": 0, "10-50%": 0, "50-%.0f%%" % (100 * cfg.success_threshold): 0,
               ">=%.0f%% (success-adjacent)" % (100 * cfg.success_threshold): 0}
    bucket_keys = list(buckets.keys())
    n_arenas_total = 0
    for worker in trainer.workers:
        for k in range(cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None:
                continue
            n_arenas_total += 1
            m = snap["node"].shape[0]
            arrived_here = getattr(worker, "simple_state", {}).get(k, {})
            arrived_count = sum(1 for l in range(m) if arrived_here.get(l) == "arrived")
            cov = arrived_count / m if m > 0 else 0.0
            if cov < 0.10:
                buckets[bucket_keys[0]] += 1
            elif cov < 0.50:
                buckets[bucket_keys[1]] += 1
            elif cov < cfg.success_threshold:
                buckets[bucket_keys[2]] += 1
            else:
                buckets[bucket_keys[3]] += 1
    print("  [debug] arena stages (%d arenas, each arena's own real-time arrived%%): %s" %
          (n_arenas_total, "  ".join("%s=%d" % (k, v) for k, v in buckets.items())))

    # Batch progress: episodes that completed within THIS SPECIFIC
    # iteration's own rollout window, not a running total -- ep_records
    # resets at the start of every collect() call (trainer.py's own
    # collect()), so by the time this is called it holds exactly this
    # iteration's own completions, nothing carried over from before.
    ep_records = oracle_pay.get("ep_records", [])
    n_completed = len(ep_records)
    n_success = sum(1 for e in ep_records if e["success"])
    n_timeout = n_completed - n_success
    if n_completed > 0:
        mean_cov_at_completion = sum(e["coverage"] for e in ep_records) / n_completed
        mean_length = sum(e["length"] for e in ep_records) / n_completed
        print("  [debug] episodes completed this iteration: %d  (success=%d  timeout=%d)  "
              "mean coverage at completion=%.4f  mean length=%.0f ticks" %
              (n_completed, n_success, n_timeout, mean_cov_at_completion, mean_length))
    else:
        print("  [debug] episodes completed this iteration: 0")

    # Timing breakdown: where this iteration's own wall-clock time actually
    # went. Already computed internally (Trainer.collect_timing()), never
    # previously surfaced in the main training log.
    timing = trainer.collect_timing()
    total_t = timing["step"] + timing["parse"] + timing["getsteps"] + timing["snap"] + timing["act"]
    if total_t > 0:
        print("  [debug] timing (s): step=%.2f (%.0f%%)  parse=%.2f (%.0f%%)  getsteps=%.2f (%.0f%%)  "
              "snap=%.2f (%.0f%%)  act=%.2f (%.0f%%)  msgs=%d" %
              (timing["step"], 100 * timing["step"] / total_t, timing["parse"], 100 * timing["parse"] / total_t,
               timing["getsteps"], 100 * timing["getsteps"] / total_t, timing["snap"], 100 * timing["snap"] / total_t,
               timing["act"], 100 * timing["act"] / total_t, timing["msgs"]))

    # Reception-event breakdown: what fraction of this iteration's own
    # decisions were triggered by a genuine seed/wall reception vs the
    # heartbeat timeout vs an ordinary neighbor message (the remainder).
    # Already computed internally, never previously surfaced.
    total_events = oracle_pay.get("split_total_events", 0)
    seed_events = oracle_pay.get("split_seed_events", 0)
    heartbeat_events = oracle_pay.get("split_heartbeat_events", 0)
    if total_events > 0:
        neighbor_events = total_events - seed_events - heartbeat_events
        print("  [debug] reception events: seed/wall=%.1f%%  neighbor=%.1f%%  heartbeat=%.1f%%  (of %d total)" %
              (100 * seed_events / total_events, 100 * neighbor_events / total_events,
               100 * heartbeat_events / total_events, total_events))

    # Belief confidence summary: mean position confidence and the fraction
    # of decisions where the filter is genuinely "localized" (above
    # belief.py's own LOCALIZED_CONF_THRESHOLD). Already computed
    # internally, never previously surfaced.
    belief_count = oracle_pay.get("belief_count", 0)
    if belief_count > 0:
        mean_conf = oracle_pay.get("conf_pos_sum", 0.0) / belief_count
        localized_frac = oracle_pay.get("localized_count", 0) / belief_count
        print("  [debug] belief: mean conf_pos=%.4f  localized fraction=%.1f%%  (of %d belief reads)" %
              (mean_conf, 100 * localized_frac, belief_count))


def log_high_arrived_arenas(trainer, cfg, threshold):
    # Written after a real discrepancy: one arena's state_pct showed 96%
    # arrived while oracle_cov, in the same log, stayed near 0.20-0.25. oracle_cov is averaged across
    # every arena in every worker/instance combined -- with --instances 20
    # --arenas 8, that is 160 arenas' worth of coverage folded into one
    # number, so a single arena sitting at a real, high coverage right now
    # can be genuinely invisible in that aggregate if the other 159 are not
    # also there yet. This prints a direct, per-arena breakdown instead,
    # for any arena whose own state_pct['arrived'] is at or above threshold.
    # Originally also printed this same arena's own coverage() value
    # alongside arrived% specifically to diagnose that discrepancy -- since
    # Everything oracle-related reports arrival% throughout this project
    # (trainer.py's success/reset check and its cov_sum/cov_count reporting),
    # so a ground-truth comparison here is not the relevant standard.
    max_seen = (-1.0, None, None, None)  # (arrived_pct, wi, k, total) -- tracked regardless of threshold
    for wi, worker in enumerate(trainer.workers):
        states_by_arena = getattr(worker, "simple_state", {})
        for k in range(cfg.num_arenas):
            states = states_by_arena.get(k, {})
            if not states:
                continue
            total = len(states)
            arrived = sum(1 for s in states.values() if s == "arrived")
            arrived_pct = arrived / total
            if arrived_pct > max_seen[0]:
                max_seen = (arrived_pct, wi, k, total)
            if arrived_pct >= threshold:
                print("  [debug per-arena] instance %d arena %d: arrived=%.1f%% (n=%d robots)" %
                      (wi, k, 100 * arrived_pct, total))
    # Unconditional -- fires every single iteration regardless of threshold,
    # specifically to separate two, otherwise indistinguishable explanations
    # for silence above: this function never being reached at all (a wiring
    # problem), versus being reached correctly but genuinely never crossing
    # threshold yet. If this line itself never appears, the call site isn't
    # being hit. If it appears but its own arrived_pct never approaches
    # threshold, the mechanism is working and threshold=0.5 just hasn't
    # been reached yet -- two real possibilities, so print the maximum seen
    # rather than assume either.
    if max_seen[1] is not None:
        print("  [debug per-arena] highest arrived%% seen this iteration, any arena: "
              "instance %d arena %d: %.1f%% (n=%d)" % (max_seen[1], max_seen[2], 100*max_seen[0], max_seen[3]))
    else:
        print("  [debug per-arena] no worker had any simple_state populated this iteration at all")
    # Minimal, direct, unconditional -- worker.image_id[k] is set on every
    # single _reset_arena call regardless of whether self.image_names
    # exists (that gate only affects the separate "arena %d: formation %d"
    # print in trainer.py, which is why it can go silent even on a genuine
    # reset). This has no such gate: if the same formation is genuinely
    # being reselected across resets, the same image_id will repeat here,
    # directly, iteration to iteration -- checkable by reading the printed
    # numbers, not by trusting a claim about what the code does.
    for wi, worker in enumerate(trainer.workers):
        ids = getattr(worker, "image_id", {})
        for k in range(cfg.num_arenas):
            if k in ids:
                print("  [debug per-arena] instance %d arena %d: image_id=%s" % (wi, k, ids[k]))


def print_eval_log_summary(it, audit_log, probe_log, pos_log = None):
    # Split out from run_eval, which lives in launch.py because it reads that
    # module's resolved configuration. This half is pure formatting of the
    # audit/probe/pos logs and needs no Unity connection, so it can be
    # exercised directly.
    n_decisions = sum(mean.shape[0] for mean, _, _ in audit_log)
    print("EVAL_LOG iter %d: %d decision(s) captured across %d collect-internal batch(es)"
          % (it, n_decisions, len(audit_log)))
    if n_decisions == 0:
        print("EVAL_LOG iter %d: no decisions at all this collect() call -- either no "
              "robots reached a decision tick, or decision_steps came back empty from "
              "Unity. Check the Unity Editor console for errors, and confirm the scene "
              "is actually in Play mode before this script starts." % it)
        return
    motor = np.concatenate([env_action[:, MESSAGE_SIZE:] for _, _, env_action in audit_log], axis=0)
    print("EVAL_LOG iter %d: motor (sent to Unity) -- mean=%s std=%s min=%.4f max=%.4f  "
          "(near-zero mean/std here means the actor itself is computing near-nothing, "
          "before anything reaches Unity)"
          % (it, np.array2string(motor.mean(axis=0), precision=4),
             np.array2string(motor.std(axis=0), precision=4),
             float(motor.min()), float(motor.max())))
    print("EVAL_LOG iter %d: first %d raw (mean, env_action motor) pairs:"
          % (it, min(3, len(audit_log))))
    for mean, _, env_action in audit_log[:3]:
        print("  policy mean[0] = %s   ->   motor sent[0] = %s"
              % (np.array2string(mean[0], precision=4),
                 np.array2string(env_action[0, MESSAGE_SIZE:], precision=4)))
    if probe_log:
        obs = np.concatenate([node_b for node_b, _ in probe_log], axis=0)
        nan_count = int(np.isnan(obs).sum())
        print("EVAL_LOG iter %d: raw observation (node_b) -- shape=%s mean=%.4f "
              "min=%.4f max=%.4f nan_count=%d  (all-zero, all-identical, or a nonzero "
              "nan_count here points at Unity's own observation side, not the actor)"
              % (it, str(obs.shape), float(obs.mean()), float(obs.min()), float(obs.max()), nan_count))
        # NODE_FEATURES's own comment (kilobot_gnn.py): 19 = P(2) + H(2) +
        # |D|(1) + dir_D(2) + C(1) + M(2) + T(9) -- breaking out by group
        # rather than one whole-array stat, since a problem confined to one
        # group (e.g. T, the incoming transmission) points somewhere very
        # different than the same pattern across all of them.
        groups = [("P (true position)", slice(0, 2)), ("H (true heading)", slice(2, 4)),
                  ("|D| (distance)", slice(4, 5)), ("dir_D (direction)", slice(5, 7)),
                  ("C", slice(7, 8)), ("M", slice(8, 10)), ("T (transmission)", slice(10, 19))]
        for label, sl in groups:
            g = obs[:, sl]
            print("  %-20s mean=%.4f std=%.4f min=%.4f max=%.4f"
                  % (label, float(g.mean()), float(g.std()), float(g.min()), float(g.max())))
        # matches diagnostics.py's own probe_run: is the robot's own true
        # heading (H) at least pointed toward the direction it needs to go
        # (dir_D)? mean(dot) near 1 = steering correctly, near 0 = steering
        # roughly perpendicular/random, near -1 = steering the wrong way
        # entirely. This can be genuinely informative even if position
        # itself turns out not to be changing -- it separates "the robot
        # wants to go the right way but isn't moving" from "the robot
        # doesn't even want to go the right way."
        heading = obs[:, 2:4]
        dirv = obs[:, 5:7]
        hn = heading / (np.linalg.norm(heading, axis=1, keepdims=True) + 1e-6)
        dn = dirv / (np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-6)
        dot = (hn * dn).sum(axis=1)
        print("  steering quality: mean(dot(heading, dir_to_target))=%.4f  "
              "(1=pointed straight at it, 0=perpendicular/undirected, -1=pointed away)"
              % float(dot.mean()))
    if pos_log:
        # ground-truth position, identity-tagged -- fine here specifically
        # because it's printed for a human, never fed to the network (see
        # act()'s own comment on pos_track for the full rationale). This is
        # the most direct possible answer to "is anything physically
        # moving in Unity at all", independent of whether the actor's own
        # motor output or steering looks reasonable.
        # Direct follow-up request: split into separate, continuous
        # segments wherever a robot's own image_id changes -- that's
        # exactly what a genuine episode reset looks like (image_id is set
        # only in _reset_arena), and comparing a position from before a
        # reset against one from after it is not real, continuous
        # movement, it's the distance between where the old episode ended
        # and where the new one began. Each segment needs its own 2+
        # observations to contribute a displacement, same bar the
        # unsegmented version already used, just applied within a segment
        # rather than across the whole call.
        by_robot = {}
        for entry in pos_log:
            step, arena_ids, locals_, pos, image_ids = entry
            for a, l, p, iid in zip(arena_ids, locals_, pos, image_ids):
                by_robot.setdefault((int(a), int(l)), []).append((step, p, int(iid)))
        displacements = []
        n_boundaries = 0
        for key, entries in by_robot.items():
            entries.sort(key=lambda e: e[0])
            segment = [entries[0]]
            for prev, cur in zip(entries, entries[1:]):
                if cur[2] != prev[2]:
                    n_boundaries += 1
                    if len(segment) >= 2:
                        displacements.append(float(np.linalg.norm(segment[-1][1] - segment[0][1])))
                    segment = [cur]
                else:
                    segment.append(cur)
            if len(segment) >= 2:
                displacements.append(float(np.linalg.norm(segment[-1][1] - segment[0][1])))
        boundary_note = ("; %d episode boundary/boundaries excluded from this measurement" % n_boundaries) if n_boundaries else ""
        if not displacements:
            print("  EVAL_LOG iter %d: position tracking -- every robot seen only once this "
                  "call, no displacement measurable yet (normal for early iterations; check "
                  "later iters too)%s" % (it, boundary_note))
        else:
            displacements = np.array(displacements)
            n_still = int((displacements < 1e-4).sum())
            print("  EVAL_LOG iter %d: true position displacement (first seen -> last seen "
                  "this call, never across an episode reset), %d segment(s) measured -- "
                  "mean=%.4f max=%.4f %d/%d essentially unchanged (<1e-4)%s  (if this is ~0 "
                  "for everyone while motor above is clearly nonzero, the command is not "
                  "reaching/being applied "
                  "in Unity -- a C#-side issue, not this codebase's own Python)"
                  % (it, len(displacements), float(displacements.mean()), float(displacements.max()),
                     n_still, len(displacements), boundary_note))
