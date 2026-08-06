"""Behaviour cloning: fit the actor to the scripted oracle.

Both halves live here -- bc_update/bc_update_replay, the gradient step over one
batch of (observation, oracle target) pairs, and bc_train, the loop that
collects, fits, evaluates, logs and checkpoints. bc_train used to live in
diagnostics.py, which made the project's main warm-start path look like an
instrumentation helper.

The teacher is simple_oracle.py; the sample store is bc_replay.py; the held-out
score is val_tape.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kilobot_gnn import MESSAGE_SIZE, actor_forward_batch, recurrent_forward_batch, split_forward_batch
from policy import squash_action
from ppo import _stack_decisions


def _bc_step(policy, actor_opt, cfg, tc, prop, h_prev, targets, arrived, arrived_valid,
             is_arrived = None, natural_arrived_rate = None):
    # One gradient step, shared by both fit paths below so the loss itself is
    # defined in exactly one place and the reservoir path can never quietly
    # drift from the original.
    mean, _ = split_forward_batch(policy.actor, tc, prop, h_prev)
    motors = squash_action(mean)[:, MESSAGE_SIZE:]
    # config.py's own bc_motor_skip_arrived has the full rationale. The motor
    # head is simply never shown the arrived rows -- nothing is fabricated in
    # their place, they are dropped from this term entirely.
    if getattr(cfg, "bc_motor_skip_arrived", False) and is_arrived is not None:
        keep = (~is_arrived).nonzero().squeeze(-1)
        if int(keep.numel()) == 0:
            return 0.0, 0.0, 0.0
        motor_loss = ((motors[keep] - targets[keep]) ** 2).mean()
    else:
        motor_loss = ((motors - targets) ** 2).mean()
    loss = motor_loss
    motor_only = float(motor_loss.detach())
    if arrived is not None and getattr(policy.actor, "head_arrived", None) is not None:
        have = arrived_valid.nonzero().squeeze(-1)
        if int(have.numel()) > 0:
            logits = policy.actor._arrived_logit[have].squeeze(-1)
            labels = arrived[have].squeeze(-1)
            # config.py's own bc_arrived_natural_prior has the full rationale.
            # A balanced minibatch is not a sample from the real distribution,
            # and for a binary head the batch composition IS the class prior it
            # calibrates to. These weights undo the sampler for this term only,
            # so balancing keeps helping the motor head without teaching the
            # arrived head that arrival is far more common than it is.
            if (getattr(cfg, "bc_arrived_natural_prior", False)
                    and natural_arrived_rate is not None and 0.0 < natural_arrived_rate < 1.0):
                batch_rate = float(labels.mean())
                if 0.0 < batch_rate < 1.0:
                    w_pos = natural_arrived_rate / batch_rate
                    w_neg = (1.0 - natural_arrived_rate) / (1.0 - batch_rate)
                    w = torch.where(labels > 0.5,
                                    torch.full_like(labels, w_pos),
                                    torch.full_like(labels, w_neg))
                    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction = "none")
                    arrived_loss = (w * raw).sum() / w.sum().clamp(min = 1e-8)
                else:
                    arrived_loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                arrived_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + cfg.arrived_loss_weight * arrived_loss
    actor_opt.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    actor_opt.step()
    policy.clamp_log_std()
    return float(loss.detach()), float(grad_norm), motor_only


def bc_update_replay(policy, actor_opt, reservoir, cfg, steps, extra = None):
    # config.py's own bc_replay_capacity has the full rationale. Same number
    # of gradient steps and same minibatch size the caller would have used
    # without the reservoir, so this is a change of which samples a fit sees,
    # not of how much fitting happens per iteration.
    device = next(policy.parameters()).device
    mb = max(1, int(cfg.minibatch))
    last = 0.0
    last_grad_norm = 0.0
    loss_sum = 0.0
    motor_sum = 0.0
    loss_count = 0
    nat_rate = reservoir.natural_arrived_rate()
    for _ in range(int(steps)):
        batch = reservoir.sample(mb, device)
        if batch is None:
            break
        last, last_grad_norm, motor_only = _bc_step(policy, actor_opt, cfg, batch["tc"], batch["prop"],
                                                    batch["h"], batch["tgt"], batch["arrived"],
                                                    batch["arrived_valid"],
                                                    is_arrived = batch.get("is_arrived"),
                                                    natural_arrived_rate = nat_rate)
        loss_sum = loss_sum + last
        motor_sum = motor_sum + motor_only
        loss_count = loss_count + 1
    if extra is not None:
        extra["mean_loss"] = loss_sum / max(loss_count, 1)
        extra["grad_norm"] = last_grad_norm
        extra["n_decisions"] = reservoir.total()
        extra["reservoir_counts"] = reservoir.counts()
        extra["motor_mse"] = motor_sum / max(loss_count, 1)
        extra["arrived_rate"] = nat_rate
        extra["steps"] = loss_count
    return last


def bc_update(policy, actor_opt, buffer, cfg, epochs, extra = None, reservoir = None):
    # Supervised: fit the actor's motor output to the stored oracle targets.
    # extra, if given a dict, is populated in place with mean_loss (averaged
    # across every minibatch of the fit, not just the last one -- the
    # returned "last" value is noisy by construction, a single minibatch's
    # loss, not an aggregate), grad_norm (clip_grad_norm_'s own pre-clip
    # return value, free to capture -- verified empirically it's the real
    # pre-clip total norm, not clipped or otherwise altered), and
    # n_decisions (how many oracle-labeled samples this iteration's fit
    # actually had). Optional and additive -- every existing caller keeps
    # working completely unchanged, since none of them pass this argument.
    decisions = [d for d in buffer.decisions if d.get("bc_target") is not None]
    # config.py's own bc_replay_capacity has the full rationale. The step
    # budget is derived from this iteration's own decision count exactly as
    # the direct path below would have spent it (epochs x minibatches over
    # the fresh rollout), so replay changes the mixture a fit sees without
    # changing how much compute an iteration costs.
    if reservoir is not None and reservoir.enabled:
        per_epoch = max(1, int(math.ceil(len(decisions) / max(1, int(cfg.minibatch)))))
        return bc_update_replay(policy, actor_opt, reservoir, cfg,
                                int(epochs) * per_epoch, extra = extra)
    # config.py's own turning_duplicate_factor has the full rationale.
    # Every duplicated entry here is the exact same, real decision --
    # same real observation, same real, oracle-computed target -- nothing
    # synthesized. A decision appearing n+1 times total (1 original + n
    # extra copies) contributes to the averaged BC loss below exactly
    # n+1 times over, the same effect a per-example loss weight of n+1
    # would have, without touching the loss computation itself.
    turning_dup = getattr(cfg, "turning_duplicate_factor", 0)
    if turning_dup > 0:
        extra_copies = [d for d in decisions if d.get("was_turning")] * turning_dup
        decisions = decisions + extra_copies
    if not decisions:
        if extra is not None:
            extra["mean_loss"] = 0.0
            extra["grad_norm"] = 0.0
            extra["n_decisions"] = 0
        return 0.0
    data = _stack_decisions(decisions)
    targets = torch.stack([d["bc_target"] for d in decisions])
    # config.py's own use_arrived_head has the full rationale. None for any
    # decision that doesn't have one -- kept as a list, not stacked here,
    # since a per-minibatch mask (below) needs to know which chunk entries
    # actually have a real label, not just substitute a placeholder value.
    arrived_targets_all = [d.get("arrived_target") for d in decisions]
    device = next(policy.parameters()).device
    for k in data:
        data[k] = data[k].to(device)
    targets = targets.to(device)
    n = targets.shape[0]
    mb = max(1, int(cfg.minibatch))
    actor_type = getattr(cfg, "actor_type", "deepset")
    last = 0.0
    last_grad_norm = 0.0
    loss_sum = 0.0
    loss_count = 0
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, mb):
            chunk = perm[s:s + mb]
            z = data["z"][chunk]
            if actor_type == "gru_split_observation":
                tc = data["tx"][chunk]
                prop = data["prop"][chunk]
                h_prev = data["prev_hidden"][chunk]
                mean, _ = split_forward_batch(policy.actor, tc, prop, h_prev)
            elif actor_type == "gru":
                seed = data["seed"][chunk]
                tx = data["tx"][chunk]
                msg = tx[:, :MESSAGE_SIZE]
                prop = data["prop"][chunk]
                h_prev = data["prev_hidden"][chunk]
                mean, _ = recurrent_forward_batch(policy.actor, z, seed, msg, prop, h_prev)
            else:
                seed = data["seed"][chunk]
                tx = data["tx"][chunk]
                msg = tx[:, :MESSAGE_SIZE]
                sender = tx[:, MESSAGE_SIZE]
                db_rows = data["db_rows"][chunk]
                db_valid = data["db_valid"][chunk]
                mean, _, _ = actor_forward_batch(policy.actor, z, seed, msg, sender, db_rows, db_valid)
            motors = squash_action(mean)[:, MESSAGE_SIZE:]
            loss = ((motors - targets[chunk]) ** 2).mean()
            # config.py's own use_arrived_head has the full rationale.
            # policy.actor._arrived_logit was just populated as a side
            # effect of split_forward_batch's own call above (same,
            # already-established pattern as _motor_preact) -- None when
            # the head doesn't exist, in which case this whole block is a
            # no-op and the loss above is completely unchanged.
            if actor_type == "gru_split_observation" and getattr(policy.actor, "head_arrived", None) is not None:
                chunk_targets = [arrived_targets_all[i] for i in chunk.tolist()]
                have_label = [i for i, t in enumerate(chunk_targets) if t is not None]
                if have_label:
                    logits = policy.actor._arrived_logit[have_label].squeeze(-1)
                    labels = torch.stack([chunk_targets[i] for i in have_label]).squeeze(-1).to(device)
                    arrived_loss = F.binary_cross_entropy_with_logits(logits, labels)
                    loss = loss + cfg.arrived_loss_weight * arrived_loss
            actor_opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            actor_opt.step()
            policy.clamp_log_std()
            last = float(loss.detach())
            last_grad_norm = float(grad_norm)
            loss_sum = loss_sum + last
            loss_count = loss_count + 1
    if extra is not None:
        extra["mean_loss"] = loss_sum / max(loss_count, 1)
        extra["grad_norm"] = last_grad_norm
        extra["n_decisions"] = n
    return last

def bc_train(trainer, policy, actor_opt, cfg, iterations, summary, bc_epochs, bc_out, logger = None,
            checkpoint_every = 1, teacher = "oracle", on_iteration = None, debug_per_arena_threshold = None,
            debug_iteration_detail = False):
    # Clone the actor to the oracle controller. Each iteration: collect a rollout
    # where the env follows the oracle (storing oracle motors as targets), fit the
    # actor's motors to them, then eval the cloned actor driving on its own and log
    # its coverage. The summary's coverage curve is the ACTOR's, so it shows whether
    # cloning actually lifts the actor toward the oracle ceiling.
    #
    # logger is optional (None for callers that do not have one).
    #
    # Three things are logged beyond the fit loss itself:
    #
    # 1. The ORACLE's rollout_payload(), captured immediately after the first
    # collect() and before the actor-eval collect() resets the same
    # accumulators. That gives a per-iteration ceiling -- how the same formation
    # and spawn scored under the oracle -- rather than the actor's number in
    # isolation. Run through metrics.rollout_stats, the same function the RL
    # path uses, under an "oracle/" prefix so it cannot collide with the
    # actor's keys.
    #
    # 2. bc_update's mean_loss (averaged over the whole fit; the "motor_mse"
    # key is a single minibatch by construction and noisy on its own) and
    # grad_norm (clip_grad_norm_'s pre-clip return value).
    #
    # 3. Periodic checkpointing to the SAME bc_out path every checkpoint_every
    # iterations, so the existing restart workflow
    # (KILOBOT_INIT_ACTOR=<bc_out>) keeps working unchanged. export_actor
    # writes atomically -- torch.save to a temp file then os.replace -- so an
    # interruption cannot leave a corrupted checkpoint. It costs ~1.3ms against
    # a multi-minute iteration, hence the default of every iteration.
    import os
    from bc_replay import BCReservoir, BC_STATES, UNLABELLED
    from checkpoint import export_actor
    from metrics import rollout_stats
    print("BC: cloning actor to %s for %d iterations" % (teacher, iterations))
    # config.py's own bc_replay_capacity has the full rationale. Constructed
    # unconditionally but inert unless the capacity is set, so a run that
    # leaves it at the default behaves exactly as before.
    last_actor_cov = [0.0]
    # config.py's own val_tape_path has the full rationale. bc_train is shared
    # by run_bc_monitored.py and launch.py, but until now only the former had
    # any held-out imitation metric -- launch.py's own KILOBOT_MODE=bc against
    # real Unity printed coverage alone, which measures the task rather than
    # the imitation this phase is actually optimising. A tape is just recorded
    # observations, so one recorded once on the replica against held-out
    # formations can be replayed against a Unity-trained checkpoint unchanged.
    tape = None
    tape_hist = []
    best_tape = [None]
    tape_path = getattr(cfg, "val_tape_path", "") or None
    tape_interval = max(1, int(getattr(cfg, "val_tape_interval", 5)))
    if tape_path:
        from val_tape import load_tape, replay_tape, tape_state_counts
        tape = load_tape(tape_path)
        if tape is None:
            print("BC: val tape at %s missing or version-mismatched -- continuing without it"
                  % tape_path, flush = True)
        else:
            print("BC: val tape %d sequences x %d steps, states %s%s"
                  % (tape["valid"].shape[1], tape["valid"].shape[0], tape_state_counts(tape),
                     "  (arrived EXCLUDED from the criterion)"
                     if getattr(cfg, "bc_motor_skip_arrived", False) else ""), flush = True)
    reservoir = BCReservoir(cfg)
    reservoir_path = getattr(cfg, "bc_replay_path", "") or None
    persist = bool(getattr(cfg, "bc_replay_persist", False)) and reservoir_path is not None
    if reservoir.enabled:
        print("BC: replay reservoir on -- capacity %d per state, balanced=%s, max_age=%s, evict=%s" %
              (reservoir.capacity, reservoir.balanced,
               reservoir.max_age if reservoir.max_age > 0 else "unbounded", reservoir.evict), flush = True)
        if persist:
            actor = policy.actor
            # Only the split-observation actor exposes up1/hidden_size. The
            # deepset and plain-gru actors do not, and reading them
            # unconditionally crashed a real Unity run on its first iteration
            # ("'Actor' object has no attribute 'up1'"). Falling back to None
            # skips the shape check rather than the load, which is correct:
            # the check is a safety net for mismatched observation widths, not
            # a prerequisite.
            up1 = getattr(actor, "up1", None)
            ok, why = reservoir.load(reservoir_path,
                                     expect_input = int(up1.in_features) if up1 is not None else None,
                                     expect_h = int(getattr(actor, "hidden_size", 0)) or None)
            if ok:
                counts = reservoir.counts()
                print("BC: reloaded reservoir from %s -- %d samples %s"
                      % (reservoir_path, reservoir.total(), counts), flush = True)
                # A reservoir whose samples carry no oracle state label cannot
                # be balanced across states, so balanced sampling silently
                # degrades to uniform -- which the phase-155 ablation measured
                # as WORSE than no replay at all on the rarest state. Refused
                # rather than warned: a real run reloaded 15978 such samples
                # written by an earlier run using a teacher that never
                # produced labels, and would have trained on them unnoticed.
                if reservoir.total() > 0 and set(counts) == {UNLABELLED}:
                    raise SystemExit(
                        "reservoir at %s contains only unlabelled samples (%d), which cannot be "
                        "state-balanced -- it was written by a run whose teacher never produced "
                        "oracle state labels. Delete it and re-run with teacher=simple_oracle."
                        % (reservoir_path, reservoir.total()))
            else:
                print("BC: starting reservoir empty (%s), will persist to %s every %d iterations"
                      % (why, reservoir_path, max(1, int(getattr(cfg, "bc_replay_save_interval", 20)))),
                      flush = True)
    trainer.setup()
    for it in range(iterations):
        # teacher, default "oracle" (every
        # existing caller's exact prior behavior, since none of them pass
        # this argument), also accepts "simple_oracle" -- the non-
        # privileged, from-scratch oracle --
        # letting this same collect/fit/eval loop clone either teacher
        # without duplicating it. Both reach act() through the identical
        # scripted/executed_motor/bc_target machinery; which oracle
        # actually runs is entirely this one assignment.
        cfg.motor_override = teacher
        trainer._bc_capture = True
        buffer = trainer.collect(policy, None, deterministic=False)
        if debug_per_arena_threshold is not None:
            log_high_arrived_arenas(trainer, cfg, debug_per_arena_threshold)
        oracle_pay = trainer.rollout_payload()   # captured before the next collect() resets it
        if debug_iteration_detail:
            log_iteration_diagnostics(trainer, cfg, oracle_pay)
        bc_extra = {}
        # The reservoir argument is only ever passed when it is actually on,
        # so a default run's own call into bc_update stays byte-identical to
        # what it was, including for callers that substitute their own
        # bc_update (tests/test_fixes.py does exactly this).
        if reservoir.enabled:
            reservoir.add(buffer.decisions, it)
            loss = bc_update(policy, actor_opt, buffer, cfg, bc_epochs, extra = bc_extra,
                             reservoir = reservoir)
        else:
            loss = bc_update(policy, actor_opt, buffer, cfg, bc_epochs, extra = bc_extra)

        # config.py's own bc_actor_eval_interval has the full rationale. This
        # second, actor-driven collect produces no training data at all -- it
        # exists only to report actor_eval_cov and the arrived_agreement
        # census -- yet it costs the same rollout_steps of simulation as the
        # oracle-driven collect above, so running it every iteration is
        # roughly half of BC's entire wall clock.
        cfg.motor_override = "none"
        trainer._bc_capture = False
        actor_eval_interval = max(1, int(getattr(cfg, "bc_actor_eval_interval", 1)))
        run_actor_eval = ((it + 1) % actor_eval_interval == 0 or it == 0 or it == iterations - 1)
        if run_actor_eval:
            trainer.collect(policy, None, deterministic=True)
            actor_pay = trainer.rollout_payload()
            actor_stats = rollout_stats(actor_pay)
            cov = actor_stats.get("rollout/mean_coverage", 0.0)
            last_actor_cov[0] = cov
        else:
            # Carries the last measured value rather than reporting 0.0, which
            # would read as a collapse in the printed line and in summary's own
            # coverage curve rather than as "not measured this iteration".
            cov = last_actor_cov[0]
        oracle_stats = rollout_stats(oracle_pay)
        oracle_cov = oracle_stats.get("rollout/mean_coverage", 0.0)

        # "motor_mse" has always been the TOTAL loss whenever use_arrived_head
        # is on -- bc_update returns it after the arrived BCE has been added --
        # so the name was misleading on its own. Kept under the same name for
        # continuity with already-collected history, with the genuine
        # motor-only figure printed next to it when it is available.
        motor_only = bc_extra.get("motor_mse")
        suffix = "" if motor_only is None else "  motor_only %.5f" % motor_only
        print("bc iter %d  motor_mse %.5f (mean %.5f)%s  grad_norm %.3f  actor_eval_cov %.4f  oracle_cov %.4f" %
              (it, loss, bc_extra.get("mean_loss", 0.0), suffix, bc_extra.get("grad_norm", 0.0),
               cov, oracle_cov), flush=True)
        if reservoir.enabled:
            counts = bc_extra.get("reservoir_counts", {})
            print("  reservoir %d samples over %d states  %s" %
                  (reservoir.total(), len(counts),
                   "  ".join("%s=%d" % (k, counts[k]) for k in sorted(counts))), flush = True)

        if tape is not None and ((it + 1) % tape_interval == 0 or it == 0 or it == iterations - 1):
            sc = replay_tape(policy, tape, device = cfg.device,
                             skip_arrived = getattr(cfg, "bc_motor_skip_arrived", False),
                             arrived_threshold = cfg.arrived_confidence_threshold)
            bal = sc.get("balanced")
            tape_hist.append(bal)
            win = [x for x in tape_hist[-3:] if x is not None]
            sm = sum(win) / len(win) if win else None
            scored = sc.get("scored_states", [])
            per = "  ".join("%s%s %.4f" % (n[:4], "" if n in scored else "*", sc[n])
                            for n in BC_STATES if n in sc)
            ah = ("  arrived_head P %.3f R %.3f F1 %.3f"
                  % (sc["arrived_precision"], sc["arrived_recall"], sc["arrived_f1"])
                  if "arrived_f1" in sc else "")
            print("  val_tape %s (smoothed %s)  %s%s"
                  % ("n/a" if bal is None else "%.5f" % bal,
                     "n/a" if sm is None else "%.5f" % sm, per, ah), flush = True)
            if sm is not None and (best_tape[0] is None or sm < best_tape[0]):
                best_tape[0] = sm
                if bc_out:
                    best_path = os.path.join(os.path.dirname(os.path.abspath(bc_out)) or ".",
                                             "actor_best.pt")
                    export_actor(best_path, policy, iteration = it)
                    print("  NEW BEST val_tape %.5f -> %s" % (sm, best_path), flush = True)
        if summary is not None:
            summary.update(it, {"rollout/mean_coverage": cov})
        if logger is not None:
            log_stats = dict(actor_stats)
            for k, v in oracle_stats.items():
                log_stats["oracle/" + k] = v
            log_stats["bc/motor_mse"] = loss
            log_stats["bc/motor_mse_mean"] = bc_extra.get("mean_loss", 0.0)
            log_stats["bc/grad_norm"] = bc_extra.get("grad_norm", 0.0)
            log_stats["bc/n_decisions"] = float(bc_extra.get("n_decisions", 0))
            logger.log_scalars(log_stats, it)
        if persist and reservoir.enabled:
            interval = max(1, int(getattr(cfg, "bc_replay_save_interval", 20)))
            if (it + 1) % interval == 0 or it == iterations - 1:
                mb, secs = reservoir.save(reservoir_path)
                print("  reservoir saved: %.0f MB in %.1fs -> %s" % (mb, secs, reservoir_path), flush = True)
        if bc_out and checkpoint_every > 0 and (it + 1) % checkpoint_every == 0:
            export_actor(bc_out, policy, iteration = it)
            print("checkpoint: saved progress through iter %d to %s" % (it, bc_out), flush=True)
        if on_iteration is not None:
            on_iteration(it, {"train_loss": loss, "train_loss_mean": bc_extra.get("mean_loss", 0.0),
                              "grad_norm": bc_extra.get("grad_norm", 0.0), "train_eval_cov": cov,
                              "oracle_cov": oracle_cov,
                              "oracle_success_rate": oracle_stats.get("rollout/success_rate"),
                              "oracle_mean_final_coverage": oracle_stats.get("rollout/mean_final_coverage"),
                              "train_eval_success_rate": actor_stats.get("rollout/success_rate"),
                              "train_eval_mean_final_coverage": actor_stats.get("rollout/mean_final_coverage")})
    if summary is not None:
        summary.finalize("ok")
    if logger is not None:
        logger.close()
    if bc_out:
        export_actor(bc_out, policy, iteration = it)
        print("saved cloned actor to %s" % bc_out)
