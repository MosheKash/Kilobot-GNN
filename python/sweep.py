"""Hyperparameter search: launch runs, score their summaries, suggest the next.

Each trial is a real training run in its own process, configured through
KILOBOT_* environment variables and scored from the RunSummary JSON it writes.
Owns no training logic of its own.
"""

# Hyperparameter sweep for the Kilobot trainer.
#
# Each trial launches `python launch.py` as a subprocess with a set of
# KILOBOT_* overrides, polls the run's summary.json while it trains, kills the
# run early if it is clearly worse than its peers, and scores the finished run.
#
# The score rewards coverage gained over the run and penalizes entropy that
# climbs across the run, so a high-scoring trial is exactly what we want: a
# config that trains and whose entropy does not blow up. Running the sweep
# therefore does both jobs at once. It first finds configs that train without
# entropy climbing, then the sampler concentrates on improving them.
#
# Optuna is imported lazily so this module (and its pure objective function)
# can be imported and tested without Optuna installed.
#
# Examples:
#   Explore on a small proxy task (few shapes, short runs):
#     python sweep.py --trials 40 --iters 40 --formations 8 --workers 2 --device cuda
#
#   Refine: keep adding trials to the same study on a harder task:
#     python sweep.py --trials 30 --iters 120 --formations 64 --device cuda
#
#   Confirm the best config across seeds before trusting it:
#     python sweep.py --trials 0 --confirm-seeds 3 --iters 120 --formations 64
#
#   Distributed: point several machines at one study and storage URL:
#     python sweep.py --storage postgresql://user:pw@host/db --study-name kilobot ...

import os
import sys
import json
import time
import signal
import subprocess


FAILED = None  # compute_objective returns this score when a run produced nothing usable


def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def compute_objective(summary, ent_penalty=0.5, ev_bonus=0.0):
    # Pure scoring function. Higher is better.
    #   coverage_gain: how much shape coverage improved over the run.
    #   entropy climb: the fitted total rise in entropy across the run; only the
    #                  positive part is penalized, since a falling entropy is fine.
    #   ev term:       a small optional bonus for a critic that explains variance,
    #                  which is a sign the learning signal is real.
    # Returns (score, breakdown). score is None if coverage is missing.
    cov_i = summary.get("coverage_initial")
    cov_f = summary.get("coverage_final")
    if cov_i is None or cov_f is None:
        return FAILED, {"reason": "no coverage in summary"}
    cov_gain = cov_f - cov_i
    span = max(1, int(summary.get("iterations", 1)) - 1)
    ent_slope = summary.get("entropy_slope", 0.0) or 0.0
    ent_trend = ent_slope * span
    ent_climb = max(0.0, ent_trend)
    ev = summary.get("explained_variance_final", 0.0) or 0.0
    ev_term = ev_bonus * max(0.0, min(ev, 1.0))
    score = cov_gain - ent_penalty * ent_climb + ev_term
    breakdown = {
        "score": score,
        "coverage_gain": cov_gain,
        "coverage_final": cov_f,
        "entropy_slope": ent_slope,
        "entropy_trend": ent_trend,
        "entropy_penalty": ent_penalty * ent_climb,
        "ev_final": ev,
        "ev_term": ev_term,
    }
    return score, breakdown


def progress_gain(summary):
    # The intermediate value reported to the pruner: coverage gained so far.
    cov_i = summary.get("coverage_initial")
    cov_f = summary.get("coverage_final")
    if cov_i is None or cov_f is None:
        return None
    return cov_f - cov_i


def _kill(proc):
    # Kill the whole process group so the subprocess's Unity workers die too.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def trial_env(params, args, run_dir, summary_path, seed):
    env = dict(os.environ)
    for k, v in params.items():
        env[k] = str(v)
    env.update({
        "KILOBOT_SMOKE": "0",
        "KILOBOT_ITERATIONS": str(args.iters),
        "KILOBOT_NUM_WORKERS": str(args.workers),
        "KILOBOT_NUM_ARENAS": str(args.arenas),
        "KILOBOT_MAX_FORMATIONS": str(args.formations),
        "KILOBOT_ROLLOUT_STEPS": str(args.rollout_steps),
        "KILOBOT_MAX_EPISODE_STEPS": str(args.max_episode_steps),
        "KILOBOT_DEVICE": str(args.device),
        "KILOBOT_LOGDIR": run_dir,
        "KILOBOT_SUMMARY": summary_path,
        "KILOBOT_CKPT_EVERY": "0",
        "KILOBOT_SEED": str(seed),
    })
    # Unity's ImageLibrary and the Python image pool both read KILOBOT_FORMATIONS.
    # Resolve to an absolute path so it works regardless of where the trial runs.
    fpath = args.formations_path
    if not os.path.isabs(fpath):
        fpath = os.path.abspath(os.path.join(args.cwd, fpath))
    env["KILOBOT_FORMATIONS"] = fpath
    return env


def launch_run(env, args):
    # Start launch.py in its own process group and return the Popen handle.
    return subprocess.Popen([sys.executable, "launch.py"], cwd=args.cwd, env=env,
                            start_new_session=True)


def run_once(params, args, run_dir, seed, trial=None):
    # Run one training subprocess to completion (or prune). Returns the finalized
    # summary dict, or None if the run produced nothing. When `trial` is given,
    # report progress to it and honor pruning.
    import optuna
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    env = trial_env(params, args, run_dir, summary_path, seed)
    proc = launch_run(env, args)
    start = time.time()
    last_reported = -1
    try:
        while True:
            ret = proc.poll()
            summ = load_summary(summary_path)
            if summ is not None and trial is not None:
                done = int(summ.get("iterations", 0))
                gain = progress_gain(summ)
                if done > last_reported and gain is not None:
                    trial.report(gain, step=done)
                    last_reported = done
                    if trial.should_prune():
                        _kill(proc)
                        raise optuna.TrialPruned()
            if ret is not None:
                break
            if time.time() - start > args.trial_timeout:
                _kill(proc)
                break
            time.sleep(args.poll)
        return load_summary(summary_path)
    finally:
        if proc.poll() is None:
            _kill(proc)


def suggest_params(trial):
    # The sensitive few, in the ranges worth searching first. k_pos (the
    # strength of the off-shape penalty gradient) and the entropy coefficient
    # have the most leverage on whether the run trains at all and whether
    # entropy stays bounded. k_pos is not the same knob as Config.reward_shaping
    # -- the separate potential-based term k*(prev_dist - gamma*dist) that
    # is what actually gives a robot far from
    # the target any gradient to move at all, since off_penalty saturates at
    # -k_pos beyond l_scale regardless of k_pos's value. reward_shaping is not
    # swept here and is not set anywhere in this file; it must be set by hand
    # in the launching shell (KILOBOT_REWARD_SHAPING=5.0, the project's
    # standing recommendation) or every trial silently runs at Config's
    # default of 0.0. See docs/sweep.md.
    return {
        "KILOBOT_ACTOR_LR": trial.suggest_float("actor_lr", 3e-5, 1e-3, log=True),
        "KILOBOT_CRITIC_LR": trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True),
        "KILOBOT_ENTROPY_COEF": trial.suggest_float("entropy_coef", 1e-5, 3e-2, log=True),
        "KILOBOT_CLIP": trial.suggest_float("clip", 0.1, 0.3),
        "KILOBOT_PPO_EPOCHS": trial.suggest_int("ppo_epochs", 3, 12),
        "KILOBOT_GAE_LAMBDA": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "KILOBOT_LOG_STD_INIT": trial.suggest_float("log_std_init", -1.5, 0.0),
        "KILOBOT_K_POS": trial.suggest_float("k_pos", 0.25, 4.0, log=True),
        "KILOBOT_R_PACK": trial.suggest_float("r_pack", 0.25, 4.0, log=True),
        "KILOBOT_PACK_RANGE": trial.suggest_float("pack_range", 0.08, 0.5),
    }


def make_objective(args):
    import optuna

    def objective(trial):
        params = suggest_params(trial)
        run_dir = os.path.join(args.out, "trial_%05d" % trial.number)
        summ = run_once(params, args, run_dir, args.seed, trial=trial)
        if summ is None:
            raise optuna.TrialPruned()
        score, breakdown = compute_objective(summ, args.ent_penalty, args.ev_bonus)
        if score is None:
            raise optuna.TrialPruned()
        for k, v in breakdown.items():
            trial.set_user_attr(k, v)
        return score

    return objective


def confirm_best(study, args):
    # Re-run the best config across several seeds to check it is not a fluke.
    # RL is high variance, so a winner should hold up across seeds.
    best = dict(study.best_params)
    params = {
        "KILOBOT_ACTOR_LR": best["actor_lr"],
        "KILOBOT_CRITIC_LR": best["critic_lr"],
        "KILOBOT_ENTROPY_COEF": best["entropy_coef"],
        "KILOBOT_CLIP": best["clip"],
        "KILOBOT_PPO_EPOCHS": best["ppo_epochs"],
        "KILOBOT_GAE_LAMBDA": best["gae_lambda"],
        "KILOBOT_LOG_STD_INIT": best["log_std_init"],
        "KILOBOT_K_POS": best["k_pos"],
        "KILOBOT_R_PACK": best["r_pack"],
        "KILOBOT_PACK_RANGE": best["pack_range"],
    }
    scores = []
    for s in range(args.confirm_seeds):
        run_dir = os.path.join(args.out, "confirm_seed_%d" % s)
        summ = run_once(params, args, run_dir, s, trial=None)
        if summ is None:
            print("  seed %d: run produced no summary" % s)
            continue
        score, breakdown = compute_objective(summ, args.ent_penalty, args.ev_bonus)
        scores.append(score)
        print("  seed %d: score %.4f (cov_gain %.4f, entropy_trend %.4f)"
              % (s, score, breakdown["coverage_gain"], breakdown["entropy_trend"]))
    if scores:
        mean = sum(scores) / len(scores)
        var = sum((x - mean) ** 2 for x in scores) / len(scores)
        print("confirm: mean %.4f  std %.4f  over %d seeds" % (mean, var ** 0.5, len(scores)))
    return scores


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Optuna hyperparameter sweep for the Kilobot trainer")
    ap.add_argument("--trials", type=int, default=40, help="number of trials to run (0 to only confirm)")
    ap.add_argument("--iters", type=int, default=40, help="training iterations per trial")
    ap.add_argument("--workers", type=int, default=2, help="KILOBOT_NUM_WORKERS per trial")
    ap.add_argument("--arenas", type=int, default=9, help="KILOBOT_NUM_ARENAS per trial")
    ap.add_argument("--formations", type=int, default=8, help="proxy task size: how many target shapes")
    ap.add_argument("--formations-path", default=os.path.join("..", "data", "formations"),
                    help="folder of target images; sets KILOBOT_FORMATIONS for each trial")
    ap.add_argument("--rollout-steps", type=int, default=256,
                    help="decision steps collected per iteration; smaller is faster (config default is 4096)")
    ap.add_argument("--max-episode-steps", type=int, default=256,
                    help="max steps per episode for the proxy task (config default is 2048)")
    ap.add_argument("--device", default="cuda", help="KILOBOT_DEVICE for the learner")
    ap.add_argument("--seed", type=int, default=0, help="seed for trials and the TPE sampler")
    ap.add_argument("--ent-penalty", type=float, default=0.5, help="weight on entropy climbing in the score")
    ap.add_argument("--ev-bonus", type=float, default=0.0, help="small bonus for critic explained variance")
    ap.add_argument("--out", default=os.path.join("..", "results", "sweeps", "run"),
                    help="directory for per-trial logs, summaries, and the study db")
    ap.add_argument("--study-name", default="kilobot")
    ap.add_argument("--storage", default=None,
                    help="optuna storage URL; default is a sqlite file under --out")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between summary polls")
    ap.add_argument("--trial-timeout", type=float, default=3600.0, help="hard cap per trial in seconds")
    ap.add_argument("--warmup", type=int, default=8, help="iterations before pruning may start")
    ap.add_argument("--confirm-seeds", type=int, default=0,
                    help="after the sweep, re-run the best config across this many seeds")
    ap.add_argument("--cwd", default=os.path.dirname(os.path.abspath(__file__)),
                    help="working directory to launch.py from")
    args = ap.parse_args()

    import optuna
    os.makedirs(args.out, exist_ok=True)
    storage = args.storage or ("sqlite:///" + os.path.abspath(os.path.join(args.out, "sweep.db")))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.warmup),
    )
    print("study '%s' at %s" % (args.study_name, storage))

    if args.trials > 0:
        study.optimize(make_objective(args), n_trials=args.trials)

    if len(study.trials) > 0 and any(t.value is not None for t in study.trials):
        print("best score : %.4f" % study.best_value)
        print("best params:", json.dumps(study.best_params, indent=2))
        with open(os.path.join(args.out, "best.json"), "w") as f:
            json.dump({"value": study.best_value, "params": study.best_params,
                       "user_attrs": study.best_trial.user_attrs}, f, indent=2)
        if args.confirm_seeds > 0:
            print("confirming best config across %d seeds" % args.confirm_seeds)
            confirm_best(study, args)
    else:
        print("no completed trials yet")


if __name__ == "__main__":
    main()
