"""
run_bc_monitored.py -- BC training (cloning the actor to simple_oracle.py,
), same collect/fit/eval loop and --instances/
--arenas scaling as run_bc_simple_oracle.py, with three things added: a
held-out validation split, so "best" means "generalizes to formations never
trained on," not "fits the training batches best"; a best-by-validation
checkpoint kept alongside the usual latest-progress one; and a live-
updating progress image so a run left going for days can be checked on at
a glance without touching a terminal.

Usage: python run_bc_monitored.py <out-dir> [--iterations 300]
    [--instances 4] [--arenas 4] [--formations ../data/formations]
    [--encoder ../data/image_encoder.pt] [--limit N] [--min-bots 40]
    [--max-bots 60] [--val-count 2000] [--eval-interval 5]
    [--plot-interval 30] [--bc-epochs 4] [--device cpu]

--instances/--arenas mean exactly what they already mean in
run_bc_simple_oracle.py: --instances separate headless Unity players (each
with its own worker id, and its own --swarm-rng offset so their rollouts
don't correlate), --arenas is cfg.num_arenas,
how many parallel arenas each one runs. All of them feed the SAME actor and
optimizer -- this is one training run collecting instances*arenas parallel
rollouts per iteration, not several independent runs. Total parallel
arenas collected per iteration is instances * arenas.

Writes to <out-dir>/:
    actor_latest.pt   most recent completed iteration, for resuming
    actor_best.pt     the iteration with the best held-out validation
                      coverage seen so far
    history.jsonl     one JSON object per completed iteration
    progress.png      four-panel plot, redrawn every --plot-interval
                      seconds: train loss, train-distribution coverage
                      against the oracle's own ceiling, held-out
                      validation coverage (the actual overfitting
                      signal), and a status line

Validation formations: --val-count of them (default 2000), held out of
training and used only for the periodic eval below, carved out once via a
seeded shuffle into their own directory the first time this is run for a
given <out-dir>, and reused unchanged on every resume. Training draws from
the full, original formation folder rather than a second, near-total
symlink tree excluding those -- at typical pool sizes the chance any
training rollout also happens to draw one of a few thousand held-out
formations is small enough not to meaningfully affect what the validation
number measures, and was not judged worth a second, much larger symlink
tree just to close.

Safe to interrupt (Ctrl+C, a crash, closing the terminal) and resume:
re-running the same command with the same <out-dir> continues from
actor_latest.pt and history.jsonl rather than starting over.
"""
import argparse
import atexit
import json
import os
import random
import resource
import time

import torch


def formation_split(folder, val_count, seed):
    names = sorted(n for n in os.listdir(folder) if n.endswith(".png"))
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    val_set = set(shuffled[:val_count])
    return [n for n in names if n in val_set]


def ensure_val_dir(folder, val_dir, val_count, seed):
    marker = os.path.join(val_dir, "_names.json")
    if os.path.exists(marker):
        with open(marker) as f:
            return json.load(f)
    names = formation_split(folder, val_count, seed)
    os.makedirs(val_dir, exist_ok = True)
    for n in names:
        link = os.path.join(val_dir, n)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(folder, n)), link)
    with open(marker, "w") as f:
        json.dump(names, f)
    return names


def load_history(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def draw_progress(history, total_iterations, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize = (14, 25))
    ax_loss, ax_train_cov = axes[0][0], axes[0][1]
    ax_train_final, ax_val_cov = axes[1][0], axes[1][1]
    ax_val_final, ax_status = axes[2][0], axes[2][1]
    ax_oracle_state_pct, ax_oracle_state_raw = axes[3][0], axes[3][1]
    ax_actor_state_pct, ax_actor_state_raw = axes[4][0], axes[4][1]
    ax_memory, ax_spare = axes[5][0], axes[5][1]
    ax_spare.axis("off")

    state_names = ["go_north", "turning", "wall_following", "navigating", "arrived"]
    state_colors = ["tab:gray", "tab:purple", "tab:brown", "tab:cyan", "tab:green"]

    if history:
        its = [r["iteration"] for r in history]
        ax_loss.plot(its, [r["train_loss_mean"] for r in history], color = "tab:blue", linewidth = 1)
        ax_train_cov.plot(its, [r["train_eval_cov"] for r in history], color = "tab:blue",
                          label = "actor (mean, per-tick)", linewidth = 1)
        ax_train_cov.plot(its, [r["oracle_cov"] for r in history], color = "tab:orange",
                          label = "oracle ceiling", linewidth = 1, linestyle = ":")
        ax_train_cov.legend(fontsize = 8)

        # Oracle and actor state panels stacked one on top
        # of the other (same column, adjacent rows) for direct comparison,
        # not side by side -- oracle above, actor below, in that order to
        # match the loss/coverage panels above them (actor's own current
        # behavior is always what is being compared AGAINST the oracle,
        # not the other way around).
        state_rows = [r for r in history if r.get("state_total")]
        if state_rows:
            state_its = [r["iteration"] for r in state_rows]
            pct_series = [[(r["state_pct"].get(s) or 0.0) * 100.0 for r in state_rows] for s in state_names]
            ax_oracle_state_pct.stackplot(state_its, *pct_series, labels = state_names, colors = state_colors)
            ax_oracle_state_pct.legend(fontsize = 7, loc = "upper left", bbox_to_anchor = (1.0, 1.0))
            for s, color in zip(state_names, state_colors):
                raw_series = [r["state_raw"].get(s, 0) for r in state_rows]
                ax_oracle_state_raw.plot(state_its, raw_series, color = color, label = s, linewidth = 1)
            ax_oracle_state_raw.legend(fontsize = 7, loc = "upper left", bbox_to_anchor = (1.0, 1.0))

        actor_state_rows = [r for r in history if r.get("actor_state_total")]
        if actor_state_rows:
            actor_state_its = [r["iteration"] for r in actor_state_rows]
            actor_pct_series = [[(r["actor_state_pct"].get(s) or 0.0) * 100.0 for r in actor_state_rows]
                                for s in state_names]
            ax_actor_state_pct.stackplot(actor_state_its, *actor_pct_series, labels = state_names, colors = state_colors)
            ax_actor_state_pct.legend(fontsize = 7, loc = "upper left", bbox_to_anchor = (1.0, 1.0))
            for s, color in zip(state_names, state_colors):
                actor_raw_series = [r["actor_state_raw"].get(s, 0) for r in actor_state_rows]
                ax_actor_state_raw.plot(actor_state_its, actor_raw_series, color = color, label = s, linewidth = 1)
            ax_actor_state_raw.legend(fontsize = 7, loc = "upper left", bbox_to_anchor = (1.0, 1.0))

        mem_rows = [r for r in history if r.get("rss_mb") is not None]
        if mem_rows:
            ax_memory.plot([r["iteration"] for r in mem_rows], [r["rss_mb"] for r in mem_rows],
                          color = "tab:red", label = "system RAM (RSS)", linewidth = 1)
        vram_rows = [r for r in history if r.get("vram_mb") is not None]
        if vram_rows:
            ax_memory.plot([r["iteration"] for r in vram_rows], [r["vram_mb"] for r in vram_rows],
                          color = "tab:orange", label = "VRAM (reserved)", linewidth = 1)
        if mem_rows or vram_rows:
            ax_memory.legend(fontsize = 8)

        # success_rate (fraction of episodes reaching success_threshold) is
        # NOT plotted here -- with success_threshold at its own default of
        # 1.1 (disabled, "don't be hasty"), it is structurally always 0
        # regardless of how well training is going, so a panel for it would
        # always be a flat, uninformative line. mean_final_coverage (actual
        # coverage at the moment each episode ends, success or timeout) is
        # the metric that still varies meaningfully under that default, and
        # is what actor_best.pt is actually kept by below -- plotted here
        # instead, which a prior pass at this plot omitted entirely despite
        # already computing and logging it every iteration.
        final_rows = [r for r in history if r.get("train_eval_mean_final_coverage") is not None]
        if final_rows:
            ax_train_final.plot([r["iteration"] for r in final_rows],
                                [r["train_eval_mean_final_coverage"] for r in final_rows],
                                color = "tab:blue", label = "actor", linewidth = 1)
            oracle_final_rows = [r for r in history if r.get("oracle_mean_final_coverage") is not None]
            ax_train_final.plot([r["iteration"] for r in oracle_final_rows],
                                [r["oracle_mean_final_coverage"] for r in oracle_final_rows],
                                color = "tab:orange", label = "oracle ceiling", linewidth = 1, linestyle = ":")
            ax_train_final.legend(fontsize = 8)

        val_its = [r["iteration"] for r in history if r.get("val_cov") is not None]
        val_covs = [r["val_cov"] for r in history if r.get("val_cov") is not None]
        if val_its:
            ax_val_cov.plot(val_its, val_covs, color = "tab:green", marker = "o", markersize = 3, linewidth = 1)

        val_final_its = [r["iteration"] for r in history if r.get("val_final_cov") is not None]
        val_finals = [r["val_final_cov"] for r in history if r.get("val_final_cov") is not None]
        if val_final_its:
            ax_val_final.plot(val_final_its, val_finals, color = "tab:red", marker = "o", markersize = 3, linewidth = 1)

    ax_loss.set_title("train loss (bc motor mse, mean per iteration)")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha = 0.3)

    ax_train_cov.set_title("train-distribution mean coverage (per-tick average, diluted by travel time)")
    ax_train_cov.set_xlabel("iteration")
    ax_train_cov.set_ylim(0, 1)
    ax_train_cov.grid(alpha = 0.3)

    ax_train_final.set_title("train-distribution final coverage (at episode end, not diluted by travel time)")
    ax_train_final.set_xlabel("iteration")
    ax_train_final.set_ylim(0, 1)
    ax_train_final.grid(alpha = 0.3)

    ax_val_cov.set_title("held-out validation mean coverage")
    ax_val_cov.set_xlabel("iteration")
    ax_val_cov.set_ylim(0, 1)
    ax_val_cov.grid(alpha = 0.3)

    ax_val_final.set_title("held-out validation final coverage (what actor_best.pt is kept by)")
    ax_val_final.set_xlabel("iteration")
    ax_val_final.set_ylim(0, 1)
    ax_val_final.grid(alpha = 0.3)

    # An actor-driven equivalent of the state breakdown
    # below, to compare against the oracle's own -- on_iteration's own
    # comment (search "shadow observer") has the full mechanism. Worth
    # repeating here since this correction sits right next to a title that
    # otherwise would still say "actor has no equivalent," which was true
    # before this and is no longer true now.
    ax_oracle_state_pct.set_title("robot state breakdown, ORACLE-driven collection (% of robot-ticks)")
    ax_oracle_state_pct.set_xlabel("iteration")
    ax_oracle_state_pct.set_ylabel("%")
    ax_oracle_state_pct.set_ylim(0, 100)
    ax_oracle_state_pct.grid(alpha = 0.3)

    ax_oracle_state_raw.set_title("robot state breakdown, ORACLE-driven collection (raw robot-ticks)")
    ax_oracle_state_raw.set_xlabel("iteration")
    ax_oracle_state_raw.set_ylabel("robot-ticks")
    ax_oracle_state_raw.grid(alpha = 0.3)

    ax_actor_state_pct.set_title("robot state breakdown, ACTOR-driven collection (% of robot-ticks)")
    ax_actor_state_pct.set_xlabel("iteration")
    ax_actor_state_pct.set_ylabel("%")
    ax_actor_state_pct.set_ylim(0, 100)
    ax_actor_state_pct.grid(alpha = 0.3)

    ax_actor_state_raw.set_title("robot state breakdown, ACTOR-driven collection (raw robot-ticks)")
    ax_actor_state_raw.set_xlabel("iteration")
    ax_actor_state_raw.set_ylabel("robot-ticks")
    ax_actor_state_raw.grid(alpha = 0.3)

    # a real run at this file's own prior defaults died with SIGKILL and no
    # other message -- the OOM killer's own signature (see
    # max_episode_steps' own comment for the full incident). This panel is
    # the direct answer: a steady climb here, not a plateau, is the same
    # warning sign a real OOM kill gives, just visible before it happens
    # instead of after. VRAM (reserved) is reported alongside because RSS
    # alone cannot answer "is there room to turn arenas up" at all, since
    # it only ever reflected system RAM, never the GPU's own memory.
    ax_memory.set_title("memory -- should plateau, not climb steadily")
    ax_memory.set_xlabel("iteration")
    ax_memory.set_ylabel("MB")
    ax_memory.grid(alpha = 0.3)

    ax_status.axis("off")
    lines = ["status  (updated %s)" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    if not history:
        lines.append("not started yet")
    else:
        last = history[-1]
        lines.append("iteration %d / %d" % (last["iteration"] + 1, total_iterations))
        if last.get("rss_mb") is not None:
            lines.append("memory (RSS)             %.0f MB" % last["rss_mb"])
        if last.get("vram_mb") is not None:
            lines.append("memory (VRAM reserved)   %.0f MB" % last["vram_mb"])
        lines.append("train loss (mean)       %.5f" % last["train_loss_mean"])
        lines.append("train-eval final coverage %s" %
                     (("%.4f" % last["train_eval_mean_final_coverage"]) if last.get("train_eval_mean_final_coverage") is not None else "n/a"))
        lines.append("oracle final coverage      %s" %
                     (("%.4f" % last["oracle_mean_final_coverage"]) if last.get("oracle_mean_final_coverage") is not None else "n/a"))
        lines.append("")
        lines.append("best val final coverage %.4f" % last.get("best_val_final_cov", -1.0))
        lines.append("(actor_best.pt is that checkpoint)")
        if last.get("state_total"):
            lines.append("")
            lines.append("last iteration's state breakdown (oracle-driven):")
            for s in state_names:
                pct = (last["state_pct"].get(s) or 0.0) * 100.0
                raw = last["state_raw"].get(s, 0)
                lines.append("  %-15s %5.1f%%  (%d)" % (s, pct, raw))
    ax_status.text(0.02, 0.98, "\n".join(lines), va = "top", ha = "left", fontsize = 10, family = "monospace",
                  transform = ax_status.transAxes)

    fig.tight_layout()
    tmp = png_path + ".tmp.png"
    fig.savefig(tmp, dpi = 110)
    plt.close(fig)
    os.replace(tmp, png_path)


def preprocess(path):
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.asarray(img, dtype = np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def build_parser():
    """Every knob this driver exposes. Defaults are the ones actually used."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--iterations", type = int, default = 300)
    ap.add_argument("--formations", default = "../data/formations")
    ap.add_argument("--encoder", default = "../data/image_encoder.pt")
    ap.add_argument("--limit", type = int, default = 5000)
    ap.add_argument("--min-bots", type = int, default = 40)
    ap.add_argument("--max-bots", type = int, default = 60)
    ap.add_argument("--instances", type = int, default = 4)
    ap.add_argument("--arenas", type = int, default = 4)
    ap.add_argument("--rollout", type = int, default = 192)
    ap.add_argument("--val-rollout", type = int, default = 192)
    ap.add_argument("--val-max-episode-steps", type = int, default = 0,
                   help = "episode length for the held-out validation worker. 0 (default) "
                          "inherits --max-episode-steps. val_final_cov is coverage at the "
                          "moment an episode ENDS, so it only ever populates if the val "
                          "worker accumulates a full episode's worth of ticks across the run "
                          "-- see the warning main() prints when it cannot.")
    ap.add_argument("--max-episode-steps", type = int, default = 18000)
    ap.add_argument("--success-threshold", type = float, default = 1.1)
    ap.add_argument("--cold-start-injection-prob", type = float, default = 0.0,
                    help = "config.py's own cold_start_injection_prob -- fraction of BC's oracle-driven "
                           "decisions that get a forced h_prev=0, to fix a real, confirmed defect where "
                           "the actor almost never sees a genuine cold start during training (~0.27%% of "
                           "decisions naturally). 0.0 (default) is the original, unaffected behavior.")
    ap.add_argument("--turning-duplicate-factor", type = int, default = 0,
                    help = "config.py's own turning_duplicate_factor -- each real, naturally-occurring "
                           "turning-state BC example gets included this many additional times in the "
                           "same update, giving it more weight in the averaged loss without ever "
                           "synthesizing an observation or a target. Replaces the earlier "
                           "turning_injection_prob, which was found, via direct measurement, to pair "
                           "its own fixed target with an authentic wall reading only 1.7%% of the time. "
                           "0 (default) is the original, unaffected behavior.")
    ap.add_argument("--bc-replay-capacity", type = int, default = 0,
                    help = "per-oracle-state BC replay reservoir capacity (0 = off, the "
                           "previous behaviour of fitting only the current rollout window)")
    ap.add_argument("--bc-replay-unbalanced", action = "store_true",
                    help = "sample the reservoir in proportion to how much data each state "
                           "has instead of giving every state an equal share of each minibatch")
    ap.add_argument("--bc-replay-max-age", type = int, default = 0,
                    help = "drop reservoir samples older than this many iterations (0 = unbounded)")
    ap.add_argument("--bc-replay-evict", default = "random", choices = ["random", "fifo"],
                    help = "which samples to keep once a state is at capacity")
    ap.add_argument("--memoryless-actor", action = "store_true",
                    help = "ablation: replace the GRU with a parameter-matched feedforward "
                           "aggregator, isolating recurrence from capacity")
    ap.add_argument("--bc-actor-eval-interval", type = int, default = 1,
                    help = "run bc_train's actor-driven eval collect only every N iterations. it "
                           "produces no training data and costs the same simulation as the "
                           "oracle collect, so N>1 is close to a straight wall-clock saving")
    ap.add_argument("--bc-replay-persist", action = "store_true",
                    help = "save the replay reservoir to <out_dir>/bc_reservoir.pt and reload it "
                           "on resume. without this an interrupted run restarts with an empty "
                           "reservoir and loses every rare-state sample it had accumulated")
    ap.add_argument("--bc-replay-save-interval", type = int, default = 20,
                    help = "how many iterations between reservoir saves (size and duration are "
                           "printed on every save so this can be tuned from real numbers)")
    ap.add_argument("--bc-replay-min-samples", type = int, default = 512,
                    help = "a state ramps in linearly to its full equal share of a minibatch "
                           "until it holds this many samples")
    ap.add_argument("--bc-motor-skip-arrived", action = "store_true",
                    help = "drop arrived decisions from the motor loss entirely, so the motor "
                           "head never learns to output [0,0] and stopping is handled solely by "
                           "the arrived head gating the command. requires --use-arrived-head")
    ap.add_argument("--bc-arrived-natural-prior", action = "store_true",
                    help = "reweight the arrived head's BCE term back to the class prior actually "
                           "present in the reservoir, so balanced sampling does not inflate it")
    ap.add_argument("--val-tape-ticks", type = int, default = 0,
                    help = "record a held-out oracle-driven validation tape this many ticks long "
                           "ONCE, then replay it through the actor at every eval (0 = off, keep "
                           "the old coverage-based selection). 18000 covers a full episode, so "
                           "all five oracle states are represented")
    ap.add_argument("--val-tape-path", default = None,
                    help = "where to cache the tape (default <out_dir>/val_tape.pt). reused "
                           "across runs, so the one-time recording cost is paid once")
    ap.add_argument("--val-tape-max-robots", type = int, default = 64,
                    help = "cap the number of robot sequences kept in the tape")
    ap.add_argument("--val-probe-ticks", type = int, default = 384,
                    help = "short actor-driven cold-start rollout at each eval, scored on "
                           "behaviour rather than task completion (0 = off)")
    ap.add_argument("--val-probe-min-moved", type = float, default = 0.5,
                    help = "an actor whose cold-start moved-fraction is below this cannot be "
                           "written as best, however good its imitation error looks")
    ap.add_argument("--val-smooth", type = int, default = 3,
                    help = "select on the mean of the last N evals rather than a single one")
    ap.add_argument("--arrived-release-threshold", type = float, default = 0.0,
                    help = "confidence at or below which a switched-off robot switches back on "
                           "(0 = never, the original permanent behaviour)")
    ap.add_argument("--use-arrived-head", action = "store_true",
                    help = "config.py's own use_arrived_head -- adds a new, trained head that flips a "
                           "high-confidence arrived flag, forcing the motor to zero and freezing the "
                           "GRU's own hidden state from that point on, instead of relying on the "
                           "continuous motor head alone to learn 'sit here'. Off by default -- an "
                           "existing checkpoint's own state_dict still loads cleanly either way.")
    ap.add_argument("--use-turn-anchor", action = "store_true",
                    help = "config.py's own use_turn_anchor -- adds two values to prop, "
                           "sin/cos of the heading change since the actor's own real wall reading "
                           "last went from zero to nonzero, so the network no longer has to "
                           "reconstruct that relative angle from an absolute heading and its own "
                           "hidden state alone. Off by default -- expands the actor's own input "
                           "width by 2 when set, so a checkpoint trained with this flag will not "
                           "load into an actor built without it, or vice versa.")
    ap.add_argument("--debug-per-arena-threshold", type = float, default = None,
                    help = "Diagnostic only, off by default (None). When set (e.g. 0.5), prints "
                           "a direct, per-arena line -- instance index, arena index, this specific "
                           "arena's own state_pct['arrived'], and this same arena's own real "
                           "coverage() value right now -- for every arena whose own arrived share "
                           "is at or above this threshold, right after each iteration's oracle-"
                           "driven collect(). For isolating whether oracle_cov's own, all-arena "
                           "average is hiding a real, individual arena's own high coverage, versus "
                           "a genuine per-arena discrepancy between the two metrics.")
    ap.add_argument("--debug-iteration-detail", action = "store_true",
                    help = "Diagnostic only, off by default. Prints, every iteration, right after "
                           "the oracle-driven collect(): an arena-stage histogram (each arena's own "
                           "real-time coverage(), bucketed -- arena-level, distinct from state_pct's "
                           "own robot-level census), how many episodes completed within this "
                           "specific iteration's own rollout window (split success vs timeout, with "
                           "their own mean coverage and length at completion), a wall-clock timing "
                           "breakdown (step/parse/getsteps/snap/act), a reception-event breakdown "
                           "(seed/wall vs neighbor vs heartbeat-triggered decisions), and a belief "
                           "confidence summary (mean conf_pos, fraction genuinely localized). "
                           "Everything printed reuses data the training loop already computes every "
                           "iteration -- no extra rollouts or passes.")
    ap.add_argument("--arrived-confidence-threshold", type = float, default = 0.95,
                    help = "config.py's own arrived_confidence_threshold -- only used when "
                           "--use-arrived-head is set.")
    ap.add_argument("--arrived-loss-weight", type = float, default = 1.0,
                    help = "config.py's own arrived_loss_weight -- only used when --use-arrived-head "
                           "is set.")
    ap.add_argument("--heartbeat", type = int, default = 48)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--swarm-rng", type = int, default = None,
                   help = "seed the Unity player's spawn RNG, making the run's sequence of arenas "
                          "replayable (episodes still differ from one another). Distinct from "
                          "--seed, which only seeds torch/numpy on this side. Each player gets "
                          "this plus its worker id, so parallel instances stay diverse. Left "
                          "unset, spawns are unseeded, as they have always been.")
    ap.add_argument("--bc-epochs", type = int, default = 4)
    ap.add_argument("--device", default = "cpu")
    ap.add_argument("--build", default = None,
                   help = "path to the Unity player (default ../Builds/Kilobot.x86_64)")
    ap.add_argument("--base-port", type = int, default = 5005,
                   help = "ml-agents derives each player's socket from base_port + worker_id; "
                          "raise it if another run is already using these ports")
    ap.add_argument("--time-scale", type = float, default = 20.0,
                   help = "Unity time scale for headless collection")
    ap.add_argument("--val-count", type = int, default = 2000)
    ap.add_argument("--eval-interval", type = int, default = 5)
    ap.add_argument("--plot-interval", type = int, default = 30)
    return ap


def build_train_cfg(args):
    """The BC-collection config: simple_oracle drives, the actor watches."""
    from config import Config
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.seed_layout = "corners"
    cfg.heartbeat_ticks = args.heartbeat
    cfg.rollout_steps = args.rollout
    cfg.motor_override = "simple_oracle"
    cfg.oracle_known_start_heading = True
    cfg.num_arenas = args.arenas
    # real bug, found while answering a direct question about GPU use --
    # args.device was already passed to load_encoder/build_image_pool, but
    # never to cfg.device itself, which is what Trainer actually reads for
    # tensor placement (trainer.py's own torch.Generator(device=self.cfg.
    # device) and actor_io.split_obs(..., self.cfg.device)) -- meaning
    # --device cuda would have silently left the policy's own computation
    # on cpu regardless, only moving the encoder. Confirmed this is worth
    # fixing, not just a formality: a real, timed collect() call (built-in
    # Trainer.collect_timing(), not new instrumentation) showed "act" --
    # the oracle/actor decision logic, all torch, batched across every
    # deciding robot at once, including the belief filter's own particle
    # computation -- at 68.9% of total time, vs only 18.1% for "step" (the
    # numpy-only physics this file's own earlier comments assumed would
    # dominate). Batched torch computation is exactly what a GPU should
    # help with; the numpy-based simulation stepping will not speed up
    # regardless of --device, since it never touches torch at all.
    cfg.device = args.device
    # config.py's own default (2048) is shorter than a robot typically
    # needs to actually reach "arrived" under the current sparse-wall
    # configuration. Measured twice, not assumed: a first pass (3000-tick
    # test window) gave mean 2443/median 2497/p90 2905, but only 21% of
    # spawned robots ever reached arrived within that window -- the other
    # 79% were still in progress when the window ended, meaning that
    # measurement was itself right-censored (biased low; slower robots
    # never got counted at all). Re-measured with a 7500-tick window
    # instead: 87% completion, mean 3384/median 3757/p90 4823/max 5062 --
    # closer to the true distribution, though the remaining 13% still
    # uncounted means even this may understate the tail somewhat. 18000,
    # roughly 3.5x the measured max, is deliberately generous rather than
    # tightly fit to a number already known to be an underestimate --
    # Long enough to more than cover the robots converging -- don't be
    # hasty to end the episode," with substantial settled time built in
    # for the actor to see and learn "stay put" behavior, not just enough
    # margin to converge and stop.
    cfg.max_episode_steps = args.max_episode_steps
    # --rollout and --max-episode-steps are INDEPENDENT. Trainer.collect
    # allocates a fresh RolloutBuffer per call and resets only its own
    # accumulators; worker.step_count and the rest of simulation state live on
    # the worker and persist, so a robot's episode continues across collects.
    # A rollout only needs to be as long as makes sense for one training
    # update's batch size.
    #
    # Setting it to a full episode instead is what once OOM-killed a real run:
    # measured at ~202MB RSS growth per 300 ticks x 1 arena x 50-60 bots with
    # bc_capture on, extrapolating to roughly 380GB at 36000 ticks x 16 arenas.
    #
    # Note the state breakdown resets every ~47 iterations, not the ~94 that
    # max_episode_steps/rollout suggests: bc_train calls collect() TWICE per
    # iteration against the same worker -- once oracle-driven for training data,
    # once actor-driven to measure eval coverage -- and both advance the same
    # step_count. 18000/(2*192) = 46.875. The eval collect's actions are also
    # where the next iteration's training data starts from; there is no reset
    # between them.
    #
    # success_threshold is set above the maximum possible coverage of 1.0, so
    # coverage crossing a threshold never ends an episode early; only the step
    # limit does.
    cfg.success_threshold = args.success_threshold
    cfg.cold_start_injection_prob = args.cold_start_injection_prob
    cfg.turning_duplicate_factor = args.turning_duplicate_factor
    cfg.bc_replay_capacity = args.bc_replay_capacity
    cfg.bc_replay_balanced = not args.bc_replay_unbalanced
    cfg.bc_replay_max_age = args.bc_replay_max_age
    cfg.bc_replay_evict = args.bc_replay_evict
    cfg.bc_replay_min_samples = args.bc_replay_min_samples
    cfg.actor_recurrent = not args.memoryless_actor
    cfg.bc_actor_eval_interval = args.bc_actor_eval_interval
    cfg.bc_replay_persist = args.bc_replay_persist
    cfg.bc_replay_save_interval = args.bc_replay_save_interval
    cfg.bc_replay_path = os.path.join(args.out_dir, "bc_reservoir.pt")
    cfg.bc_motor_skip_arrived = args.bc_motor_skip_arrived
    cfg.bc_arrived_natural_prior = args.bc_arrived_natural_prior
    cfg.arrived_release_threshold = args.arrived_release_threshold
    cfg.seed = args.seed
    cfg.use_arrived_head = args.use_arrived_head
    cfg.use_turn_anchor = args.use_turn_anchor
    cfg.arrived_confidence_threshold = args.arrived_confidence_threshold
    cfg.arrived_loss_weight = args.arrived_loss_weight
    # config.py's own bc_motor_skip_arrived has the rationale. Refused rather
    # than silently corrected: with no arrived head there is nothing to gate
    # the motor command, so a motor head never trained to stop would leave
    # every robot driving forever -- a run that looks like it started fine and
    # is worthless hours later.
    if cfg.bc_motor_skip_arrived and not cfg.use_arrived_head:
        raise SystemExit("--bc-motor-skip-arrived requires --use-arrived-head: without the "
                         "arrived head nothing stops a robot, since the motor head is "
                         "deliberately never trained to output [0,0]")
    if cfg.arrived_release_threshold >= cfg.arrived_confidence_threshold and cfg.arrived_release_threshold > 0:
        raise SystemExit("--arrived-release-threshold must be strictly below "
                         "--arrived-confidence-threshold (it is a hysteresis floor, not a re-test)")
    # config.py's own comment on split_prop_time_scale says to re-derive
    # this if max_episode_steps changes materially. Directly measured for
    # the current max_episode_steps=18000 (not scaled proportionally from
    # an older value the way an earlier pass at this did): captured the
    # actual anchor-tracker elapsed-time distribution during a real,
    # bc_capture-enabled rollout, p90 17.1, set to 1/p90. Notably close to
    # an earlier direct measurement taken at max_episode_steps=6000 (p90
    # 20.0, 1/p90=0.05) despite the 3x difference in episode length --
    # this quantity does not appear to scale strongly with
    # max_episode_steps at all (consistent with it tracking time since a
    # landmark was last re-observed, not time since episode start), so
    # re-deriving this on every future episode-length change may matter
    # less than config.py's own comment implies, though re-measuring
    # directly remains cheap and was done again here regardless.
    cfg.split_prop_time_scale = 0.058
    return cfg


def build_workers(args, cfg, val_cfg, formation_pool, val_formation_pool, val_dir):
    """The training workers and the held-out validation worker, on either backend.

    Returns (workers, val_worker, close_envs) -- close_envs is a no-argument
    callable the caller must run at the end, since Unity players are separate
    processes that outlive this one if never closed.

    Unity is the default and the one that matters: it is what training and
    evaluation are actually measured on. Each training instance is its own
    headless player, and the validation worker is one more on top -- so
    --instances 4 means 5 Unity processes, not 4. worker_id must be distinct
    across all of them (ml-agents derives the socket port from it), which is
    why the validation worker takes the slot just past the training ones rather
    than a large arbitrary offset the way its replica env_seed does.

    Swarm size: --min-bots/--max-bots reach the player through
    KILOBOT_MIN_BOTS/KILOBOT_MAX_BOTS, which SwarmManager reads in preference to
    its Inspector fields.
    """
    import unity_env
    envs = []

    def spawn(worker_id, formations):
        # KILOBOT_FORMATIONS must be re-set immediately before EACH launch, not
        # once for the whole group: SwarmManager/ImageLibrary read it in Awake
        # from the environment the player inherits, so whatever is set at spawn
        # time is what that player is stuck with for its whole life.
        #
        # The validation player therefore needs the HELD-OUT directory, not the
        # training one. Setting it once for everything was a real bug: Python
        # fed the val worker imageIds indexed into the 2000-image val pool while
        # its Unity side had loaded the ~172500-image training directory, so
        # ImageLibrary resolved each id against a completely different sorted
        # file list. The floor shape, and the node dist column coverage is
        # measured from, then described a different formation than the actor was
        # steering toward -- making every held-out number meaningless rather
        # than merely wrong.
        #
        # swarm_rng is offset by worker_id for the same reason: every player
        # inherits this one process's environment, so one shared value would
        # give all of them the identical sequence of spawns -- the same arena
        # replicated --instances times instead of the diversity the parallelism
        # was bought for.
        unity_env.set_player_env(formations = formations, heartbeat_ticks = cfg.heartbeat_ticks,
                                 seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                                 min_bots = args.min_bots, max_bots = args.max_bots,
                                 swarm_rng = None if args.swarm_rng is None
                                              else args.swarm_rng + worker_id)
        worker, env = unity_env.make_unity_worker(
            worker_id = worker_id, num_arenas = args.arenas, build_path = args.build,
            no_graphics = True, base_port = args.base_port, time_scale = args.time_scale)
        envs.append(env)
        return worker

    workers = [spawn(i, args.formations) for i in range(args.instances)]
    val_worker = spawn(args.instances, val_dir)

    closed = []

    def close_envs():
        # idempotent, and registered with atexit as well as called explicitly:
        # a Unity player is a separate process that keeps running (and keeps
        # holding its port) if this one exits without closing it, including on
        # an exception or a Ctrl+C partway through training
        if closed:
            return
        closed.append(True)
        for env in envs:
            try:
                env.close()
            except Exception as exc:   # a player that already died must not mask the real error
                print("warning: failed to close a Unity env cleanly (%s)" % exc, flush = True)

    atexit.register(close_envs)
    return workers, val_worker, close_envs


def build_val_cfg(args, cfg, val_formation_pool):
    """The held-out config: the actor drives itself, nothing overrides it."""
    from config import Config
    val_cfg = Config()
    val_cfg.actor_type = cfg.actor_type
    val_cfg.seed_layout = cfg.seed_layout
    val_cfg.heartbeat_ticks = cfg.heartbeat_ticks
    val_cfg.rollout_steps = args.val_rollout
    val_cfg.motor_override = "none"
    val_cfg.oracle_known_start_heading = True
    val_cfg.num_arenas = args.arenas
    val_cfg.device = args.device
    val_cfg.max_episode_steps = args.val_max_episode_steps or args.max_episode_steps
    val_cfg.success_threshold = args.success_threshold
    val_cfg.split_prop_time_scale = cfg.split_prop_time_scale
    # real bug caught directly, same session: use_arrived_head and
    # use_turn_anchor both affect what gather_split_state produces on
    # every call, keyed off whichever cfg instance gets passed in at that
    # specific call site -- val_cfg is a fresh Config() with both
    # defaulting to False, so validation (motor_override="none", meaning
    # the actor's own real output is genuinely used, not overridden) was
    # feeding a 40-wide prop into an actor built expecting 42 whenever
    # either flag was set on the main cfg but not mirrored here. Confirmed
    # directly: a real run with both flags set crashed on the very first
    # validation pass with a mat1/mat2 shape mismatch (2x40 vs 42x40).
    # Latent since use_arrived_head was first built, not new to
    # use_turn_anchor -- never caught earlier because verification for
    # both features used standalone test scripts that never exercised
    # this specific, real code path.
    val_cfg.use_arrived_head = cfg.use_arrived_head
    val_cfg.use_turn_anchor = cfg.use_turn_anchor
    val_cfg.actor_recurrent = cfg.actor_recurrent
    val_cfg._oracle_formation_pool = val_formation_pool
    return val_cfg


def main():
    args = build_parser().parse_args()

    from config import Config
    from policy import GaussianPolicy
    from formations import build_formation_pool
    from trainer import Trainer
    from kilobot_gnn import build_actor
    from bc import bc_train
    from checkpoint import export_actor, load_for_eval
    from encoder import load_encoder
    from images import build_image_pool
    from kilobot_gnn import Z, SEED_SIZE, WALL_SIZE
    from metrics import rollout_stats
    import simple_oracle as SO
    import actor_io

    os.makedirs(args.out_dir, exist_ok = True)
    latest_path = os.path.join(args.out_dir, "actor_latest.pt")
    best_path = os.path.join(args.out_dir, "actor_best.pt")
    history_path = os.path.join(args.out_dir, "history.jsonl")
    png_path = os.path.join(args.out_dir, "progress.png")

    val_dir = os.path.join(args.out_dir, "val_formations")
    val_names = ensure_val_dir(args.formations, val_dir, args.val_count, seed = 12345)

    torch.manual_seed(args.seed)
    cfg = build_train_cfg(args)

    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    # exclude=val_names guarantees the training
    # pool can never overlap the held-out val split -- before this, the
    # two were only independently, separately random, which is not the
    # same thing (with --limit 5000 against --val-count 2000 out of
    # roughly 172500 total, the expected overlap by chance alone was
    # real, not negligible: confirmed as ~58 formations, not asserted).
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit, device = args.device,
                                  exclude = val_names)
    formation_pool = build_formation_pool(args.formations, limit = args.limit, exclude = val_names)
    assert len(image_pool) == len(formation_pool), \
        "encoder pool and formation pool length mismatch -- should never happen"
    print("loaded %d training formations" % len(formation_pool), flush = True)
    cfg._oracle_formation_pool = formation_pool

    val_image_pool = build_image_pool(val_dir, preprocess, device = args.device)
    val_formation_pool = build_formation_pool(val_dir)
    print("loaded %d held-out validation formations" % len(val_formation_pool), flush = True)
    val_cfg = build_val_cfg(args, cfg, val_formation_pool)

    # val_final_cov is coverage at the moment an episode ENDS. The val worker
    # only advances --val-rollout ticks per evaluation, and evaluations happen
    # every --eval-interval iterations, so unless that budget covers a whole
    # episode nothing ever ends and val_final_cov stays None for the entire
    # run -- which is exactly what a real 300-iteration run did (0/300
    # non-null), silently, while also pinning best_val_final_cov at 0.0 so
    # actor_best.pt was never updated past the very first iteration. Warn
    # loudly rather than let that recur; val_cov (mean coverage over the
    # rollout) is the meaningful criterion under the default budget.
    _val_ticks = (args.iterations // max(args.eval_interval, 1)) * args.val_rollout
    if _val_ticks < val_cfg.max_episode_steps:
        print("WARNING: the validation worker will accumulate about %d ticks over this run "
              "(%d iterations / --eval-interval %d x --val-rollout %d), but its episodes are "
              "%d ticks long -- so no val episode can complete and val_final_cov will stay "
              "n/a throughout. Checkpoint selection will fall back to val_cov, which is a "
              "real held-out signal. To get val_final_cov instead, lower "
              "--val-max-episode-steps (currently %d) below that tick budget, or raise "
              "--val-rollout." % (_val_ticks, args.iterations, args.eval_interval,
                                  args.val_rollout, val_cfg.max_episode_steps,
                                  val_cfg.max_episode_steps), flush = True)

    workers, val_worker, close_envs = build_workers(args, cfg, val_cfg,
                                                    formation_pool, val_formation_pool, val_dir)
    print("%d instance(s) x %d arena(s) = %d parallel arenas per iteration" %
          (args.instances, args.arenas, args.instances * args.arenas), flush = True)

    train_tr = Trainer.from_workers(workers, cfg, encoder, image_pool)
    val_tr = Trainer.from_workers([val_worker], val_cfg, encoder, val_image_pool)
    # bc_train (diagnostics.py) calls .setup() on the trainer it's given --
    # but that's train_tr, passed in explicitly below. val_tr is handled
    # entirely separately, right here, and was never given the same call:
    # its worker's own .arenas stayed permanently empty, so every val_tr.
    # collect() call ran on zero robots, making val_cov structurally,
    # trivially 0.0 forever regardless of the actor's own real quality --
    # confirmed directly (not guessed at) by instrumenting a live run and
    # reading val_worker.arenas back as [] right after collect() returned.
    # launch.py's own run_eval already calls
    # .setup() correctly on their own, separate Trainer instances, which
    # is why loading the same checkpoints through those tools works fine.
    val_tr.setup()

    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    # MUST happen before the optimizer is constructed below, or Adam holds this
    # policy's pre-move, now-stale cpu parameter tensors rather than the ones
    # actually being trained.
    #
    # Nothing in this codebase moved the model at all for a long time, so
    # --device only ever moved the DATA. That surfaces only on a real GPU, as
    # "mat1 on cuda:0, other tensors on cpu" -- which reads backwards at first
    # glance, since torch's addmm names its INPUT operand mat1, so the message
    # means the data was right and the model was not.
    policy = policy.to(cfg.device)
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)

    # val_tape.py has the full rationale. Recorded once against the held-out
    # formations and cached to disk, so the cost is paid a single time (and
    # not at all on a rerun), and every eval afterwards is pure network
    # compute with no simulation. Built AFTER the policy exists only because
    # trainer.collect() wants one; the tape is oracle-driven, so nothing about
    # this actor's own weights ends up in it.
    tape = None
    if args.val_tape_ticks > 0:
        from val_tape import build_tape, save_tape, load_tape, replay_tape, tape_state_counts
        from bc_replay import BC_STATES
        tape_path = args.val_tape_path or os.path.join(args.out_dir, "val_tape.pt")
        tape = load_tape(tape_path)
        if tape is None:
            print("recording validation tape (%d ticks, held-out formations) -- one time, cached to %s"
                  % (args.val_tape_ticks, tape_path), flush = True)
            tape = build_tape(val_tr, val_cfg, policy, args.val_tape_ticks,
                              max_robots = args.val_tape_max_robots)
            save_tape(tape, tape_path)
        else:
            print("loaded cached validation tape from %s" % tape_path, flush = True)
        print("  tape: %d robot sequences x %d steps, states %s" %
              (tape["valid"].shape[1], tape["valid"].shape[0], tape_state_counts(tape)), flush = True)
        if args.bc_motor_skip_arrived:
            print("  tape: arrived EXCLUDED from the selection criterion (marked * below), matching "
                  "--bc-motor-skip-arrived; the arrived head is scored separately as P/R/F1", flush = True)

    history = load_history(history_path)
    tape_history = []
    best_val_tape = [None]
    val_tape_raw = [None]
    val_tape_smoothed = [None]
    val_tape_states = [None]
    cold_start_moved = [None]
    arrived_head_f1 = [None]
    val_tape_eligible = [None]
    # Restore the best-so-far from history, exactly as the coverage criterion
    # already does above. Without this a resumed run would overwrite
    # actor_best.pt at its very first eval regardless of quality, discarding
    # whatever the pre-restart run had actually found.
    #
    # The running best is read from the recorded best_val_tape field rather
    # than recomputed as the minimum of every val_tape_smoothed ever logged.
    # Those are not the same thing and the difference is a real bug that was
    # caught on a live resume: an eval VETOED by the cold-start probe still
    # writes its score to history but never writes a checkpoint, so taking the
    # minimum set the bar to 0.02033 while actor_best.pt on disk was actually
    # the 0.02613 checkpoint -- every future checkpoint scoring between those
    # two would then be silently discarded despite being better than the saved
    # one. Only a score that actually produced a saved checkpoint may set the
    # bar.
    #
    # Older histories predate the field; those fall back to the minimum over
    # rows that would have passed the probe veto, reconstructed from the
    # recorded cold_start_moved against the current threshold.
    recorded_best = None
    for row in history:
        rb = row.get("best_val_tape")
        if rb is not None:
            recorded_best = rb
    if recorded_best is not None:
        best_val_tape[0] = recorded_best
    else:
        for row in history:
            prev = row.get("val_tape_smoothed")
            if prev is None:
                continue
            moved = row.get("cold_start_moved")
            if moved is not None and moved < args.val_probe_min_moved:
                continue
            if best_val_tape[0] is None or prev < best_val_tape[0]:
                best_val_tape[0] = prev
    for row in history:
        prev_raw = row.get("val_tape")
        if prev_raw is not None:
            tape_history.append(prev_raw)

    done = len(history)
    best_val_final_cov = -1.0
    for row in history:
        # mean_final_coverage is the actual criterion (see on_iteration's
        # own comment below for why) -- val_cov is the fallback for
        # history written before either field existed
        criterion = row.get("val_final_cov")
        if criterion is None:
            criterion = row.get("val_cov")
        if criterion is not None and criterion > best_val_final_cov:
            best_val_final_cov = criterion
    if done > 0 and os.path.exists(latest_path):
        load_for_eval(latest_path, policy, args.device)
        if best_val_tape[0] is not None:
            print("resuming from iteration %d, best val_tape so far %.5f" % (done, best_val_tape[0]), flush = True)
        else:
            print("resuming from iteration %d, best_val_final_cov so far %.4f" % (done, best_val_final_cov), flush = True)

    remaining = args.iterations - done
    if remaining <= 0:
        print("already completed %d/%d iterations" % (done, args.iterations), flush = True)
        # Explicitly, not via the atexit registration: closing a Unity env for
        # the first time during interpreter shutdown can block indefinitely
        # inside ml-agents' communicator teardown, so this early return used to
        # hang forever instead of exiting.
        close_envs()
        return

    history_file = open(history_path, "a")
    last_plot_time = [0.0]

    # Percentage and raw count of robots in each of
    # simple_oracle.py's own five states (go_north/turning/wall_following/
    # navigating/arrived), for the oracle-driven training collection here
    # -- this state machine is simple_oracle.py's own internal concept and
    # only exists while the oracle itself is actually driving decisions
    # (motor_override="simple_oracle"). An actor-driven equivalent DOES
    # now exist too (a direct, later request) -- see "shadow observer"
    # below, right after this wrapper is installed, for that mechanism and
    # why it needed a different approach than just reusing this one.
    # Wraps simple_oracle_motors itself (installed
    # once, for the single bc_train call below, which runs every
    # remaining iteration internally in one go) rather than adding a new
    # accumulator to trainer.py -- this data is oracle-specific, not a
    # general rollout concept the shared Trainer class has any business
    # knowing about. Census taken after every call (not once per tick):
    # simple_oracle_motors already only runs on ticks where at least one
    # robot is deciding, and each call censuses every robot's own current
    # state, not just the ones deciding that specific call, so this
    # samples the whole population's state distribution every time
    # anything happens, not on a fixed tick schedule -- a reasonable,
    # representative time-average given how frequently decisions occur
    # relative to the staggering (heartbeat_ticks) that spaces them per
    # robot. Read out and reset to zero inside on_iteration below, once
    # per completed iteration, so each iteration's own history/plot entry
    # reflects only that iteration's own census, not a running total
    # across the whole run.
    state_names = ["go_north", "turning", "wall_following", "navigating", "arrived"]
    state_ticks = {s: 0 for s in state_names}
    state_total = [0]
    orig_oracle_motors = SO.simple_oracle_motors
    def census_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, formation_pool, belief_attr = "belief"):
        result = orig_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng, formation_pool, belief_attr = belief_attr)
        for states_in_arena in worker.simple_state.values():
            for s in states_in_arena.values():
                if s in state_ticks:
                    state_ticks[s] = state_ticks[s] + 1
                    state_total[0] = state_total[0] + 1
        return result
    SO.simple_oracle_motors = census_oracle_motors

    # An actor graph of robot state alongside the oracle
    # one, to compare directly. The state machine above is genuinely
    # simple_oracle.py's own concept and has no actor equivalent -- BUT
    # its state-TRACKING is purely a function of a robot's true, physical
    # sensor readings (walls) plus its own prior state, not of who is
    # actually issuing the motor command that moves it. So this runs the
    # identical simple_oracle_motors as a "shadow" observer during the
    # actor-driven phase too: the actor's own output still drives the
    # robot untouched (this wrapper's own call happens AFTER the real
    # act() already ran and returned), simple_oracle_motors' own returned
    # motor tensor is discarded entirely, and only worker.simple_state
    # (which it updates as a side effect, independent of whose motor
    # command is actually used) is read. Wraps actor_io.act, not
    # split_obs, specifically because act() is the one call in this chain
    # that actually receives worker directly -- split_obs only sees the
    # raw observation tensor, with no way to know which of possibly
    # several instances' own workers a given call belongs to. Uses a
    # separate belief_attr ("actor_shadow_belief") so this shadow call's
    # own particle filter can never collide with the real oracle call's
    # "simple_belief" (same reasoning actor_io.py's own comment gives for
    # why bc_capture mode already needs two): the two are tracking
    # different things even when both are active in the same run. Calls
    # the ORIGINAL simple_oracle_motors (orig_oracle_motors, captured
    # before the census wrapper above replaced it), not SO.
    # simple_oracle_motors itself -- calling the wrapped version here
    # would double-count this call into the oracle-phase census too,
    # silently contaminating it with actor-phase data.
    actor_state_ticks = {s: 0 for s in state_names}
    actor_state_total = [0]
    # Does the actor's own, live arrived_head decision match,
    # tick by tick, the SAME belief-based criterion the oracle itself uses
    # (the shadow belief just above), for this same robot, this same tick --
    # not "does the trajectory eventually reach the same state"
    # (actor_state_pct already answers that), but whether the moment-to-
    # moment STOP decision itself agrees. "Neither" (both still in progress)
    # kept as its own bucket rather than folded into "agree", since it
    # dominates trivially for most of any episode and would otherwise mask
    # the genuinely interesting cases.
    arrived_agree = {"both": 0, "actor_only": 0, "shadow_only": 0, "neither": 0}
    orig_act = actor_io.act
    def shadow_act(buffer, policy, worker, decision_steps, cfg, rng, deterministic = False, bc_capture = False,
                   probe = False, probe_log = None, audit = False, audit_log = None,
                   pos_track = False, pos_log = None):
        # The shadow runs BEFORE act(), and the order is load-bearing.
        # simple_oracle_motors derives this tick's motion from
        # `step_count - last_dec_step[a][l]`, and act() sets last_dec_step to
        # step_count for every robot it commands -- so called afterwards the
        # shadow sees steps_since == 0 for every robot, dead-reckons zero
        # motion, never advances its particle filter and never accumulates any
        # rotation. Its turn can then never complete, which pins every robot in
        # `turning` forever and makes `shadow_says_arrived` permanently false:
        # actor_state_pct reads as go_north/turning only, and every arrived call
        # the actor makes is counted as actor_only by construction. Measured
        # directly: over an identical 1500-tick rollout the shadow reported
        # 22292 go_north / 48922 turning / 0 / 0 with the old ordering and
        # 19909 / 3097 / 39649 wall_following / 4443 navigating with this one.
        # actor_io.act's own bc_capture path calls the oracle before the same
        # update, for the same reason.
        if cfg.motor_override == "none" and len(decision_steps) > 0:
            vector, _ = actor_io.split_obs(decision_steps.obs, cfg.device)
            arena_ids = vector[:, 0].long()
            locals_ = vector[:, 1].long()
            walls = vector[:, 2 + SEED_SIZE:2 + SEED_SIZE + WALL_SIZE]
            orig_oracle_motors(worker, arena_ids, locals_, walls, None, cfg, rng,
                              getattr(cfg, "_oracle_formation_pool", None), belief_attr = "actor_shadow_belief")
        result = orig_act(buffer, policy, worker, decision_steps, cfg, rng, deterministic = deterministic,
                          bc_capture = bc_capture, probe = probe, probe_log = probe_log, audit = audit,
                          audit_log = audit_log, pos_track = pos_track, pos_log = pos_log)
        if cfg.motor_override == "none" and len(decision_steps) > 0:
            for states_in_arena in worker.simple_state.values():
                for s in states_in_arena.values():
                    if s in actor_state_ticks:
                        actor_state_ticks[s] = actor_state_ticks[s] + 1
                        actor_state_total[0] = actor_state_total[0] + 1
            switched_off = getattr(worker, "arrived_switched_off", {})
            for a, states_in_arena in worker.simple_state.items():
                off_in_arena = switched_off.get(a, {})
                for l, shadow_state in states_in_arena.items():
                    actor_says_arrived = off_in_arena.get(l, False)
                    shadow_says_arrived = (shadow_state == "arrived")
                    if actor_says_arrived and shadow_says_arrived:
                        arrived_agree["both"] = arrived_agree["both"] + 1
                    elif actor_says_arrived:
                        arrived_agree["actor_only"] = arrived_agree["actor_only"] + 1
                    elif shadow_says_arrived:
                        arrived_agree["shadow_only"] = arrived_agree["shadow_only"] + 1
                    else:
                        arrived_agree["neither"] = arrived_agree["neither"] + 1
        return result
    actor_io.act = shadow_act

    def on_iteration(local_it, stats):
        nonlocal best_val_final_cov
        global_it = done + local_it
        # Cleared every iteration, so a row written on a NON-eval iteration
        # records null rather than silently repeating the previous eval's
        # numbers. Without this, history.jsonl carried eval_interval copies of
        # each eval, and the resume path -- which rebuilds tape_history row by
        # row -- ended up smoothing over repeated copies of one eval instead of
        # over distinct evals, quietly disabling --val-smooth after any restart
        # that did not land exactly on an eval iteration. It also made
        # cold_start_moved and arrived_head_f1 wrong on most rows for anyone
        # reading the history back.
        val_tape_raw[0] = None
        val_tape_smoothed[0] = None
        val_tape_states[0] = None
        cold_start_moved[0] = None
        arrived_head_f1[0] = None
        val_tape_eligible[0] = None
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        # direct follow-up to a real SIGKILL (the OOM killer's own
        # signature, silent by nature -- see max_episode_steps' own
        # comment above for the full incident) -- printed every iteration,
        # not gated behind eval_interval, specifically so a genuinely
        # growing trend is visible as early as possible rather than only
        # discovered after the fact from whatever last made it into
        # training.log before the process was killed.
        print("iter %d  memory (RSS) %.0f MB" % (global_it, rss_mb), flush = True)
        aa = arrived_agree
        aa_denom = aa["both"] + aa["actor_only"] + aa["shadow_only"]
        aa_pct = (100.0 * aa["both"] / aa_denom) if aa_denom > 0 else float("nan")
        print("iter %d  arrived_agreement both=%d actor_only=%d shadow_only=%d neither=%d  "
              "(agreement among any-arrived cases: %.1f%%)" %
              (global_it, aa["both"], aa["actor_only"], aa["shadow_only"], aa["neither"], aa_pct), flush = True)
        # Enough real GPU runs have happened by this point
        # (phases 125-130) that "should I turn arenas up" is a real,
        # current question -- and RSS alone cannot answer it, since it
        # only ever reflected system RAM, never VRAM at all, which is the
        # actual thing that question depends on. torch.cuda.memory_
        # reserved (not memory_allocated) is what's tracked here: the
        # full pool torch's own caching allocator has actually claimed
        # from the driver, not just what's live in tensors at this exact
        # instant -- reserved is the number that predicts whether the
        # NEXT allocation might fail, which is the number this question
        # actually needs. cfg.device is only ever "cuda" or "cpu" in this
        # file's own --device flag, never a specific "cuda:1"-style index,
        # so checking the string directly is sufficient and correctly
        # skips this on the cpu path, where these calls would be
        # meaningless (or on some torch/driver combinations, error).
        vram_mb = None
        if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_reserved(cfg.device) / (1024.0 * 1024.0)
            print("iter %d  memory (VRAM reserved) %.0f MB" % (global_it, vram_mb), flush = True)
        val_cov = None
        val_success_rate = None
        val_final_cov = None
        is_last = (local_it == remaining - 1)
        if (global_it + 1) % args.eval_interval == 0 or is_last:
            val_cfg.motor_override = "none"
            val_tr._bc_capture = False
            with torch.no_grad():
                val_tr.collect(policy, None, deterministic = True)
            val_pay = val_tr.rollout_payload()
            val_stats = rollout_stats(val_pay)
            val_cov = val_stats.get("rollout/mean_coverage", 0.0)
            # mean_coverage is a per-tick average across the
            # whole rollout, diluted by however long robots spend still
            # travelling -- success_rate (fraction of episodes that actually
            # reached success_threshold before timing out) is the direct,
            # undiluted signal instead, and what actor_best.pt is now kept
            # by -- EXCEPT: success_threshold defaults to 1.1 (disabled,
            # above) per "don't be hasty", meaning success can structurally
            # never fire and success_rate would always read exactly 0,
            # never usable as a selection criterion under this project's
            # own current default. mean_final_coverage (coverage at the
            # moment an episode actually ends, success or timeout, not
            # diluted by travel time the way mean_coverage is) stays
            # meaningful either way, so that is the actual criterion below;
            # success_rate is still computed and logged, since it becomes
            # meaningful again if success_threshold is ever set below 1.0.
            # Falls back to val_cov if no episode completed within this
            # eval's own rollout at all (possible early in training, when
            # few or no episodes finish) rather than crash on a missing key.
            val_success_rate = val_stats.get("rollout/success_rate")
            val_final_cov = val_stats.get("rollout/mean_final_coverage")
            if tape is not None:
                # Selection criterion during BC is held-out imitation error,
                # not task coverage. Coverage over a 192-tick window cannot
                # separate two checkpoints when the swarm needs roughly 15000
                # ticks to converge, and the old fallback (val_cov) was
                # confounded with how far the val worker's own single episode
                # had progressed, so it rose with tick count largely
                # regardless of actor quality.
                tape_scores = replay_tape(policy, tape, device = cfg.device,
                                          skip_arrived = cfg.bc_motor_skip_arrived,
                                          arrived_threshold = cfg.arrived_confidence_threshold)
                val_tape_balanced = tape_scores.get("balanced")
                probe = {}
                if args.val_probe_ticks > 0:
                    from val_probe import cold_start_probe
                    probe = cold_start_probe(val_tr, val_cfg, policy, args.val_probe_ticks)
                tape_history.append(val_tape_balanced)
                val_tape_raw[0] = val_tape_balanced
                val_tape_states[0] = {k: tape_scores[k] for k in tape_scores if k not in ("all", "balanced")}
                cold_start_moved[0] = probe.get("moved_fraction")
                arrived_head_f1[0] = tape_scores.get("arrived_f1")
                window = [x for x in tape_history[-max(1, args.val_smooth):] if x is not None]
                smoothed = sum(window) / len(window) if window else None
                val_tape_smoothed[0] = smoothed
                moved = probe.get("moved_fraction")
                # A frozen actor can post an excellent imitation error while
                # being useless -- docs/tuning.md has exactly that failure. The
                # on-policy probe is meant to be a veto, not a tiebreak -- but
                # it is inert against Unity (see val_probe.py), so `moved` is
                # None and this currently reduces to the smoothed tape score.
                eligible = (smoothed is not None
                            and (moved is None or moved >= args.val_probe_min_moved))
                val_tape_eligible[0] = bool(eligible)
                scored = tape_scores.get("scored_states", [])
                per_state = "  ".join(
                    "%s%s %.4f" % (n[:4], "" if n in scored else "*", tape_scores[n])
                    for n in BC_STATES if n in tape_scores)
                ah = ""
                if "arrived_f1" in tape_scores:
                    ah = ("  arrived_head P %.3f R %.3f F1 %.3f"
                          % (tape_scores["arrived_precision"], tape_scores["arrived_recall"],
                             tape_scores["arrived_f1"]))
                print("iter %d  val_tape %s (smoothed %s)  %s%s  cold_start_moved %s" %
                     (global_it,
                      ("%.5f" % val_tape_balanced) if val_tape_balanced is not None else "n/a",
                      ("%.5f" % smoothed) if smoothed is not None else "n/a",
                      per_state, ah,
                      ("%.3f" % moved) if moved is not None else "n/a"), flush = True)
                if eligible and (best_val_tape[0] is None or smoothed < best_val_tape[0]):
                    best_val_tape[0] = smoothed
                    export_actor(best_path, policy, iteration = global_it)
                    print("iter %d  NEW BEST val_tape %.5f -> %s" % (global_it, smoothed, best_path), flush = True)
                elif not eligible and moved is not None:
                    print("iter %d  not eligible for best: cold-start moved_fraction %.3f < %.3f"
                          % (global_it, moved, args.val_probe_min_moved), flush = True)
            else:
                criterion = val_final_cov if val_final_cov is not None else val_cov
                if criterion > best_val_final_cov:
                    best_val_final_cov = criterion
                    export_actor(best_path, policy, iteration = global_it)
                    print("iter %d  NEW BEST val_final_cov %s val_success_rate %s -> %s" %
                         (global_it, ("%.4f" % val_final_cov) if val_final_cov is not None else "n/a",
                          ("%.4f" % val_success_rate) if val_success_rate is not None else "n/a", best_path), flush = True)
                else:
                    print("iter %d  val_final_cov %s val_success_rate %s  (best %.4f)" %
                         (global_it, ("%.4f" % val_final_cov) if val_final_cov is not None else "n/a",
                          ("%.4f" % val_success_rate) if val_success_rate is not None else "n/a", best_val_final_cov), flush = True)

        total = state_total[0]
        state_pct = {s: (state_ticks[s] / total if total > 0 else None) for s in state_names}
        actor_total = actor_state_total[0]
        actor_state_pct = {s: (actor_state_ticks[s] / actor_total if actor_total > 0 else None) for s in state_names}
        row = {"iteration": global_it, "train_loss": stats["train_loss"],
              "train_loss_mean": stats["train_loss_mean"], "grad_norm": stats["grad_norm"],
              "train_eval_cov": stats["train_eval_cov"], "oracle_cov": stats["oracle_cov"],
              "oracle_success_rate": stats.get("oracle_success_rate"),
              "oracle_mean_final_coverage": stats.get("oracle_mean_final_coverage"),
              "train_eval_success_rate": stats.get("train_eval_success_rate"),
              "train_eval_mean_final_coverage": stats.get("train_eval_mean_final_coverage"),
              "val_cov": val_cov, "val_success_rate": val_success_rate, "val_final_cov": val_final_cov,
              "best_val_final_cov": best_val_final_cov,
              "val_tape": val_tape_raw[0], "val_tape_smoothed": val_tape_smoothed[0],
              "val_tape_states": val_tape_states[0], "cold_start_moved": cold_start_moved[0],
              "arrived_head_f1": arrived_head_f1[0],
              "best_val_tape": best_val_tape[0], "val_tape_eligible": val_tape_eligible[0],
              "state_raw": dict(state_ticks), "state_pct": state_pct, "state_total": total,
              "actor_state_raw": dict(actor_state_ticks), "actor_state_pct": actor_state_pct,
              "actor_state_total": actor_total,
              "arrived_agreement": dict(arrived_agree),
              "rss_mb": rss_mb, "vram_mb": vram_mb, "time": time.time()}
        history_file.write(json.dumps(row) + "\n")
        history_file.flush()
        for s in state_names:
            state_ticks[s] = 0
            actor_state_ticks[s] = 0
        state_total[0] = 0
        actor_state_total[0] = 0
        for k in arrived_agree:
            arrived_agree[k] = 0


        now = time.time()
        if now - last_plot_time[0] >= args.plot_interval or is_last:
            last_plot_time[0] = now
            try:
                draw_progress(load_history(history_path), args.iterations, png_path)
            except Exception as e:
                print("progress plot failed this round: %s" % e, flush = True)

    print("training iterations %d..%d (of %d total), writing progress to %s" %
         (done, args.iterations - 1, args.iterations, png_path), flush = True)
    bc_train(train_tr, policy, actor_opt, cfg, remaining, None, args.bc_epochs, latest_path,
            teacher = "simple_oracle", on_iteration = on_iteration,
            debug_per_arena_threshold = args.debug_per_arena_threshold,
            debug_iteration_detail = args.debug_iteration_detail)
    SO.simple_oracle_motors = orig_oracle_motors
    actor_io.act = orig_act
    history_file.close()
    close_envs()
    if tape is not None:
        print("\ndone. best val_tape (held-out imitation error, lower is better)=%s"
              % (("%.5f" % best_val_tape[0]) if best_val_tape[0] is not None else "n/a"), flush = True)
    else:
        print("\ndone. best_val_final_cov=%.4f" % best_val_final_cov, flush = True)
    print("best checkpoint: %s" % best_path, flush = True)
    print("to continue into RL fine-tuning:", flush = True)
    print("  KILOBOT_INIT_ACTOR=%s KILOBOT_MODE=rl python launch.py" % best_path, flush = True)


if __name__ == "__main__":
    main()
