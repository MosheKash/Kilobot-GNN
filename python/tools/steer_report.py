"""steer_report.py -- does the clone reproduce the oracle's STEERING, not just its motors.

The headline imitation numbers this project reported for a long time -- "motor MSE
0.0027 in wall_following", "88.5% of decisions within 0.05 on both wheels" -- are
dominated by a channel that carries no information. A differential-drive command
is two numbers, and they are better read as two orthogonal modes:

    speed  = (L + R) / 2         what the oracle holds nearly constant
    turn   = (R - L) * 1.8 / (0.7 * (L + R))     what decides where the robot goes

`turn` is exactly simple_oracle's own steering variable, and the speed scale
cancels out of it. During wall_following the oracle's own turn has a spread of
0.0093 -- it is a stabilising controller, so it holds itself straight -- which is
about 0.1% of the variance in the wheel pair. An MSE on the pair therefore spends
essentially all of itself on the common mode, and a clone can score 0.0027 while
its steering error is ten times the entire signal it is meant to reproduce.

This tool measures that directly, for any number of checkpoints, and draws it.

usage:
  python tools/steer_report.py --out-dir ../results/bc_v2/report_steer \\
      --tape ../results/bc_v2/tape_val.pt \\
      --actor "round 10 (previous best)=../results/bc_v2/run_r10/actor_best.pt" \\
      --actor "oracle-form head=../results/bc_v2/run_o1/actor_best.pt"
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The palette the rest of this project's figures already use (bc_report.py,
# settle_report.py), kept identical so the pages read as one system.
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e2df"
SERIES = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]


def style(ax, title = None, xlabel = None, ylabel = None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color = GRID, linewidth = 0.8, alpha = 0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors = INK2, labelsize = 9, length = 0)
    if title:
        ax.set_title(title, color = INK, fontsize = 11, loc = "left", pad = 10)
    if xlabel:
        ax.set_xlabel(xlabel, color = INK2, fontsize = 9)
    if ylabel:
        ax.set_ylabel(ylabel, color = INK2, fontsize = 9)


def load_actor(path, device):
    from config import Config
    from kilobot_gnn import build_actor
    ck = torch.load(path, map_location = device, weights_only = False)
    m = ck.get("meta", {})
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.use_arrived_head = bool(m.get("use_arrived_head", True))
    cfg.use_turn_anchor = bool(m.get("use_turn_anchor", True))
    cfg.use_state_head = bool(m.get("use_state_head", False))
    cfg.use_wall_head = bool(m.get("use_wall_head", False))
    cfg.use_steer_feature = bool(m.get("use_steer_feature", False))
    cfg.use_oracle_head = bool(m.get("use_oracle_head", False))
    cfg.oracle_residual = float(m.get("oracle_residual", 0.05))
    cfg.oracle_residual_turn = float(m.get("oracle_residual_turn", 0.0))
    cfg.split_activation = m.get("activation", "relu")
    cfg.device = device
    from kilobot_gnn import widths_from_state_dict
    for k, v in widths_from_state_dict(ck["actor"]).items():
        setattr(cfg, "split_" + k, v)
    actor = build_actor(cfg).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    n = sum(p.numel() for p in actor.parameters())
    return actor, m, n


@torch.no_grad()
def roll(actor, tape, device, chunk = 256, batch = 256, keep = 400000, seed = 0):
    """Every decision's (oracle turn, actor turn, state, sequence id), on rows
    whose command actually encodes a steering angle."""
    from bc_offline import forward_chunk, turn_from_motors, steer_rows
    T, R = tape["valid"].shape
    parts = []
    per_state_sq = {}
    for b0 in range(0, R, batch):
        b1 = min(b0 + batch, R)
        h = actor.initial_hidden(b1 - b0, device = device)
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            v = tape["valid"][t0:t1, b0:b1].to(device)
            if not bool(v.any()):
                continue
            tc = tape["tc"][t0:t1, b0:b1].to(device).float()
            prop = tape["prop"][t0:t1, b0:b1].to(device).float()
            tgt = tape["tgt"][t0:t1, b0:b1].to(device).float()
            mo, _, _, h, _, _ = forward_chunk(actor, tc, prop, v, h)
            st = tape["state"][t0:t1, b0:b1].to(device)
            # full-motor squared error per state, so the old metric is on the
            # same page as the new one rather than quoted from elsewhere
            sq = ((mo - tgt) ** 2).mean(dim = 2)
            for s in range(5):
                m = v & (st == s)
                if bool(m.any()):
                    a, c = per_state_sq.get(s, (0.0, 0))
                    per_state_sq[s] = (a + float(sq[m].sum()), c + int(m.sum()))
            u = v & steer_rows(tgt)
            if not bool(u.any()):
                continue
            seq = torch.arange(b0, b1, device = device).expand(t1 - t0, -1)
            parts.append(torch.stack([turn_from_motors(tgt[..., 0], tgt[..., 1])[u],
                                      turn_from_motors(mo[..., 0], mo[..., 1])[u],
                                      st[u].float(), seq[u].float()], dim = 1).cpu())
    d = torch.cat(parts) if parts else torch.zeros(0, 4)
    if len(d) > keep:
        g = torch.Generator().manual_seed(seed)
        d = d[torch.randperm(len(d), generator = g)[:keep]]
    mse = {s: a / c for s, (a, c) in per_state_sq.items()}
    return d.numpy(), mse


def stats(d, mse, states):
    """Per-state steering statistics, plus the persistent per-robot component."""
    out = {}
    for s, name in enumerate(states):
        m = d[:, 2] == s
        n = int(m.sum())
        row = {"n": n, "motor_mse": mse.get(s)}
        if n >= 200:
            o, a = d[m, 0], d[m, 1]
            e = a - o
            var = max(float(o.var()), 1e-12)
            row["turn_sd_oracle"] = float(o.std())
            row["turn_rms_err"] = float(np.sqrt((e ** 2).mean()))
            # The rms is dominated by a rare, TRANSIENT failure -- a decision
            # where the state or wall head is momentarily wrong emits a command
            # from the wrong branch, and those are worth 1-2 in turn units. The
            # median says what a typical decision looks like, and the per-robot
            # mean below says how much of the error is the persistent kind.
            row["turn_med_err"] = float(np.median(np.abs(e)))
            row["turn_p90_err"] = float(np.quantile(np.abs(e), 0.9))
            row["turn_r2"] = 1.0 - float((e ** 2).mean()) / var
            row["turn_corr"] = float(np.corrcoef(o, a)[0, 1]) if a.std() > 0 else 0.0
            # persistent component: each robot's own mean error. A correlated
            # error of a given size is far more damaging here than an
            # independent one -- driving the ORACLE with a persistent bias of
            # 0.10 drops its stopped fraction from 0.99 to 0.40, while i.i.d.
            # noise of 0.12 leaves it at 0.98 (docs/tuning.md phase 159).
            seq = d[m, 3].astype(np.int64)
            u, inv = np.unique(seq, return_inverse = True)
            cnt = np.bincount(inv, minlength = len(u))
            mean = np.bincount(inv, weights = e, minlength = len(u)) / np.maximum(cnt, 1)
            k = cnt >= 20
            if k.sum() > 5:
                row["turn_bias_rms"] = float(np.sqrt((mean[k] ** 2).mean()))
                row["turn_resid_rms"] = float(np.sqrt(((e - mean[inv]) ** 2).mean()))
                row["_bias"] = mean[k]
        out[name] = row
    return out


# ─── figures ─────────────────────────────────────────────────────────────────

def fig_signal_vs_error(res, states, path):
    """The whole problem in one panel: the size of the signal against the size
    of the error, in the oracle's own steering variable."""
    show = [s for s in states if s in ("wall_following", "navigating")]
    fig, axes = plt.subplots(1, len(show), figsize = (5.2 * len(show), 4.2))
    axes = np.atleast_1d(axes)
    names = list(res.keys())
    for ax, st in zip(axes, show):
        sd = next((res[n][st].get("turn_sd_oracle") for n in names
                   if res[n][st].get("turn_sd_oracle")), None)
        y = [res[n][st].get("turn_rms_err", np.nan) for n in names]
        xs = np.arange(len(names))
        bars = ax.bar(xs, y, width = 0.6, color = [SERIES[i % len(SERIES)] for i in range(len(names))],
                      zorder = 3)
        if sd:
            ax.axhline(sd, color = INK, linewidth = 2, linestyle = "--", zorder = 4)
            ax.text(len(names) - 0.45, sd * 1.12,
                    "the oracle's ENTIRE steering signal (sd %.4f)" % sd,
                    color = INK, fontsize = 8.5, ha = "right", va = "bottom")
        for x, v in zip(xs, y):
            if v == v:
                ax.text(x, v * 1.05, "%.3f" % v, ha = "center", va = "bottom",
                        color = INK, fontsize = 9)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation = 20, ha = "right", fontsize = 8.5, color = INK2)
        style(ax, st.replace("_", " "), None, "rms error in the oracle's turn (log)")
    fig.suptitle("The steering channel: error against signal, on held-out oracle data",
                 color = INK, fontsize = 12.5, x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_scatter(raw, states, path, state = "wall_following", n = 9000, seed = 0):
    """Actor's turn against the oracle's, decision by decision.

    The window is set from the ORACLE's own spread, not from the actor's. That
    matters: during wall_following the teacher's turn has a standard deviation of
    0.0093, so a window wide enough to contain the clone's outliers is a window in
    which the entire signal is one pixel at the origin. The caption reports what
    share of decisions the window holds, and fig_error_cdf below shows the tail
    that it does not.
    """
    si = states.index(state)
    names = list(raw.keys())
    ref = np.concatenate([raw[n_][raw[n_][:, 2] == si][:, 0] for n_ in names])
    # 6 standard deviations of the TEACHER's own turn. Wide enough to contain the
    # signal several times over, narrow enough that the signal is not a single
    # pixel -- which is what a window sized to the clone's outliers produces.
    lim = max(6.0 * float(ref.std()), 1e-3)
    fig, axes = plt.subplots(1, len(names), figsize = (4.1 * len(names), 4.5),
                             sharex = True, sharey = True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(seed)
    for ax, name in zip(axes, names):
        d = raw[name]
        dd = d[d[:, 2] == si]
        inside = float(((np.abs(dd[:, 1]) <= lim) & (np.abs(dd[:, 0]) <= lim)).mean())
        sh = dd if len(dd) <= n else dd[rng.choice(len(dd), n, replace = False)]
        ax.plot([-lim, lim], [-lim, lim], color = INK, linewidth = 1.6, zorder = 4)
        ax.scatter(sh[:, 0], sh[:, 1], s = 7, alpha = 0.12, color = BLUE, linewidths = 0, zorder = 3)
        e = np.abs(dd[:, 1] - dd[:, 0])
        ax.text(0.03, 0.97, "median error %.5f\np90 %.4f\n%.1f%% of decisions in view"
                % (float(np.median(e)), float(np.quantile(e, 0.9)), 100 * inside),
                transform = ax.transAxes, va = "top", ha = "left", color = INK, fontsize = 9.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        style(ax, name, "the oracle's turn", "the actor's turn")
    fig.suptitle("Every %s decision, at the scale of the teacher's own steering "
                 "(sd %.4f). The black line is agreement."
                 % (state.replace("_", " "), float(ref.std())),
                 color = INK, fontsize = 12.5, x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_error_cdf(raw, states, path, state = "wall_following"):
    """How large is the steering error, over the whole distribution.

    One curve per checkpoint: the share of decisions whose steering error is below
    a given size. A log axis because the interesting range spans four decades, and
    because the honest story has two parts -- a typical decision, and a rare one
    where a discrete head is momentarily wrong and the command comes from the
    wrong branch entirely. A single rms number hides that split; this shows it.
    """
    si = states.index(state)
    names = list(raw.keys())
    ref = np.concatenate([raw[n_][raw[n_][:, 2] == si][:, 0] for n_ in names])
    sd = float(ref.std())
    fig, ax = plt.subplots(figsize = (8.6, 4.6))
    for i, name in enumerate(names):
        d = raw[name]
        e = np.abs(d[d[:, 2] == si][:, 1] - d[d[:, 2] == si][:, 0])
        e = np.sort(np.clip(e, 1e-6, None))
        y = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, y, color = SERIES[i % len(SERIES)], linewidth = 2.2, zorder = 3)
        # direct label rather than legend-only: the palette's lighter steps sit
        # under 3:1 against the surface, so identity is never colour alone.
        # Anchored at a different height per series so the labels cannot collide
        # however close together the curves run.
        frac = 0.78 - 0.22 * i
        j = min(int(frac * len(e)), len(e) - 1)
        ax.annotate(name, (e[j], y[j]), textcoords = "offset points", xytext = (9, -3),
                    color = INK, fontsize = 9.5)
    ax.axvline(sd, color = INK, linewidth = 1.8, linestyle = "--", zorder = 4)
    ax.text(sd * 1.15, 0.06, "the teacher's ENTIRE\nsteering signal (sd %.4f)" % sd,
            color = INK, fontsize = 9)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 3)
    ax.set_ylim(0, 1.02)
    style(ax, "Steering error during %s, whole distribution" % state.replace("_", " "),
          "|actor's turn - teacher's turn|  (log)", "share of decisions below")
    fig.tight_layout()
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_bias(res, path, state = "wall_following"):
    """The persistent, per-robot component -- the damaging kind of error."""
    names = [n for n in res if "_bias" in res[n].get(state, {})]
    if not names:
        return
    fig, ax = plt.subplots(figsize = (7.4, 4.2))
    lim = max(float(np.percentile(np.abs(res[n][state]["_bias"]), 99)) for n in names) * 1.15
    edges = np.linspace(-lim, lim, 61)
    for i, n in enumerate(names):
        ax.hist(np.clip(res[n][state]["_bias"], -lim, lim), bins = edges, histtype = "step",
                linewidth = 2, color = SERIES[i % len(SERIES)], label = n, zorder = 3)
    ax.axvline(0, color = INK, linewidth = 1.4, zorder = 4)
    style(ax, "Each robot's own MEAN steering error over its whole episode (%s)"
          % state.replace("_", " "),
          "mean turn error for one robot", "robots")
    leg = ax.legend(frameon = False, fontsize = 9, loc = "upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.02, 0.96, "a spike at zero is a robot that goes where the teacher sends it;\n"
                        "width here is a constant turn, which is a circle",
            transform = ax.transAxes, va = "top", ha = "left", color = INK3, fontsize = 8.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_proxy(rows, path):
    """The same runs, read by three different measures, as three stacked panels.

    Deliberately not two y-axes on one chart: the three quantities have nothing
    in common but their x-axis, and a shared scale would invent a relationship
    the data does not have. The middle panel is on a log scale because the range
    it has to cover is three decades.
    """
    if len(rows) < 3:
        return
    names = [r["name"] for r in rows]
    xs = np.arange(len(names))
    panels = [("held-out balanced motor MSE\nwhat every run before this was selected on",
               [r.get("balanced") for r in rows], BLUE, "lower is better", False),
              ("median steering error during wall following\nthe channel that decides where a robot goes",
               [r.get("steer") for r in rows], ORANGE, "lower is better", True),
              ("robots on the shape, closed loop in Unity\nthe task",
               [r.get("coverage") for r in rows], AQUA, "higher is better", False)]
    panels = [p for p in panels if any(v is not None and v == v for v in p[1])]
    fig, axes = plt.subplots(len(panels), 1, figsize = (9.0, 2.7 * len(panels)), sharex = True)
    axes = np.atleast_1d(axes)
    for ax, (title, vals, col, note, logy) in zip(axes, panels):
        v = np.array([np.nan if x is None else x for x in vals], dtype = float)
        ax.plot(xs, v, color = col, linewidth = 2, marker = "o", markersize = 7,
                markeredgecolor = "white", markeredgewidth = 1.5, zorder = 3)
        if logy:
            ax.set_yscale("log")
        for i, (x, y) in enumerate(zip(xs, v)):
            if y != y:
                continue
            txt = ("%.5f" % y) if y >= 1e-4 else ("%.6f" % y)
            ax.annotate(txt.rstrip("0").rstrip(".") if "." in txt else txt, (x, y),
                        textcoords = "offset points", xytext = (0, 11 if i % 2 == 0 else -17),
                        ha = "center", color = INK, fontsize = 8.5)
        style(ax, title + "  --  " + note)
        ax.margins(y = 0.35 if not logy else 0.6)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(names, rotation = 20, ha = "right", fontsize = 8.5, color = INK2)
    fig.suptitle("The same seven runs, read three ways", color = INK, fontsize = 12.5,
                 x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.965))
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_recoverable(tape, path, device = "cuda"):
    """Is the teacher's steering direction a function of what the actor sees?

    Two states, two answers. In wall_following the latent is the latched wall,
    which is publicly recoverable, and the observation reproduces the teacher's
    direction exactly. In navigating the teacher steers by its OWN particle
    filter -- a separate one from the filter that produces the observation --
    and the direction is unbiased but scattered over tens of degrees.
    """
    from bc_offline import wall_labels_from_targets, turn_from_motors, steer_rows
    from bc_replay import BC_STATES
    v, st = tape["valid"], tape["state"]
    tgt, prop = tape["tgt"].float(), tape["prop"].float()
    turn = turn_from_motors(tgt[..., 0], tgt[..., 1])
    ok = v & steer_rows(tgt)
    implied = torch.asin(turn.clamp(-0.999, 0.999)) * 180.0 / math.pi
    wl = wall_labels_from_targets(tape)
    tan = torch.tensor([0.0, -90.0, 180.0, 90.0], device = v.device)
    th = torch.atan2(prop[..., 10], prop[..., 11]) * 180.0 / math.pi
    obs_w = ((tan[wl.clamp(min = 0)] - th) + 180.0) % 360.0 - 180.0
    obs_n = torch.atan2(prop[..., 19], prop[..., 20]) * 180.0 / math.pi
    series = []
    m = ok & (st == BC_STATES.index("wall_following")) & (wl >= 0)
    series.append(("wall following\nlatent = the latched wall, publicly recoverable",
                   (((implied - obs_w) + 180) % 360 - 180)[m].float().cpu().numpy(), AQUA))
    m = ok & (st == BC_STATES.index("navigating"))
    series.append(("navigating\nlatent = the teacher's own particle filter, private",
                   (((implied - obs_n) + 180) % 360 - 180)[m].float().cpu().numpy(), ORANGE))
    fig, axes = plt.subplots(1, 2, figsize = (11.4, 4.3))
    edges = np.linspace(-90, 90, 121)
    for ax, (title, d, col) in zip(axes, series):
        ax.hist(np.clip(d, -90, 90), bins = edges, color = col, zorder = 3)
        ax.axvline(0, color = INK, linewidth = 1.4, zorder = 4)
        inside = float((np.abs(d) < 5).mean())
        ax.text(0.03, 0.96, "%.1f%% within 5$\\degree$\nmedian %+.2f$\\degree$, spread %.1f$\\degree$"
                % (100 * inside, float(np.median(d)), float(d.std())),
                transform = ax.transAxes, va = "top", ha = "left", color = INK, fontsize = 10)
        style(ax, title, "teacher's direction minus the actor's observable one (degrees)",
              "decisions")
    fig.suptitle("Can the teacher's steering direction be recovered from the actor's own "
                 "observation?", color = INK, fontsize = 12.5, x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_training(run_dirs, path):
    """Steering error and the two discrete heads, over training."""
    curves = []
    for name, rd in run_dirs:
        p = os.path.join(rd, "history.jsonl")
        if not os.path.exists(p):
            continue
        ep, steer, bal, sh = [], [], [], []
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "val" not in r:
                    continue
                ep.append(r["epoch"])
                steer.append(r["val"].get("turn_rms_wall_following", np.nan))
                bal.append(r["val"].get("balanced", np.nan))
                sh.append(r["val"].get("state_head_acc", np.nan))
        if ep:
            curves.append((name, np.array(ep), np.array(steer, float),
                           np.array(bal, float), np.array(sh, float)))
    if not curves:
        return
    fig, axes = plt.subplots(1, 3, figsize = (14.4, 4.2))
    for i, (name, ep, steer, bal, sh) in enumerate(curves):
        c = SERIES[i % len(SERIES)]
        axes[0].plot(ep, steer, color = c, linewidth = 2, label = name)
        axes[1].plot(ep, bal, color = c, linewidth = 2, label = name)
        if np.isfinite(sh).any():
            axes[2].plot(ep, sh, color = c, linewidth = 2, label = name)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    style(axes[0], "rms steering error, wall following", "epoch", "turn units")
    style(axes[1], "balanced motor MSE (the old headline)", "epoch", "MSE")
    style(axes[2], "state head accuracy, held out", "epoch", "share correct")
    for ax in axes:
        leg = ax.legend(frameon = False, fontsize = 8.5)
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.suptitle("Training", color = INK, fontsize = 12.5, x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.94))
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default = "../results/bc_v2/tape_val.pt")
    ap.add_argument("--actor", action = "append", default = [],
                    help = "label=path/to/actor_best.pt, repeatable; order is the figure order")
    ap.add_argument("--run-dir", action = "append", default = [],
                    help = "label=path/to/run_dir, repeatable; for the training curves")
    ap.add_argument("--proxy", default = None,
                    help = "JSON list of {name, balanced, steer, coverage} for the proxy figure")
    ap.add_argument("--out-dir", required = True)
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--scatter-state", default = "wall_following")
    args = ap.parse_args(argv)

    from bc_offline import load_tape
    from bc_replay import BC_STATES
    os.makedirs(args.out_dir, exist_ok = True)
    tape = load_tape(args.tape, args.device)

    raw, res, summary = {}, {}, {}
    for spec in args.actor:
        name, _, path = spec.partition("=")
        actor, meta, n_params = load_actor(path, args.device)
        d, mse = roll(actor, tape, args.device)
        raw[name] = d
        res[name] = stats(d, mse, BC_STATES)
        summary[name] = {"path": path, "params": n_params,
                         "oracle_head": bool(meta.get("use_oracle_head", False)),
                         "states": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                    for k, v in res[name].items()}}
        st = res[name].get("wall_following", {})
        print("%-30s params %5d  wall: mse %.5f  turn med %.5f  rms %.4f  bias %.4f  p90 %.4f"
              % (name, n_params, st.get("motor_mse", float("nan")),
                 st.get("turn_med_err", float("nan")), st.get("turn_rms_err", float("nan")),
                 st.get("turn_bias_rms", float("nan")), st.get("turn_p90_err", float("nan"))),
              flush = True)

    if res:
        fig_signal_vs_error(res, BC_STATES, os.path.join(args.out_dir, "steer_signal_vs_error.png"))
        fig_scatter(raw, BC_STATES, os.path.join(args.out_dir, "steer_scatter.png"),
                    state = args.scatter_state)
        fig_error_cdf(raw, BC_STATES, os.path.join(args.out_dir, "steer_error_cdf.png"),
                      state = args.scatter_state)
        fig_bias(res, os.path.join(args.out_dir, "steer_bias.png"))
    fig_recoverable(tape, os.path.join(args.out_dir, "steer_recoverable.png"))
    if args.run_dir:
        fig_training([tuple(s.split("=", 1)) for s in args.run_dir],
                     os.path.join(args.out_dir, "steer_training.png"))
    if args.proxy and os.path.exists(args.proxy):
        with open(args.proxy) as f:
            fig_proxy(json.load(f), os.path.join(args.out_dir, "steer_proxy.png"))
    with open(os.path.join(args.out_dir, "steer_summary.json"), "w") as f:
        json.dump(summary, f, indent = 2)
    print("wrote %s" % args.out_dir)


if __name__ == "__main__":
    main()
