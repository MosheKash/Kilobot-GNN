"""bc_report.py -- the figures and the summary for a BC run.

Two inputs, both produced elsewhere and both plain files, so a report can be
regenerated without re-running anything:

  <run-dir>/history.jsonl   bc_offline.py's per-epoch record
  eval_*.json               tools/eval_closed_loop.py, one per driver

Writes PNGs plus summary.json/summary.md into --out-dir. Charts follow one
convention throughout: the oracle is orange, the trained actor is blue, and
every multi-series figure carries both a legend and end-of-line labels so the
series are never identified by colour alone.

usage:
  python tools/bc_report.py --run-dir ../results/bc_v2/run1 \
      --eval-oracle ../results/bc_v2/eval_oracle.json \
      --eval-actor  ../results/bc_v2/eval_actor.json \
      --out-dir ../results/bc_v2/report
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The validated categorical slots (references/palette.md): blue, orange, aqua,
# yellow, magenta. Roles are fixed for the whole report -- actor blue, oracle
# orange -- so a colour means the same thing in every figure.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a",
                                               "#eda100", "#e87ba4", "#4a3aa7")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e2df"
STATE_COLORS = {"go_north": BLUE, "turning": ORANGE, "wall_following": AQUA,
                "navigating": VIOLET, "arrived": MAGENTA}


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


_LABEL_SLOTS = {}


def end_label(ax, x, y, text, color):
    """A series name at the end of its own line, so identity never rests on colour.

    Labels are nudged apart when two series end at the same value -- which they
    do whenever both drivers converge, and an unreadable overlap there would
    defeat the point of having the label at all.
    """
    if len(x) == 0:
        return
    yv = y[-1]
    if yv != yv:   # NaN
        return
    used = _LABEL_SLOTS.setdefault(id(ax), [])
    span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
    dy = 0
    while any(abs(yv - u) < 0.035 * span and abs(dy - d) < 9 for u, d in used):
        dy += 11
    used.append((yv, dy))
    ax.annotate(text, (x[-1], yv), textcoords = "offset points", xytext = (6, dy),
                color = color, fontsize = 9, va = "center", fontweight = "bold")


# Slot order matters: the palette's adjacent-pair gates were validated in this
# order (references/palette.md), and the roles are fixed across every figure --
# the oracle is always orange, the actor being judged always blue.
ACTOR_SERIES = [BLUE, AQUA, VIOLET, YELLOW, MAGENTA]


def _driver_order(evals):
    """Oracle first, then the actors in the order given: reference, then judged."""
    rest = [n for n in evals if n != "oracle"]
    return (["oracle"] if "oracle" in evals else []) + rest


def _color(name, evals):
    if name == "oracle":
        return ORANGE
    rest = [n for n in _driver_order(evals) if n != "oracle"]
    # The final actor keeps the primary blue however many earlier rounds are
    # also plotted, so "the actor" means the same colour in every figure.
    idx = rest.index(name)
    return ACTOR_SERIES[0] if idx == len(rest) - 1 else ACTOR_SERIES[1 + idx % (len(ACTOR_SERIES) - 1)]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─── training figures ────────────────────────────────────────────────────────

def fig_training(hist, out_dir):
    # One row per epoch, last write wins. Two runs pointed at the same output
    # directory append to the same history, and the curve then doubles back on
    # itself; bc_offline.py now refuses that, but histories written before it did
    # still have to plot.
    by_epoch = {}
    for r in hist:
        by_epoch[r.get("epoch")] = r
    hist = [by_epoch[k] for k in sorted(by_epoch)]
    ev = [r for r in hist if r.get("val")]
    if not ev:
        return {}
    ep = [r["epoch"] for r in ev]
    val = [r["val"]["balanced"] for r in ev]
    trf = [r.get("train_eval_balanced") for r in ev]

    fig, axes = plt.subplots(1, 2, figsize = (13, 4.6))
    ax = axes[0]
    ax.plot(ep, val, color = BLUE, linewidth = 2, label = "held-out formations")
    if any(t is not None for t in trf):
        ax.plot(ep, [t if t is not None else np.nan for t in trf], color = ORANGE,
                linewidth = 2, label = "training formations")
        end_label(ax, ep, [t if t is not None else np.nan for t in trf], "train", ORANGE)
    end_label(ax, ep, val, "held-out", BLUE)
    ax.set_yscale("log")
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2)
    style(ax, "Imitation error against the oracle (balanced over states, lower is better)",
          "epoch", "mean squared motor error")

    ax = axes[1]
    from bc_replay import BC_STATES
    for name in BC_STATES:
        ys = [r["val"].get(name) for r in ev]
        if all(y is None for y in ys):
            continue
        ys = [y if y is not None else np.nan for y in ys]
        ax.plot(ep, ys, color = STATE_COLORS.get(name, MUTED), linewidth = 2, label = name)
        end_label(ax, ep, ys, name.replace("_", " "), STATE_COLORS.get(name, MUTED))
    ax.set_yscale("log")
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2, ncol = 2)
    style(ax, "Held-out imitation error by oracle state", "epoch", "mean squared motor error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_error.png"), dpi = 150, facecolor = "white")
    plt.close(fig)

    # the arrived head, which is what actually stops a robot
    fig, axes = plt.subplots(1, 2, figsize = (13, 4.6))
    ax = axes[0]
    for key, color, label in (("arrived_precision", BLUE, "precision"),
                              ("arrived_recall", ORANGE, "recall"),
                              ("arrived_f1", AQUA, "F1")):
        ys = [r["val"].get(key) for r in ev]
        if all(y is None for y in ys):
            continue
        ys = [y if y is not None else np.nan for y in ys]
        ax.plot(ep, ys, color = color, linewidth = 2, label = label)
        end_label(ax, ep, ys, label, color)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2)
    style(ax, "Arrived head on held-out data (at the deployed 0.95 threshold)", "epoch", "")

    ax = axes[1]
    dead = [r["val"].get("dead_units") for r in ev]
    if any(d is not None for d in dead):
        ax.plot(ep, [d if d is not None else np.nan for d in dead], color = MAGENTA, linewidth = 2)
        end_label(ax, ep, [d if d is not None else np.nan for d in dead], "dead units", MAGENTA)
    ax.set_ylim(bottom = 0)
    style(ax, "Permanently-zero units in the shared head layer (phase 154's failure mode)",
          "epoch", "units out of 40")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_arrived_and_health.png"), dpi = 150,
                facecolor = "white")
    plt.close(fig)

    last = ev[-1]["val"]
    bestv = min(val)
    return {"epochs": len(hist), "best_val_balanced": bestv,
            "final_val": {k: v for k, v in last.items() if isinstance(v, (int, float))}}


# ─── closed-loop figures ─────────────────────────────────────────────────────

def series(ev, key):
    xs = [s["tick"] for s in ev["samples"]]
    ys = [s[key] for s in ev["samples"]]
    return np.asarray(xs, float), np.asarray(ys, float)


def band(ev, key):
    xs = [s["tick"] for s in ev["samples"]]
    lo, hi = [], []
    for s in ev["samples"]:
        vals = [a[key] for a in s["per_arena"]]
        lo.append(np.percentile(vals, 25) if vals else np.nan)
        hi.append(np.percentile(vals, 75) if vals else np.nan)
    return np.asarray(xs, float), np.asarray(lo, float), np.asarray(hi, float)


def fig_eval(evals, out_dir):
    """Coverage, distance and stopping, oracle against actor, on the same axes."""
    panels = [("coverage", "Robots on the target shape (ground truth)", "fraction of the swarm"),
              ("mean_dist", "Mean distance to the shape (normalised)", "distance"),
              ("stopped", "Robots that have stopped themselves", "fraction of the swarm")]
    fig, axes = plt.subplots(1, 3, figsize = (18, 4.8))
    for ax, (key, title, ylab) in zip(axes, panels):
        for name in _driver_order(evals):
            ev = evals[name]
            color = _color(name, evals)
            x, y = series(ev, key)
            xb, lo, hi = band(ev, key)
            ax.fill_between(xb, lo, hi, color = color, alpha = 0.13, linewidth = 0)
            ax.plot(x, y, color = color, linewidth = 2, label = name)
            end_label(ax, x, y, name, color)
        ax.legend(frameon = False, fontsize = 9, labelcolor = INK2, loc = "best")
        style(ax, title, "environment tick", ylab)
        if key != "mean_dist":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom = 0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "closed_loop_curves.png"), dpi = 150, facecolor = "white")
    plt.close(fig)


def coverage_from_positions(ev, tau = 5.0):
    """Recompute final coverage python-side, from stored positions and shapes.

    An independent check on the number Unity reports, using formations.py's own
    geometry -- the geometry the oracle actually steers by. The two agreed to
    0.7 units per robot once the player's bake rotation was aligned
    (KILOBOT_BAKE_ROTATION_STEPS=0); at the player's default of 1 they do not
    agree at all, because Unity's field is then that geometry rotated 90
    degrees. Keeping the check in the report is what makes that visible rather
    than assumed.
    """
    shapes = {}
    for e in ev.get("final", []):
        if e.get("shape"):
            shapes[(e["worker"], e["arena"])] = np.asarray(e["shape"], float)
    out = []
    for s in ev["samples"]:
        cov = []
        for a in s["per_arena"]:
            sh = shapes.get((a["worker"], a["arena"]))
            if sh is None or "pos" not in a:
                continue
            pos = np.asarray(a["pos"], float)
            d = np.linalg.norm(pos[:, None, :] - sh[None, :, :], axis = 2).min(axis = 1)
            cov.append(float((d < tau).mean()))
        out.append((s["tick"], float(np.mean(cov)) if cov else float("nan")))
    return out


def fig_paired(evals, out_dir):
    """Per-arena final coverage, oracle against actor, as a paired dot plot.

    The arenas are the same in both runs (same --swarm-rng, same --seed, so the
    same formations and the same spawns), which is what makes pairing them
    legitimate -- and what makes the spread visible rather than averaged away.
    """
    actors = [n for n in _driver_order(evals) if n != "oracle"]
    if "oracle" not in evals or not actors:
        return {}
    final_actor = actors[-1]
    o = {(a["worker"], a["arena"]): a["coverage"] for a in evals["oracle"]["samples"][-1]["per_arena"]}
    a = {(a["worker"], a["arena"]): a["coverage"]
         for a in evals[final_actor]["samples"][-1]["per_arena"]}
    keys = sorted(set(o) & set(a))
    if not keys:
        return {}
    ov = np.array([o[k] for k in keys])
    av = np.array([a[k] for k in keys])
    order = np.argsort(-ov)
    ov, av = ov[order], av[order]
    y = np.arange(len(keys))

    fig, ax = plt.subplots(figsize = (9, max(3.5, 0.32 * len(keys) + 1.5)))
    for i in range(len(keys)):
        ax.plot([av[i], ov[i]], [y[i], y[i]], color = GRID, linewidth = 2, zorder = 1)
    ax.scatter(ov, y, s = 70, color = ORANGE, zorder = 3, label = "oracle",
               edgecolor = "white", linewidth = 1.5)
    ax.scatter(av, y, s = 70, color = BLUE, zorder = 3, label = final_actor,
               edgecolor = "white", linewidth = 1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(["arena %d/%d" % (keys[i][0], keys[i][1]) for i in order], fontsize = 8)
    ax.set_xlim(0, 1.02)
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2, loc = "lower right")
    style(ax, "Final coverage per arena -- same formation and spawn for both drivers",
          "fraction of robots on the shape", None)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "closed_loop_per_arena.png"), dpi = 150, facecolor = "white")
    plt.close(fig)
    return {"per_arena_oracle_mean": float(ov.mean()), "per_arena_actor_mean": float(av.mean()),
            "arenas": len(keys),
            "actor_within_90pct_of_oracle": int(np.sum(av >= 0.9 * ov))}


def fig_demo(evals, out_dir, n = 6):
    """What the swarm actually ended up looking like, per arena, for each driver."""
    actors = [x for x in _driver_order(evals) if x != "oracle"]
    if not actors:
        return
    evals = {k: v for k, v in evals.items() if k == "oracle" or k == actors[-1]}
    keyed = {}
    for name, ev in evals.items():
        for entry in ev["final"]:
            keyed.setdefault((entry["worker"], entry["arena"]), {})[name] = entry
    keys = [k for k in sorted(keyed) if len(keyed[k]) == len(evals)][:n]
    if not keys:
        return
    rows = len(evals)
    fig, axes = plt.subplots(rows, len(keys), figsize = (2.9 * len(keys), 3.1 * rows),
                             squeeze = False)
    for r, name in enumerate(_driver_order(evals)):
        for c, k in enumerate(keys):
            ax = axes[r][c]
            e = keyed[k][name]
            shape = np.asarray(e.get("shape", []), float)
            pos = np.asarray(e["pos"], float)
            if shape.size:
                ax.scatter(shape[:, 0], shape[:, 1], s = 6, color = "#d7d6d2", marker = "s", linewidths = 0)
            color = _color(name, evals)
            ax.scatter(pos[:, 0], pos[:, 1], s = 16, color = color,
                       edgecolor = "white", linewidth = 0.5)
            ax.set_xlim(-105, 105); ax.set_ylim(-105, 105)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            cov = float(np.mean(np.asarray(e["dist"], float) < 0.05))
            if r == 0:
                ax.set_title(e.get("formation_name", "arena %d/%d" % k) or "", fontsize = 8,
                             color = INK2)
            ax.set_xlabel("%s -- %.0f%% on shape" % (name, 100 * cov), fontsize = 8,
                          color = color, fontweight = "bold")
    fig.suptitle("Final swarm positions on held-out formations (grey = target shape)",
                 color = INK, fontsize = 12, x = 0.01, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "demo_final_shapes.png"), dpi = 150, facecolor = "white")
    plt.close(fig)


def fig_progression(evals, out_dir, arena_index = 0, frames = 6):
    """One arena, both drivers, over time -- the assembly as it happens."""
    actors = [x for x in _driver_order(evals) if x != "oracle"]
    if actors:
        evals = {k: v for k, v in evals.items() if k == "oracle" or k == actors[-1]}
    cols = []
    for name in _driver_order(evals):
        ev = evals[name]
        samples = ev["samples"]
        if not samples or "pos" not in samples[0]["per_arena"][0]:
            return
        idx = np.linspace(0, len(samples) - 1, frames).astype(int)
        cols.append((name, ev, idx))
    shape = {}
    for name, ev in evals.items():
        for entry in ev["final"]:
            if (entry["worker"], entry["arena"]) == (0, arena_index):
                shape[name] = np.asarray(entry.get("shape", []), float)
    fig, axes = plt.subplots(len(cols), frames, figsize = (2.4 * frames, 2.7 * len(cols)),
                             squeeze = False)
    for r, (name, ev, idx) in enumerate(cols):
        color = _color(name, evals)
        for c, i in enumerate(idx):
            ax = axes[r][c]
            s = ev["samples"][i]
            rec = None
            for a in s["per_arena"]:
                if (a["worker"], a["arena"]) == (0, arena_index):
                    rec = a
            if rec is None:
                continue
            sh = shape.get(name)
            if sh is not None and sh.size:
                ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s", linewidths = 0)
            pos = np.asarray(rec["pos"], float)
            ax.scatter(pos[:, 0], pos[:, 1], s = 12, color = color,
                       edgecolor = "white", linewidth = 0.4)
            ax.set_xlim(-105, 105); ax.set_ylim(-105, 105)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.set_xlabel("tick %d -- %.0f%%" % (s["tick"], 100 * rec["coverage"]),
                          fontsize = 8, color = INK2)
            if c == 0:
                ax.set_ylabel(name, fontsize = 10, color = color, fontweight = "bold")
    fig.suptitle("One held-out arena assembling, oracle above, actor below",
                 color = INK, fontsize = 12, x = 0.01, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "demo_progression.png"), dpi = 150, facecolor = "white")
    plt.close(fig)


def write_gif(evals, out_dir, arena_index = 0, fps = 6):
    """An animation of the same arena, side by side, if pillow is available."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return
    names = _driver_order(evals)
    n_frames = min(len(evals[n]["samples"]) for n in names)
    if n_frames < 2:
        return
    shapes = {}
    for name in names:
        for entry in evals[name]["final"]:
            if (entry["worker"], entry["arena"]) == (0, arena_index):
                shapes[name] = np.asarray(entry.get("shape", []), float)
    fig, axes = plt.subplots(1, len(names), figsize = (4.2 * len(names), 4.6), squeeze = False)
    scatters, titles = {}, {}
    for i, name in enumerate(names):
        ax = axes[0][i]
        sh = shapes.get(name)
        if sh is not None and sh.size:
            ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s", linewidths = 0)
        color = _color(name, evals)
        scatters[name] = ax.scatter([], [], s = 18, color = color, edgecolor = "white",
                                    linewidth = 0.4)
        ax.set_xlim(-105, 105); ax.set_ylim(-105, 105); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        titles[name] = ax.set_title(name, color = color, fontsize = 11)

    def frame(fi):
        for name in names:
            s = evals[name]["samples"][fi]
            rec = None
            for a in s["per_arena"]:
                if (a["worker"], a["arena"]) == (0, arena_index):
                    rec = a
            if rec is None or "pos" not in rec:
                continue
            scatters[name].set_offsets(np.asarray(rec["pos"], float))
            titles[name].set_text("%s -- tick %d, %.0f%% on shape"
                                  % (name, s["tick"], 100 * rec["coverage"]))
        return list(scatters.values())

    anim = FuncAnimation(fig, frame, frames = n_frames, blit = False)
    try:
        anim.save(os.path.join(out_dir, "demo_animation.gif"), writer = PillowWriter(fps = fps))
    except Exception as exc:
        print("gif not written (%s)" % exc, flush = True)
    plt.close(fig)


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default = None)
    ap.add_argument("--eval-oracle", default = None)
    ap.add_argument("--eval-actor", default = None)
    ap.add_argument("--eval", action = "append", default = [], metavar = "NAME=PATH",
                    help = "an extra labelled curve, e.g. --eval 'after round 1=run1.json'. "
                           "Repeatable, kept in the order given; the last actor curve is the "
                           "one the paired plot, the demo grid and the animation use")
    ap.add_argument("--out-dir", required = True)
    ap.add_argument("--demo-arenas", type = int, default = 6)
    ap.add_argument("--gif", action = "store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok = True)

    summary = {}
    if args.run_dir:
        hist = load_jsonl(os.path.join(args.run_dir, "history.jsonl"))
        summary["training"] = fig_training(hist, args.out_dir)
        cfg_path = os.path.join(args.run_dir, "args.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                summary["training"]["args"] = json.load(f)

    evals = {}
    pairs = [("oracle", args.eval_oracle)]
    for spec in args.eval:
        name, _, path = spec.partition("=")
        pairs.append((name.strip(), path.strip()))
    pairs.append(("actor", args.eval_actor))
    for name, path in pairs:
        if path and os.path.exists(path):
            with open(path) as f:
                evals[name] = json.load(f)
    if evals:
        fig_eval(evals, args.out_dir)
        summary["closed_loop"] = {}
        for name, ev in evals.items():
            last = ev["samples"][-1]
            summary["closed_loop"][name] = {
                "final_coverage": last["coverage"], "final_mean_dist": last["mean_dist"],
                "final_stopped": last["stopped"], "arenas": last["arenas"],
                "ticks": last["tick"],
                "peak_coverage": max(s["coverage"] for s in ev["samples"]),
                "spawn_coverage": ev["samples"][0]["coverage"]}
            try:
                cpy = coverage_from_positions(ev)
                summary["closed_loop"][name]["final_coverage_python_frame"] = cpy[-1][1]
                summary["closed_loop"][name]["peak_coverage_python_frame"] = max(
                    c for _, c in cpy if c == c)
            except Exception as exc:
                summary["closed_loop"][name]["coverage_python_frame_error"] = str(exc)
        summary["closed_loop"].update(fig_paired(evals, args.out_dir))
        fig_demo(evals, args.out_dir, n = args.demo_arenas)
        fig_progression(evals, args.out_dir)
        if args.gif:
            write_gif(evals, args.out_dir)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent = 2)
    _write_md(summary, args.out_dir)
    print("wrote report to %s" % args.out_dir, flush = True)


def _write_md(summary, out_dir):
    lines = ["# Behaviour cloning report", ""]
    tr = summary.get("training") or {}
    if tr:
        fv0 = tr.get("final_val", {})
        lines += ["## Imitation of the oracle (held-out formations)", "",
                  "- best balanced motor MSE: **%.5f**" % tr.get("best_val_balanced", float("nan")),
                  "- decisions matching the oracle within %.2f on both wheels: **%.1f%%**"
                  % (fv0.get("within_tol", 0.05), 100 * fv0.get("within_all", float("nan"))),
                  "- arrived head: precision %.3f, recall %.3f (at the deployed 0.95 threshold)"
                  % (fv0.get("arrived_precision", float("nan")), fv0.get("arrived_recall", float("nan"))),
                  "- permanently-zero units in the shared head layer: %d of 40"
                  % fv0.get("dead_units", -1),
                  "- epochs: %d" % tr.get("epochs", 0), ""]
        fv = tr.get("final_val", {})
        if fv:
            lines.append("| metric | value |")
            lines.append("|---|---|")
            for k in sorted(fv):
                lines.append("| %s | %.5f |" % (k, fv[k]))
            lines.append("")
    cl = summary.get("closed_loop") or {}
    if cl:
        lines += ["## Closed loop in Unity (held-out formations)", "",
                  "| driver | final coverage | peak coverage | mean distance | stopped |",
                  "|---|---|---|---|---|"]
        for name in _driver_order(cl):
            if name in cl and isinstance(cl[name], dict):
                d = cl[name]
                lines.append("| %s | %.3f | %.3f | %.4f | %.3f |"
                             % (name, d["final_coverage"], d["peak_coverage"],
                                d["final_mean_dist"], d["final_stopped"]))
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
