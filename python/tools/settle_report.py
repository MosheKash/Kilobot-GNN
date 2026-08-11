"""settle_report.py -- the distribution of "did the robots stop where they were sent".

The task metric this project actually cares about is not coverage. Coverage asks
whether a robot is near ANY on-pixel of the target drawing, which a robot sitting
on someone else's part of the shape satisfies. The question that matters is
per-robot: did it stop, and did it stop near the point IT was assigned.

And the aggregate over robots hides the thing worth seeing, because the oracle is
strongly bimodal across arenas -- some arenas place 85% of their robots within 5
units and some place none. So this reports the DISTRIBUTION over arenas, not the
mean over robots:

  * per-arena share of robots settled within 5 / 10 / 20 units, as a strip plot
  * the empirical CDF of that share, which answers "in what fraction of arenas
    does the oracle settle at least X% of its robots"
  * the pooled per-robot error distribution, with the thresholds marked

Reads tools/eval_closed_loop.py output. Runs recorded before that tool stored
per-robot targets are handled by recomputing the assignment offline through the
same observation.ensure_target path, which is deterministic in (formation,
robot index, image id).

usage:
  python tools/settle_report.py --eval oracle=../results/bc_v2/eval_oracle.json \
      --out-dir ../results/bc_v2/report_settle
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

BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e2df"
TOLS = (5.0, 10.0, 20.0)


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


def _offline_targets(formations, limit, image_id, n, _cache = {}):
    """Recompute the assignment for runs recorded before targets were stored."""
    from formations import build_formation_pool
    from observation import TARGET_POOL_SIZE
    from spatial_hash import hilbert_order, assign_target_index
    key = (formations, limit)
    if key not in _cache:
        _cache[key] = build_formation_pool(formations, limit = limit)
    pool = _cache[key]
    ck = ("pts", key, image_id)
    if ck not in _cache:
        pts = pool[image_id % len(pool)].sample_points(TARGET_POOL_SIZE)
        _cache[ck] = (pts, hilbert_order(pts))
    pts, order = _cache[ck]
    return np.stack([pts[assign_target_index(order, l, image_id)] for l in range(n)])


def _split_pooled(ev, args):
    """Per-arena error arrays, in the same order as ev["final"]."""
    out = []
    for e in ev.get("final", []):
        pos = np.asarray(e["pos"], float)
        n = len(pos)
        tg = e.get("target")
        if tg is not None and all(t is not None for t in tg):
            tgt = np.asarray(tg, float)
        elif e.get("image_id") is not None:
            tgt = _offline_targets(args.formations, args.limit, int(e["image_id"]), n)
        else:
            out.append(np.zeros(0))
            continue
        out.append(np.linalg.norm(pos - tgt, axis = 1))
    return out


def arena_stats(ev, formations, limit):
    """Per-arena settled shares and the pooled per-robot errors."""
    rows = []
    pooled = []
    for e in ev.get("final", []):
        pos = np.asarray(e["pos"], float)
        n = len(pos)
        tg = e.get("target")
        if tg is not None and all(t is not None for t in tg):
            tgt = np.asarray(tg, float)
        elif e.get("image_id") is not None:
            tgt = _offline_targets(formations, limit, int(e["image_id"]), n)
        else:
            continue
        err = np.linalg.norm(pos - tgt, axis = 1)
        flags = e.get("stopped_flags")
        if flags is not None:
            stopped = np.asarray(flags, bool)
            stopped_share = float(stopped.mean())
        else:
            # Recorded before per-robot flags existed. The per-arena stopped
            # SHARE is still in the last sample, which is what the >=95% filter
            # needs; the per-robot conjunction is not, so `settled` degrades to
            # `near` for those runs -- honest only because the filter keeps just
            # arenas that are essentially fully stopped anyway.
            stopped_share = None
            for a in (ev.get("samples") or [{}])[-1].get("per_arena", []):
                if (a.get("worker"), a.get("arena")) == (e["worker"], e["arena"]):
                    stopped_share = float(a.get("stopped", 0.0))
            stopped = np.ones(n, bool)
            if stopped_share is None:
                stopped_share = 0.0
        pooled.append(err)
        row = {"arena": "%d/%d" % (e["worker"], e["arena"]), "robots": n,
               "median": float(np.median(err)), "stopped": stopped_share,
               "per_robot_flags": flags is not None}
        for tol in TOLS:
            row["near_%d" % int(tol)] = float((err < tol).mean())
            row["settled_%d" % int(tol)] = float(((err < tol) & stopped).mean())
        rows.append(row)
    return rows, (np.concatenate(pooled) if pooled else np.zeros(0))


def draw(all_rows, all_pooled, out_dir, key_prefix):
    names = list(all_rows)
    colors = {n: c for n, c in zip(names, [ORANGE, BLUE, AQUA, VIOLET, MAGENTA])}
    fig, axes = plt.subplots(1, 3, figsize = (18, 5.2))

    # 1. strip plot: one dot per arena, per tolerance
    ax = axes[0]
    for xi, tol in enumerate(TOLS):
        for si, name in enumerate(names):
            vals = np.array([r["%s_%d" % (key_prefix, int(tol))] for r in all_rows[name]])
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.26
            x = xi + (si - (len(names) - 1) / 2) * 0.3 + jitter
            ax.scatter(x, 100 * vals, s = 46, color = colors[name], alpha = 0.75,
                       edgecolor = "white", linewidth = 1.0,
                       label = name if xi == 0 else None)
            ax.plot([xi + (si - (len(names) - 1) / 2) * 0.3 - 0.13,
                     xi + (si - (len(names) - 1) / 2) * 0.3 + 0.13],
                    [100 * np.median(vals)] * 2, color = colors[name], linewidth = 2.5)
    ax.set_xticks(range(len(TOLS)))
    ax.set_xticklabels(["within %d units" % int(t) for t in TOLS])
    ax.set_ylim(-3, 103)
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2, loc = "upper right")
    style(ax, "Every arena is a dot; the bar is the median arena",
          None, "% of that arena's robots settled")

    # 2. ECDF -- "in what share of arenas does it settle at least X% of robots"
    ax = axes[1]
    for name in names:
        for tol, ls in zip(TOLS, ["-", "--", ":"]):
            vals = np.sort(np.array([r["%s_%d" % (key_prefix, int(tol))] for r in all_rows[name]]))
            if not len(vals):
                continue
            # share of arenas achieving AT LEAST x
            y = 1.0 - np.arange(len(vals)) / len(vals)
            ax.step(100 * vals, 100 * y, where = "post", color = colors[name],
                    linestyle = ls, linewidth = 2,
                    label = "%s, within %d u" % (name, int(tol)))
    ax.set_xlim(0, 100); ax.set_ylim(0, 103)
    ax.legend(frameon = False, fontsize = 8, labelcolor = INK2, loc = "upper right")
    style(ax, "Share of arenas reaching at least a given settle rate",
          "% of robots settled in an arena", "% of arenas at or above")

    # 3. pooled per-robot error
    ax = axes[2]
    bins = np.linspace(0, 120, 49)
    for name in names:
        p = all_pooled[name]
        ax.hist(np.clip(p, 0, 119), bins = bins, histtype = "step", linewidth = 2,
                color = colors[name], density = True, label = name)
    for tol in TOLS:
        ax.axvline(tol, color = INK3, linewidth = 1, linestyle = ":")
        ax.annotate("%du" % int(tol), (tol, ax.get_ylim()[1] * 0.94), fontsize = 8,
                    color = INK3, ha = "left")
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2)
    style(ax, "Every robot: distance to its own assigned target",
          "distance (arena units)", "density")

    fig.tight_layout()
    path = os.path.join(out_dir, "settle_distribution.png")
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)
    return path


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action = "append", required = True, metavar = "NAME=PATH")
    ap.add_argument("--out-dir", required = True)
    ap.add_argument("--formations", default = "../results/bc_v2/val_formations")
    ap.add_argument("--limit", type = int, default = 2000)
    ap.add_argument("--min-stopped", type = float, default = 0.95,
                    help = "drop any arena where fewer than this share of robots have stopped. "
                           "An arena still in motion has not answered the question being asked -- "
                           "the point is how well the swarm places itself once it is DONE, not "
                           "how far through the episode the run happened to get")
    ap.add_argument("--metric", default = "settled", choices = ["settled", "near"],
                    help = "settled = stopped AND near its target; near = near it regardless")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok = True)

    all_rows, all_pooled, summary = {}, {}, {}
    for spec in args.eval:
        name, _, path = spec.partition("=")
        with open(path) as f:
            ev = json.load(f)
        rows, pooled = arena_stats(ev, args.formations, args.limit)
        kept = [r for r in rows if r["stopped"] >= args.min_stopped]
        dropped = len(rows) - len(kept)
        if dropped:
            print("%s: dropped %d of %d arenas below %.0f%% stopped (%s)"
                  % (name, dropped, len(rows), 100 * args.min_stopped,
                     ", ".join("%s=%.0f%%" % (r["arena"], 100 * r["stopped"])
                               for r in rows if r["stopped"] < args.min_stopped)))
        rows = kept
        if not rows:
            print("%s: NO arena reached %.0f%% stopped -- nothing to report"
                  % (name, 100 * args.min_stopped))
            continue
        keep_ids = {r["arena"] for r in rows}
        pooled = np.concatenate([p for p, e in zip(_split_pooled(ev, args), ev.get("final", []))
                                 if "%d/%d" % (e["worker"], e["arena"]) in keep_ids] or [np.zeros(0)])
        all_rows[name] = rows
        all_pooled[name] = pooled
        s = {"arenas": len(rows), "arenas_dropped_not_finished": dropped,
             "min_stopped_required": args.min_stopped,
             "robots": int(sum(r["robots"] for r in rows)),
             "median_error": float(np.median(pooled)),
             "stopped_mean": float(np.mean([r["stopped"] for r in rows]))}
        for tol in TOLS:
            per = np.array([r["%s_%d" % (args.metric, int(tol))] for r in rows])
            s["robots_%d" % int(tol)] = float(np.mean(np.concatenate(
                [[r["%s_%d" % (args.metric, int(tol))]] * r["robots"] for r in rows])))
            s["arena_median_%d" % int(tol)] = float(np.median(per))
            for bar in (0.5, 0.8, 0.9):
                s["arenas_above_%d_at_%du" % (int(100 * bar), int(tol))] = float((per >= bar).mean())
        summary[name] = s
        print("== %s: %d arenas, %d robots" % (name, s["arenas"], s["robots"]))
        print("   %-28s %s" % ("per-arena settle rate:",
              "  ".join("<%du median %.0f%%" % (int(t), 100 * s["arena_median_%d" % int(t)]) for t in TOLS)))
        for tol in TOLS:
            print("   within %2du: arenas above 50%% / 80%% / 90%% of robots:  %.0f%% / %.0f%% / %.0f%%"
                  % (int(tol), 100 * s["arenas_above_50_at_%du" % int(tol)],
                     100 * s["arenas_above_80_at_%du" % int(tol)],
                     100 * s["arenas_above_90_at_%du" % int(tol)]))
    path = draw(all_rows, all_pooled, args.out_dir, args.metric)
    with open(os.path.join(args.out_dir, "settle_summary.json"), "w") as f:
        json.dump({"metric": args.metric, "summary": summary, "per_arena": all_rows}, f, indent = 2)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
