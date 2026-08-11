"""settle_curve.py -- the task metric over time, one line per driver.

tools/settle_report.py answers "where did the swarm end up"; this answers "how
did it get there", which is what says whether a clone is reproducing the
teacher's BEHAVIOUR or merely landing on a similar score. The oracle's own curve
has a distinctive shape -- coverage dips first, because every robot drives north
into a wall before it can localize, and only then climbs -- and a clone that
tracks that shape is running the same state machine.

usage:
  python tools/settle_curve.py --out ../results/bc_v2/report_steer/settle_curve.png \\
      --eval "oracle=../results/bc_v2/eval_oracle_settled.json" \\
      --eval "actor=../results/bc_v2/eval_o3_settled.json"
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
SERIES = [ORANGE, BLUE, AQUA, VIOLET, MAGENTA]


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


PANELS = [("settled_5", "robots stopped within 5 units of their OWN target", "the task metric"),
          ("stopped", "robots that have stopped", "the oracle finishes; the clone does not"),
          ("coverage", "robots on the shape (any part of it)", "the old headline number")]


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action = "append", required = True, metavar = "NAME=PATH")
    ap.add_argument("--out", required = True)
    args = ap.parse_args(argv)

    runs = []
    for spec in args.eval:
        name, _, path = spec.partition("=")
        with open(path) as f:
            runs.append((name, json.load(f).get("samples", [])))

    fig, axes = plt.subplots(1, len(PANELS), figsize = (5.0 * len(PANELS), 4.3))
    axes = np.atleast_1d(axes)
    for ax, (key, title, note) in zip(axes, PANELS):
        for i, (name, samples) in enumerate(runs):
            t = np.array([s["tick"] for s in samples], float)
            y = np.array([(s.get(key) if s.get(key) is not None else np.nan) for s in samples], float)
            if not np.isfinite(y).any():
                continue
            ax.plot(t, y, color = SERIES[i % len(SERIES)], linewidth = 2.2, zorder = 3)
            j = int(len(t) * (0.60 + 0.12 * i))
            j = min(max(j, 1), len(t) - 1)
            ax.annotate(name, (t[j], y[j]), textcoords = "offset points", xytext = (6, 7),
                        color = INK, fontsize = 9.5)
        ax.set_ylim(-0.02, 1.02)
        style(ax, title + "\n" + note, "environment tick", "share of robots")
    fig.suptitle("Closed loop on held-out formations: the same 8 arenas, the same spawns",
                 color = INK, fontsize = 12.5, x = 0.02, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok = True)
    fig.savefig(args.out, dpi = 150, facecolor = "white")
    plt.close(fig)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
