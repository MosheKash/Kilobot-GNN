"""render_eval.py -- turn ONE eval_closed_loop JSON into visual frames.

tools/eval_closed_loop.py runs headless and prints the coverage / stopped /
settled numbers, and it already stores every per-tick robot position in the
output JSON (unless run with --no-positions). This script draws those stored
positions so you can SEE one driver's behaviour across arenas and across time,
with the same per-arena numbers echoed to the console.

Outputs, all keyed off the final per-robot states and the per-tick samples:

  <out>/arena_<wi>_<k>_<formation>.png   each arena's final frame
  <out>/arena_grid.png                   all arenas side by side (montage)
  <out>/demo_animation.gif               one arena animated across ticks (--gif)

Visual conventions match the rest of the repo (bc_report / hybrid_report):
grey squares are the target shape and each robot's dot is tinted by how far it
ended up from ITS OWN assigned point:

  <5u    the driver's colour (on-target)
  5-10u  yellow   |  10-15u  orange  |  15-20u  red  |  20u+  maroon

A magenta ring additionally marks a robot that STOPPED yet is >=5u off -- the
failure to watch for. The same bands are echoed as per-arena counts on the
console (off 5-10 / 10-15 / 15-20 / 20+).

usage:
  python tools/render_eval.py ../results/vis/eval_hyb.json --out-dir ../results/vis
  python tools/render_eval.py ../results/vis/eval_hyb.json --out-dir ../results/vis --gif
  python tools/render_eval.py ../results/vis/eval_hyb.json --out-dir ../results/vis \
      --arena 0 --gif            # just worker 0 arena 0, plus its animation
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

from bc_report import BLUE, ORANGE, MAGENTA, GRID, INK, INK2
from bc_report import style

SAME_DRIVER = "#2a78d6"          # the actor (hybrid) blue; identical convention to the reports
FAR_RING = MAGENTA               # stopped but >=5u from its own point
FAR_TOL = 5.0                    # units -- matches the report's stopped-beyond-5u failure
ARENA_LIM = 105.0

# Distance-to-own-target error bands, colour-coded by severity (light -> severe).
# <5u robots keep the driver's colour; robots 5u+ off are tinted along this ramp.
OFF_BANDS = [(5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, None)]
OFF_COLORS = ["#fdb863", "#e08214", "#d73027", "#800026"]
OFF_LABELS = ["5-10u off", "10-15u off", "15-20u off", "20u+ off"]


def _band_index(err):
    """Vectorised band index per error: -1 = <5u (on-target), else 0..3."""
    err = np.asarray(err, float)
    out = np.full(err.shape, -1, dtype = int)
    for i, (lo, hi) in enumerate(OFF_BANDS):
        m = err >= lo
        if hi is not None:
            m = m & (err < hi)
        out[m] = i
    return out


def _band_counts(err):
    idx = _band_index(err)
    return {lab: int(np.sum(idx == i)) for i, lab in enumerate(OFF_LABELS)}


def _driver_name(ev):
    mode = ev.get("mode", "actor")
    hyb = (ev.get("args") or {}).get("closed_form_hybrid", False)
    if mode == "oracle":
        return "oracle"
    return "actor (hybrid)" if hyb else "actor"


def _driver_color(ev):
    return ORANGE if ev.get("mode") == "oracle" else SAME_DRIVER


def _entry_errors(entry):
    pos = np.asarray(entry["pos"], float)
    tgt = np.asarray(entry["target"], float)
    flag = np.asarray(entry.get("stopped_flags"), bool)
    err = np.linalg.norm(pos - tgt, axis = 1)
    return err, flag


def draw_arena(ax, entry, ev, title = None, legend = False):
    """One final frame: shape, robots, far-stopped rings.

    Robots are tinted by how far they ended up from their own assigned point:
    the driver colour when <5u, then a yellow->orange->red->maroon ramp for
    the 5-10 / 10-15 / 15-20 / 20+u bands. A magenta ring additionally marks
    a robot that STOPPED yet is >=5u off -- the failure to watch for.
    """
    sh = np.asarray(entry.get("shape", []), float)
    if sh.size:
        ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s", linewidths = 0)
    pos = np.asarray(entry["pos"], float)
    base = _driver_color(ev)
    err, flag = _entry_errors(entry)
    idx = _band_index(err)
    # on-target first so off-target dots are never hidden behind the swarm
    on = idx < 0
    if on.any():
        ax.scatter(pos[on, 0], pos[on, 1], s = 13, color = base, alpha = 0.85,
                   edgecolor = "white", linewidth = 0.4)
    for i, c in enumerate(OFF_COLORS):
        m = idx == i
        if m.any():
            ax.scatter(pos[m, 0], pos[m, 1], s = 13, color = c, alpha = 0.95,
                       edgecolor = "white", linewidth = 0.4)
    far = flag & (err >= FAR_TOL)
    if far.any():
        ax.scatter(pos[far, 0], pos[far, 1], s = 40, facecolor = "none",
                   edgecolor = FAR_RING, linewidth = 1.1, zorder = 4)
    ax.set_xlim(-ARENA_LIM, ARENA_LIM); ax.set_ylim(-ARENA_LIM, ARENA_LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if title is not None:
        ax.set_title(title, fontsize = 8, color = INK2)
    if legend:
        _band_legend(ax, base)
    if flag.size:
        settled = float(np.mean(flag & (err < FAR_TOL)))
        ax.set_xlabel("%.0f%% settled<5u" % (100 * settled), fontsize = 7, color = base)


def _band_handles(base):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker = "o", color = "w", markerfacecolor = base,
                      markersize = 7, label = "<5u on-target")]
    for c, lab in zip(OFF_COLORS, OFF_LABELS):
        handles.append(Line2D([0], [0], marker = "o", color = "w", markerfacecolor = c,
                              markersize = 7, label = lab))
    handles.append(Line2D([0], [0], marker = "o", color = "w", markerfacecolor = "none",
                          markeredgecolor = FAR_RING, markersize = 8, label = "stopped, off"))
    return handles


def _band_legend(ax, base):
    ax.legend(handles = _band_handles(base), loc = "upper right", frameon = True,
              fontsize = 7, borderaxespad = 0.4, handletextpad = 0.3)


def build_parser():
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("eval", help = "eval JSON from tools/eval_closed_loop.py")
    ap.add_argument("--out-dir", default = "../results/vis", help = "where the frames land")
    ap.add_argument("--arena", type = int, default = None,
                    help = "only render (worker, arena): worker is position i // arenas, arena is i %% arenas")
    ap.add_argument("--gif", action = "store_true",
                    help = "also render an animation GIF of one arena across ticks")
    ap.add_argument("--fps", type = float, default = 6.0)
    return ap


def main(argv = None):
    args = build_parser().parse_args(argv)
    with open(args.eval) as f:
        ev = json.load(f)
    final = ev.get("final", [])
    if not final:
        raise SystemExit("no 'final' records in %s -- resurrect it or re-run without --no-positions"
                         % args.eval)
    os.makedirs(args.out_dir, exist_ok = True)
    name = _driver_name(ev)
    color = _driver_color(ev)

    entries = []
    for e in final:
        if args.arena is not None and (e["worker"], e["arena"]) != (0, args.arena):
            continue
        entries.append(e)
    print("%s: rendering %d arena(s) to %s" % (name, len(entries), os.path.abspath(args.out_dir)),
          flush = True)

    # per-arena numbers + individual PNG
    for e in entries:
        err, flag = _entry_errors(e)
        stopped = float(flag.mean())
        settled = float(np.mean(flag & (err < FAR_TOL))) if flag.size else 0.0
        med = float(np.median(err)) if err.size else float("nan")
        fname = e.get("formation_name") or "%d" % e.get("image_id", "")
        fname = fname[:-4] if fname.endswith(".png") else fname
        title = "w%d/a%d  %s" % (e["worker"], e["arena"], fname)
        counts = _band_counts(err)
        print("  arena %d/%d  %-12s stopped=%.0f%%  settled<5u=%.0f%%  median_err=%.1f  "
              "off 5-10=%d 10-15=%d 15-20=%d 20+=%d"
              % (e["worker"], e["arena"], fname, 100 * stopped, 100 * settled, med,
                 counts["5-10u off"], counts["10-15u off"], counts["15-20u off"], counts["20u+ off"]),
              flush = True)
        fig, ax = plt.subplots(figsize = (4.2, 4.2))
        draw_arena(ax, e, ev, title = title, legend = True)
        fig.tight_layout()
        path = os.path.join(args.out_dir, "arena_%d_%d_%s.png" % (e["worker"], e["arena"], fname))
        fig.savefig(path, dpi = 140, facecolor = "white")
        plt.close(fig)
        print("    wrote %s" % path, flush = True)

    # montage of every arena, one run, side by side (skips the cross-driver legend)
    if len(entries) > 1:
        ncols = min(len(entries), 5)
        nrows = int(np.ceil(len(entries) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize = (2.6 * ncols, 2.6 * nrows),
                                 squeeze = False)
        for i, e in enumerate(entries):
            ax = axes[i // ncols][i % ncols]
            title = e.get("formation_name") or ""
            draw_arena(ax, e, ev, title = title)
        for ax in axes.flat[len(entries):]:
            ax.axis("off")
        fig.suptitle("%s -- where each swarm ended up (dot colour = distance off its own point; "
                     "ring = stopped but >=%.0fu from its point)" % (name, FAR_TOL),
                     color = INK, fontsize = 12)
        fig.legend(handles = _band_handles(_driver_color(ev)), loc = "lower center",
                   ncol = 6, frameon = False, fontsize = 8, handletextpad = 0.3,
                   columnspacing = 1.2)
        fig.tight_layout(rect = (0, 0.035, 1, 0.95))
        grid = os.path.join(args.out_dir, "arena_grid.png")
        fig.savefig(grid, dpi = 140, facecolor = "white")
        plt.close(fig)
        print("wrote %s" % grid, flush = True)

    if args.gif:
        _write_gif(ev, entries, args.out_dir, args.fps, name, color)

    print("done. open %s/*.png (and demo_animation.gif) to confirm visually."
          % os.path.abspath(args.out_dir), flush = True)


def _write_gif(ev, entries, out_dir, fps, name, color):
    """Animate the per-tick samples for a single (worker, arena) = (0, args.arena or 0).

    Dots are tinted live by each robot's distance to its OWN (fixed, per-episode)
    assigned target using the same band ramp as the stills. Needs target[], which
    eval_closed_loop only writes to `final`, so the animation needs the final
    record to provide them.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        print("gif skipped (no pillow: %s)" % exc, flush = True)
        return
    arena_index = entries[0]["arena"] if entries else 0
    samples = ev.get("samples", [])
    frames = [s for s in samples
              if any((a["worker"], a["arena"]) == (0, arena_index) and "pos" in a
                     for a in s.get("per_arena", []))]
    if len(frames) < 2:
        print("gif skipped (fewer than 2 sampled frames for arena 0/%d)" % arena_index, flush = True)
        return
    entry = next(e for e in entries if (e["worker"], e["arena"]) == (0, arena_index))
    sh = np.asarray(entry.get("shape", []), float)
    target = np.asarray(entry["target"], float) if entry.get("target") else None
    fig, ax = plt.subplots(figsize = (3.6, 3.6))
    if sh.size:
        ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s", linewidths = 0)
    sc = ax.scatter([], [], s = 16, edgecolor = "white", linewidth = 0.4)
    ax.set_xlim(-ARENA_LIM, ARENA_LIM); ax.set_ylim(-ARENA_LIM, ARENA_LIM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    _band_legend(ax, color)
    title = ax.set_title("", color = INK2)

    def frame(fi):
        s = frames[fi]
        rec = next(a for a in s["per_arena"] if (a["worker"], a["arena"]) == (0, arena_index))
        pos = np.asarray(rec["pos"], float)
        if target is not None and len(target) == len(pos):
            idx = _band_index(np.linalg.norm(pos - target, axis = 1))
            fc = np.where(idx < 0, color, np.asarray(OFF_COLORS)[np.clip(idx, 0, None)])
            sc.set_facecolors(fc)
        sc.set_offsets(pos)
        title.set_text("tick %d  %.0f%% on shape  stopped %.0f%%"
                       % (s["tick"], 100 * rec["coverage"], 100 * rec["stopped"]))
        return (sc, title)

    anim = FuncAnimation(fig, frame, frames = len(frames), blit = False)
    path = os.path.join(out_dir, "demo_animation.gif")
    try:
        anim.save(path, writer = PillowWriter(fps = fps))
    except Exception as exc:
        print("gif not written (%s)" % exc, flush = True)
        path = None
    plt.close(fig)
    if path:
        print("wrote %s" % path, flush = True)


if __name__ == "__main__":
    main()
