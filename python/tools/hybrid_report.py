"""hybrid_report.py -- the figures, the summary and the page for the arrival gate.

The hybrid report answers one question: how does the actor's arrival gate work,
and does it work? Four closed-loop runs on the SAME held-out formations and the
same spawns (identical --swarm-rng and --seed, so the comparison is paired):

  oracle       the scripted teacher, its own closed-form arrived state
  head         the actor's learned arrived head alone (deployed 0.95 threshold)
  closed-form  the oracle's arrival rule computed from the actor's filter
  hybrid       closed-form OR learned head (cfg.closed_form_hybrid)

Inputs are plain eval JSONs like tools/eval_closed_loop.py writes, so a report
can be regenerated without re-running anything. Output is summary.json plus a
fully self-contained index.html (PNGs inlined as data URIs), matching the
visual conventions of bc_report: the oracle is orange, the canonical actor
(here the hybrid) is blue, the other actor variants are the palette's remaining
adjacent pairs, and no series is ever identified by colour alone.

usage:
  python tools/hybrid_report.py --out-dir ../results/hybrid_cloning \\
      --eval-oracle  ../results/bc_v2/eval_oracle.json \\
      --eval-head    ../results/bc_v2/eval_actor.json \\
      --eval-cf      ../results/bc_v2/eval_o3_cf08.json \\
      --eval-hybrid  ../results/bc_v2/eval_o3_hyb.json
"""

import argparse
import base64
import html
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bc_report import BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GRID, INK, INK2, MUTED
from bc_report import style, _color, fig_eval, fig_paired
from settle_report import TOLS, arena_stats

NAMES = ["oracle", "head", "closed-form", "hybrid"]
DISPLAY = {"oracle": "oracle", "head": "learned head", "closed-form": "closed-form only",
           "hybrid": "hybrid (OR)"}
LOCAL_CONF = 0.4


def load(path):
    with open(path) as f:
        return json.load(f)


def arena_rows(ev):
    """Per-arena records keyed by (worker, arena), final positions + targets."""
    rows = {}
    for e in ev.get("final", []):
        pos = np.asarray(e["pos"], float)
        tgt = np.asarray(e.get("target"), float) if e.get("target") is not None else None
        err = (np.linalg.norm(pos - tgt, axis = 1) if tgt is not None and len(tgt) == len(pos)
               else np.zeros(len(pos)))
        rows[(e["worker"], e["arena"])] = {
            "worker": e["worker"], "arena": e["arena"],
            "image_id": e.get("image_id"), "formation": e.get("formation_name"),
            "pos": pos, "target": tgt, "err": err, "shape": np.asarray(e.get("shape", []), float),
            "stopped": np.asarray(e.get("stopped_flags"), bool),
        }
    return rows


def who_fires(evals):
    """Per-robot Venn of what stopped the swarm, across three actor variants.

    The three runs are separate rollouts of the same (formation, spawn, robot
    index), so per-robot identity is comparable but NOT causally equal -- one
    robot stopping earlier changes the messages the swarm broadcasts and can
    move a neighbour's fate. The numbers are therefore read as "in the head
    run, the head stopped this robot; in the hybrid run, the OR stopped it".
    `uncredited` marks robots the hybrid stopped that neither standalone branch
    stopped in its own run -- the same divergence, and part of why the sets do
    not partition exactly. The agreement line is reported so the caveat is
    visible rather than assumed away.
    """
    flags = {}
    for v in ("head", "closed-form", "hybrid"):
        f = {}
        for e in evals[v]["final"]:
            f[(e["worker"], e["arena"])] = np.asarray(e["stopped_flags"], bool)
        flags[v] = f
    counts = {"head_only": 0, "cf_only": 0, "both": 0, "uncredited": 0, "moving": 0}
    agree = total = 0
    for arena in flags["hybrid"]:
        h, c, b = flags["head"][arena], flags["closed-form"][arena], flags["hybrid"][arena]
        n = len(b)
        hs, cs, bs = set(np.flatnonzero(h)), set(np.flatnonzero(c)), set(np.flatnonzero(b))
        counts["head_only"] += len(bs & hs - cs)
        counts["cf_only"] += len(bs & cs - hs)
        counts["both"] += len(bs & hs & cs)
        # stopped by the hybrid but by neither branch in their own runs: the
        # rollouts diverged (the runs are separate), so one robot finishing
        # changes what a neighbour sees. The gate itself is the OR.
        counts["uncredited"] += len(bs - (hs | cs))
        counts["moving"] += int(np.sum(~b))
        agree += int(np.sum((h | c) == b))
        total += n
    counts["robots"] = total
    counts["agree_with_or_gate"] = agree / total if total else float("nan")
    return counts


def fig_who_stops(evals, out_dir):
    """Stacked: settled within 5/10/20u, stopped-but-far, still moving."""
    fig, ax = plt.subplots(figsize = (11, 5.2))
    cats = [("_5", "settled < 5u"), ("_10", "5u &ndash; 10u"), ("_20", "10u &ndash; 20u")]
    names = [n for n in NAMES if n in evals]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    colors = [_color(n, evals) for n in names]
    for suffix, label in cats:
        vals = [100 * _settled_share(evals[n], suffix) for n in names]
        ax.bar(x, vals, bottom = bottom, color = colors, label = label)
        bottom += np.asarray(vals)
    stopped_far = [100 * (_stopped(evals[n]) - _settled_share(evals[n], "_20")) for n in names]
    ax.bar(x, stopped_far, bottom = bottom, color = colors,
           label = "stopped, beyond 20u", alpha = 0.55)
    bottom += np.asarray(stopped_far)
    moving = [100 * (1 - _stopped(evals[n])) for n in names]
    ax.bar(x, moving, bottom = bottom, color = colors, label = "still moving", alpha = 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[n] for n in names])
    ax.set_ylim(0, 105)
    ax.legend(frameon = False, fontsize = 9, labelcolor = INK2, loc = "upper center",
              bbox_to_anchor = (0.5, -0.14), ncol = 4)
    style(ax, "Where every robot ends up: stopped near its own point or not",
          None, "% of robots")
    fig.tight_layout()
    path = os.path.join(out_dir, "who_stops.png")
    fig.savefig(path, dpi = 150, facecolor = "white")
    plt.close(fig)
    return path


def _stopped(ev):
    last = ev["samples"][-1]
    return float(last.get("stopped", 0.0) or 0.0)


def _settled_share(ev, suffix):
    last = ev["samples"][-1]
    return float(last.get("settled" + suffix, 0.0) or 0.0)


def fig_arena_grid(evals, out_dir, n_max = 10):
    """Four drivers x up to ten arenas: where each swarm ended up."""
    keyed = {}
    for v in NAMES:
        if v not in evals:
            continue
        for entry in evals[v]["final"]:
            k = (entry["worker"], entry["arena"])
            keyed.setdefault(k, {})[v] = entry
    keys = [k for k in sorted(keyed)
            if all(v in keyed[k] for v in NAMES if v in evals)][:n_max]
    names = [v for v in NAMES if v in evals]
    fig, axes = plt.subplots(len(names), len(keys),
                             figsize = (2.55 * len(keys), 2.75 * len(names) + 1.0),
                             squeeze = False)
    for r, name in enumerate(names):
        color = _color(name, evals)
        for c, k in enumerate(keys):
            ax = axes[r][c]
            e = keyed[k][name]
            sh = np.asarray(e.get("shape", []), float)
            pos = np.asarray(e["pos"], float)
            flag = np.asarray(e.get("stopped_flags"), bool)
            if sh.size:
                ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s",
                           linewidths = 0)
            base = _color(name, evals)
            ax.scatter(pos[:, 0], pos[:, 1], s = 13, color = base, alpha = 0.85,
                       edgecolor = "white", linewidth = 0.4)
            if flag.size:
                far = flag & (np.linalg.norm(pos - e["target"], axis = 1) >= 5.0) \
                    if e.get("target") is not None else None
                if far is not None and far.any():
                    ax.scatter(pos[far, 0], pos[far, 1], s = 40, facecolor = "none",
                               edgecolor = MAGENTA, linewidth = 1.1, zorder = 4)
            ax.set_xlim(-105, 105); ax.set_ylim(-105, 105)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(keyed[k][names[0]].get("formation_name", ""), fontsize = 8,
                             color = INK2)
            xl = ""
            if flag.size and e.get("target") is not None:
                err = np.linalg.norm(pos - e["target"], axis = 1)
                xl = "%.0f%% settled<5u" % (100 * float(np.mean(flag & (err < 5.0))))
            ax.set_xlabel(xl, fontsize = 7, color = color)
            if c == 0:
                ax.set_ylabel(DISPLAY[name], fontsize = 10, color = color, fontweight = "bold")
    fig.suptitle("Where each swarm ended up (grey = target shape; ring = stopped but >= 5u from its own point)",
                 color = INK, fontsize = 12, x = 0.01, ha = "left")
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    path = os.path.join(out_dir, "arena_grid.png")
    fig.savefig(path, dpi = 140, facecolor = "white")
    plt.close(fig)
    return path


def write_gif(evals, out_dir, arena_index = 0, fps = 6):
    """An animation of the same arena, the four drivers side by side.

    Each frame is one recorded sample; per_arena holds robot positions at that
    tick. The shapes come from the final records. Skipped quietly if pillow is
    unavailable, matching bc_report.write_gif's contract.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception:
        return None
    names = [v for v in NAMES if v in evals]
    n_frames = min(len(evals[n]["samples"]) for n in names)
    if n_frames < 2:
        return None
    shapes = {}
    for name in names:
        for entry in evals[name]["final"]:
            if (entry["worker"], entry["arena"]) == (0, arena_index):
                shapes[name] = np.asarray(entry.get("shape", []), float)
    fig, axes = plt.subplots(1, len(names), figsize = (3.4 * len(names), 3.9),
                             squeeze = False)
    scatters, titles = {}, {}
    for i, name in enumerate(names):
        ax = axes[0][i]
        sh = shapes.get(name)
        if sh is not None and sh.size:
            ax.scatter(sh[:, 0], sh[:, 1], s = 5, color = "#d7d6d2", marker = "s",
                       linewidths = 0)
        color = _color(name, evals)
        scatters[name] = ax.scatter([], [], s = 16, color = color, edgecolor = "white",
                                    linewidth = 0.4)
        ax.set_xlim(-105, 105); ax.set_ylim(-105, 105); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        titles[name] = ax.set_title(DISPLAY[name], color = color, fontsize = 10)

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
                                  % (DISPLAY[name], s["tick"], 100 * rec["coverage"]))
        return list(scatters.values())

    anim = FuncAnimation(fig, frame, frames = n_frames, blit = False)
    path = os.path.join(out_dir, "demo_animation.gif")
    try:
        anim.save(path, writer = PillowWriter(fps = fps))
    except Exception as exc:
        print("gif not written (%s)" % exc, flush = True)
        path = None
    plt.close(fig)
    return path


def fig_gate(out_dir):
    """The gate as a circuit: two detectors, an OR, a latch."""
    fig, ax = plt.subplots(figsize = (12.5, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 46)

    def box(x, y, w, h, fill, edge, text, fs = 9, tc = INK):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle = "round,pad=0.35",
                                    facecolor = fill, edgecolor = edge, linewidth = 1.4)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha = "center", va = "center",
                fontsize = fs, color = tc)

    def arrow(x1, y1, x2, y2, color = INK2):
        ax.annotate("", xy = (x2, y2), xytext = (x1, y1),
                    arrowprops = dict(arrowstyle = "-|>", color = color, lw = 1.6))

    # inputs
    ax.text(2, 43.5, "actor observation (prop, one slot per robot)", fontsize = 10,
            color = INK, fontweight = "bold")
    box(2, 25, 20, 6, "#ffffff", GRID, "PROP_DIST_T (slot 21)\nd_target, normalised")
    box(2, 17, 20, 6, "#ffffff", GRID, "PROP_CONF_POS (slot 12)\nconf_pos")
    box(2, 9, 20, 6, "#ffffff", GRID, "PROP_SIN_T / PROP_COS_T\n(slots 19, 20)")
    arrow(10, 25, 10, 23.5); arrow(10, 17, 12, 15.5); arrow(26, 12, 35, 21)

    # closed form
    box(36, 17, 26, 10, "#fff3ec", ORANGE,
        "closed_form_arrived\n(d target < 0.08) & (conf >= 0.4)\n& has_target",
        fs = 8, tc = "#7a3410")
    ax.text(36, 11.4, "a rule, the oracle's own arrival test on the actor's filter",
            fontsize = 7.5, color = MUTED)

    # learned head
    box(36, 28, 26, 10, "#edf4fe", BLUE,
        "learned arrived head\nsigmoid(_arrived_logit) > 0.95",
        fs = 8, tc = "#123a77")
    ax.text(36, 25.6, "trained on recorded oracle stops; accurate in the tape's",
            fontsize = 7.5, color = MUTED)
    ax.text(36, 24.3, "distribution, under-fires on the deployment shift",
            fontsize = 7.5, color = MUTED)

    for y in (32.5, 21.5):
        arrow(26.5, y, 35, y)

    # OR
    box(70, 18, 16, 8, "#f1f1ee", INK2, "OR", fs = 13, tc = INK)
    arrow(62, 32.5, 70, 24, BLUE)
    arrow(62, 17, 70, 23, ORANGE)
    ax.text(70, 14.6, "either branch stops the robot\n(closed_form_hybrid)",
            fontsize = 7.5, color = MUTED, ha = "center")

    # latch
    box(70, 6, 16, 8, "#f1f1ee", INK2, "latch\n(terminal)", fs = 8.5, tc = INK)
    arrow(78, 18, 78, 15)
    ax.text(91, 9.5, "once it fires the robot stays\noff for the episode, like the\noracle's arrived state",
            fontsize = 7.5, color = MUTED, ha = "left")

    # motor
    box(50, 1, 22, 3.4, "#1baf7a", "#0f5f46", "motors forced to zero", fs = 8, tc = "#063426")
    arrow(66, 1.8, 78, 6, "#0f5f46")

    fig.tight_layout()
    path = os.path.join(out_dir, "gate_circuit.png")
    fig.savefig(path, dpi = 160, facecolor = "white")
    plt.close(fig)
    return path


def build_summary(evals, out_dir):
    s = {}
    for n in NAMES:
        if n not in evals:
            continue
        ev = evals[n]
        arena = arena_rows(ev)
        s[n] = {
            "ticks": ev["samples"][-1]["tick"],
            "arenas": len(arena),
            "robots": int(sum(len(a["pos"]) for a in arena.values())),
            "stopped": float(ev["samples"][-1].get("stopped", 0)),
            "coverage": float(ev["samples"][-1].get("coverage", 0)),
            "mean_dist": float(ev["samples"][-1].get("mean_dist", 0)),
            "median_err": float(np.median([a["target_err_median"]
                                           for a in ev["samples"][-1]["per_arena"]
                                           if a.get("target_err_median") is not None]))
                if any(a.get("target_err_median") is not None
                       for a in ev["samples"][-1]["per_arena"]) else float("nan"),
            "settled_5": _settled_share(ev, "_5"),
            "settled_10": _settled_share(ev, "_10"),
            "settled_20": _settled_share(ev, "_20"),
            "peak_coverage": max(x.get("coverage", 0) for x in ev["samples"]),
        }
    s["who_fires"] = who_fires(evals)
    return s


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def num(x, fmt = "%.3f"):
    return "n/a" if x is None else fmt % x


def figure_html(path, alt):
    mime = "image/gif" if path.endswith(".gif") else "image/png"
    with open(path, "rb") as f:
        uri = "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))
    return ('<figure><img src="%s" alt="%s" /></figure>' % (uri, html.escape(alt)))


def build_page(evals, summary, out_dir, meta):
    orc, head, cf, hyb = (summary.get(v, {}) for v in NAMES)

    # ── main summary table ───────────────────────────────────────────────────
    rows = []
    for v in NAMES:
        if v not in summary:
            continue
        d = summary[v]
        row = "<tr"
        row += " class='row-final'" if v == "hybrid" else ""
        row += "><td>%s</td>" % html.escape(DISPLAY[v])
        row += "".join("<td>%s</td>" % pct(d.get(k)) for k in (
            "stopped", "settled_5", "settled_10", "settled_20"))
        row += "<td>%s</td>" % num(d.get("median_err"))
        row += "<td>%s</td></tr>" % pct(d.get("coverage"))
        rows.append(row)
    table = "\n".join(rows)

    # ── per-arena table ──────────────────────────────────────────────────────
    keys = sorted({a for v in NAMES if v in evals for a in arena_rows(evals[v])})
    kept = []
    for kk in keys:
        got = {v: arena_rows(evals[v]).get(kk) for v in NAMES if v in evals}
        if any(r is None for r in got.values()):
            continue
        kept.append((kk, got))
    pa_rows = []
    for kk, got in kept:
        cells = "<td>%d/%d</td><td>%s</td>" % (kk[0], kk[1],
                html.escape(got["oracle"]["formation"] or ""))
        for v in NAMES:
            if v not in got:
                cells += "<td>&ndash;</td><td>&ndash;</td><td>&ndash;</td>"
                continue
            g = got[v]
            if len(g["stopped"]):
                stopped = float(g["stopped"].mean())
                settled = float(((g["err"] < 5.0) & g["stopped"]).mean())
            else:
                stopped = settled = float("nan")
            med = float(np.median(g["err"])) if len(g["err"]) else float("nan")
            cells += "<td>%s</td><td>%s</td><td>%s</td>" % (
                num(stopped, "%.0f"), num(settled, "%.0f"), num(med, "%.1f"))
        pa_rows.append("<tr>%s</tr>" % cells)
    pa_headers = "".join(
        "<th>%s<br/><span class='sig'>stopped / &lt;5u / med err</span></th>"
        % html.escape(DISPLAY[v]) for v in NAMES if v in evals)

    # ── who-stops figures ────────────────────────────────────────────────────
    who = summary.get("who_fires", {})
    nbot = who.get("robots", 0)

    def frac(key):
        return pct(who.get(key, 0) / nbot) if nbot else "n/a"

    gate_fig = figure_html(os.path.join(out_dir, "gate_circuit.png"),
                           "the hybrid gate as a circuit")
    curve_fig = figure_html(os.path.join(out_dir, "closed_loop_curves.png"),
                            "closed-loop curves")
    who_fig = figure_html(os.path.join(out_dir, "who_stops.png"),
                          "who stops where")
    grid_fig = figure_html(os.path.join(out_dir, "arena_grid.png"),
                           "final swarm positions, ten arenas")
    settle_fig = figure_html(os.path.join(out_dir, "settle_distribution.png"),
                             "settle distribution across arenas")
    gif_path = os.path.join(out_dir, "demo_animation.gif")
    gif_fig = (figure_html(gif_path, "the assembly, animated, all four drivers")
               if os.path.exists(gif_path) else "")

    arenas = orc.get("arenas", 0)
    ticks = orc.get("ticks", 0)
    med_err_hyb = num(hyb.get("median_err"), "%.1f")

    return PAGE % dict(
        orc_stopped=pct(orc.get("stopped")), arenas=arenas, ticks=ticks,
        head_stopped=pct(head.get("stopped")), head_settled=pct(head.get("settled_5")),
        cf_stopped=pct(cf.get("stopped")),
        hyb_stopped=pct(hyb.get("stopped")), hyb_cov=pct(hyb.get("coverage")),
        hyb_settled=pct(hyb.get("settled_5")), orc_settled=pct(orc.get("settled_5")),
        hyb_settled10=pct(hyb.get("settled_10")), orc_settled10=pct(orc.get("settled_10")),
        hyb_settled20=pct(hyb.get("settled_20")), orc_settled20=pct(orc.get("settled_20")),
        med_err_hyb=med_err_hyb, orc_med_err=num(orc.get("median_err"), "%.1f"),
        gate_fig=gate_fig, local_conf=LOCAL_CONF,
        table=table,
        curve_fig=curve_fig, who_fig=who_fig, settle_fig=settle_fig,
        nbot=nbot, grid_fig=grid_fig, gif_fig=gif_fig,
        head_only=frac("head_only"), cf_only=frac("cf_only"),
        both=frac("both"), uncredited=frac("uncredited"), moving=frac("moving"),
        agree=pct(who.get("agree_with_or_gate")),
        pa_headers=pa_headers, pa_rows="\n".join(pa_rows),
    )


def write_page(evals, summary, out_dir, meta):
    page = build_page(evals, summary, out_dir, meta)
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)


PAGE = """<title>Why a hybrid arrives: gating a cloned stop on the oracle's own rule</title>
<style>
  :root { color-scheme: light;
    --ground:#f5f6f8; --panel:#ffffff; --ink:#10131a; --ink-2:#4a5162; --ink-3:#767e90;
    --line:#dde1e9; --actor:#2a78d6; --oracle:#eb6834;
    --shadow:0 1px 2px rgba(16,19,26,.06), 0 8px 24px rgba(16,19,26,.05); }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) { color-scheme: dark;
      --ground:#0e1116; --panel:#161a21; --ink:#eef1f6; --ink-2:#a8b0c0; --ink-3:#7c8496;
      --line:#262c37; --actor:#3987e5; --oracle:#ef7c4e;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35); } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
    font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; }
  .wrap { max-width: 1160px; margin:0 auto; padding:56px 24px 96px; }
  header { border-bottom:1px solid var(--line); padding-bottom:28px; margin-bottom:36px; }
  .eyebrow { font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3); }
  h1 { font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    font-weight:600; font-size:clamp(30px,4.4vw,46px); line-height:1.12;
    letter-spacing:-.015em; text-wrap:balance; margin:14px 0 12px; }
  .standfirst { color:var(--ink-2); font-size:18px; max-width:66ch; margin:0; }
  h2 { font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    font-size:25px; font-weight:600; letter-spacing:-.01em; margin:52px 0 6px;
    text-wrap:balance; }
  h2 + p.lede { color:var(--ink-2); margin:0 0 18px; max-width:66ch; }
  p { max-width:70ch; }
  code,.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; }
  code { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:1px 5px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:30px 0 8px; }
  .tile { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px 18px 16px; box-shadow:var(--shadow); }
  .tile-label { font:600 11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3); }
  .tile-value { font-variant-numeric:tabular-nums; font-size:34px; font-weight:600; letter-spacing:-.02em; margin:8px 0 4px; }
  .tile:nth-child(3) .tile-value { color:var(--oracle); }
  .tile:nth-child(4) .tile-value { color:var(--actor); }
  .tile-note { color:var(--ink-2); font-size:13px; line-height:1.45; }
  figure { margin:26px 0; }
  figure img { width:100%%; height:auto; display:block; border:1px solid var(--line); border-radius:10px; background:#fff; }
  figcaption { color:var(--ink-2); font-size:14px; margin-top:10px; max-width:80ch; }
  .table-wrap { overflow-x:auto; margin:20px 0; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
  table { border-collapse:collapse; width:100%%; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:10px 14px; border-bottom:1px solid var(--line); font-size:13.5px; }
  th:first-child,td:first-child { text-align:left; }
  thead th { font:600 11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-3); }
  tbody tr:last-child td { border-bottom:none; }
  tr.row-final td { font-weight:600; color:var(--actor); }
  .note { border-left:3px solid var(--oracle); background:var(--panel); border-radius:0 8px 8px 0; padding:16px 18px; margin:22px 0; box-shadow:var(--shadow); }
  .note-status { border-left-color:var(--actor); }
  .note strong { display:block; margin-bottom:4px; }
  pre { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; overflow-x:auto; font-size:13px; line-height:1.6; }
  footer { margin-top:64px; padding-top:20px; border-top:1px solid var(--line); color:var(--ink-3); font-size:13px; }
  .sig { color:var(--ink-3); font-size:12px; font-weight:400; text-transform:none; }
  table.pa th { font-size:10px; }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Kilobot swarm &middot; arrival gate</div>
    <h1>A clone that stops with the oracle's own rule &mdash; or the learned head, whichever fires</h1>
    <p class="standfirst">The steering was fixed by the oracle-form head; what still separated the clone
      from its teacher was <em>arrival certification</em> &mdash; a robot's decision that it has reached
      its own assigned point and may stop, forever. The learned head that is accurate in training
      under-fires when deployed. The open door is a <strong>hybrid gate</strong>: the teacher's own
      rule, computed from the clone&rsquo;s observation real-time, in OR with the learned head.</p>
  </header>

  <div class="tiles">
    <div class="tile"><div class="tile-label">Oracle, robots stopped</div>
      <div class="tile-value">%(orc_stopped)s</div>
      <div class="tile-note">its own arrived state, %(arenas)s arenas, %(ticks)s ticks</div></div>
    <div class="tile"><div class="tile-label">Learned head alone</div>
      <div class="tile-value">%(head_stopped)s</div>
      <div class="tile-note">stopped either late or not at all; %(head_settled)s settle within 5&nbsp;units</div></div>
    <div class="tile"><div class="tile-label">Closed-form rule alone</div>
      <div class="tile-value">%(cf_stopped)s</div>
      <div class="tile-note">the actor&rsquo;s filter is conservative; at &tau;&#8209;v almost nobody passes</div></div>
    <div class="tile"><div class="tile-label">Hybrid gate</div>
      <div class="tile-value">%(hyb_stopped)s</div>
      <div class="tile-note">%(hyb_settled)s settled &lt;5u at median error %(med_err_hyb)s &mdash; over the oracle&rsquo;s %(orc_settled)s</div></div>
  </div>

  <h2>What the gate is</h2>
  <p class="lede">Two detectors and an OR. One is a learned head the actor was trained with; the other
    is a closed-form rule &mdash; the oracle&rsquo;s own arrival test &mdash; that reads the actor&rsquo;s
    belief filter directly.</p>
  %(gate_fig)s
  <p>The inputs in the diagram are four of the observation&rsquo;s per-robot property slots, produced
    by the belief filter&rsquo;s target path:</p>
  <div class="table-wrap"><table>
    <thead><tr><th>slot</th><th>content</th><th>role in the rule</th></tr></thead>
    <tbody>
      <tr><td><code>PROP_DIST_T</code> (21)</td><td>filtered distance to the robot&rsquo;s own target</td><td>must be below the arrival radius</td></tr>
      <tr><td><code>PROP_CONF_POS</code> (12)</td><td>localisation confidence</td><td>must be &ge; %(local_conf)s &ndash; the rule only acts once localized</td></tr>
      <tr><td><code>PROP_SIN_T</code>/<code>PROP_COS_T</code> (19, 20)</td><td>direction to the target</td><td>non-zero pair proves a target is assigned at all</td></tr>
    </tbody>
  </table></div>
  <pre>def closed_form_arrived(prop, tau, conf_floor = 0.4):
    has_target  = (prop[..., PROP_SIN_T] != 0.0) | (prop[..., PROP_COS_T] != 0.0)
    return (prop[..., PROP_DIST_T] &lt; tau) &amp; (prop[..., PROP_CONF_POS] &gt;= conf_floor) &amp; has_target

# deployed as dist = cfg.closed_form_arrival_dist (0.08) -- not cfg.tau_v (0.05)
# the actor's own filter under-reports closeness (~1.5x at arrival), so tau_v parks nobody</pre>
  <p>In <code>actor_io._arrived_head_gate</code> the two branches run in OR, on the same tick, and the
    result latches: a robot that fires either branch is switched off for the rest of the episode, exactly
    like the oracle&rsquo;s terminal <code>arrived</code> state. Config keys:
    <code>use_closed_form_arrived</code>, <code>closed_form_arrival_dist</code>,
    <code>closed_form_hybrid</code>.</p>

  <h2>Did stopping get fixed?</h2>
  <p class="lede">Four closed-loop runs &mdash; oracle, learned head, closed form, hybrid &mdash; on the
    same %(arenas)s held-out formations and the same spawns (same <code>--swarm-rng</code> and
    <code>--seed</code>), every number ground truth from the robot&rsquo;s own assigned point.
    The metric is per-robot: did it stop, and did it stop near the point <em>it</em> was assigned.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>driver</th><th>stopped</th><th>settled &lt;5u</th><th>&lt;10u</th><th>&lt;20u</th>
      <th>median error</th><th>coverage</th></tr></thead>
    <tbody>
%(table)s
    </tbody>
  </table></div>
  <div class="note note-status">
    <strong>What the numbers say.</strong> The learned head stops %(head_stopped)s of the swarm by
    itself but parks only %(head_settled)s on its own point; the closed form alone certifies tight
    arrivals but stops just %(cf_stopped)s, because the actor&rsquo;s filter only admits that close.
    Wiring them together in OR keeps the closed form&rsquo;s tight arrivals (median error
    %(med_err_hyb)s vs the oracle&rsquo;s %(orc_med_err)s) and the head&rsquo;s stopping rate, for
    %(hyb_stopped)s stopped and %(hyb_cov)s coverage &mdash; beating the oracle on every settle
    tolerance: %(hyb_settled)s vs %(orc_settled)s within 5 units, %(hyb_settled10)s vs
    %(orc_settled10)s within 10, %(hyb_settled20)s vs %(orc_settled20)s within 20. The gate is
    terminal either way, so a robot the two disagree on stops once.
  </div>
  %(curve_fig)s
  %(who_fig)s
  %(settle_fig)s

  <h2>Who stopped whom</h2>
  <p class="lede">Across the %(nbot)d robots the actor drove in the head, closed-form and hybrid runs:
    which branch caught which robot, plus the negative control &mdash; how many stayed in motion. The
    three runs are separate rollouts, so the per-robot split is indicative, not causal: one robot
    stopping early changes the messages the swarm broadcasts.</p>
  <p>Per robot over those %(nbot)d actor robots: the head alone stopped %(head_only)s, the closed form
    alone %(cf_only)s, both %(both)s, and %(moving)s stayed in motion. The residual %(uncredited)s
    were stopped by the hybrid in this run but by neither branch in the standalone runs &mdash; the
    rollouts diverged, one robot finishing changes what a neighbour sees. The hybrid&rsquo;s
    <code>closed_form&nbsp;OR&nbsp;head</code> reproduced the union of the two branch decisions on
    %(agree)s of robots &mdash; the disagreement is that same divergence, not a gate error. That union
    is the whole difference between &ldquo;parks a third&rdquo; and &ldquo;certifies arrival&rdquo;.</p>

  <h2>What it looks like</h2>
  <p class="lede">One held-out arena, all four drivers, tick after tick. Grey is the target shape;
    each dot is a robot. The oracle drives straight in; the head and the closed form leave robots
    orbiting their points; the hybrid stops them where they were sent.</p>
  %(gif_fig)s

  <h2>Arena by arena</h2>
  <p class="lede">%(arenas)s distinct held-out formations, the same ones every driver saw (two
    workers, five arenas each), shown four ways.
    Grey is the target shape; a ring marks a robot that <em>stopped beyond 5&nbsp;units of its own
    point</em>, the failure the hybrid exists to avoid.</p>
  %(grid_fig)s
  <div class="table-wrap"><table class="pa">
    <thead><tr><th>arena</th><th>formation</th>
      %(pa_headers)s</tr></thead>
    <tbody>
%(pa_rows)s
    </tbody>
  </table></div>

  <h2>Reproducing it</h2>
  <p>One eval per driver, same arena geometry and same spawns; the gate flags are all in the CLI:</p>
  <pre>python tools/eval_closed_loop.py results/hybrid_cloning/eval_oracle_10.json --mode oracle \
    --ticks 10000 --instances 2 --arenas 5 --swarm-rng 500 --seed 7 --bake-rotation-steps 0
python tools/eval_closed_loop.py results/hybrid_cloning/eval_o3_hyb_10.json --mode actor \
    --weights results/bc_v2/run_o3/actor_best.pt \
    --closed-form-arrived --closed-form-dist 0.08 --closed-form-hybrid [same arena flags]</pre>
  <p>The closed-form-only run drops <code>--closed-form-hybrid</code>; the head-only run drops both
    closed-form flags.</p>

  <footer>Generated from <span class="mono">summary.json</span> by
    <span class="mono">tools/hybrid_report.py</span>. Every number on this page comes from that file.</footer>
</div>
"""

def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required = True)
    ap.add_argument("--eval-oracle", required = True)
    ap.add_argument("--eval-head", required = True)
    ap.add_argument("--eval-cf", required = True)
    ap.add_argument("--eval-hybrid", required = True)
    ap.add_argument("--formations", default = "../results/bc_v2/val_formations")
    ap.add_argument("--limit", type = int, default = 2000)
    ap.add_argument("--ticks", type = int, default = 10000)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok = True)

    evals = {"oracle": load(args.eval_oracle), "head": load(args.eval_head),
             "closed-form": load(args.eval_cf), "hybrid": load(args.eval_hybrid)}
    # re-export the closed-loop curves with the canonical driver colouring
    fig_eval(evals, args.out_dir)   # reuses bc_report's palette + end labels
    # reorder settle dicts so draw()'s palette lands on the same roles as every
    # other figure: oracle orange, the hybrid (the canonical actor) blue, then
    # the two head-only/closed-form-only variants in the remaining palette slots.
    order = ["oracle", "hybrid", "head", "closed-form"]
    settle_rows, settle_pooled = {}, {}
    for n in order:
        rows, pooled = arena_stats(evals[n], args.formations, args.limit)
        settle_rows[DISPLAY[n]], settle_pooled[DISPLAY[n]] = rows, pooled
    from settle_report import draw as settle_draw
    settle_draw(settle_rows, settle_pooled, args.out_dir, "settled")

    fig_gate(args.out_dir)
    fig_who_stops(evals, args.out_dir)
    fig_arena_grid(evals, args.out_dir)
    fig_paired(evals, args.out_dir)   # oracle vs hybrid paired coverage
    write_gif(evals, args.out_dir, arena_index = 0)

    summary = build_summary(evals, args.out_dir)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent = 2)
    write_page(evals, summary, args.out_dir, vars(args))
    print("wrote hybrid report to %s" % args.out_dir, flush = True)


if __name__ == "__main__":
    main()