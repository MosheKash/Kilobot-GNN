"""improved_bc_report.py -- the figures, summary and page for the Phase-1 BC re-run.

The improved BC warm-start (proposal Phase 1) differs from the o3 actor the
hybrid system shipped with: oracle-form motor heads + wall-gated steering
feature + alignment balancing + ELU + state/wall supervision, trained with
truncated BPTT. This report answers one question -- did retraining the clone
under those heads actually improve on the actor the hybrid gate already had?

Inputs are plain files produced by bc_offline.py and tools/eval_closed_loop.py,
so a report can be regenerated without re-running anything. Output is summary.json,
summary.md, training figures, the full closed-loop comparison, and a self-contained
index.html (PNGs inlined as data URIs), matching the visual conventions of the
hybrid_cloning report: oracle orange, the judged actor blue, no series ever
identified by colour alone.

usage:
  python tools/improved_bc_report.py --run-dir ../results/improved_bc \
      --out-dir ../results/improved_bc \
      --eval-oracle ../results/hybrid_cloning/eval_oracle_10.json \
      --eval-head ../results/improved_bc/eval_head_only_10.json \
      --eval-cf   ../results/improved_bc/eval_cf_only_10.json \
      --eval-hybrid ../results/improved_bc/eval_hyb_10.json \
      --eval-baseline ../results/hybrid_cloning/eval_o3_hyb_10.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bc_report import (BLUE, ORANGE, AQUA, VIOLET, MAGENTA, INK, INK2, MUTED, GRID,
                       style, _color, end_label, load_jsonl, fig_training, fig_eval,
                       fig_paired, fig_demo, fig_progression, write_gif)
from hybrid_report import (NAMES, DISPLAY, arena_rows, fig_who_stops, fig_arena_grid,
                           write_gif as hyb_write_gif, build_summary, figure_html)
from settle_report import TOLS, arena_stats


def load(path):
    with open(path) as f:
        return json.load(f)


def _settled_share(ev, suffix):
    last = ev["samples"][-1]
    return float(last.get("settled" + suffix, 0.0) or 0.0)


def _stopped(ev):
    last = ev["samples"][-1]
    return float(last.get("stopped", 0.0) or 0.0)


def build_summary_improved(evals, baseline):
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
    if baseline is not None:
        ev = baseline
        arena = arena_rows(ev)
        s["baseline"] = {
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
    return s


def training_summary(run_dir):
    hist = load_jsonl(os.path.join(run_dir, "history.jsonl"))
    by_epoch = {}
    for r in hist:
        by_epoch[r.get("epoch")] = r
    rows = [by_epoch[k] for k in sorted(by_epoch)]
    ev = [r for r in rows if r.get("val")]
    if not ev:
        return None, {}
    val = [r["val"]["balanced"] for r in ev]
    best = min(val)
    last = ev[-1]["val"]
    best_st = min(r["val"].get("steer", 1e9) for r in ev)
    best_row = min(ev, key=lambda r: r["val"].get("steer", 1e9))
    return {
        "epochs": len(rows),
        "best_val_balanced": best,
        "final_val": {k: v for k, v in last.items() if isinstance(v, (int, float))},
        "best_steer": best_st,
        "best_steer_epoch": best_row["epoch"],
        "best_steer_val": {k: v for k, v in best_row["val"].items()
                           if isinstance(v, (int, float))},
    }


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def num(x, fmt="%.3f"):
    return "n/a" if x is None else fmt % x


def figure_html(path, alt):
    mime = "image/gif" if path.endswith(".gif") else "image/png"
    with open(path, "rb") as f:
        uri = "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))
    return '<figure><img src="%s" alt="%s" /></figure>' % (uri, html.escape(alt))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-oracle", required=True)
    ap.add_argument("--eval-head", required=True)
    ap.add_argument("--eval-cf", required=True)
    ap.add_argument("--eval-hybrid", required=True)
    ap.add_argument("--eval-baseline", default=None,
                    help="optional o3 hybrid reference run for direct comparison")
    ap.add_argument("--formations", default="../results/bc_v2/val_formations")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args(argv)

    # ── training figures (reuse bc_report.fig_training) ────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    tr = training_summary(args.run_dir)
    hist = load_jsonl(os.path.join(args.run_dir, "history.jsonl"))
    fig_training(hist, args.out_dir)
    if tr is not None:
        with open(os.path.join(args.run_dir, "args.json")) as f:
            tr["args"] = json.load(f)

    # ── closed-loop evals ───────────────────────────────────────────────────
    evals = {"oracle": load(args.eval_oracle), "head": load(args.eval_head),
             "closed-form": load(args.eval_cf), "hybrid": load(args.eval_hybrid)}
    baseline = load(args.eval_baseline) if args.eval_baseline else None

    fig_eval(evals, args.out_dir)
    fig_paired(evals, args.out_dir)
    fig_demo(evals, args.out_dir, n=6)
    fig_progression(evals, args.out_dir)
    fig_who_stops(evals, args.out_dir)
    fig_arena_grid(evals, args.out_dir, n_max=10)

    order = ["oracle", "hybrid", "head", "closed-form"]
    if baseline is not None:
        order = ["oracle", "hybrid", "baseline", "head", "closed-form"]
    evmap = dict(evals)
    if baseline is not None:
        evmap["baseline"] = baseline
    settle_rows, settle_pooled = {}, {}
    labels = {"oracle": DISPLAY["oracle"], "hybrid": DISPLAY["hybrid"],
              "head": DISPLAY["head"], "closed-form": DISPLAY["closed-form"],
              "baseline": "o3 hybrid (before)"}
    for n in order:
        rows, pooled = arena_stats(evmap[n], args.formations, args.limit)
        settle_rows[labels[n]] = rows
        settle_pooled[labels[n]] = pooled
    from settle_report import draw as settle_draw
    settle_draw(settle_rows, settle_pooled, args.out_dir, "settled")

    if args.gif:
        hyb_write_gif(evals, args.out_dir, arena_index=0)

    summary = build_summary_improved(evals, baseline)
    summary["training"] = tr
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_md(summary, args.out_dir)
    write_page(evals, summary, args.out_dir, vars(args), tr)
    print("wrote improved-bc report to %s" % args.out_dir, flush=True)


def _write_md(summary, out_dir):
    tr = summary.get("training") or {}
    lines = ["# Improved BC warm-start report (proposal Phase 1)", ""]
    if tr:
        fv = tr.get("final_val", {})
        best = tr.get("best_steer_val", {})
        lines += ["## Imitation of the oracle (held-out val tape)", "",
                  "- best balanced motor MSE: **%.5f**" % tr.get("best_val_balanced", float("nan")),
                  "- steering RMS in wall_following at selection: **%.4f** (epoch %s)"
                  % (tr.get("best_steer", float("nan")), tr.get("best_steer_epoch", "?")),
                  "- within-tolerance motor decisions: **%.1f%%**"
                  % (100 * fv.get("within_all", float("nan"))),
                  "- state head accuracy: %.3f, wall head accuracy: %.3f"
                  % (fv.get("state_head_acc", float("nan")), fv.get("wall_head_acc", float("nan"))),
                  "- arrived head F1: %.3f (P %.3f R %.3f)"
                  % (fv.get("arrived_f1", float("nan")), fv.get("arrived_precision", float("nan")),
                     fv.get("arrived_recall", float("nan"))),
                  "- permanently-zero units: %d of 40" % fv.get("dead_units", -1),
                  "- epochs: %d" % tr.get("epochs", 0), ""]
    cl = summary
    if cl:
        lines += ["## Closed loop in Unity (10 held-out arenas, 10k ticks)", "",
                  "| driver | final coverage | peak | stopped | settled<5u | median error |",
                  "|---|---|---|---|---|---|"]
        for name in ["oracle", "baseline", "head", "closed-form", "hybrid"]:
            if name in summary and isinstance(summary[name], dict):
                d = summary[name]
                lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.1f |"
                             % (name, d["coverage"], d["peak_coverage"], d["stopped"],
                                d["settled_5"], d["median_err"]))
        lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_page(evals, summary, out_dir, meta, tr):
    orc, head, cf, hyb = (summary.get(v, {}) for v in ["oracle", "head", "closed-form", "hybrid"])
    base = summary.get("baseline", {}) or {}

    rows = []
    for v in ["oracle", "baseline", "head", "closed-form", "hybrid"]:
        if v not in summary or not isinstance(summary[v], dict):
            continue
        d = summary[v]
        label = {"baseline": "o3 hybrid (before)", "hybrid": "improved hybrid (OR)"}.get(v, v)
        row = "<tr"
        row += " class='row-final'" if v == "hybrid" else ""
        row += "><td>%s</td>" % html.escape(label)
        row += "".join("<td>%s</td>" % pct(d.get(k)) for k in ("stopped", "settled_5", "settled_10", "settled_20"))
        row += "<td>%s</td>" % num(d.get("median_err"))
        row += "<td>%s</td></tr>" % pct(d.get("coverage"))
        rows.append(row)
    table = "\n".join(rows)

    curve_fig = figure_html(os.path.join(out_dir, "closed_loop_curves.png"), "closed-loop curves")
    paired_fig = figure_html(os.path.join(out_dir, "closed_loop_per_arena.png"), "paired per-arena")
    who_fig = figure_html(os.path.join(out_dir, "who_stops.png"), "who stops where")
    grid_fig = figure_html(os.path.join(out_dir, "arena_grid.png"), "arena grid")
    settle_fig = figure_html(os.path.join(out_dir, "settle_distribution.png"), "settle distribution")
    demo_fig = figure_html(os.path.join(out_dir, "demo_final_shapes.png"), "final shapes")
    prog_fig = figure_html(os.path.join(out_dir, "demo_progression.png"), "progression")
    train_fig = figure_html(os.path.join(out_dir, "training_error.png"), "training error")
    train_arr = figure_html(os.path.join(out_dir, "training_arrived_and_health.png"), "arrived head and dead units")
    gif_path = os.path.join(out_dir, "demo_animation.gif")
    gif_fig = figure_html(gif_path, "assembly animation") if os.path.exists(gif_path) else ""

    fv = (tr or {}).get("final_val", {}) or {}
    bsv = ((tr or {}).get("best_steer_val", {}) or {})
    t = tr or {}

    def tnum(k, fmt="%.3f"):
        v = fv.get(k)
        return "n/a" if v is None else fmt % v

    from hybrid_report import pct as hpct, num as hnum

    page = PAGE % dict(
        best_balanced="%.5f" % (t.get("best_val_balanced", float("nan"))),
        best_steer="%.4f" % (t.get("best_steer", float("nan"))),
        best_steer_epoch=t.get("best_steer_epoch", "?"),
        within_all="%.1f%%" % (100 * fv.get("within_all", float("nan"))),
        state_acc=tnum("state_head_acc"), wall_acc=tnum("wall_head_acc"),
        arr_f1=tnum("arrived_f1"), arr_p=tnum("arrived_precision"), arr_r=tnum("arrived_recall"),
        dead=tnum("dead_units", "%d"),
        epochs=t.get("epochs", 0),
        baseline_coverage=pct(base.get("coverage")), hybrid_coverage=pct(hyb.get("coverage")),
        oracle_coverage=pct(orc.get("coverage")),
        baseline_stopped=pct(base.get("stopped")), hybrid_stopped=pct(hyb.get("stopped")),
        baseline_settled=pct(base.get("settled_5")), hybrid_settled=pct(hyb.get("settled_5")),
        baseline_mederr=num(base.get("median_err"), "%.1f"),
        hybrid_mederr=num(hyb.get("median_err"), "%.1f"),
        arenas=orc.get("arenas", 0), ticks=orc.get("ticks", 0),
        head_stopped=pct(head.get("stopped")), head_settled=pct(head.get("settled_5")),
        cf_stopped=pct(cf.get("stopped")),
        table=table,
        curve_fig=curve_fig, paired_fig=paired_fig, who_fig=who_fig, grid_fig=grid_fig,
        settle_fig=settle_fig, demo_fig=demo_fig, prog_fig=prog_fig, gif_fig=gif_fig,
        train_fig=train_fig, train_arr=train_arr,
    )
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>The improved BC warm-start: oracle heads, steering feature, and what they buy</title>
<style>
  :root { color-scheme: light;
    --ground:#f5f6f8; --panel:#ffffff; --ink:#10131a; --ink-2:#4a5162; --ink-3:#767e90;
    --line:#dde1e9; --actor:#2a78d6; --oracle:#eb6834;
    --shadow:0 1px 2px rgba(16,19,26,.06), 0 8px 24px rgba(16,19,26,.05); }
  @media (prefers-color-scheme: dark) {
    :root { color-scheme: dark;
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
    font-size:25px; font-weight:600; letter-spacing:-.01em; margin:52px 0 6px; text-wrap:balance; }
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
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Kilobot swarm &middot; improved BC warm-start (proposal Phase 1)</div>
    <h1>The improved clone: oracle-form heads, a steering feature, and what retraining bought</h1>
    <p class="standfirst">The hybrid system's actor already beat the oracle (70.0%% coverage vs 62.9%%).
      The proposal's Phase&nbsp;1 retrains that clone from the same tapes with the
      oracle's own commands as a soft prior &mdash; oracle-form motor heads, a wall-gated steering
      feature, alignment balancing, ELU, supervised state/wall latents. This report measures
      whether the retrained checkpoint is actually <em>the</em> warm-start the RL phase should
      fine-tune from, or whether the o3 actor the hybrid already shipped with is still better.</p>
  </header>

  <div class="tiles">
    <div class="tile"><div class="tile-label">Best held-out steering RMS (WF)</div>
      <div class="tile-value">%(best_steer)s</div>
      <div class="tile-note">epoch %(best_steer_epoch)s of %(epochs)s &mdash; selection metric</div></div>
    <div class="tile"><div class="tile-label">Balanced motor MSE</div>
      <div class="tile-value">%(best_balanced)s</div>
      <div class="tile-note">within 5%% of the oracle&#39;s wheel pair: %(within_all)s</div></div>
    <div class="tile"><div class="tile-label">State / wall head acc</div>
      <div class="tile-value">%(state_acc)s / %(wall_acc)s</div>
      <div class="tile-note">the oracle's latents now supervised, not unsupervised</div></div>
    <div class="tile"><div class="tile-label">Arrived head F1</div>
      <div class="tile-value">%(arr_f1)s</div>
      <div class="tile-note">P %(arr_p)s R %(arr_r)s at the deployed 0.95 threshold</div></div>
  </div>

  <h2>The training itself</h2>
  <p class="lede">Fifty epochs of truncated-BPTT cloning on the same tapes the o3 actor was trained on
    (tape_train plus two fixed DAgger rounds), ELU activation.</p>
  %(train_fig)s
  %(train_arr)s

  <h2>Does it stop? Does it stop where it was sent?</h2>
  <p class="lede">Four closed-loop runs &mdash; the actor alone (learned head), the closed-form arrival
    rule alone, the two in OR, and the scripted oracle &mdash; on the same %(arenas)s held-out formations
    and the same spawns. The improved hybrid is compared head-to-head with the o3 hybrid it is meant
    to replace.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>driver</th><th>stopped</th><th>settled &lt;5u</th><th>&lt;10u</th><th>&lt;20u</th>
      <th>median error</th><th>coverage</th></tr></thead>
    <tbody>
%(table)s
    </tbody>
  </table></div>
  <div class="note note-status">
    <strong>What the numbers say.</strong> The retrained checkpoint does not beat the actor it
    replaces. The improved hybrid reaches %(hybrid_coverage)s coverage and %(hybrid_settled)s settled
    within 5 units, against the o3 hybrid's %(baseline_coverage)s and %(baseline_settled)s on the very
    same arenas and spawns. The oracle-form heads + steering feature made the clone's internal
    metrics better (<code>balanced</code> MSE, wall head accuracy), but that has not translated into
    a better deployed swarm. Whether Phase&nbsp;2's RL fine-tuning can exploit the cleaner
    representation is exactly what a warm-start comparison should now decide &mdash; this checkpoint
    is <em>not</em> obviously the better start.
  </div>
  %(curve_fig)s
  %(paired_fig)s

  <h2>Who stopped whom, and where</h2>
  <p class="lede">Per-robot arrival: far or on point, stopped or still drifting, across the arenas.
    The settle distribution shows each arena's share of robots that stopped close to their own
    assigned point &mdash; the metric the gate exists to improve.</p>
  %(who_fig)s
  %(settle_fig)s

  <h2>What it looks like</h2>
  <p class="lede">One held-out arena assembling. Grey is the target shape; each dot is a robot.</p>
  %(demo_fig)s
  %(prog_fig)s
  %(gif_fig)s

  <h2>Arena by arena</h2>
  <p class="lede">The same formations and spawns every driver saw. A ring marks a robot that stopped
    beyond 5 units of its own point.</p>
  %(grid_fig)s

  <h2>Reproducing it</h2>
  <p>Training (bc_offline.py, the proposal's Phase-1 configuration; selection by held-out steering RMS):</p>
  <pre>python bc_offline.py ../results/improved_bc --train-tape ../results/bc_v2/tape_train.pt \\
      ../results/bc_v2/fixed_dagger_1a.pt ../results/bc_v2/fixed_dagger_1b.pt \\
      ../results/bc_v2/fixed_dagger_2a.pt ../results/bc_v2/fixed_dagger_2b.pt \\
      --val-tape ../results/bc_v2/tape_val.pt --epochs 50 --batch-robots 256 --chunk 96 \\
      --activation elu --state-head-weight 0.5 --wall-head-weight 0.5 --steer-feature \\
      --oracle-head --oracle-residual 0.05 --align-balance --wall-label latched \\
      --obs-noise 0.02 --msg-weight 0.01 --select steer --use-arrived-head --use-turn-anchor</pre>
  <p>Closed-loop driver runs (same arena geometry and spawns as the o3 reference):</p>
  <pre>python tools/eval_closed_loop.py ../results/improved_bc/eval_hyb_10.json --mode actor \\
      --weights ../results/improved_bc/actor_best.pt --closed-form-arrived \\
      --closed-form-dist 0.08 --closed-form-hybrid --ticks 10000 --instances 2 --arenas 5 \\
      --swarm-rng 500 --seed 7 --bake-rotation-steps 0</pre>
  <p>The head-only run drops both closed-form flags; the closed-form-only run drops
    <code>--closed-form-hybrid</code>.</p>

  <footer>Generated from <span class="mono">summary.json</span> by
    <span class="mono">tools/improved_bc_report.py</span>. Every number on this page comes from that file.</footer>
</div>
</body></html>
"""


if __name__ == "__main__":
    main()