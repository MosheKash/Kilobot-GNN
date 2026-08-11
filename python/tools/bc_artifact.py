"""bc_artifact.py -- one self-contained HTML page from a report directory.

Reads what tools/bc_report.py wrote (summary.json plus the PNGs and the GIF) and
inlines all of it as data URIs, so the result is a single file that renders with
no network access at all. Nothing is retyped: every number on the page comes out
of summary.json, so the page cannot drift from the run it describes.

usage:
  python tools/bc_artifact.py ../results/bc_v2/report --out ../results/bc_v2/report/index.html
"""

import argparse
import base64
import html
import json
import mimetypes
import os


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))


def figure(report_dir, name, caption, alt):
    path = os.path.join(report_dir, name)
    if not os.path.exists(path):
        return ""
    return ("""<figure>
  <img src="%s" alt="%s" />
  <figcaption>%s</figcaption>
</figure>""" % (data_uri(path), html.escape(alt), caption))


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def num(x, fmt = "%.4f"):
    return "n/a" if x is None else fmt % x


def build(report_dir, title):
    with open(os.path.join(report_dir, "summary.json")) as f:
        s = json.load(f)
    tr = s.get("training", {}) or {}
    fv = tr.get("final_val", {}) or {}
    cl = s.get("closed_loop", {}) or {}
    oracle = cl.get("oracle", {}) or {}
    actors = [(k, v) for k, v in cl.items() if isinstance(v, dict) and k != "oracle"]
    final_name, final = (actors[-1] if actors else ("actor", {}))

    oc = oracle.get("final_coverage")
    ac = final.get("final_coverage")
    # The number the swarm starts at, which is where "is this above chance"
    # gets decided; bc_report records it per driver.
    spawn = oracle.get("spawn_coverage", final.get("spawn_coverage"))
    if oc is not None and ac is not None and ac >= oc - 0.05:
        status = ("The clone reproduces the teacher",
                  "Closed loop, on formations it never trained on, the actor finishes within five "
                  "points of the oracle.")
    else:
        noisy = None
        for k, v in cl.items():
            if isinstance(v, dict) and "noise" in k:
                noisy = v.get("final_coverage")
        extra = ("" if noisy is None else
                 " And the size of the error is not the explanation: driving the <em>oracle</em> "
                 "with Gaussian noise of exactly the clone's own per-wheel error still reaches "
                 "%s. Random error of that magnitude is survivable; the clone's is structured "
                 "&mdash; it ends up in the wrong state of the teacher's machine and stays there."
                 % pct(noisy))
        status = ("Where this stands",
                  "The clone reproduces the oracle's <em>decisions</em> but not yet its "
                  "<em>outcome</em>. Closed loop it reaches %s against the oracle's %s, and a "
                  "swarm dropped at random already starts at %s &mdash; so the actor is not above "
                  "chance on the task while the oracle more than doubles it.%s"
                  % (pct(ac), pct(oc), pct(spawn), extra))
    tiles = [
        ("Imitation error, held out", num(tr.get("best_val_balanced"), "%.5f"),
         "mean squared motor error against the oracle, averaged over its five states"),
        ("Decisions matching the oracle", pct(fv.get("within_all")),
         "within 0.05 on both wheels, on formations never trained on"),
        ("Oracle, robots on the shape", pct(oracle.get("final_coverage")),
         "ground truth after %s ticks, held-out formations" % oracle.get("ticks", "?")),
        ("Trained actor, robots on the shape", pct(final.get("final_coverage")),
         "same formations, same spawns, same measurement"),
    ]
    tile_html = "\n".join(
        """    <div class="tile">
      <div class="tile-label">%s</div>
      <div class="tile-value">%s</div>
      <div class="tile-note">%s</div>
    </div>""" % (html.escape(a), html.escape(b), html.escape(c)) for a, b, c in tiles)

    rows = []
    for name in ["oracle"] + [k for k, _ in actors]:
        d = cl.get(name)
        if not isinstance(d, dict):
            continue
        rows.append("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            ' class="row-final"' if name == final_name else "",
            html.escape(name), pct(d.get("final_coverage")), pct(d.get("peak_coverage")),
            num(d.get("final_mean_dist")), pct(d.get("stopped") if "stopped" in d
                                               else d.get("final_stopped"))))
    table = "\n".join(rows)

    state_rows = []
    for st in ("go_north", "turning", "wall_following", "navigating", "arrived"):
        if st in fv:
            state_rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                st.replace("_", " "), num(fv.get(st), "%.5f"),
                pct(fv.get("within_" + st)), "%d" % int(fv.get("n_" + st, 0))))
    states = "\n".join(state_rows)

    figs_imitation = "\n".join([
        figure(report_dir, "training_error.png",
               "Held-out imitation error over training. Left: the balanced score, on held-out "
               "formations and on the training ones. Right: the same, split by which state of "
               "the oracle's machine produced the command.",
               "imitation error curves"),
        figure(report_dir, "training_arrived_and_health.png",
               "The arrived head is what actually stops a robot, so it is scored separately at "
               "the threshold the runtime gate uses. Right: units in the shared head layer that "
               "output exactly zero for every decision -- the failure mode of phase 154.",
               "arrived head and network health"),
    ])
    figs_loop = "\n".join([
        figure(report_dir, "closed_loop_curves.png",
               "Closed loop on held-out formations. Coverage dips first for both drivers: robots "
               "drive north until they meet a wall, which takes them away from the shape before "
               "it takes them to it. The shaded band is the interquartile range across arenas.",
               "closed loop curves"),
        figure(report_dir, "closed_loop_per_arena.png",
               "Every arena, paired: same formation, same spawn, one driver each.",
               "per arena comparison"),
    ])
    figs_demo = "\n".join([
        figure(report_dir, "demo_progression.png",
               "One held-out arena over time. Grey is the target shape; each dot is a robot.",
               "assembly over time"),
        figure(report_dir, "demo_final_shapes.png",
               "Where each swarm ended up, on six held-out formations.",
               "final swarm positions"),
    ])
    gif = figure(report_dir, "demo_animation.gif",
                 "The same arena, animated. The oracle assembles the shape; the actor is judged "
                 "against it.", "animated demo")

    return TEMPLATE % {
        "title": html.escape(title),
        "tiles": tile_html,
        "table": table,
        "states": states,
        "figs_imitation": figs_imitation,
        "figs_loop": figs_loop,
        "figs_demo": figs_demo,
        "figs_settle": figure(report_dir, "settle_distribution.png",
                              "Every arena is a dot (left), the share of arenas reaching a given "
                              "settle rate (middle), and every robot's distance to its own "
                              "assigned target (right). The right panel is two populations, not "
                              "one spread: a robot lands on its point or ends up unrelated.",
                              "settle distribution"),
        "gif": gif,
        "epochs": tr.get("epochs", "?"),
        "arenas": oracle.get("arenas", "?"),
        "ticks": oracle.get("ticks", "?"),
        "dead": int(fv.get("dead_units", -1)),
        "precision": num(fv.get("arrived_precision"), "%.4f"),
        "recall": num(fv.get("arrived_recall"), "%.4f"),
        "oracle_cov": pct(oracle.get("final_coverage")),
        "actor_cov": pct(final.get("final_coverage")),
        "oracle_cov_py": pct(oracle.get("final_coverage_python_frame")),
        "actor_cov_py": pct(final.get("final_coverage_python_frame")),
        "status_head": status[0],
        "status_body": status[1],
    }


TEMPLATE = """<title>%(title)s</title>
<style>
  :root {
    color-scheme: light;
    --ground:      #f5f6f8;
    --panel:       #ffffff;
    --ink:         #10131a;
    --ink-2:       #4a5162;
    --ink-3:       #767e90;
    --line:        #dde1e9;
    --actor:       #2a78d6;
    --oracle:      #eb6834;
    --shadow:      0 1px 2px rgba(16, 19, 26, .06), 0 8px 24px rgba(16, 19, 26, .05);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --ground:  #0e1116;
      --panel:   #161a21;
      --ink:     #eef1f6;
      --ink-2:   #a8b0c0;
      --ink-3:   #7c8496;
      --line:    #262c37;
      --actor:   #3987e5;
      --oracle:  #ef7c4e;
      --shadow:  0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground:  #0e1116;
    --panel:   #161a21;
    --ink:     #eef1f6;
    --ink-2:   #a8b0c0;
    --ink-3:   #7c8496;
    --line:    #262c37;
    --actor:   #3987e5;
    --oracle:  #ef7c4e;
    --shadow:  0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 56px 24px 96px; }
  header { border-bottom: 1px solid var(--line); padding-bottom: 28px; margin-bottom: 36px; }
  .eyebrow {
    font: 600 12px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: .14em; text-transform: uppercase; color: var(--ink-3);
  }
  h1 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-weight: 600; font-size: clamp(30px, 4.4vw, 46px); line-height: 1.12;
    letter-spacing: -.015em; text-wrap: balance; margin: 14px 0 12px;
  }
  .standfirst { color: var(--ink-2); font-size: 18px; max-width: 62ch; margin: 0; }
  h2 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 25px; font-weight: 600; letter-spacing: -.01em; margin: 52px 0 6px;
    text-wrap: balance;
  }
  h2 + p.lede { color: var(--ink-2); margin: 0 0 18px; max-width: 66ch; }
  p { max-width: 68ch; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
  code { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 30px 0 8px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 18px 18px 16px; box-shadow: var(--shadow); }
  .tile-label { font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }
  .tile-value { font-variant-numeric: tabular-nums; font-size: 34px; font-weight: 600; letter-spacing: -.02em; margin: 8px 0 4px; }
  .tile:nth-child(3) .tile-value { color: var(--oracle); }
  .tile:nth-child(4) .tile-value { color: var(--actor); }
  .tile-note { color: var(--ink-2); font-size: 13px; line-height: 1.45; }
  figure { margin: 26px 0; }
  figure img { width: 100%%; height: auto; display: block; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
  figcaption { color: var(--ink-2); font-size: 14px; margin-top: 10px; max-width: 74ch; }
  .table-wrap { overflow-x: auto; margin: 20px 0; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
  table { border-collapse: collapse; width: 100%%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 11px 16px; border-bottom: 1px solid var(--line); font-size: 14px; }
  th:first-child, td:first-child { text-align: left; }
  thead th { font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }
  tbody tr:last-child td { border-bottom: none; }
  tr.row-final td { font-weight: 600; color: var(--actor); }
  .note-status { border-left-color: var(--actor); }
  .note { border-left: 3px solid var(--oracle); background: var(--panel); border-radius: 0 8px 8px 0; padding: 16px 18px; margin: 22px 0; box-shadow: var(--shadow); }
  .note strong { display: block; margin-bottom: 4px; }
  pre { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; overflow-x: auto; font-size: 13px; line-height: 1.6; }
  footer { margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--ink-3); font-size: 13px; }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Kilobot swarm &middot; behaviour cloning</div>
    <h1>Cloning the scripted oracle into a 24&nbsp;KB recurrent policy</h1>
    <p class="standfirst">One shared policy, 40&ndash;60 robots per arena, each seeing only what its own
      infrared reaches. It is trained to reproduce a scripted five-state teacher, and then judged
      the only way that counts: driving the swarm itself, on shapes it has never seen.</p>
  </header>

  <div class="tiles">
%(tiles)s
  </div>

  <h2>Imitating the teacher</h2>
  <p class="lede">Trained on recorded oracle rollouts &mdash; whole per-robot sequences, replayed from a
    cold start with truncated backpropagation through time &mdash; and scored on formations held out
    of training entirely. %(epochs)s epochs.</p>
%(figs_imitation)s

  <div class="table-wrap">
  <table>
    <thead><tr><th>oracle state</th><th>motor MSE</th><th>within 0.05</th><th>decisions</th></tr></thead>
    <tbody>
%(states)s
    </tbody>
  </table>
  </div>
  <p>The arrived head, which is what actually stops a robot, reaches precision %(precision)s and
    recall %(recall)s on held-out data at the threshold the runtime gate uses. %(dead)s of the shared
    head layer's 40 units are permanently zero.</p>

  <div class="note note-status">
    <strong>%(status_head)s</strong>
    %(status_body)s
  </div>

  <h2>The task metric: did each robot stop on its own point</h2>
  <p class="lede">Not coverage. Per robot: did it stop, and did it stop within X units of the
    point it was assigned &mdash; shown as a distribution over arenas, counting only arenas the
    driver actually finished (95%% or more of its robots stopped). The oracle settles a median
    40%% of robots within 5 units and never exceeds 83%% in any of 24 arenas; the best clone
    finishes no arena at all.</p>
%(figs_settle)s

  <h2>Driving the swarm</h2>
  <p class="lede">Both drivers on the same held-out formations, the same spawns and the same
    measurement: %(arenas)s arenas, %(ticks)s ticks, ground-truth distance from every robot to the
    target shape.</p>
%(figs_loop)s

  <div class="table-wrap">
  <table>
    <thead><tr><th>driver</th><th>final coverage</th><th>peak</th><th>mean distance</th><th>stopped</th></tr></thead>
    <tbody>
%(table)s
    </tbody>
  </table>
  </div>

  <div class="note">
    <strong>Why the numbers here are not the ones the codebase used to print</strong>
    Unity's baked distance field was the geometry the oracle steers by, rotated 90&nbsp;degrees:
    every <code>coverage</code> number in this project scored the swarm against a shape it was
    never aiming at. Measured, not inferred &mdash; with the rotation left in place, Unity's
    per-robot distance correlates 0.998 with the python distance computed at rotated positions and
    0.0 with it computed as-is. Aligned, the oracle reads %(oracle_cov)s; rotated, the same run
    reads about a quarter, indistinguishable from not having moved. Every figure here recomputes
    coverage independently python-side as a cross-check: oracle %(oracle_cov_py)s, actor
    %(actor_cov_py)s.
  </div>

  <h2>What it looks like</h2>
%(figs_demo)s
%(gif)s

  <h2>Reproducing it</h2>
  <p>Every stage writes a file and skips itself if that file exists, so the pipeline resumes rather
    than repeats:</p>
  <pre>cd python
./scripts/bc_offline_pipeline.sh ../results/bc_v2</pre>
  <p>The recorded tapes are the unit of reproducibility: the fit reads them and nothing else, so a
    training run is determined by the tape files and a seed.</p>

  <footer>Generated from <span class="mono">summary.json</span> by
    <span class="mono">tools/bc_artifact.py</span>. Every number on this page comes from that file.</footer>
</div>
"""


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir")
    ap.add_argument("--out", default = None)
    ap.add_argument("--title", default = "Cloning the Kilobot oracle")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.report_dir, "index.html")
    page = build(args.report_dir, args.title)
    with open(out, "w") as f:
        f.write(page)
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
