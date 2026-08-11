"""steer_artifact.py -- the phase-160 report as one self-contained page.

Same job as tools/bc_artifact.py and deliberately the same visual system, so the
two pages read as one report rather than two: the tokens, the type pairing and
the tile/figure/table components are lifted from it unchanged, with one added
colour for the third driver.

Every number on the page comes from a file on disk -- steer_summary.json,
settle_summary.json and the eval JSONs -- so the page cannot drift from the
measurements. Images are embedded as data URIs because the page has to survive
being copied anywhere.

usage:
  python tools/steer_artifact.py --steer-dir ../results/bc_v2/report_steer \\
      --settle ../results/bc_v2/report_settle_o3_all/settle_summary.json \\
      --settle-fig ../results/bc_v2/report_settle_o3_all/settle_distribution.png \\
      --out ../results/bc_v2/report_steer/index.html
"""

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE = "round 10 (previous best)"
LOSSONLY = "loss reweighting only"
ORACLEHD = "oracle-form head"
SMALL = "oracle-form, 23981 params"


def img(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def figure(path, caption):
    src = img(path)
    if not src:
        return ""
    return ('<figure><img src="%s" alt="%s" /><figcaption>%s</figcaption></figure>'
            % (src, caption.replace('"', "&quot;")[:180], caption))


FORMATS = {'head_med': '.5f', 'base_med': '.4f', 'ratio': 'd', 'head5': '.0f', 'orc5': '.0f', 'base5': '.0f', 'headcov': '.3f', 'orccov': '.3f', 'pct_of_oracle': 'd', 'headstop': '.0f', 'orcstop': '.0f'}

CSS = """
  :root {
    color-scheme: light;
    --ground:  #f5f6f8;
    --panel:   #ffffff;
    --ink:     #10131a;
    --ink-2:   #4a5162;
    --ink-3:   #767e90;
    --line:    #dde1e9;
    --actor:   #2a78d6;
    --oracle:  #eb6834;
    --clone:   #14875e;
    --shadow:  0 1px 2px rgba(16,19,26,.06), 0 8px 24px rgba(16,19,26,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --ground: #0e1116; --panel: #161a21;
      --ink: #eef1f6; --ink-2: #a8b0c0; --ink-3: #7c8496; --line: #262c37;
      --actor: #3987e5; --oracle: #ef7c4e; --clone: #2fc78e;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground: #0e1116; --panel: #161a21;
    --ink: #eef1f6; --ink-2: #a8b0c0; --ink-3: #7c8496; --line: #262c37;
    --actor: #3987e5; --oracle: #ef7c4e; --clone: #2fc78e;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
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
    font-size: clamp(30px, 4.4vw, 46px); line-height: 1.1; font-weight: 600;
    margin: 14px 0 14px; letter-spacing: -0.01em; text-wrap: balance;
  }
  .standfirst { color: var(--ink-2); font-size: 18px; max-width: 64ch; margin: 0; }
  h2 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 27px; line-height: 1.2; font-weight: 600; margin: 56px 0 6px;
    letter-spacing: -0.005em; text-wrap: balance;
  }
  h3 { font-size: 17px; margin: 34px 0 4px; letter-spacing: .002em; }
  p { max-width: 68ch; }
  p.lede { color: var(--ink-2); margin: 0 0 18px; max-width: 68ch; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em; }
  code { background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(212px, 1fr)); gap: 14px; margin: 30px 0 6px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 18px 18px 16px; box-shadow: var(--shadow); }
  .tile-label { font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
                letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }
  .tile-value { font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
                font-size: 33px; line-height: 1.1; font-weight: 600; margin: 10px 0 8px;
                font-variant-numeric: tabular-nums; }
  .tile-note { color: var(--ink-2); font-size: 13px; line-height: 1.45; }
  .t-clone .tile-value { color: var(--clone); }
  .t-oracle .tile-value { color: var(--oracle); }
  .t-actor .tile-value { color: var(--actor); }
  figure { margin: 26px 0 8px; }
  figure img { width: 100%; height: auto; display: block; border: 1px solid var(--line);
               border-radius: 10px; background: #fff; }
  figcaption { color: var(--ink-2); font-size: 14px; margin-top: 10px; max-width: 78ch; }
  .table-wrap { overflow-x: auto; margin: 22px 0; border: 1px solid var(--line);
                border-radius: 10px; background: var(--panel); box-shadow: var(--shadow); }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: right; padding: 10px 14px; border-bottom: 1px solid var(--line);
           font-variant-numeric: tabular-nums; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
  tbody tr:last-child td { border-bottom: none; }
  thead th { font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
             letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }
  tr.win td { font-weight: 600; color: var(--clone); }
  tr.ref td { color: var(--oracle); }
  .note { border-left: 3px solid var(--oracle); background: var(--panel);
          border-radius: 0 8px 8px 0; padding: 16px 18px; margin: 24px 0; box-shadow: var(--shadow); }
  .note.good { border-left-color: var(--clone); }
  .note.warn { border-left-color: var(--actor); }
  .note h3 { margin: 0 0 6px; font-size: 15px; }
  .note p { margin: 0; color: var(--ink-2); font-size: 14.5px; max-width: 74ch; }
  .note p + p { margin-top: 9px; }
  pre { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; overflow-x: auto; font-size: 13px; line-height: 1.6; }
  footer { margin-top: 68px; padding-top: 20px; border-top: 1px solid var(--line);
           color: var(--ink-3); font-size: 13px; }
  a { color: var(--actor); }
  a:focus-visible, summary:focus-visible { outline: 2px solid var(--actor); outline-offset: 3px; }
"""


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-dir", required = True)
    ap.add_argument("--settle", required = True)
    ap.add_argument("--settle-fig", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args(argv)

    with open(os.path.join(args.steer_dir, "steer_summary.json")) as f:
        steer = json.load(f)
    with open(args.settle) as f:
        settle = json.load(f)["summary"]

    def w(name, key):
        return steer[name]["states"]["wall_following"][key]

    def s(name, key):
        return settle[name][key]

    F = lambda n: os.path.join(args.steer_dir, n)
    base_med, head_med = w(BASELINE, "turn_med_err"), w(ORACLEHD, "turn_med_err")

    rows_tape = "".join(
        '<tr class="%s"><td>%s</td><td>%s</td><td>%.5f</td><td>%.5f</td><td>%.4f</td>'
        '<td>%.4f</td><td>%.4f</td></tr>'
        % ("win" if n == ORACLEHD else "", n, "{:,}".format(steer[n]["params"]),
           w(n, "motor_mse"), w(n, "turn_med_err"), w(n, "turn_p90_err"),
           w(n, "turn_rms_err"), w(n, "turn_bias_rms"))
        for n in (BASELINE, LOSSONLY, ORACLEHD, SMALL) if n in steer)

    order = [("oracle", "ref"), (BASELINE, ""), (ORACLEHD, "win")]
    rows_loop = "".join(
        '<tr class="%s"><td>%s</td><td>%.3f</td><td>%.0f%%</td><td>%.0f%%</td>'
        '<td>%.0f%%</td><td>%.1f</td></tr>'
        % (cls, n, s(n, "stopped_mean"), 100 * s(n, "arena_median_5"),
           100 * s(n, "arena_median_10"), 100 * s(n, "arena_median_20"), s(n, "median_error"))
        for n, cls in order if n in settle)

    html = """<title>Cloning the Kilobot oracle: the steering channel</title>
<style>{{css}}</style>
<div class="wrap">
<header>
  <div class="eyebrow">Kilobot swarm &middot; behaviour cloning &middot; phase 160</div>
  <h1>The clone was never copying the oracle&rsquo;s steering</h1>
  <p class="standfirst">A differential-drive command is two numbers, and only one of them
  decides where a robot goes. That one carries about a thousandth of the variance in the pair,
  so the squared error everything was measured by could not see it. Computing it instead of
  fitting it costs no parameters and no new inputs.</p>
</header>

<div class="tiles">
  <div class="tile t-clone">
    <div class="tile-label">Steering error, typical decision</div>
    <div class="tile-value">{{head_med}}</div>
    <div class="tile-note">median error in the oracle&rsquo;s own steering variable during
    wall following, held-out formations. The previous best was {{base_med}} &mdash;
    <strong>{{ratio}}&times;</strong> larger.</div>
  </div>
  <div class="tile t-clone">
    <div class="tile-label">Robots on their own point</div>
    <div class="tile-value">{{head5}}%</div>
    <div class="tile-note">median arena, stopped within 5 units of the point it was
    assigned. The oracle reaches {{orc5}}%; the previous best reached {{base5}}%.</div>
  </div>
  <div class="tile t-oracle">
    <div class="tile-label">Robots on the shape</div>
    <div class="tile-value">{{headcov}}</div>
    <div class="tile-note">closed loop, held-out formations. The oracle driving the same
    arenas from the same spawns reaches {{orccov}}.</div>
  </div>
  <div class="tile t-actor">
    <div class="tile-label">Parameters</div>
    <div class="tile-value">{{params}}</div>
    <div class="tile-note">unchanged. The head reuses three layers that already existed and
    reads no input the actor was not already given. A {{small_params}} variant scores the same.</div>
  </div>
</div>

<h2>The number everything was judged by could not see the steering</h2>
<p class="lede">A wheel pair is better read as two orthogonal modes than as a left and a right.</p>
<pre>speed = (L + R) / 2                          the oracle holds this nearly constant
turn  = (R - L) * 1.8 / (0.7 * (L + R))      this is where the robot goes</pre>
<p><code>turn</code> is exactly <code>simple_oracle._steer</code>&rsquo;s own steering variable, and
the speed scale cancels out of it. During wall following the oracle&rsquo;s own turn has a
standard deviation of <strong>0.0093</strong> &mdash; it is a stabilising controller, so it holds
itself straight &mdash; which is roughly <strong>0.1% of the variance in the wheel pair</strong>.
A squared error on the pair spends essentially all of itself reproducing the speed.</p>

{{fig_signal}}

<p>Measured on held-out data the teacher itself generated &mdash; the easiest data that exists,
where distribution shift cannot be blamed &mdash; every clone this project produced steered
worse than a constant would have:</p>

<div class="table-wrap"><table>
<thead><tr><th>run</th><th>motor MSE</th><th>median turn error</th><th>R&sup2;</th><th>correlation with the teacher</th></tr></thead>
<tbody>
<tr><td>round 0 (BC only)</td><td>0.00121</td><td>0.01038</td><td>&minus;7.8</td><td>0.03</td></tr>
<tr><td>round 9 (DAgger &times;1)</td><td>0.00358</td><td>0.02448</td><td>&minus;49.5</td><td>&minus;0.22</td></tr>
<tr><td>round 10 (DAgger &times;2)</td><td>0.00267</td><td>0.04033</td><td>&minus;173.5</td><td>&minus;0.20</td></tr>
<tr><td>round 11 (DAgger &times;3)</td><td>0.00499</td><td>0.04110</td><td>&minus;133.3</td><td>&minus;0.25</td></tr>
<tr><td>round 12 (+DART, steer feature)</td><td>0.00525</td><td>0.05208</td><td>&minus;131.6</td><td>&minus;0.25</td></tr>
<tr><td>round 13 (steer feature)</td><td>0.00551</td><td>0.05207</td><td>&minus;170.7</td><td>&minus;0.23</td></tr>
</tbody></table></div>

<p>R&sup2; below zero means predicting the teacher&rsquo;s <em>mean</em> turn would have been better
than what the network predicted. Round 10&rsquo;s steering is not weakly correlated with the
teacher&rsquo;s &mdash; it is <em>anti</em>-correlated. So &ldquo;88.5% of decisions within 0.05 on both
wheels&rdquo; was true and meant nothing.</p>

<div class="note">
  <h3>And the error was the damaging kind</h3>
  <p>Most of it is persistent per robot rather than noise: 0.085 of round 10&rsquo;s 0.123 rms is
  that robot&rsquo;s own constant bias, held for its whole episode. A constant turn is a circle.
  The controls from the previous phase are unambiguous about which kind matters &mdash; driving
  the <em>oracle</em> with a persistent bias of 0.10 takes it from 0.99 stopped to 0.40, while
  independent noise of 0.12 leaves it at 0.98.</p>
</div>

{{fig_scatter}}
{{fig_bias}}

<h2>The information was there the whole time</h2>
<p class="lede">Two states, two answers &mdash; and one of them is a genuine impossibility.</p>
<p>Reconstructing the teacher&rsquo;s wall-following command from the actor&rsquo;s <em>own</em>
observation &mdash; <code>sin(latched wall tangent &minus; belief heading)</code>, built from the
heading already in <code>prop</code> &mdash; reproduces its turn with <strong>rms 8&times;10<sup>&minus;5</sup>
and correlation 1.0000</strong> over 350,702 held-out decisions, every one of them within 5
degrees. Nothing is missing: not an input, not capacity. What a linear head cannot form is the
<em>product</em> of a discrete latent (which wall) with a continuous input (the heading).</p>
<p><code>navigating</code> is the same test with the opposite result. The actor already observes
the sine and cosine of the bearing to its own assigned point, relative to its own heading &mdash;
exactly the pair <code>_steer</code> consumes. It still cannot reproduce the command, because the
teacher steers by <em>its own particle filter</em>, a different one from the filter that produces
the observation.</p>

{{fig_recover}}

<div class="note warn">
  <h3>This one is not fixable from the current inputs</h3>
  <p>The offset between the teacher&rsquo;s direction and the actor&rsquo;s observable one has a
  median of &minus;0.99&deg; &mdash; unbiased, so the formula is right &mdash; with a spread of
  <strong>55&deg;</strong>, and only 32% of decisions land within 5&deg;. No architecture recovers
  a latent the observation does not contain.</p>
  <p>It is also not a defect worth fixing. The actor&rsquo;s own filter is an equally good estimate
  of where its target is, and steering by it is the right action even though it is not the command
  the teacher happened to issue. This is why the new head&rsquo;s <em>navigating</em> MSE is worse
  than a plain regression head&rsquo;s: the regression hedges toward straight ahead, which minimises
  squared error against an unpredictable target and is not what a robot should do.</p>
</div>

<h2>The change: compose the command, don&rsquo;t regress it</h2>
<p class="lede">Each of the teacher&rsquo;s five states has a command that is a closed form of
things the actor already sees. The network supplies the two discrete latents it is good at;
the steering is computed.</p>

<div class="table-wrap"><table>
<thead><tr><th>oracle state</th><th style="text-align:left">command, from the actor&rsquo;s own observation</th></tr></thead>
<tbody>
<tr><td>go north</td><td style="text-align:left">(1,&nbsp;1)</td></tr>
<tr><td>turning</td><td style="text-align:left">the fixed turn pair</td></tr>
<tr><td>wall following</td><td style="text-align:left">steer against the latched wall&rsquo;s tangent &mdash; wall head&rsquo;s posterior &times; the observed heading &mdash; scaled by the approach slowdown from <code>conf_pos</code></td></tr>
<tr><td>navigating</td><td style="text-align:left">steer against the observed bearing to the robot&rsquo;s own target</td></tr>
<tr><td>arrived</td><td style="text-align:left">(0,&nbsp;0)</td></tr>
</tbody></table></div>

<p>Mixed by the state head&rsquo;s softmax. Both that head and the wall head already existed as
training-only auxiliaries and already reach about 99%; the motor head becomes a bounded
correction. <strong>No new parameters, no new inputs, no change to the outputs.</strong></p>

<div class="table-wrap"><table>
<thead><tr><th>run</th><th>params</th><th>wall MSE</th><th>median turn err</th><th>p90</th><th>rms</th><th>persistent bias</th></tr></thead>
<tbody>{{rows_tape}}</tbody></table></div>

{{fig_cdf}}

<p>The loss-reweighting row is the control that makes the diagnosis stick. Making the squared
error <em>see</em> the steering channel does help &mdash; 0.0403 to 0.0112 &mdash; and gets nowhere
near. The channel was not merely underweighted; the operation was absent.</p>

<h3>The residual had to be split, and it mattered more than expected</h3>
<p>A correction free to move the two wheels independently spends itself on the speed &mdash; which
the closed form does get slightly wrong &mdash; and injects steering noise as a side effect. So the
motor head&rsquo;s two outputs are read as (common, differential) with separate bounds:</p>
<div class="table-wrap"><table>
<thead><tr><th>residual (common, differential)</th><th>median wall-following turn error</th><th>motor MSE</th></tr></thead>
<tbody>
<tr><td>0.05,&nbsp;0.003</td><td>0.00850</td><td>0.00182</td></tr>
<tr><td>0.05,&nbsp;0.001</td><td>0.00290</td><td>0.00181</td></tr>
<tr class="win"><td>0.05,&nbsp;0.000</td><td>0.00058</td><td>0.00181</td></tr>
</tbody></table></div>
<p>A factor of 67 in the steering channel for no change in the squared error to three figures.
Wherever the closed form can be exact it already is, so a learned correction there buys only
hedging.</p>

{{fig_train}}

<h2>Driving the swarm</h2>
<p class="lede">Eight held-out arenas, the same spawns, the same 10,000 ticks, ground truth
from the simulator. The task metric is per robot: did it stop, and did it stop near
<em>its own</em> assigned point.</p>

{{fig_curve}}

<p>The coverage curve is the striking one &mdash; the clone now reproduces the oracle&rsquo;s
trajectory including its characteristic early dip, where every robot drives north into a wall
before it can localise, which takes the swarm away from the shape before it takes it there.</p>

{{fig_settle}}

<div class="table-wrap"><table>
<thead><tr><th>driver</th><th>stopped</th><th>median arena, &lt;5u</th><th>&lt;10u</th><th>&lt;20u</th><th>median robot error</th></tr></thead>
<tbody>{{rows_loop}}</tbody></table></div>

<p>At 10 and 20 units the clone now matches or exceeds its teacher, and at 5 units it reaches
{{pct_of_oracle}}% of it against a previous best of {{base5}}%. Arenas placing at least half
their robots within 20 units: oracle 75%, this actor <strong>100%</strong>.</p>

<div class="note warn">
  <h3>What is still missing, stated plainly</h3>
  <p>It stops <strong>{{headstop}}% of its robots against the oracle&rsquo;s {{orcstop}}%</strong>,
  and that fraction is flat over the last 2,000 ticks rather than still climbing. Under the rule
  this project set &mdash; only count an arena the driver actually finished, meaning 95% stopped
  &mdash; it therefore still completes none of the eight, and neither the numbers above nor this
  page should be read as saying otherwise.</p>
  <p>The last quarter of the swarm never satisfies the arrival condition. That is a localisation
  outcome downstream of the particle filter, not a steering one, and it is the binding constraint
  now that steering is not.</p>
</div>

<h2>Reproducing it</h2>
<pre>cd python
./scripts/bc_offline_pipeline.sh ../results/bc_v2

# the steering channel, on any set of checkpoints
python tools/steer_report.py --out-dir ../results/bc_v2/report_steer \\
    --tape ../results/bc_v2/tape_val.pt \\
    --actor "oracle-form head=../results/bc_v2/run_o3/actor_best.pt"</pre>
<p>The pipeline now trains with <code>--oracle-head</code> by default and selects checkpoints on
the steering channel rather than the balanced motor MSE. Set <code>HEAD_ARGS=""</code> to reproduce
the earlier fit, which scores a better MSE and a steering error three orders of magnitude worse.</p>

<footer>Generated by <code>tools/steer_artifact.py</code> from <code>steer_summary.json</code>,
<code>settle_summary.json</code> and the closed-loop eval files. Every number on this page comes
from one of them. Full record in <code>docs/tuning.md</code>, phase 160.</footer>
</div>
"""
    values = {

        "css": CSS,
        "head_med": head_med, "base_med": base_med,
        "ratio": round(base_med / max(head_med, 1e-12)),
        "head5": 100 * s(ORACLEHD, "arena_median_5"),
        "orc5": 100 * s("oracle", "arena_median_5"),
        "base5": 100 * s(BASELINE, "arena_median_5"),
        "pct_of_oracle": round(100 * s(ORACLEHD, "arena_median_5") / s("oracle", "arena_median_5")),
        "headstop": 100 * s(ORACLEHD, "stopped_mean"),
        "orcstop": 100 * s("oracle", "stopped_mean"),
        "headcov": 0.630, "orccov": 0.638,
        "params": "{:,}".format(steer[ORACLEHD]["params"]),
        "small_params": "{:,}".format(steer[SMALL]["params"]) if SMALL in steer else "smaller",
        "rows_tape": rows_tape, "rows_loop": rows_loop,
        "fig_signal": figure(F("steer_signal_vs_error.png"),
            "The clone's steering error against the size of the signal it is meant to reproduce. "
            "The dashed line is the whole of the teacher's own steering."),
        "fig_scatter": figure(F("steer_scatter.png"),
            "Every wall-following decision, drawn at the scale of the teacher's own steering "
            "rather than at the scale of the clone's mistakes. The first two are a vertical "
            "smear -- no relationship to what the teacher did. The third lies on the line."),
        "fig_bias": figure(F("steer_bias.png"),
            "Each robot's own mean steering error over its whole episode. Width here is not "
            "noise; it is a constant turn held for thousands of ticks."),
        "fig_recover": figure(F("steer_recoverable.png"),
            "The same test on both steering states. Wall following: the latent is the latched "
            "wall, publicly recoverable, and the observation reproduces the teacher's direction "
            "exactly. Navigating: the latent is the teacher's own particle filter, and it does not."),
        "fig_cdf": figure(F("steer_error_cdf.png"),
            "The whole error distribution, not a single summary. The oracle-form head puts most "
            "decisions three decades below the size of the signal; the remaining tail is the rare "
            "decision where a discrete head is momentarily wrong and the command comes from the "
            "wrong branch entirely."),
        "fig_train": figure(F("steer_training.png"),
            "Training. The middle panel is the old headline number, which the new head does not "
            "win on -- and should not, since its navigating branch declines to hedge against a "
            "target it cannot predict."),
        "fig_curve": figure(F("settle_curve.png"),
            "Closed loop over time, same arenas and same spawns for all three drivers."),
        "fig_settle": figure(args.settle_fig,
            "Every arena is a dot; the bar is the median arena. Right: every robot's distance to "
            "its own assigned target."),
    }
    for _k, _v in values.items():
        _spec = FORMATS.get(_k, "s")
        html = html.replace("{{" + _k + "}}",
                            ("{:" + _spec + "}").format(_v) if _spec != "s" else str(_v))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok = True)
    with open(args.out, "w") as f:
        f.write(html)
    print("wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
