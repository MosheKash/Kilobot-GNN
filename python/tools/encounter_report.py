"""Gate-1 encounter-flux analysis: read the probe blocks from the cell JSONs
and produce the window-matching verdict.

The question: during a robot's LOST phase (own conf < 0.4 -- the phase where
evidence would change its behavior), how often does it hear from an evidence
holder (a sender whose beacon content carries conf >= bar, under the active
writer gate)? Zero flux kills the evidence-sharing design; nonzero flux is
what the homing mechanism (Gate 2: ~100% conditional conversion) exploits.

Reports per cell:
  robots, ever-lost robots
  median lost-phase share of ticks with >=1 usable(0.7) reception
  robots with >=1 / >=2 usable(0.7) receptions while lost
  median first-usable tick vs median first-localized tick (window overlap)
"""

import json
import sys
import numpy as np


def load_cell(path):
    d = json.loads(open(path).read())
    rows = d.get("encounter_probe")
    if rows is None:
        print("%s: NO PROBE BLOCK" % path)
        return None
    return rows


def summarize(path):
    rows = load_cell(path)
    if rows is None:
        return
    lost = [r for r in rows if r["lost_ticks"] > 0]
    print("== %s" % path)
    print("  robots: %d, ever lost: %d, ever localized: %d" %
          (len(rows), len(lost), sum(1 for r in rows if r["first_localized_tick"] >= 0)))
    if not lost:
        print("  (no robot was ever lost -- nothing to measure)")
        return
    for bar in ("0.6", "0.7", "0.8"):
        shares = [r["usable_ticks_lost"][bar] / r["lost_ticks"] for r in lost]
        hit1 = sum(1 for r in lost if r["first_usable_tick_lost"][bar] >= 0)
        hit2 = sum(1 for r in lost
                   if r["usable_ticks_lost"][bar] >= 2)
        print("  bar %s: median lost-share of usable ticks %.4f | robots hit >=1: %d/%d "
              "(%.1f%%) | >=2: %d (%.1f%%) | median share when hit: %.4f" %
              (bar, float(np.median(shares)), hit1, len(lost), 100.0 * hit1 / len(lost),
               hit2, 100.0 * hit2 / len(lost),
               float(np.median([s for s in shares if s > 0])) if hit1 else 0.0))
    fl = [r["first_usable_tick_lost"]["0.7"] for r in lost if r["first_usable_tick_lost"]["0.7"] >= 0]
    ll = [r["first_localized_tick"] for r in lost if r["first_localized_tick"] >= 0]
    if fl:
        print("  first usable(0.7) tick: median %.0f (n=%d) | first localized tick: median %.0f (n=%d)"
              % (float(np.median(fl)), len(fl),
                 float(np.median(ll)) if ll else -1, len(ll)))
    # window matching: of the robots that localized, how many heard something usable BEFORE localizing
    loc_robots = [r for r in lost if r["first_localized_tick"] >= 0]
    before = sum(1 for r in loc_robots if 0 <= r["first_usable_tick_lost"]["0.7"] < r["first_localized_tick"])
    print("  robots that heard usable(0.7) BEFORE their own localization: %d/%d localized robots (%.1f%%)"
          % (before, len(loc_robots), 100.0 * before / len(loc_robots) if loc_robots else 0.0))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
        print()