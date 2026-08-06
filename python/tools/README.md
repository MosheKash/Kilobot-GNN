# python/tools

Diagnostics that run *against* the live pipeline. Nothing in `python/` imports
these — they are run by hand, read a checkpoint, a tape or a player, print a
number, and are done. They live here so the top level of `python/` is only the
pipeline itself.

Each script puts `python/` on `sys.path` at import time, so they run from any
directory.

| script | what it answers |
| --- | --- |
| `verify_unity_pipeline.py` | is the whole loop wired up? 12 checks, from "no Python simulator survives" to a Unity → Python → Unity round trip. Run this after rebuilding the player. |
| `check_dead_reckoning.py` | is dead reckoning still exact? Drives fixed motor pairs against a live player and compares displacement, heading change AND travel direction against what `split_tick_motion` predicts. Deterministic on both sides, so the target is zero error, not "small". Exits non-zero if it collects no samples, rather than reporting a vacuous 0%. |
| `calibrate_kinematics.py` | what does the current build actually measure for `prop_max_speed` and `prop_wheelbase`? Both are now DERIVED from Unity's constants in `config.py`, so this is the independent check on that derivation rather than the source of the values. |
| `check_actor.py` | replays a saved validation tape through a checkpoint and reports its motor/arrived agreement, per oracle state |
| `check_differential.py` | the same replay, isolating the left/right motor differential — how sharply the actor turns compared with its teacher |
| `sweep_threshold.py` | sweeps the arrived-head decision threshold over a tape and prints the precision/recall curve |
| `compare_runs.py` | diffs the printed per-iteration metrics of two or more run logs |

All of them take their arguments positionally and print a usage line with
`--help`.

For offline data and encoder preparation — which never touches a running
pipeline — see `data-prep/`. For larger, still-evolving probe scripts see
`python/temp_test_material/` (gitignored). Run output goes to `results/`.
