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
| `steer_report.py` | does the clone reproduce the oracle's **steering**, not just its motors? Rolls any number of checkpoints through a tape and scores them in the oracle's own steering variable `turn = (R-L)*1.8/(0.7*(L+R))`, whose spread during `wall_following` is 0.0093 -- about 0.1% of the variance in the wheel pair, which is why a motor MSE cannot see it. Draws the error against the size of the signal, the decision-by-decision scatter, the persistent per-robot component, and whether the teacher's direction is recoverable from the observation at all. See `tuning.md` phase 160. |
| `check_differential.py` | the same replay, isolating the left/right motor differential — how sharply the actor turns compared with its teacher |
| `sweep_threshold.py` | sweeps the arrived-head decision threshold over a tape and prints the precision/recall curve |
| `compare_runs.py` | diffs the printed per-iteration metrics of two or more run logs |
| `record_tape.py` | records a BC tape against real Unity — oracle-driven for plain cloning, or actor-driven with oracle labels for DAgger. This one *produces* the training data `bc_offline.py` consumes, so it is the one script here that is part of a pipeline rather than a read-out. |
| `eval_closed_loop.py` | runs one driver (the oracle, or a checkpoint) on held-out formations and records coverage, distance-to-shape and stopping over time, plus positions for the demo plots. Both drivers are measured identically, which the in-training numbers are not. |
| `bc_report.py` | turns a `bc_offline.py` history and a pair of `eval_closed_loop.py` JSONs into the figures and the summary |
| `settle_report.py` | **the objective**: per robot, did it stop AND land within X of its own assigned target — reported as a distribution over arenas, filtered to arenas the driver actually finished. Coverage is not this. |
| `tape_eval.py` | scores checkpoints against tapes as a table — the off-policy/on-policy gap side by side, which is the number a DAgger round is actually moving |

All of them take their arguments positionally and print a usage line with
`--help`.

For offline data and encoder preparation — which never touches a running
pipeline — see `data-prep/`. For larger, still-evolving probe scripts see
`python/temp_test_material/` (gitignored). Run output goes to `results/`.
