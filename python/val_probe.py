"""Cold-start probe: does the actor do the right thing from a zeroed hidden state?

A robot's very first decision is a vanishingly small share of BC's data, so the
fit can be good on average while the cold-start behaviour is badly wrong. This
is meant to score that case directly rather than infer it from the loss.

INERT AGAINST UNITY. Scoring needs a per-tick sequence of true robot positions,
which it reads from `worker.arenas` -- an attribute only the deleted Python
replica ever had. A Unity `EnvWorker` does not expose one, so `_sample_positions`
returns immediately, `_score` gets an empty track set, and `cold_start_probe`
returns `{}`. `run_bc_monitored.py` then sees `moved_fraction = None` and skips
the eligibility veto entirely -- no crash, but no signal either.

Reviving it needs per-tick true positions from the player. The snapshot channel
already carries them (`node[:, 0:2]`), so the fix is to sample from
`worker.snapshot(k)` rather than from a replica arena; nothing Unity-side is
missing. `cold_start_probe` warns once when it cannot sample, so enabling
--val-probe-ticks does not silently do nothing.
"""

import numpy as np
import torch

# The tape (val_tape.py) scores imitation accuracy under the ORACLE's own state
# distribution, which is exactly the blind spot of behaviour cloning: it cannot
# see compounding error, where a small mistake puts the actor somewhere the
# oracle never went. This is the cheap on-policy counterweight -- a short,
# actor-driven rollout from a genuine cold start, scored on behaviour rather
# than on task completion, so it says something useful in a few hundred ticks
# instead of needing the swarm to converge.
#
# All three measures already exist as diagnostics in this project; they are
# collected here in one place and turned into numbers a selection rule can use.


def cold_start_probe(trainer, cfg, policy, ticks, loop_threshold = 0.3):
    prev_override = cfg.motor_override
    prev_rollout = cfg.rollout_steps
    cfg.motor_override = "none"
    trainer._bc_capture = False
    for k in range(len(trainer.workers[0].arenas) if hasattr(trainer.workers[0], "arenas") else 0):
        trainer._reset_arena(trainer.workers[0], k)
    tracks = {}
    cfg.rollout_steps = int(ticks)
    worker = trainer.workers[0]
    _sample_positions(worker, tracks)
    with torch.no_grad():
        trainer.collect(policy, None, deterministic = True)
    _sample_positions(worker, tracks)
    cfg.motor_override = prev_override
    cfg.rollout_steps = prev_rollout
    return _score(tracks, loop_threshold)


_warned = []


def _sample_positions(worker, tracks):
    arenas = getattr(worker, "arenas", None)
    if not arenas:
        if not _warned:
            _warned.append(True)
            print("WARNING: cold-start probe cannot sample positions -- it reads worker.arenas, "
                  "which a Unity EnvWorker does not provide, so it will report nothing and the "
                  "eligibility veto will not fire. See val_probe.py's module docstring.",
                  flush = True)
        return
    for k, arena in enumerate(arenas):
        pos = np.asarray(arena.pos, dtype = np.float64).copy()
        for l in range(pos.shape[0]):
            tracks.setdefault((k, l), []).append(pos[l].copy())


def _score(tracks, loop_threshold):
    if not tracks:
        return {}
    moved = 0
    total = 0
    net = []
    for key in tracks:
        pts = tracks[key]
        if len(pts) < 2:
            continue
        total = total + 1
        d = float(np.linalg.norm(pts[-1] - pts[0]))
        net.append(d)
        if d > 1e-3:
            moved = moved + 1
    if total == 0:
        return {}
    return {"moved_fraction": moved / float(total),
            "mean_net_displacement": float(np.mean(net)),
            "robots": total}
