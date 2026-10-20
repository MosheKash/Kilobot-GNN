"""The closed-form latch's release hysteresis (KILOBOT_CLOSED_FORM_RELEASE).

Release v3: STRONG REPUDIATION ONLY. Three measured facts shaped the rule:

1. The S1b diagnostic: 59% of wrong-stopped robots no longer satisfied the
   gate at final (conf median 0.20 vs 0.80 for settled-right) -- parked on a
   latch their own current belief would refuse.
2. Release v1 (stay while the closed form holds) churned 55,890 release
   decisions in one episode and LOST 0.16 settled: settled robots' conf dips
   below the 0.4 gate floor often enough that plain re-testing wakes them
   (29 previously-settled robots disturbed per 12 wrong stops fixed), and a
   moving robot's conf cannot re-reach the parked latch floor -- a poverty
   trap (moving 0.043 -> 0.262).
3. The learned head saturates at p=1.000 on every already-off evaluation
   (786/786 debug lines) -- it keys on the parked observation itself, so it
   can never vote to release.

v3 therefore wakes a robot only when its own belief ACTIVELY REPUDIATES the
parking spot: conf < release_repudiate_conf AND believed dist >
release_repudiate_dist. Everything else stays parked, whatever the head says.
"""

import math
from types import SimpleNamespace

import torch

import actor_io


def _cfg(release=False, persist=1, rep_conf=0.15, rep_dist=0.12):
    return SimpleNamespace(
        motor_override="none",
        use_closed_form_arrived=True,
        closed_form_and_gate=False,
        closed_form_hybrid=True,
        closed_form_release=release,
        release_repudiate_conf=rep_conf,
        release_repudiate_dist=rep_dist,
        arrived_release_threshold=0.5,
        arrived_confidence_threshold=0.95,
        arrival_persist=persist,
        arrival_min_anchored=False,
    )


def _policy(p_arrived):
    p_arrived = min(max(p_arrived, 1e-6), 1.0 - 1e-6)
    logit = math.log(p_arrived / (1.0 - p_arrived))
    actor = SimpleNamespace(_arrived_logit=torch.tensor([logit]), head_arrived=object())
    return SimpleNamespace(actor=actor)


def _worker(off=False):
    return SimpleNamespace(arrived_switched_off={0: {2: off}})


def _gate(worker, cfg, policy, closed_form, idx=0, prop_dist=None, prop_conf=None):
    return actor_io._arrived_head_gate(worker, policy, cfg, 0, 2, idx,
                                       closed_form=torch.tensor([closed_form]),
                                       prop_dist=prop_dist, prop_conf=prop_conf)


def test_latches_once_then_stays_terminal_when_release_disabled():
    cfg = _cfg(release=False)
    policy = _policy(0.2)   # learned head disagrees; the closed form does the latching
    w = _worker()
    assert _gate(w, cfg, policy, True) is True       # first fire latches
    assert _gate(w, cfg, policy, False) is True      # evidence gone: terminal anyway
    assert _gate(w, cfg, policy, False) is True
    assert w.arrived_switched_off[0][2] is True


def test_strong_repudiation_wakes_the_robot():
    cfg = _cfg(release=True)
    policy = _policy(1.0)   # the head is saturated; it must not matter
    w = _worker(off=True)
    # conf collapsed AND the belief says the target is far: wake
    assert _gate(w, cfg, policy, False, prop_dist=0.30, prop_conf=0.02) is False
    assert w.arrived_switched_off[0][2] is False
    assert not getattr(w, "pending_stop_bonus", {})


def test_borderline_conf_dip_does_not_wake():
    # the v1 failure: settled robots whose conf dips below the 0.4 gate
    # floor must stay parked; only strong repudiation (low conf AND far)
    # wakes anything
    cfg = _cfg(release=True)
    policy = _policy(0.3)
    w = _worker(off=True)
    assert _gate(w, cfg, policy, False, prop_dist=0.06, prop_conf=0.38) is True
    assert _gate(w, cfg, policy, False, prop_dist=0.06, prop_conf=0.01) is True   # close but no conf: stay
    assert _gate(w, cfg, policy, False, prop_dist=0.30, prop_conf=0.38) is True   # far but confident: stay


def test_head_confidence_never_holds_or_wakes():
    cfg = _cfg(release=True)
    w = _worker(off=True)
    # saturated head: cannot keep a repudiated robot parked...
    assert _gate(w, cfg, _policy(0.999), False, prop_dist=0.30, prop_conf=0.02) is False
    # ...and cannot wake a borderline one
    w2 = _worker(off=True)
    assert _gate(w2, cfg, _policy(0.999), False, prop_dist=0.06, prop_conf=0.38) is True


def test_without_prop_readout_release_never_fires():
    # the predicate needs the believed dist/conf; absent, stay parked
    cfg = _cfg(release=True)
    w = _worker(off=True)
    assert _gate(w, cfg, _policy(0.2), False) is True


def test_released_robot_re_earns_the_latch_through_protection():
    cfg = _cfg(release=True, persist=2)
    policy = _policy(0.2)
    w = _worker(off=True)
    assert _gate(w, cfg, policy, False, prop_dist=0.30, prop_conf=0.02) is False   # released
    # re-latching needs two consecutive satisfying decisions
    assert _gate(w, cfg, policy, True, prop_dist=0.06, prop_conf=0.5) is False     # persistence 1/2
    assert _gate(w, cfg, policy, True, prop_dist=0.06, prop_conf=0.5) is True      # persistence 2/2
    # the re-latch transition raises the stop-bonus flag again (the trainer's
    # stop_bonus_paid dedupes the payment to once per episode)
    assert w.pending_stop_bonus[0][2] is True
    assert w._arrival_persist[0][2] >= 2
