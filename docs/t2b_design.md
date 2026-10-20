# T2b design draft — learned content on the evidence channel (scope (b))

Status (2026-10-18): IMPLEMENTED, UNIT-TESTED, AND STOPPED BY THE OWNER
BEFORE TRAINING. The machinery is banked: slot-7 self-trust (evidence.py
store + materialization), the actor_io writer keeping the policy's slot-7
sample, config/launch/eval flags with guards, the veto-init knob
(KILOBOT_VETO_INIT) with resume-path re-application, and the parity smoke
(results/t2/smoke_t2b_*.log + smoke_t2b.sh). One design change landed
during implementation: the multiplier carries a DEADBAND (t >= -0.25 ->
1.0) because the first parity smoke caught the BC head's fit noise around
zero demoting every claim by ~0.4% and drifting the tick lines -- legacy
wire content now reads as full trust exactly. The four [DECIDE] points
below were resolved as recommended when the design was approved. Phase-174
of docs/tuning.md records the program's state and the conclusion this
design was part of.

## 0. The v1 postmortem this design answers

T2 v1 (55 iterations, whole actor trainable, 256-tick windows) scored
settled 0.0617/0.0600 (fs7/fs1) vs T1-fixed's 0.4225/0.4175 — a −0.36
catastrophe on both seeds, with stopped collapsing to ~0.56. Mechanism: at
256-tick windows the reward never observes a settled outcome at N=600
(settled ≈ 0.002 in every logged window; localization needs ~1500+ ticks), so
PPO optimized only the belief-shaping proxy and drifted the motor/arrival
surface. T2a (scope (a), in flight) freezes everything but `head_msg` — that
removes the drift. What T2a cannot do is ADD anything: its only learnable
output is the veto, which can silence a row, never speak.

This document designs the smallest honest version of "the policy says
something the fixed protocol cannot": what to add, where it rides, what the
receiver does with it, and why the motors-respond-to-speech chain still holds.

## 1. The core constraint the design must respect

Under `evidence_comms` the writer env-writes slots 0-5:

- slots 0-2 = the honest beacon: the sender's OWN believed position and
  confidence, read from its belief state (prop 8/9/12).
- slots 3-5 = the disclosure: the sender's last DIRECT sighting (landmark
  id, measured range, recency).

Both are already the sender's best knowledge at broadcast time. A policy
that REPLACED them could only distort — that is the measured lesson of the
phase-168 era (the learned-content warm-start collapse) and of phase 170
(position claims need accurate means; there are none to improve on). So
**T2b does not touch slots 0-5.** The learnable content lives in the two
reserved floats:

- **slot 7: the learned self-trust annotation t ∈ [-1, 1]** (post-squash).
- slot 8: reserved, stays zero (future: the parked-bit, §6).

This is annotation-class content: the policy never states a position, only
grades its own claim. It cannot "lie louder" — the same envelope the veto
has, one level finer.

## 2. Receiver semantics (the only new machinery)

The claim store (`evidence.py`) records each sender's latest claim. T2b adds
the annotation to the record and consumes it at materialization:

    multiplier = 0.5 * (1 + t)          # t=-1 -> 0.0, t=+1 -> 1.0
    effective_conf = claim.conf * multiplier

- `t = -1`: the claim is dropped at materialization (finer than the row
  veto, which silences the whole row including the disclosure).
- `t = +1`: multiplier 1.0 = the claim's stored conf unchanged.
- **No amplification**: the multiplier is capped at 1.0 — subtract-only, by
  construction. [DECIDE: keep the cap for v1 (recommended; matches the veto
  philosophy and the ladder's finding that MORE transmission is not the
  missing ingredient) or allow up-weighting (e.g. cap 1.25) in v2.]
- The consensus pre-gate (both-ring agreement >= 0.10) runs on the
  effective conf — a down-weighted claim is *softer* evidence, not hidden
  evidence; the pre-gate's own veto logic is untouched.
- Per-claim, not per-sender: the store keys by sender and keeps the latest
  annotation with each claim (a sender's trust can change between its
  broadcasts — that is the signal).

## 3. The honesty envelope (what the policy may and may not move)

- Slots 0-5: env-written. Unreachable. (actor_io writes them after the
  policy's sample, exactly as today.)
- Slot 6 (veto): unchanged from T2a — the policy silences its row.
- Slot 7: the policy grades its own claim, only downward in effect.
- Slot 8: zero.
- There is no recipient addressing and no third-party rating: a sender
  annotates only its OWN claim. [DECIDE: third-party trust ratings
  (sender A grades sender B's claim) are the richer design and the bigger
  attack surface; defer to v2 if v1's self-grading shows signal.]

## 4. Why this is learnable at the v1-failed recipe (the frozen-motor case)

The v1 failure was the motor drift, not the absence of signal. With motors
frozen:

- The response function speech → beliefs → outcomes is FIXED, so the credit
  path from a slot-7 annotation to a receiver's belief error is stable and
  repeatable — the well-posed case for learning an allocation.
- The learning signal that IS visible at window scale is belief error:
  the claim store's materialization directly shapes the receiver's belief
  cloud, and `berr` responds at decision timescale (v1's berr floor moved
  0.63 → 0.55 inside 256-tick windows). The annotation's job — demote the
  claims that would have poisoned a fix — is exactly a berr-level event.
- Settled outcomes remain the EVAL's verdict (paired harness, both seeds),
  never a training-time quantity. Same protocol as T2a.

## 5. Initialization and the T1-parity gap

Same discipline as the veto init (bias +2.0, P(veto) ≈ 4e-4 at iteration 0):

- slot 7's bias initializes at +2.0 → post-squash mean 0.962 →
  multiplier 0.981 — a ~2% down-weight of every claim at iteration 0.
- This is NOT exact T1 parity (T1 multiplier = 1.0). At the deterministic
  eval the gap is fixed and measurable (≈0.02 claim weight); the smoke test
  quantifies it. [DECIDE: +2.0 (consistent with the veto's init; ~2% parity
  gap) vs +3.0 (multiplier 0.995, parity gap ~0.5%, slower exploration).
  Recommendation: +2.0, with the smoke measuring the actual gap before the
  real run.]

## 6. What slot 8 could become (the parked-bit, phase 172's fork (b))

Recorded for completeness, NOT in v1: slot 8 as the parked/stationary bit
reopens homing-to-sender under the structural refutation's own answer —
a sender that advertises "my evidence site does not move" is the one
population Gate-2's ~100% conditional conversion applies to. The honest
envelope requires the bit to be env-cleared the moment the robot's motor
sample leaves the stop state (a moving robot cannot advertise parked-ness),
with the policy deciding WHEN a stopped robot volunteers to be homed-to —
the beacon-volunteer decision, a genuinely new allocation choice with a real
opportunity cost (parking = not navigating). This needs its own probe cell
before any training (phase 172's own gate on fork (b)).

## 7. Tests (the implementation's gate)

1. Wire codec: slot 7 transports through Unity's 9-float message unchanged;
   a pre-T2b checkpoint's broadcasts (slot 7 = 0) read as multiplier 0.5 —
   document that legacy wire values are NOT trusted as "+1 trust".
   **[DECIDE: this is the one real wire-ambiguity risk — a T1-era writer
   emits 7=0, which T2b's receiver reads as half-trust. Two mitigations:
   (i) the multiplier maps 7=0 → 1.0 and only t < 0 demotes (one-sided
   encoding; loses half the range), (ii) a protocol bit elsewhere marks
   T2b-capable senders. Recommendation: (i) — multiplier = max(0, t), so
   the legacy 0 is a no-op and demotion lives in (-1, 0). Re-derive the
   init: bias +2.0 → multiplier 0.96 ≈ T1-exact to 4%.]**
2. Store: annotation rides with the claim; latest-wins; eviction unaffected.
3. Materialization: multiplier semantics (t=0 → 1.0, t=-1 → 0, cap at 1.0),
   effective conf feeds the consensus pre-gate unchanged.
4. Veto interop: slot-6 veto still silences the whole row first.
5. Warm-start parity: A7 → T2b actor loads (head_msg keys popped, as today);
   init +2.0 → the smoke's deterministic tick-by-tick settles match T1-fixed
   within the measured multiplier gap.
6. The freeze interop: FREEZE_REST=1 keeps slot 7 inside head_msg (it is a
   message dimension — trainable under the freeze, like the veto).

## 8. Run plan

- Implementation: evidence.py (store + materialization), actor_io.py (writer
  slot 7), config.py + launch.py env wiring (one flag,
  KILOBOT_LEARNED_SELF_TRUST), eval_closed_loop.py flag + guard, tests (~6),
  parity smoke at N=240 (~30 min) — then the real run.
- Recipe = T2a's, with the slot-7 surface added: 55 iterations × 256-tick
  rollouts at N=600, frozen motors, veto + self-trust both live, entropy
  0.003, msg_lr 5e-5. ~11.5 h on an idle box.
- Evals: T2b-final at 600 on fs7 + fs1 (2 × 4.5 h) vs the T1-fixed bar
  (0.4225/0.4175) and B0 (0.3642/0.3733).
- Sequencing note: the box's GPU makes concurrent trainings ~2-3× slower
  each (measured on the concurrent evals); T2a finishes ~10:30 and (b)
  starts after it — sequential strictly dominates concurrent here.

## 9. What would falsify this design

- The smoke shows the annotation channel cannot move outcomes: slot 7's
  gradient stays flat across 10+ iterations (msg_g ~ 0 on the slot-7 rows)
  → the annotation carries no exploitable signal at this recipe; the
  honest close is "self-trust is not learnable here".
- T2b-final ≈ T1-fixed on both seeds: the fixed thresholds already extract
  the channel's value; the curation family (a+b) closes with instruments
  and the density story as the deliverables.
- T2b > T1-fixed on both seeds: the learning attribution exists — the
  five-arm table (B0/T1/T2a/T2b) carries the headline, and the parked-bit
  (§6) becomes the next authorized probe.
