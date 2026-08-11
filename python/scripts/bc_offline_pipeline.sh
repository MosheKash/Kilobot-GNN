#!/bin/bash
# The whole offline-BC pipeline, end to end and reproducible from an empty
# results/ directory: hold out formations, record oracle tapes against real
# Unity, fit the actor to them as sequences, run both the oracle and the
# trained actor closed-loop on the held-out formations, and draw the report.
#
# Every stage is skipped if its output already exists, so re-running after an
# interruption resumes rather than repeats, and a single stage can be redone by
# deleting just its output.
#
# The two Unity stages (tapes, eval) are the slow ones -- roughly 2 hours and 1
# hour respectively on one machine. Training itself is minutes.
#
# usage: ./scripts/bc_offline_pipeline.sh [out-dir] [stage ...]
#   ./scripts/bc_offline_pipeline.sh                       # everything
#   ./scripts/bc_offline_pipeline.sh ../results/bc_v2 train report

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-../results/bc_v2}
shift || true
STAGES=${*:-"split tapes train dagger eval report"}

FORMATIONS=${KILOBOT_FORMATIONS:-../data/formations}
DEVICE=${KILOBOT_DEVICE:-cuda}
TICKS=${TICKS:-8000}
EVAL_TICKS=${EVAL_TICKS:-12000}
EPOCHS=${EPOCHS:-120}
ACTIVATION=${ACTIVATION:-elu}
# Observation noise during fitting. 0.03 gave both the best held-out imitation
# error and the best closed loop of anything tried in phase 156; 0 reproduces
# the unregularised fit.
OBS_NOISE=${OBS_NOISE:-0.03}
# The oracle-form motor head and the two auxiliary heads it mixes by. Set
# HEAD_ARGS="" to reproduce the pre-phase-160 fit, which scores a better balanced
# MSE and a steering error three orders of magnitude worse.
HEAD_ARGS=${HEAD_ARGS---state-head-weight 1.0 --wall-head-weight 1.0 --oracle-head --select turn_bias_wall_following}
RUN=$OUT/run_r0

echo "pipeline -> $OUT   stages: $STAGES"

run_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# 1. the held-out split: 2000 formations, seeded, carved out once and reused.
if run_stage split; then
    python - "$FORMATIONS" "$OUT/val_formations" <<'PY'
import sys
from run_bc_monitored import ensure_val_dir
names = ensure_val_dir(sys.argv[1], sys.argv[2], 2000, seed = 12345)
print("held-out formations: %d -> %s" % (len(names), sys.argv[2]))
PY
fi

# 2. the tapes. Training data excludes every held-out name; the validation tape
#    is recorded against the held-out directory itself, with its own swarm RNG
#    so it is a different set of spawns as well as a different set of shapes.
if run_stage tapes; then
    if [ ! -f "$OUT/tape_train.pt" ]; then
        python tools/record_tape.py "$OUT/tape_train.pt" --ticks "$TICKS" \
            --instances 4 --arenas 4 --limit 4000 \
            --exclude "$OUT/val_formations/_names.json" \
            --use-arrived-head --use-turn-anchor --device "$DEVICE" \
            --base-port 5300 --swarm-rng 0 --seed 1
    fi
    if [ ! -f "$OUT/tape_val.pt" ]; then
        python tools/record_tape.py "$OUT/tape_val.pt" --ticks "$TICKS" \
            --instances 2 --arenas 4 --formations "$OUT/val_formations" --limit 2000 \
            --use-arrived-head --use-turn-anchor --device "$DEVICE" \
            --base-port 5400 --swarm-rng 100 --seed 2
    fi
fi

# 3. the fit itself: sequences, truncated BPTT, held-out selection.
#
#    --oracle-head is what makes the fit reproduce the teacher's STEERING rather
#    than only its motors. A differential-drive command is a common mode and a
#    differential mode, and during wall_following the differential -- which is
#    the only channel that decides where a robot goes -- carries about 0.1% of
#    the variance in the pair, so an MSE on the pair cannot see it. Every clone
#    before phase 160 scored a motor MSE of 0.0027 with a steering R^2 of -173.
#    The head composes the command from the oracle's own five closed forms
#    instead, for no extra parameters and no extra inputs, and --select follows
#    it: the balanced MSE is not the quantity to rank runs by.
if run_stage train; then
    python bc_offline.py "$RUN" \
        --train-tape "$OUT/tape_train.pt" --val-tape "$OUT/tape_val.pt" \
        --epochs "$EPOCHS" --activation "$ACTIVATION" --device "$DEVICE" --seed 0 \
        --obs-noise "$OBS_NOISE" $HEAD_ARGS
fi

# 3b. DAgger. One-step imitation accuracy does not imply closed-loop
#     competence here, and the gap is not small: the round-0 actor matched the
#     oracle's command on 97% of held-out decisions and still stopped 72% of
#     its swarm in the wrong place within 2800 ticks, because the arrived head's
#     rare false positives are ABSORBING -- a robot that wrongly latches
#     "arrived" is stopped for good. The only cure is training data drawn from
#     the states the actor's own mistakes lead to, which is what these rounds
#     record: the actor drives, the oracle labels.
if run_stage dagger; then
    PREV=$RUN
    for R in $(seq 1 "${DAGGER_ROUNDS:-2}"); do
        if [ ! -f "$OUT/dagger_$R.pt" ]; then
            # The warm-up is what makes a round see the LATER states: without it
            # robots stall in the first phase they fail in and no round ever
            # produces on-policy wall_following or navigating data.
            python tools/record_tape.py "$OUT/dagger_$R.pt" --ticks "${DAGGER_TICKS:-6000}" \
                --oracle-warmup-ticks "${DAGGER_WARMUP:-2500}" \
                --instances 2 --arenas 4 --limit 4000 \
                --exclude "$OUT/val_formations/_names.json" \
                --driver actor --weights "$PREV/actor_best.pt" \
                --arrived-threshold 0.95 --arrived-release-threshold 0.5 \
                --device "$DEVICE" --base-port 5300 --swarm-rng "$((200 * R))" --seed 1
        fi
        TAPES="$OUT/tape_train.pt"
        for P in $(seq 1 "$R"); do TAPES="$TAPES $OUT/dagger_$P.pt"; done
        if [ ! -f "$OUT/run_r$R/actor_best.pt" ]; then
            python bc_offline.py "$OUT/run_r$R" --train-tape $TAPES \
                --val-tape "$OUT/tape_val.pt" --epochs "$EPOCHS" \
                --activation "$ACTIVATION" --device "$DEVICE" --seed 0 \
                --obs-noise "$OBS_NOISE" $HEAD_ARGS
        fi
        PREV=$OUT/run_r$R
    done
    RUN=$PREV
fi

# 4. closed loop, both drivers, same formations and same spawns.
if run_stage eval; then
    for MODE in oracle actor; do
        [ -f "$OUT/eval_$MODE.json" ] && continue
        EXTRA=""
        [ "$MODE" = actor ] && EXTRA="--weights $RUN/actor_best.pt"
        # Phase 161: gate the actor's stop on the oracle's own closed-form
        # arrival rule (config.py's use_closed_form_arrived) instead of the
        # learned arrived head, whose tape accuracy (0.99 recall @0.95) does not
        # survive the deployment localisation shift. CLOSED_FORM_ARRIVED=1 opts
        # into the experiment; the default keeps the learned-head gate.
        [ "$MODE" = actor ] && [ "${CLOSED_FORM_ARRIVED:-0}" = 1 ] \
            && EXTRA="$EXTRA --closed-form-arrived"
        # optional calibrated arrival radius (normalized); 0 leaves cfg.tau_v
        [ "$MODE" = actor ] && [ "${CLOSED_FORM_ARRIVED:-0}" = 1 ] \
            && [ -n "${CLOSED_FORM_DIST:-}" ] && EXTRA="$EXTRA --closed-form-dist $CLOSED_FORM_DIST"
        [ "$MODE" = actor ] && [ "${CLOSED_FORM_HYBRID:-0}" = 1 ] \
            && EXTRA="$EXTRA --closed-form-hybrid"
        # --bake-rotation-steps 0 and --arrived-release-threshold 0.5 are both
        # deliberate; see the flags' own help text for why the player defaults
        # are wrong here.
        python tools/eval_closed_loop.py "$OUT/eval_$MODE.json" --mode "$MODE" $EXTRA \
            --ticks "$EVAL_TICKS" --sample-every 200 --instances 2 --arenas 4 \
            --formations "$OUT/val_formations" --limit 2000 --bake-rotation-steps 0 \
            --arrived-release-threshold 0.5 \
            --swarm-rng 500 --seed 7 --device "$DEVICE" --base-port 5500
    done
fi

# 5. the figures and the summary.
if run_stage report; then
    python tools/bc_report.py --run-dir "$RUN" \
        --eval-oracle "$OUT/eval_oracle.json" --eval-actor "$OUT/eval_actor.json" \
        --out-dir "$OUT/report" --gif
    # the steering channel, which the report above does not measure
    python tools/steer_report.py --out-dir "$OUT/report_steer" \
        --tape "$OUT/tape_val.pt" --actor "actor=$RUN/actor_best.pt" \
        --run-dir "actor=$RUN"
    # the task metric itself: stopped AND near its OWN assigned point, per arena
    python tools/settle_report.py --out-dir "$OUT/report_settle" \
        --eval "oracle=$OUT/eval_oracle.json" --eval "actor=$OUT/eval_actor.json"
fi

echo "done -> $OUT"
