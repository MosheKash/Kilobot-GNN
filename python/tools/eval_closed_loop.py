"""eval_closed_loop.py -- run one driver (the oracle, or a trained actor) in
real Unity on held-out formations and record how the swarm actually does.

This is the measurement the BC metrics are a proxy for. Imitation error says
how closely the actor reproduces the oracle's motor command on recorded
observations; this says whether the swarm the actor drives ends up in the
shape -- closed loop, its own mistakes compounding, on formations it never
trained on.

Both drivers are measured the SAME way, which the in-training numbers are not:
`oracle_cov` in bc.py is the oracle's own belief-based arrived fraction while
`actor_eval_cov` is ground-truth coverage (trainer.py's _record_snapshots picks
one or the other by who is driving), so the two are not comparable. Here every
run reports:

  coverage      fraction of robots within cfg.tau_v of the target shape,
                ground truth from the critic snapshot -- the task itself
  mean_dist     mean normalised distance to the shape, which keeps moving
                after coverage saturates
  stopped       fraction of robots that have stopped: the oracle's own
                `arrived` state, or the actor's arrived-head gate

Run it once per driver with the SAME --swarm-rng and --seed and the two runs
see the same formations and the same spawns, so the comparison is paired.

usage:
  python tools/eval_closed_loop.py ../results/bc_v2/eval_oracle.json --mode oracle
  python tools/eval_closed_loop.py ../results/bc_v2/eval_actor.json  --mode actor \
      --weights ../results/bc_v2/run1/actor_best.pt
"""

import argparse
import atexit
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices = ["oracle", "renewal", "actor"], required = True)
    ap.add_argument("--weights", default = None, help = "actor checkpoint (required for --mode actor)")
    ap.add_argument("--ticks", type = int, default = 12000)
    ap.add_argument("--sample-every", type = int, default = 200,
                    help = "environment ticks between metric samples")
    ap.add_argument("--instances", type = int, default = 2)
    ap.add_argument("--arenas", type = int, default = 4)
    ap.add_argument("--formations", default = "../results/bc_v2/val_formations")
    ap.add_argument("--encoder", default = "../data/image_encoder.pt")
    ap.add_argument("--limit", type = int, default = 2000)
    ap.add_argument("--min-bots", type = int, default = 40)
    ap.add_argument("--max-bots", type = int, default = 60)
    ap.add_argument("--heartbeat", type = int, default = 48)
    ap.add_argument("--activation", default = None,
                    help = "override the activation; default reads the checkpoint's meta")
    ap.add_argument("--use-arrived-head", action = "store_true", default = None)
    ap.add_argument("--use-turn-anchor", action = "store_true", default = None)
    ap.add_argument("--use-oracle-head", action = "store_true", default = None,
                    help = "config.py's own use_oracle_head: the motor command as a soft "
                           "mixture over the oracle's five closed-form commands. NOT "
                           "detectable from the state_dict (it reuses existing heads and "
                           "adds no parameters), so old RL checkpoints whose meta predates "
                           "checkpoint.architecture_meta need this passed explicitly")
    ap.add_argument("--arrived-threshold", type = float, default = 0.95)
    ap.add_argument("--arrived-release-threshold", type = float, default = 0.0)
    ap.add_argument("--closed-form-arrived", action = "store_true",
                    help = "config.py's own use_closed_form_arrived: gate the actor's stop on the "
                           "oracle's own closed-form arrival rule (filter distance to the robot's "
                           "own target below cfg.tau_v, with conf_pos past the localization floor) "
"computed from the actor's observation, instead of the learned arrived "
                            "head. Terminal, like the oracle's arrived state")
    ap.add_argument("--closed-form-dist", type = float, default = 0.0,
                    help = "arrival radius for --closed-form-arrived, in normalized units. 0 means "
                           "cfg.tau_v (the oracle's own rule); the actor's own filter under-reports "
                           "closeness, so a larger value parks robots where this filter does "
                           "certify arrival")
    ap.add_argument("--closed-form-hybrid", action = "store_true",
                    help = "with --closed-form-arrived, run the closed-form rule in OR with the "
                           "learned arrived head instead of replacing it (config.py's own "
                           "closed_form_hybrid)")
    ap.add_argument("--closed-form-release", action = "store_true",
                    help = "remove the closed-form latch's terminality: a latched robot stays off "
                           "only while the latch evidence still holds (closed form, or learned head "
                           ">= arrived_release_threshold); otherwise it releases and may re-latch "
"through the protected fire path. Off keeps the historical terminal "
                        "latch. The S1b diagnostic: 59%% of wrong stops are stale latches")
    ap.add_argument("--closed-form-and", action = "store_true",
                    help = "with --closed-form-arrived, run the closed-form rule in AND with the "
                           "learned arrived head: stop only when BOTH branches agree")
    ap.add_argument("--freeze-hidden", dest = "freeze_hidden", action = "store_true", default = False,
                    help = "config.py's own arrived_freeze_hidden. Off by default here, unlike the "
                           "Config default: a checkpoint trained by bc_offline.py has seen the "
                           "arrived stretches with the recurrence running, so freezing it at "
                           "deployment would be a recurrence the training never saw")
    ap.add_argument("--belief-comms", action = "store_true",
                    help = "run the receivers' particle filters with peer-beacon fusion "
                           "(config.py's belief_comms): a broadcast (x, y, conf) past "
                           "PEER_CONF_MIN is fused as a range-bearing fix. The default (off) "
                           "is the historical harness; a transmission-trained actor is meant "
                           "to be evaluated BOTH ways -- off measures the motor fine-tune, "
                           "on measures what the learned protocol adds")
    ap.add_argument("--belief-comms-learned", action = "store_true",
                    help = "with --belief-comms, let the policy's own message head supply the "
                           "beacon content instead of actor_io's honest-position override "
                           "(config.py's belief_comms_learned)")
    ap.add_argument("--tc-peer-zero", action = "store_true",
                    help = "zero a peer row's message content (slots 0-8) in the actor's Tc, "
                           "keeping only strength (config.py's tc_peer_zero): the beacon then "
                           "reaches the receiver only through its belief filter, never through "
                           "the motor trunk's input -- isolates the input-distribution shift "
                           "phase 168's driver D measured")
    ap.add_argument("--beacon-anchored-min-conf", type = float, default = 0.0,
                    help = "Stage 2b sender gate: with --belief-comms, a robot broadcasts the honest "
                           "beacon when anchored this episode AND its own belief conf >= this bar, "
                           "instead of only after the arrival latch fired. Measured (S2a): the "
                           "arrived-only pool is geometrically separated from receivers in need -- "
                           "senders cluster at the shape, receivers are mid-arena, 7u IR cannot "
                           "bridge it. Takes precedence over --beacon-arrived-only when > 0")
    ap.add_argument("--recv-gate", action = "store_true",
                    help = "run the learned receiver's trust gate (config.py's recv_gate): the "
                           "policy's own message slot 3 is the robot's fusion-trust output, g in "
                           "[0,1], scaling its fused-ring sigma by 1/g. The warm-start readout "
                           "plants the gate at ~0.98 (status quo); the checkpoint must have been "
                           "trained with the gate live for it to differ")
    ap.add_argument("--hop-comms", action = "store_true",
                    help = "run the robust hop-count flood (config.py's hop_comms): robots maintain "
                           "their IR-hop distance to each corner seed by min-relaxation over received "
                           "message slots and broadcast it in slots 0-3; receivers fuse the counts as "
                           "soft upper bounds on distance. The measured honest-beacon space is "
                           "exhausted (position claims carry sender mean error), so this is the "
                           "information-only alternative")
    ap.add_argument("--beacon-arrived-only", action = "store_true",
                    help = "with --belief-comms, only a robot whose (protected) arrival latch "
                           "has fired broadcasts the honest beacon; moving robots stay silent "
                           "(config.py's beacon_arrived_only). The wavefront protocol: settled "
                           "robots are accurate mobile seeds")
    ap.add_argument("--peer-conf-min", type = float, default = 0.85,
                    help = "the receiver's usability gate for a peer beacon (belief.PEER_CONF_MIN): "
                           "a broadcast is fused only above this confidence. Measured: the honest "
                           "beacon's conf has median 0.80 for settled robots, so 0.85 filters out "
                           "most of the honest sender pool (96%% of receiver updates saw zero "
                           "usable peers) -- the fusion starvation was substantially a threshold "
                           "mismatch")
    ap.add_argument("--peer-innovation-min", type = float, default = 0.0,
                    help = "add_peer_range's innovation gate (config.py's peer_innovation_min): "
                           "fuse a peer ring only if at least this fraction of the receiver's "
                           "particles lie within 2*sigma of it; 0 keeps the fuse-always default")
    ap.add_argument("--arrival-min-anchored", action = "store_true",
                    help = "the arrival latch requires one direct seed/wall sighting this "
                           "episode (config.py's arrival_min_anchored): pure chained-hearsay "
                           "must never stop a robot terminally")
    ap.add_argument("--arrival-persist", type = int, default = 1,
                    help = "the arrival condition must hold this many consecutive decisions "
                           "before the latch fires (config.py's arrival_persist); 1 keeps the "
                           "current behaviour")
    ap.add_argument("--peer-fix", action = "store_true",
                    help = "replace single-ring peer fusion with the multi-beacon position fix "
                           "(config.py's peer_fix): a Gauss-Newton LSQ fix over the usable "
                           "beacons, residual-gated at >= 3 beacons, delivered as a bounded-mass "
                           "injected hypothesis for spread clouds only")
    ap.add_argument("--peer-fix-topk", type = int, default = 6,
                    help = "beacons in the fix (top-K usable rows by strength)")
    ap.add_argument("--peer-fix-resid-max", type = float, default = 1.5,
                    help = "weighted-RMS residual (sigma units) above which a >= 3-beacon fix "
                           "is refused; 0 disables")
    ap.add_argument("--peer-inject-min-spread", type = float, default = 0.2,
                    help = "only receivers whose cloud spread is at least this receive fix "
                           "hypotheses")
    ap.add_argument("--peer-inject-cooldown-ticks", type = int, default = 400,
                    help = "a robot receives at most one fresh fix hypothesis per this many ticks")
    ap.add_argument("--encounter-probe", action = "store_true",
                    help = "Gate-1 flux instrument (config.encounter_probe): per decision tick, "
                           "count each robot's peer receptions and -- when a beacon writer is on "
                           "(--beacon-arrived-only or --beacon-anchored-min-conf) -- receptions "
                           "from evidence holders (beacon conf >= 0.6/0.7/0.8), split into the "
                           "robot's lost phase (own conf < 0.4) vs localized phase. Fuses "
                           "nothing; the output JSON gains a per-robot 'encounter_probe' block")
    ap.add_argument("--evidence-comms", action = "store_true",
                    help = "the evidence-sharing channel (config.evidence_comms): honest beacon "
                           "under the anchored-conf sender gate (evidence_beacon_conf), "
                           "disclosure slots 3-5 (last direct sighting: which landmark, its "
                           "measured range, recency), and receiver-side consensus consumption "
                           "-- a per-robot claim store whose claims are propagated through the "
                           "receiver's own odometry and jointly consumed by belief.add_peer_fix "
                           "(never fused alone; pair separation cap / >=3 residual gate / "
                           "spread floor / bounded-mass injection). Implies --belief-comms")
    ap.add_argument("--evidence-beacon-conf", type = float, default = 0.7,
                    help = "sender gate confidence bar for --evidence-comms (Gate 1 measured "
                           "the anchored-0.7 pool at 57%% lost-phase hit rate vs 19%% "
                           "arrived-only)")
    ap.add_argument("--disclosure-recency-ticks", type = int, default = 2000,
                    help = "a referral's effective recency decays over this many ticks in the "
                           "receiver's clock, then the referral is dropped")
    ap.add_argument("--claim-store-capacity", type = int, default = 8,
                    help = "claims retained per receiver, keyed by sender, oldest evicted")
    ap.add_argument("--claim-odo-var-rate", type = float, default = 0.004,
                    help = "stored-claim sigma aging: odometry variance per unit path under "
                           "the belief code's 0.02 variance convention")
    ap.add_argument("--consensus-agreement-min", type = float, default = 0.10,
                    help = "the consensus pre-gate: a claim reaches the fix only when a "
                           "DISTINCT sender's claim agrees with it on the receiver's own cloud "
                           "(fraction of particles within BOTH rings' 2-sigma bands); 0 "
                           "disables the pre-gate")
    ap.add_argument("--referral-homing", action = "store_true",
                    help = "T1b: the scripted homing override (Gate 2's controller) replaces "
                           "the policy's motors for LOST robots (own conf < 0.4) holding a "
                           "fresh referral with a trustworthy strength-law fit; requires "
                           "--evidence-comms")
    ap.add_argument("--homing-proprio", action = "store_true",
                    help = "T2: append the +3 homing proprioception (distance, sin, cos to the "
                           "fitted referral beacon) to the actor's input. CHANGES INPUT WIDTH: "
                           "only checkpoints trained/fitted with the same width load")
    ap.add_argument("--homing-grace-ticks", type = int, default = 120,
                    help = "the homing law's grace: out of contact with the target past this "
                           "many decisions, the controller sweeps back through the disc")
    ap.add_argument("--homing-budget-ticks", type = int, default = 400,
                    help = "the homing law's abandonment budget: out of contact past this many "
                           "decisions, the referral is dropped and the robot returns to its "
                           "own policy (the measured T1b failure was capture without surrender)")
    ap.add_argument("--learned-veto", action = "store_true",
                    help = "T2: the policy's message slot 6 becomes a broadcast veto (sample "
                           "< 0 silences the whole row); requires --evidence-comms")
    ap.add_argument("--learned-self-trust", action = "store_true",
                    help = "T2b (docs/t2b_design.md): the policy's slot-7 graded self-trust "
                           "annotation demotes its own claim at the receiver's claim store "
                           "(one-sided: 1 + min(0, t)); requires --evidence-comms")
    ap.add_argument("--swarm-rng", type = int, default = 500)
    ap.add_argument("--seed", type = int, default = 7)
    ap.add_argument("--filter-seed", type = int, default = None,
                    help = "re-seed the python RNG right AFTER trainer.setup(), so the "
                           "formation picks and Unity spawns stay paired with --seed while "
                           "the filter's own randomness (resample jitter, injection draws) "
                           "is redrawn. This is the harness's first way to measure the NOISE "
                           "SCALE of the settled metric under stream divergence -- any "
                           "treatment that consumes the shared generator differently (peer "
                           "fix injections, ring fusion) desynchronizes the stream, and "
                           "single-run deltas are only interpretable against this scale. "
                           "Two runs differing only in --filter-seed are the same treatment "
                           "on the same shapes; their delta is pure filter noise")
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--base-port", type = int, default = 5500)
    # The amended task (true wheel slip, MotionNoise.cs). The slip flags are
    # exported as KILOBOT_SLIP_* before any player starts (the player reads
    # them in Awake/Start); --motion-noise / --heading-noise are the Python
    # filter's model of that slip (belief.MOTION_NOISE and the predict/update
    # heading_noise_scale the renewal oracle passes). Defaults read the same
    # env vars launch.py does, so a launch-style env reproduces here.
    ap.add_argument("--slip-bias", type = float,
                    default = float(os.environ.get("KILOBOT_SLIP_SIGMA_BIAS", "0") or 0),
                    help = "KILOBOT_SLIP_SIGMA_BIAS: per-robot persistent wheel mismatch sigma")
    ap.add_argument("--slip-tick", type = float,
                    default = float(os.environ.get("KILOBOT_SLIP_SIGMA_TICK", "0") or 0),
                    help = "KILOBOT_SLIP_SIGMA_TICK: per-FixedUpdate random wheel slip sigma")
    ap.add_argument("--slip-seed", type = int,
                    default = int(os.environ.get("KILOBOT_SLIP_SEED", "0") or 0))
    ap.add_argument("--motion-noise", type = float,
                    default = float(os.environ.get("KILOBOT_MOTION_NOISE", "0") or 0),
                    help = "belief.MOTION_NOISE: the filter's position-noise model of the slip")
    ap.add_argument("--heading-noise", type = float,
                    default = float(os.environ.get("KILOBOT_RENEWAL_HEADING_NOISE", "0") or 0),
                    help = "cfg.renewal_heading_noise: the renewal oracle's filter heading noise")
    ap.add_argument("--cfg", action = "append", default = [], metavar = "FIELD=VALUE",
                    help = "override any existing Config field (repeatable), typed by the "
                           "field's current value, e.g. --cfg renewal_heading_drift=0.003. "
                           "Meant for fields no other flag sets (the renewal_* knobs): it is "
                           "applied with the slip flags, so a field a later flag assigns "
                           "(the actor/gate ones) is overwritten by that flag. Unknown "
                           "field = error")
    ap.add_argument("--seed-layout", default = "corners", choices = ["corners", "quarters"],
                    help = "KILOBOT_SEED_LAYOUT for the player and belief.SEED_LAYOUTS: 'quarters' "
                           "moves the four seeds to mid-arena landmarks (phase 175 experiment)")
    ap.add_argument("--landmarks", action = "store_true",
                    help = "KILOBOT_LANDMARKS=1: four mid-arena landmark beacons broadcasting their "
                           "position over the neighbour message channel (phase 175)")
    ap.add_argument("--debug-wall-log", action = "store_true",
                    help = "cfg.oracle_debug_wall_log: the oracles' per-robot transition / "
                           "diagnostic prints (renewal: RENEWAL_TURN_DONE true heading error "
                           "after the fixed 90-degree turn)")
    ap.add_argument("--headed", action = "store_true",
                    help = "show a LIVE Unity window (no_graphics=False) and stream the same "
                           "coverage/stopped/settled console numbers as the headless run. Use "
                           "--time-scale 1 (the default here) to watch in real time; a single "
                           "--instances 1 arena is what you want for an eyeball check. This is "
                           "the numeric + visual counterpart to scripts/watch_actor.sh")
    ap.add_argument("--time-scale", type = float, default = None,
                    help = "Unity engine time_scale. Default: 1 when --headed (real-time viewing), "
                           "else 20 for headless speed")
    ap.add_argument("--bake-rotation-steps", type = int, default = 0,
                    help = "KILOBOT_BAKE_ROTATION_STEPS for the player. The default here is 0, "
                           "NOT the player's own default of 1: at 1, Unity's baked distance "
                           "field is formations.py's geometry rotated 90 degrees CCW (phase 31 "
                           "added the rotation in C#, phase 33 removed the matching one in "
                           "Python and the two were never reconciled), so `coverage` scores the "
                           "swarm against a shape 90 degrees from the one the oracle is steering "
                           "it into. Measured, not assumed: with this at 1, Unity's per-robot "
                           "distance correlates 0.998 with the python distance computed at "
                           "90-degree-rotated positions, and 0.0 with it computed as-is")
    ap.add_argument("--wall-seed-position", action = "store_true",
                    help = "config.py's own oracle_wall_seed_position. Off in Config and in every "
                           "tape recorded here, which means wall contact constrains one axis only "
                           "and a robot has to reach a CORNER before its belief collapses -- so "
                           "how accurately it localizes depends on how precisely it follows the "
                           "wall. On, the nearest wall seed's own broadcast position constrains "
                           "both axes, which decouples localization from trajectory precision")
    ap.add_argument("--motor-bias", type = float, default = 0.0,
                    help = "with --mode oracle: give each robot a PERSISTENT differential offset "
                           "drawn once from N(0, this), instead of fresh noise every tick. A "
                           "clone's error is not i.i.d. -- it is a steering error that persists "
                           "while the robot stays in the same state -- and a correlated error of "
                           "a given size is far more damaging than an independent one, which is "
                           "what this measures")
    ap.add_argument("--motor-noise", type = float, default = 0.0,
                    help = "with --mode oracle: perturb the ORACLE's own motor command by "
                           "Gaussian noise of this standard deviation before it is executed. "
                           "This is the control for a behaviour-cloning result -- a clone that "
                           "reproduces its teacher's command to within e per wheel is, "
                           "dynamically, the teacher driving with e of noise. If the oracle "
                           "collapses at the clone's own error level, the clone was never going "
                           "to work at that accuracy and the objective, not the fit, is what "
                           "needs changing")
    ap.add_argument("--no-positions", action = "store_true",
                    help = "do not store per-sample robot positions (they are what the "
                           "demo animation is rendered from)")
    return ap


def main(argv = None):
    args = build_parser().parse_args(argv)
    if args.mode == "actor" and not args.weights:
        raise SystemExit("--mode actor needs --weights")

    import unity_env
    from config import Config
    from encoder import load_encoder
    from formations import build_formation_pool
    from images import build_image_pool
    from kilobot_gnn import Z, build_actor
    from policy import GaussianPolicy
    from reward import coverage
    from run_bc_monitored import preprocess
    from trainer import Trainer

    meta = {}
    blob = None
    width_homing = False
    if args.weights:
        blob = torch.load(args.weights, map_location = "cpu", weights_only = False)
        meta = blob.get("meta", {}) or {}
    activation = args.activation or meta.get("activation", "relu")
    # The head flags are read from meta, with a state_dict fallback: an RL
    # full checkpoint saved before checkpoint.architecture_meta existed
    # carries meta {z, action_size} only -- no flags -- and build_actor then
    # constructs a differently-shaped actor (up1 40 vs the checkpoint's 42,
    # unexpected head keys). The tensors themselves are the source of truth
    # for everything but use_oracle_head (which adds no parameters at all --
    # it reuses existing heads, so no tensor can tell you it was on; for
    # such checkpoints pass --use-oracle-head explicitly).
    use_arrived_head = (args.use_arrived_head if args.use_arrived_head is not None
                        else bool(meta.get("use_arrived_head", False)))
    use_turn_anchor = (args.use_turn_anchor if args.use_turn_anchor is not None
                       else bool(meta.get("use_turn_anchor", False)))
    use_state_head = bool(meta.get("use_state_head", False))
    use_wall_head = bool(meta.get("use_wall_head", False))
    use_steer_feature = bool(meta.get("use_steer_feature", False))
    use_oracle_head = (args.use_oracle_head if args.use_oracle_head is not None
                       else bool(meta.get("use_oracle_head", False)))
    msg_head_prop_input = bool(meta.get("msg_head_prop_input", False))
    oracle_residual = float(meta.get("oracle_residual", 0.05))
    oracle_residual_turn = float(meta.get("oracle_residual_turn", 0.0))
    if blob is not None:
        sd = blob["policy"] if "policy" in blob else blob.get("actor", {})
        # a full checkpoint's policy state_dict prefixes actor keys with
        # "actor."; a bare actor export does not -- check both spellings
        def _has(key):
            return (key in sd) or (("actor." + key) in sd)
        if _has("head_arrived.weight"):
            use_arrived_head = True
        if _has("head_state.weight"):
            use_state_head = True
        if _has("head_wall.weight"):
            use_wall_head = True
        if _has("up1.weight"):
            from kilobot_gnn import SPLIT_TC_SIZE, SPLIT_ODOM_SIZE, HOMING_PROP_SIZE
            up_key = "actor.up1.weight" if ("actor.up1.weight" in sd) else "up1.weight"
            up_in = int(sd[up_key].shape[1])
            # The raw-width inference the first version used
            # (up_in == TC + ODOM + 2) misreported use_turn_anchor for any
            # T2-era actor whose up1 also carries the homing columns. The
            # valid split-actor widths are enumerated explicitly instead:
            # base, +2 (turn anchor), +3 (homing), +5 (both). Anything else
            # is a loud refusal, not a silent misbuild.
            base = SPLIT_TC_SIZE + SPLIT_ODOM_SIZE
            if up_in not in (base, base + 2, base + HOMING_PROP_SIZE,
                             base + 2 + HOMING_PROP_SIZE):
                raise SystemExit("up1 input width %d is not a valid split-actor "
                                 "shape (base %d; +2 = turn anchor; +%d = "
                                 "homing_proprio)" % (up_in, base, HOMING_PROP_SIZE))
            if args.use_turn_anchor is None:
                use_turn_anchor = up_in in (base + 2, base + 2 + HOMING_PROP_SIZE)
            width_homing = up_in in (base + HOMING_PROP_SIZE,
                                     base + 2 + HOMING_PROP_SIZE)
        # msg_head_prop_input (the transmission phase): recoverable from the
        # head_msg input width against head1's output width, so even a
        # checkpoint whose meta predates the field builds the right actor.
        if _has("head_msg.weight") and _has("head1.weight"):
            hm_key = "actor.head_msg.weight" if ("actor.head_msg.weight" in sd) else "head_msg.weight"
            h1_key = "actor.head1.weight" if ("actor.head1.weight" in sd) else "head1.weight"
            msg_head_prop_input = bool(int(sd[hm_key].shape[1]) > int(sd[h1_key].shape[0]))
        else:
            msg_head_prop_input = bool(meta.get("msg_head_prop_input", False))

    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.seed_layout = args.seed_layout
    cfg.landmarks = bool(args.landmarks)
    os.environ["KILOBOT_LANDMARKS"] = "1" if cfg.landmarks else "0"
    cfg.heartbeat_ticks = args.heartbeat
    cfg.rollout_steps = args.sample_every
    cfg.num_arenas = args.arenas
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.oracle_known_start_heading = True
    cfg.belief_comms = bool(args.belief_comms)
    cfg.belief_comms_learned = bool(args.belief_comms_learned)
    cfg.tc_peer_zero = bool(args.tc_peer_zero)
    cfg.beacon_arrived_only = bool(args.beacon_arrived_only)
    cfg.peer_innovation_min = float(args.peer_innovation_min)
    cfg.peer_conf_min = float(getattr(args, "peer_conf_min", 0.85))
    from belief import set_peer_conf_min
    set_peer_conf_min(cfg.peer_conf_min)
    cfg.arrival_min_anchored = bool(args.arrival_min_anchored)
    cfg.arrival_persist = int(args.arrival_persist)
    cfg.encounter_probe = bool(args.encounter_probe)
    cfg.evidence_comms = bool(args.evidence_comms)
    if cfg.evidence_comms:
        # The evidence channel's guard: it owns the slots the measured-out
        # protocols used, and its receiver path replaces theirs.
        if bool(args.hop_comms) or bool(args.recv_gate):
            raise SystemExit("evidence_comms is incompatible with hop_comms/recv_gate: "
                             "both measured out (docs/tuning.md phase 170) and their slot "
                             "usage collides with the disclosure/veto slots")
        if args.homing_proprio and not args.weights:
            raise SystemExit("--homing-proprio changes the actor's input width; a matching "
                             "checkpoint is required")
        # implies the honest beacon writer under the anchored sender gate, and
        # the receiver's conf floor drops to the sender gate's bar (the
        # bimodal honest conf peaks ~0.8; 0.85 filtered most of the pool)
        cfg.belief_comms = True
        cfg.beacon_anchored_min_conf = max(float(args.evidence_beacon_conf),
                                           float(getattr(cfg, "beacon_anchored_min_conf", 0.0) or 0.0))
    cfg.evidence_beacon_conf = float(args.evidence_beacon_conf)
    cfg.disclosure_recency_ticks = int(args.disclosure_recency_ticks)
    cfg.claim_store_capacity = int(args.claim_store_capacity)
    cfg.claim_odo_var_rate = float(args.claim_odo_var_rate)
    cfg.consensus_agreement_min = float(args.consensus_agreement_min)
    cfg.homing_grace_ticks = int(args.homing_grace_ticks)
    cfg.homing_budget_ticks = int(args.homing_budget_ticks)
    cfg.referral_homing = bool(args.referral_homing) and cfg.evidence_comms
    # homing_proprio: explicit flag wins; otherwise trust the checkpoint's
    # own meta (checkpoint.architecture_meta records it by attribute) so a
    # T2-era checkpoint builds its full-width actor automatically and the
    # strict loader matches. Pre-fix checkpoints (whose meta predates the
    # field, or whose use_turn_anchor was width-inferred and wrong) keep
    # relying on the explicit flags.
    # homing_proprio: explicit flag wins; else the state dict's own width
    # (the tensors are the source of truth); else the checkpoint meta.
    # The meta fallback covers pre-fix checkpoints only -- width inference is
    # authoritative and cannot lie about the architecture it is loading.
    cfg.homing_proprio = bool(args.homing_proprio) if args.homing_proprio else \
        (width_homing if width_homing else bool(meta.get("homing_proprio", False)))
    cfg.learned_veto = bool(args.learned_veto) and cfg.evidence_comms
    cfg.learned_self_trust = bool(args.learned_self_trust) and cfg.evidence_comms
    if cfg.evidence_comms:
        from belief import set_peer_conf_min as _spcm
        _spcm(cfg.evidence_beacon_conf)
    cfg.peer_fix = bool(args.peer_fix)
    cfg.peer_fix_topk = int(args.peer_fix_topk)
    cfg.peer_fix_resid_max = float(args.peer_fix_resid_max)
    cfg.peer_inject_min_spread = float(args.peer_inject_min_spread)
    cfg.peer_inject_cooldown_ticks = int(args.peer_inject_cooldown_ticks)
    # "renewal" selects the amended-task baseline (renewal_oracle.py) -- pair
    # it with the slip env vars (KILOBOT_SLIP_*) for the noisy-odometry task.
    cfg.motor_override = {"oracle": "simple_oracle",
                          "renewal": "renewal_oracle"}.get(args.mode, "none")
    cfg.slip_sigma_bias = float(args.slip_bias)
    cfg.slip_sigma_tick = float(args.slip_tick)
    cfg.slip_seed = int(args.slip_seed)
    cfg.motion_noise = float(args.motion_noise)
    cfg.renewal_heading_noise = float(args.heading_noise)
    cfg.oracle_debug_wall_log = bool(args.debug_wall_log)
    for kv in args.cfg:
        key, _, val = kv.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit("--cfg: no Config field %r" % key)
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            parsed = val.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            parsed = int(val)
        elif isinstance(cur, float):
            parsed = float(val)
        else:
            parsed = val
        setattr(cfg, key, parsed)
        print("cfg override: %s = %r" % (key, parsed), flush = True)
    os.environ["KILOBOT_SLIP_SIGMA_BIAS"] = repr(cfg.slip_sigma_bias)
    os.environ["KILOBOT_SLIP_SIGMA_TICK"] = repr(cfg.slip_sigma_tick)
    os.environ["KILOBOT_SLIP_SEED"] = str(cfg.slip_seed)
    import belief as _belief_mod
    _belief_mod.MOTION_NOISE = cfg.motion_noise
    print("slip: bias=%r tick=%r seed=%d | filter MOTION_NOISE=%r heading_noise=%r"
          % (cfg.slip_sigma_bias, cfg.slip_sigma_tick, cfg.slip_seed,
             cfg.motion_noise, cfg.renewal_heading_noise), flush = True)
    # Nothing may end an episode during an evaluation: a reset mid-run would
    # restart the swarm on a new formation and the curve would measure two
    # different problems glued together. Coverage cannot exceed 1.0 and the
    # step limit is set past the run length.
    cfg.success_threshold = 1.1
    cfg.max_episode_steps = args.ticks * 10
    cfg.use_arrived_head = use_arrived_head
    cfg.use_turn_anchor = use_turn_anchor
    # Training-only head, but it is in the state_dict, so the actor has to be
    # built with it or load_state_dict rejects the checkpoint.
    cfg.use_state_head = use_state_head
    cfg.use_wall_head = use_wall_head
    cfg.use_steer_feature = use_steer_feature
    cfg.use_oracle_head = use_oracle_head
    cfg.msg_head_prop_input = msg_head_prop_input
    cfg.oracle_residual = oracle_residual
    cfg.oracle_residual_turn = oracle_residual_turn
    cfg.split_activation = activation
    # Widths are recovered from the checkpoint's own tensors, so anything
    # trained with --gru-hidden loads instead of failing on a shape mismatch.
    if blob is not None:
        from kilobot_gnn import widths_from_state_dict
        for _k, _v in widths_from_state_dict(blob["actor"] if "actor" in blob else blob).items():
            setattr(cfg, "split_" + _k, _v)
    cfg.arrived_confidence_threshold = args.arrived_threshold
    cfg.arrived_release_threshold = args.arrived_release_threshold
    cfg.use_closed_form_arrived = args.closed_form_arrived
    cfg.closed_form_arrival_dist = args.closed_form_dist
    cfg.closed_form_hybrid = args.closed_form_hybrid
    cfg.closed_form_and_gate = args.closed_form_and
    cfg.closed_form_release = args.closed_form_release
    cfg.beacon_anchored_min_conf = float(args.beacon_anchored_min_conf)
    cfg.hop_comms = bool(args.hop_comms)
    cfg.recv_gate = bool(args.recv_gate)
    cfg.arrived_freeze_hidden = args.freeze_hidden
    cfg.oracle_wall_seed_position = args.wall_seed_position
    # matches run_bc_monitored.build_train_cfg, which is what every tape and
    # every trained checkpoint's observations were scaled by
    cfg.split_prop_time_scale = 0.058

    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit, device = args.device)
    formation_pool = build_formation_pool(args.formations, limit = args.limit)
    cfg._oracle_formation_pool = formation_pool
    # Every architecture switch that changes the deployed forward pass is
    # printed, not just the ones that change the state_dict's shape. use_oracle_head
    # does not change any tensor's shape, so a checkpoint trained with it loads
    # cleanly into an actor built without it and silently runs a different
    # function -- the failure would look like a bad policy, not a bad load.
    print("%s: %d formations from %s, activation=%s arrived_head=%s turn_anchor=%s "
          "oracle_head=%s (residual %.3f / %.4f) steer_feature=%s"
          % (args.mode, len(formation_pool), args.formations, activation,
             use_arrived_head, use_turn_anchor, use_oracle_head, oracle_residual,
             oracle_residual_turn, use_steer_feature), flush = True)

    # Must be set before any player is launched: ImageLibrary reads it while
    # baking, in the environment the player inherited at start-up.
    os.environ["KILOBOT_BAKE_ROTATION_STEPS"] = str(int(args.bake_rotation_steps))
    time_scale = args.time_scale if args.time_scale is not None else (1.0 if args.headed else 20.0)
    if args.headed:
        print("HEADED: showing a live Unity window; time_scale=%.1f (Ctrl+C to stop)" % time_scale,
              flush = True)
    envs, workers = [], []
    for i in range(args.instances):
        unity_env.set_player_env(formations = args.formations, heartbeat_ticks = cfg.heartbeat_ticks,
                                 seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                                 min_bots = args.min_bots, max_bots = args.max_bots,
                                 swarm_rng = args.swarm_rng + i)
        w, env = unity_env.make_unity_worker(worker_id = i, num_arenas = args.arenas,
                                             no_graphics = not args.headed, base_port = args.base_port,
                                             time_scale = time_scale)
        workers.append(w)
        envs.append(env)

    closed = []

    def close_envs():
        if closed:
            return
        closed.append(True)
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                print("warning: env close failed (%s)" % exc, flush = True)

    atexit.register(close_envs)

    if args.motor_bias > 0 and args.mode == "oracle":
        import simple_oracle as _sob
        _orig_b = _sob.simple_oracle_motors
        _bias = {}

        def _biased(worker, arena_ids, locals_, *a, **kw):
            motor = _orig_b(worker, arena_ids, locals_, *a, **kw)
            if motor is None:
                return motor
            out = motor.clone()
            for i in range(int(arena_ids.shape[0])):
                key = (int(arena_ids[i]), int(locals_[i]))
                if key not in _bias:
                    _bias[key] = float(torch.randn(1)) * args.motor_bias
                b = _bias[key]
                out[i, 0] = out[i, 0] + b
                out[i, 1] = out[i, 1] - b
            return out.clamp(0.0, 1.0)

        _sob.simple_oracle_motors = _biased
        print("oracle driving with a PERSISTENT per-robot steering bias, sigma=%.3f" % args.motor_bias,
              flush = True)

    if args.motor_noise > 0 and args.mode == "oracle":
        # Wraps the oracle's own motor function rather than the action written to
        # the worker, so the noise lands in exactly one place: the command that
        # is executed. The oracle's belief and heading tracking dead-reckon from
        # the executed motor, so it sees the perturbed motion the way a real
        # controller would -- this is a noisy driver, not a driver being lied to.
        import simple_oracle as _so
        _orig_motors = _so.simple_oracle_motors

        def _noisy(*a, **kw):
            motor = _orig_motors(*a, **kw)
            if motor is None:
                return motor
            return (motor + torch.randn_like(motor) * args.motor_noise).clamp(0.0, 1.0)

        _so.simple_oracle_motors = _noisy
        print("oracle driving with motor noise sigma=%.3f" % args.motor_noise, flush = True)

    trainer = Trainer.from_workers(workers, cfg, encoder, image_pool, formations_dir = os.path.abspath(args.formations))
    trainer._save_positions = not args.no_positions
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init).to(cfg.device)
    if args.weights:
        from checkpoint import load_for_eval
        load_for_eval(args.weights, policy, cfg.device)
        print("loaded %s" % args.weights, flush = True)
    # Re-seeded here, immediately before the first _pick_image, and not only at
    # the top: building the policy draws from the same global generator, and the
    # oracle run builds a different-shaped actor from the one a checkpoint asks
    # for (no arrived head, no turn anchor), so it consumed a different number of
    # draws and every arena then got a different formation. Two runs meant to be
    # paired were comparing different shapes. Spawns were never affected -- those
    # come from the player's own KILOBOT_SWARM_RNG.
    torch.manual_seed(args.seed)
    trainer.setup()
    if args.filter_seed is not None:
        # after setup(): formations and spawns are already fixed, so this only
        # redraws the filter's own randomness -- the noise-scale probe the
        # argument's help text describes. The filter's draws all pass
        # generator=trainer.sample_rng (belief.py shares that one Generator via
        # _init_globals), NOT the global stream -- so torch.manual_seed alone
        # was measured to be a NO-OP: nf1/nf2 reproduced A7 bit-identically.
        # The trainer's own generator must be reseeded for this to do anything.
        torch.manual_seed(args.filter_seed)
        trainer.sample_rng.manual_seed(int(args.filter_seed))

    samples = []
    ticks = 0
    t0 = time.time()
    try:
        # A sample before anything has moved. Coverage has a large floor -- a
        # robot dropped at random in an arena whose target shape is a QuickDraw
        # stroke is already within tau_v of it a third of the time -- so the
        # number is only readable against where it started.
        first = _sample(trainer, workers, cfg, coverage, 0)
        if first["arenas"]:
            samples.append(first)
            print("  tick %6d  coverage %.4f  mean_dist %.4f  stopped %.4f  (spawn)"
                  % (0, first["coverage"], first["mean_dist"], first["stopped"]), flush = True)
        while ticks < args.ticks:
            with torch.no_grad():
                trainer.collect(policy, None, deterministic = True)
            ticks += args.sample_every
            samples.append(_sample(trainer, workers, cfg, coverage, ticks))
            s = samples[-1]
            print("  tick %6d  coverage %.4f  mean_dist %.4f  stopped %.4f  |  settled<5u %.4f  "
                  "near<5u %.4f  median_err %.1f  (%.1f min)"
                  % (ticks, s["coverage"], s["mean_dist"], s["stopped"], s["settled_5"], s["near_5"],
                     float(np.median([a["target_err_median"] for a in s["per_arena"]
                                      if a.get("target_err_median") is not None] or [float("nan")])),
                     (time.time() - t0) / 60.0), flush = True)
    finally:
        close_envs()

    from images import formation_paths
    names = [os.path.basename(p) for p in formation_paths(args.formations, ".png", args.limit, None)]
    final = _final_state(trainer, workers, cfg, formation_pool, names)
    blob = {"mode": args.mode, "weights": args.weights, "meta": meta,
            "args": vars(args), "samples": samples, "final": final,
            "seconds": time.time() - t0}
    if args.encounter_probe:
        probe_rows = []
        for wi, worker in enumerate(workers):
            rec = getattr(worker, "_encounter_probe", None)
            if not rec:
                continue
            for (a, l), st in sorted(rec.items()):
                probe_rows.append({"worker": wi, "arena": a, "robot": l, **st})
        blob["encounter_probe"] = probe_rows
        n_r = len(probe_rows)
        lost = [r for r in probe_rows if r["lost_ticks"] > 0]
        if lost and n_r:
            usable7 = [r["usable_ticks_lost"][0.7] / r["lost_ticks"] for r in lost]
            hit7 = sum(1 for r in lost if r["first_usable_tick_lost"][0.7] >= 0)
            print("encounter probe: %d robots, %d ever lost; during lost phase "
                  "usable(0.7) receptions: median share %.3f, robots hit %d/%d"
                  % (n_r, len(lost), float(np.median(usable7)), hit7, len(lost)),
                  flush = True)
        else:
            print("encounter probe: %d robots recorded, none ever lost" % n_r, flush = True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump(blob, f)
    last = samples[-1]
    print("wrote %s -- final coverage %.4f, mean_dist %.4f, stopped %.4f over %d arenas"
          % (args.out, last["coverage"], last["mean_dist"], last["stopped"], last["arenas"]),
          flush = True)


def _per_robot_stopped(worker, k, m, mode):
    """Which robots have decided they are done, one flag each."""
    if mode in ("oracle", "renewal"):
        st = getattr(worker, "simple_state", {}).get(k, {})
        return [bool(st.get(l) == "arrived") for l in range(m)]
    off = getattr(worker, "arrived_switched_off", {}).get(k, {})
    return [bool(off.get(l, False)) for l in range(m)]


def _assigned_targets(worker, k, m):
    """Each robot's own assigned target point, from the authoritative cache.

    observation.ensure_target writes worker.simple_target, and it is the single
    shared resolution point -- simple_oracle steers by it and gather_split_state
    builds the actor's relative-target observation from it, so this is exactly
    the point the robot was told to go to, not a reconstruction of it.
    """
    cache = getattr(worker, "simple_target", {}).get(k, {})
    out = []
    for l in range(m):
        t = cache.get(l)
        out.append(None if t is None else [float(t[0]), float(t[1])])
    return out


def _stopped_fraction(worker, k, m, mode):
    """Fraction of an arena's robots that have decided they are done.

    Deliberately per-driver, because the two drivers stop for different
    reasons and there is no single mechanism to read: the oracle's own
    terminal `arrived` state, and the actor's arrived head having latched the
    gate in actor_io._arrived_head_gate. Both mean "this robot has stopped
    itself", which is what the number is for; neither is ground truth about
    whether it stopped in the right place -- `coverage` is.
    """
    if m <= 0:
        return 0.0
    if mode in ("oracle", "renewal"):
        st = getattr(worker, "simple_state", {}).get(k, {})
        return sum(1 for l in range(m) if st.get(l) == "arrived") / float(m)
    off = getattr(worker, "arrived_switched_off", {}).get(k, {})
    return sum(1 for l in range(m) if off.get(l)) / float(m)


def _sample(trainer, workers, cfg, coverage, ticks):
    mode = {"simple_oracle": "oracle", "renewal_oracle": "renewal"}.get(cfg.motor_override, "actor")
    covs, dists, stops, per_arena = [], [], [], []
    for wi, worker in enumerate(workers):
        for k in range(cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None or snap.get("node") is None:
                continue
            node = snap["node"]
            m = node.shape[0]
            c = float(coverage(node, cfg))
            d = float(node[:, 4].mean())
            s = _stopped_fraction(worker, k, m, mode)
            covs.append(c); dists.append(d); stops.append(s)
            rec = {"worker": wi, "arena": k, "robots": int(m),
                   "coverage": c, "mean_dist": d, "stopped": s}
            # THE metric: a robot has settled when it has stopped itself AND is
            # actually near the point it was assigned. Distance to the shape
            # (coverage) is a weaker question -- a robot can sit on someone
            # else's part of the drawing and count.
            from belief import ARENA_HALF as _AH
            tg = _assigned_targets(worker, k, m)
            st_flags = _per_robot_stopped(worker, k, m, mode)
            pos_raw = node[:, 0:2].cpu().numpy() * _AH
            errs = [float(np.linalg.norm(pos_raw[l] - np.asarray(tg[l])))
                    if tg[l] is not None else None for l in range(m)]
            have = [e for e in errs if e is not None]
            rec["target_err_median"] = float(np.median(have)) if have else None
            for tol in (5.0, 10.0, 20.0):
                near = [e is not None and e < tol for e in errs]
                rec["near_%d" % int(tol)] = float(np.mean(near)) if errs else 0.0
                rec["settled_%d" % int(tol)] = (float(np.mean([n and s2 for n, s2 in zip(near, st_flags)]))
                                                if errs else 0.0)
            if trainer._save_positions:
                # Every sample's positions, in raw arena units, so the run can
                # be replayed as an animation afterwards. ~50 robots x 2 floats
                # per arena per sample: a few MB for a whole evaluation, which
                # is worth being able to SEE the swarm assemble rather than
                # only reading its coverage curve.
                from belief import ARENA_HALF
                rec["pos"] = np.round(node[:, 0:2].cpu().numpy() * ARENA_HALF, 2).tolist()
            per_arena.append(rec)
    agg = {"tick": ticks, "coverage": float(np.mean(covs)) if covs else 0.0,
           "mean_dist": float(np.mean(dists)) if dists else 0.0,
           "stopped": float(np.mean(stops)) if stops else 0.0,
           "arenas": len(covs), "per_arena": per_arena}
    for key in ("near_5", "near_10", "near_20", "settled_5", "settled_10", "settled_20"):
        vals = [a[key] for a in per_arena if a.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else 0.0
    return agg


def _final_state(trainer, workers, cfg, formation_pool, image_names = None):
    """Robot positions and the formation each arena was solving, for the demo plots.

    Positions come out of the snapshot normalised to [-1, 1]; the formation's
    on-pixels are in raw arena units, so positions are scaled by ARENA_HALF to
    put both in the same frame. The shape is subsampled -- a QuickDraw stroke
    can be tens of thousands of pixels and the plot cannot show that anyway.
    """
    from belief import ARENA_HALF, belief_conf
    out = []
    for wi, worker in enumerate(workers):
        for k in range(cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None or snap.get("node") is None:
                continue
            node = snap["node"].cpu().numpy()
            image_id = worker.image_id.get(k) if hasattr(worker, "image_id") else None
            m = node.shape[0]
            entry = {"worker": wi, "arena": k,
                     "image_id": int(image_id) if image_id is not None else None,
                     "pos": (node[:, 0:2] * ARENA_HALF).tolist(), "dist": node[:, 4].tolist(),
                     "target": _assigned_targets(worker, k, m),
                      "stopped_flags": _per_robot_stopped(
                          worker, k, m, {"simple_oracle": "oracle",
                                         "renewal_oracle": "renewal"}.get(cfg.motor_override, "actor"))}
            # Believed-vs-true diagnostic (the wrong-stop attribution): the
            # filter's OWN mean position and confidence per robot, so the
            # latch decision's inputs can be compared with truth after the
            # run. Believed distance here is to the robot's ASSIGNED target
            # (the same frame `target` is in), not to the nearest on-pixel
            # (`dist` above). Robots with no belief entry get null.
            bel = getattr(worker, "belief", {}).get(k, {}) if hasattr(worker, "belief") else {}
            bdist = [None] * m
            bconf = [None] * m
            tgts = entry["target"]
            for l in range(m):
                parts = bel.get(l)
                if parts is None:
                    continue
                mean_xy = parts[:, 0:2].mean(dim = 0).cpu() * ARENA_HALF
                tgt = tgts[l]
                if tgt is not None:
                    bdist[l] = float((mean_xy - torch.tensor(tgt)).norm())
                bconf[l] = float(belief_conf(parts.unsqueeze(0))[0])
            entry["believed_dist"] = bdist
            entry["believed_conf"] = bconf
            if image_id is not None and formation_pool:
                idx = int(image_id) % len(formation_pool)
                if image_names is not None and idx < len(image_names):
                    entry["formation_name"] = image_names[idx]
                pts = np.asarray(formation_pool[idx].points, dtype = float)
                if pts.shape[0] > 3000:
                    pts = pts[::int(np.ceil(pts.shape[0] / 3000.0))]
                entry["shape"] = pts.tolist()
            out.append(entry)
    return out


if __name__ == "__main__":
    main()
