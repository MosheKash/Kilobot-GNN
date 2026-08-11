"""tape_eval.py -- score checkpoints against tapes, as a table.

The one number that matters for behaviour cloning is not "how well does this
checkpoint fit the oracle's own trajectories" but "how well does it fit the
trajectories IT produces", and those two are wildly different here: a checkpoint
that matched the oracle on 97% of held-out decisions matched on 12% of the
decisions along its own rollouts, with the arrived head firing wrongly on 59% of
them. Reading both, side by side, is what makes a DAgger round's effect legible
-- each round fixes the previous round's distribution, and the table shows
exactly that.

Every tape is scored the same way bc_offline.evaluate scores: roll the network
through each recorded sequence from a cold start.

usage:
  python tools/tape_eval.py --checkpoints ../results/bc_v2/run_r*/actor_best.pt \
      --tapes ../results/bc_v2/tape_val.pt ../results/bc_v2/dagger_*.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bc_offline import evaluate, load_tape
from config import Config
from kilobot_gnn import build_actor


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs = "+", required = True)
    ap.add_argument("--tapes", nargs = "+", required = True)
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--arrived-threshold", type = float, default = 0.95)
    args = ap.parse_args(argv)

    tapes = [(os.path.basename(p), p) for p in args.tapes]
    for ck in args.checkpoints:
        blob = torch.load(ck, map_location = "cpu", weights_only = False)
        meta = blob.get("meta", {}) or {}
        cfg = Config()
        cfg.actor_type = "gru_split_observation"
        cfg.use_arrived_head = bool(meta.get("use_arrived_head", True))
        cfg.use_turn_anchor = bool(meta.get("use_turn_anchor", True))
        cfg.split_activation = meta.get("activation", "relu")
        cfg.use_state_head = bool(meta.get("use_state_head", False))
        cfg.use_wall_head = bool(meta.get("use_wall_head", False))
        cfg.use_steer_feature = bool(meta.get("use_steer_feature", False))
        cfg.use_oracle_head = bool(meta.get("use_oracle_head", False))
        cfg.oracle_residual = float(meta.get("oracle_residual", 0.05))
        cfg.oracle_residual_turn = float(meta.get("oracle_residual_turn", 0.0))
        from kilobot_gnn import widths_from_state_dict
        for _k, _v in widths_from_state_dict(blob["actor"] if "actor" in blob else blob).items():
            setattr(cfg, "split_" + _k, _v)
        actor = build_actor(cfg).to(args.device)
        actor.load_state_dict(blob["actor"] if "actor" in blob else blob)
        print("== %s  (activation=%s)" % (ck, cfg.split_activation), flush = True)
        print("   %-30s %9s %8s %9s %10s" % ("tape", "balanced", "within", "turning", "arrivedFP%"))
        for name, path in tapes:
            tape = load_tape(path, args.device)
            sc = evaluate(actor, tape, args.device, arrived_threshold = args.arrived_threshold)
            fp = sc.get("arrived_fp", 0)
            tn = sc.get("arrived_tn", 0)
            print("   %-30s %9.4f %8.3f %9s %10.3f"
                  % (name, sc.get("balanced", float("nan")), sc.get("within_all", float("nan")),
                     ("%.4f" % sc["turning"]) if "turning" in sc else "n/a",
                     100.0 * fp / max(fp + tn, 1)), flush = True)
            del tape
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
