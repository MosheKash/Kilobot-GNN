"""diag_arrived_head.py -- arrived-head calibration sweep on a recorded tape.

Loads a run_o3-style actor checkpoint into a GaussianPolicy built with the same
cfg flags used in eval (activation=elu, use_arrived_head=True, oracle_head=True,
etc.), rolls it over tape_val.pt (or any tape file) from cold start, and reports
the raw head_arrived logit distribution and PRECISION/RECALL/F1 at a sweep of
thresholds (0.5, 0.7, 0.9, 0.95, 0.99) against the tape's arrived labels.

This is the diagnostic for the "under-stops" problem: if the head's val-tape
recall at 0.95 is already 0.99 (as reported in history.jsonl), then the
deployment recall gap (stopped 0.73 vs oracle 0.99) is NOT the head's
calibration -- it is a LOCALISATION shift in the actor's particle-filter
features that changes what the head sees at deployment.

Usage:
  python tools/diag_arrived_head.py [--weights PATH] [--tape PATH] [--device cuda]

Defaults:
  --weights  ../results/bc_v2/run_o3/actor_best.pt
  --tape     ../results/bc_v2/tape_val.pt
  --device   cuda
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kilobot_gnn import PROP_CONF_POS, PROP_SIN_T, PROP_COS_T
from bc_offline import load_tape, forward_chunk


def build_actor_from_checkpoint(blob, device="cpu"):
    """Reconstruct the SplitObservationActor from a checkpoint's state dict.

    Reads the widths from the state dict (widths_from_state_dict) and builds the
    actor with the cfg flags recorded in meta.
    """
    from kilobot_gnn import (
        SplitObservationActor, widths_from_state_dict,
    )
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    widths = widths_from_state_dict(blob["actor"] if "actor" in blob else blob)
    if not widths:
        raise SystemExit("Not a split-actor checkpoint (no gru/ head1/ up1 weights)")

    actor = SplitObservationActor(
        upscale_hidden=widths.get("upscale_hidden", 40),
        gru_hidden=widths.get("gru_hidden", 59),
        head_hidden=widths.get("head_hidden", 40),
        use_arrived_head=bool(meta.get("use_arrived_head", False)),
        use_turn_anchor=bool(meta.get("use_turn_anchor", False)),
        recurrent=True,
        activation=meta.get("activation", "elu"),
        use_state_head=bool(meta.get("use_state_head", False)),
        use_wall_head=bool(meta.get("use_wall_head", False)),
        use_steer_feature=bool(meta.get("use_steer_feature", False)),
        use_oracle_head=bool(meta.get("use_oracle_head", False)),
        oracle_residual=float(meta.get("oracle_residual", 0.05)),
        oracle_residual_turn=float(meta.get("oracle_residual_turn", 0.0)),
    )
    return actor.to(device)


def feature_margin_analysis(conf_pos, sin_cos):
    """Among arrived-labeled rows, compute feature margins.

    (a) CONF_POS distribution: how confident the particle filter is about position
    (b) Target bearing resultant length rt = sqrt(sin_t^2 + cos_t^2) -- this is
        the mean-resultant length of the belief's target bearing.

    Returns a dict with "conf_pos" and/or "target_rt" sub-dicts.
    """
    result = {}
    if conf_pos.numel() == 0:
        return {"error": "No arrived-positive rows"}

    result["conf_pos"] = {
        "mean": float(conf_pos.mean()),
        "median": float(conf_pos.median()),
        "p10": float(conf_pos.quantile(0.10)),
        "p25": float(conf_pos.quantile(0.25)),
        "p75": float(conf_pos.quantile(0.75)),
        "p90": float(conf_pos.quantile(0.90)),
        "min": float(conf_pos.min()),
        "max": float(conf_pos.max()),
    }

    if sin_cos.numel() > 0:
        rt = (sin_cos ** 2).sum(dim=-1).sqrt()
        result["target_rt"] = {
            "mean": float(rt.mean()),
            "median": float(rt.median()),
            "p10": float(rt.quantile(0.10)),
            "p25": float(rt.quantile(0.25)),
            "p75": float(rt.quantile(0.75)),
            "p90": float(rt.quantile(0.90)),
            "min": float(rt.min()),
            "max": float(rt.max()),
        }
        result["rt_gt_09"] = float((rt > 0.9).float().mean())
        result["rt_gt_095"] = float((rt > 0.95).float().mean())
        result["rt_lt_05"] = float((rt < 0.5).float().mean())

    return result


def calibration_sweep(actor, tape, device, thresholds=None):
    """Roll actor over tape, compute head_arrived P/R/F1 at each threshold.

    Returns (results_dict, logits, conf_pos, sin_cos).
    """
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]

    actor.eval()
    T, R = tape["valid"].shape

    all_logits = []
    all_labels = []
    conf_pos_vals = []
    sin_t_vals = []
    cos_t_vals = []

    for b0 in range(0, R, 128):
        b1 = min(b0 + 128, R)
        h = actor.initial_hidden(b1 - b0, device=device)
        for t0 in range(0, T, 256):
            t1 = min(t0 + 256, T)
            v = tape["valid"][t0:t1, b0:b1].to(device)
            if not bool(v.any()):
                continue
            tc = tape["tc"][t0:t1, b0:b1].to(device).float()
            prop = tape["prop"][t0:t1, b0:b1].to(device).float()

            motors, logit, g, h, s_log, w_log = forward_chunk(actor, tc, prop, v, h)

            if logit is not None:
                av = v & tape["arrived_valid"][t0:t1, b0:b1].to(device)
                if bool(av.any()):
                    sig = torch.sigmoid(logit).squeeze(-1)
                    lab = tape["arrived"][t0:t1, b0:b1].to(device).float() > 0.5

                    all_logits.append(sig[av].float().cpu())
                    all_labels.append(lab[av].float().cpu())
                    conf_pos_vals.append(prop[av, PROP_CONF_POS].float().cpu())
                    sin_t_vals.append(prop[av, PROP_SIN_T].float().cpu())
                    cos_t_vals.append(prop[av, PROP_COS_T].float().cpu())

    if not all_logits:
        print("ERROR: no arrived-validated rows found in tape")
        return {}, torch.tensor([]), torch.tensor([]), torch.tensor([])

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    conf_pos = torch.cat(conf_pos_vals)
    sin_t = torch.cat(sin_t_vals)
    cos_t = torch.cat(cos_t_vals)
    sin_cos = torch.stack([sin_t, cos_t], dim=-1)

    results = {}
    for thr in thresholds:
        pred = logits > thr
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[thr] = {
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    return results, logits, conf_pos, sin_cos


def print_sweep_table(results):
    """Print a human-readable table of the calibration sweep."""
    print("\n" + "=" * 80)
    print("ARRIVED-HEAD CALIBRATION SWEEP (tape_val.pt, run_o3 actor)")
    print("=" * 80)
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  "
          f"{'TP':>8}  {'FP':>8}  {'FN':>8}  {'TN':>8}")
    print("-" * 80)
    for thr in sorted(results.keys()):
        r = results[thr]
        print(f"{thr:>10.2f}  {r['precision']:>10.4f}  {r['recall']:>10.4f}  "
              f"{r['f1']:>10.4f}  {r['tp']:>8d}  {r['fp']:>8d}  "
              f"{r['fn']:>8d}  {r['tn']:>8d}")
    print("=" * 80)

    # Recall degradation analysis
    print("\nRecall degradation as threshold rises:")
    prev_rec = None
    prev_thr = None
    for thr in sorted(results.keys()):
        r = results[thr]
        if prev_rec is not None:
            delta = prev_rec - r["recall"]
            print(f"  {prev_thr:.2f} -> {thr:.2f}: recall drops {delta:+.4f} "
                  f"({r['fn']} false negatives)")
        else:
            print(f"  {thr:.2f}: recall = {r['recall']:.4f} "
                  f"({r['fn']} false negatives)")
        prev_rec = r["recall"]
        prev_thr = thr
    print()


def print_feature_analysis(analysis):
    """Print feature margin analysis results."""
    print("\n" + "=" * 80)
    print("FEATURE-MARGIN ANALYSIS (arrived-positive rows on tape)")
    print("=" * 80)

    if "conf_pos" in analysis:
        cp = analysis["conf_pos"]
        print("\n(a) CONF_POS distribution (PROP_CONF_POS = index 12):")
        print(f"    mean={cp['mean']:.4f}  median={cp['median']:.4f}  "
              f"p10={cp['p10']:.4f}  p25={cp['p25']:.4f}  "
              f"p75={cp['p75']:.4f}  p90={cp['p90']:.4f}  "
              f"min={cp['min']:.4f}  max={cp['max']:.4f}")

    if "target_rt" in analysis:
        rt = analysis["target_rt"]
        print(f"\n(b) Target bearing resultant length rt = sqrt(sin_t^2 + cos_t^2):")
        print(f"    mean={rt['mean']:.4f}  median={rt['median']:.4f}  "
              f"p10={rt['p10']:.4f}  p25={rt['p25']:.4f}  "
              f"p75={rt['p75']:.4f}  p90={rt['p90']:.4f}  "
              f"min={rt['min']:.4f}  max={rt['max']:.4f}")
        print(f"    P(rt > 0.9) = {analysis.get('rt_gt_09', 0):.4f}")
        print(f"    P(rt > 0.95) = {analysis.get('rt_gt_095', 0):.4f}")
        print(f"    P(rt < 0.5) = {analysis.get('rt_lt_05', 0):.4f}")

    print("\nInterpretation:")
    print("  - High conf_pos (> 0.8) means the particle filter is confident.")
    print("  - High rt (> 0.9) means particles strongly agree on target bearing.")
    print("  - A closed-form rule (dist < tau_v * ARENA_HALF) fires when both high.")
    print("  - If arrived rows have low conf_pos or low rt, the filter may")
    print("    genuinely mislocalize at deployment, causing the head to under-fire.")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Arrived-head calibration sweep on a recorded tape")
    parser.add_argument("--weights", default=None,
                        help="Actor checkpoint path")
    parser.add_argument("--tape", default=None,
                        help="Tape file path")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Sweep thresholds (default: 0.5 0.7 0.9 0.95 0.99)")
    args = parser.parse_args(argv)

    if args.weights is None:
        args.weights = "/home/moshe/Projects/Kilobot-GNN/results/bc_v2/run_o3/actor_best.pt"
    if args.tape is None:
        args.tape = "/home/moshe/Projects/Kilobot-GNN/results/bc_v2/tape_val.pt"

    device = torch.device(args.device)

    # Load checkpoint
    print("Loading actor checkpoint: %s" % args.weights, flush=True)
    blob = torch.load(args.weights, map_location="cpu", weights_only=False)
    actor = build_actor_from_checkpoint(blob, device=device)
    n_params = sum(p.numel() for p in actor.parameters())
    meta = blob.get("meta", {})
    print("Actor: %d parameters, activation=%s, arrived_head=%s, oracle_head=%s" % (
        n_params,
        meta.get("activation", "?"),
        bool(meta.get("use_arrived_head", False)),
        bool(meta.get("use_oracle_head", False)),
    ), flush=True)

    # Load tape
    print("Loading tape: %s" % args.tape, flush=True)
    tape = load_tape(args.tape, device=device)
    print("Tape: %d steps x %d robots, %d arrived-validated decisions" % (
        tape["valid"].shape[0], tape["valid"].shape[1],
        int((tape["valid"] & tape["arrived_valid"]).sum())), flush=True)

    # Run sweep
    thresholds = args.thresholds if args.thresholds else [0.5, 0.7, 0.9, 0.95, 0.99]
    print("\nRunning calibration sweep at thresholds: %s" %
          ", ".join("%.2f" % t for t in thresholds), flush=True)
    results, logits, conf_pos, sin_cos = calibration_sweep(
        actor, tape, device, thresholds=thresholds)

    # Print tables
    print_sweep_table(results)
    analysis = feature_margin_analysis(conf_pos, sin_cos)
    print_feature_analysis(analysis)


if __name__ == "__main__":
    main()
