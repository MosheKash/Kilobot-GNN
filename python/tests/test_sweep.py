import os
import ast
import json
import tempfile

from sweep import compute_objective, progress_gain
from metrics import RunSummary, _slope


def test_confirm_best_carries_forward_every_swept_parameter():
    # regression test: confirm_best previously only carried 8 of the 10
    # parameters suggest_params actually sweeps (r_pack and pack_range were
    # silently dropped), meaning the multi-seed confirmation step never
    # actually tested the sweep's true best config for those two -- it
    # silently fell back to Config's defaults instead. AST-based rather than
    # calling confirm_best directly, since that requires a real optuna study
    # and launches real subprocesses; this checks the same invariant
    # structurally: every name suggest_params suggests must appear as a
    # best["name"] lookup somewhere in confirm_best's source.
    src = open(os.path.join(os.path.dirname(__file__), "..", "sweep.py")).read()
    tree = ast.parse(src)

    suggested_names = set()
    confirm_best_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "suggest_params":
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and hasattr(n.func, "attr") and "suggest" in n.func.attr:
                    if n.args:
                        suggested_names.add(ast.literal_eval(n.args[0]))
        if isinstance(node, ast.FunctionDef) and node.name == "confirm_best":
            confirm_best_src = ast.get_source_segment(src, node)

    assert suggested_names, "suggest_params produced no names -- test itself is broken"
    assert confirm_best_src is not None, "confirm_best not found"
    for name in suggested_names:
        assert ('best["%s"]' % name) in confirm_best_src, \
            "suggest_params sweeps '%s' but confirm_best never reads best['%s']" % (name, name)


def test_slope_sign():
    assert _slope([1.0, 2.0, 3.0, 4.0]) > 0.0
    assert _slope([4.0, 3.0, 2.0, 1.0]) < 0.0
    assert abs(_slope([2.0, 2.0, 2.0])) < 1e-9
    assert _slope([5.0]) == 0.0


def test_objective_rewards_coverage_gain():
    summ = {"coverage_initial": 0.20, "coverage_final": 0.40, "iterations": 20,
            "entropy_slope": 0.0, "explained_variance_final": 0.3}
    score, bd = compute_objective(summ, ent_penalty=0.5)
    assert abs(score - 0.20) < 1e-9
    assert abs(bd["coverage_gain"] - 0.20) < 1e-9
    assert bd["entropy_penalty"] == 0.0


def test_objective_penalizes_entropy_climb():
    flat = {"coverage_initial": 0.20, "coverage_final": 0.30, "iterations": 21,
            "entropy_slope": 0.0}
    climb = {"coverage_initial": 0.20, "coverage_final": 0.30, "iterations": 21,
             "entropy_slope": 0.02}  # rises ~0.4 total over 20 steps
    s_flat, _ = compute_objective(flat, ent_penalty=0.5)
    s_climb, bd = compute_objective(climb, ent_penalty=0.5)
    assert s_climb < s_flat
    # penalty = 0.5 * (0.02 * 20) = 0.2
    assert abs(bd["entropy_penalty"] - 0.2) < 1e-9


def test_objective_falling_entropy_not_penalized():
    summ = {"coverage_initial": 0.20, "coverage_final": 0.25, "iterations": 21,
            "entropy_slope": -0.05}
    score, bd = compute_objective(summ, ent_penalty=0.5)
    assert bd["entropy_penalty"] == 0.0
    assert abs(score - 0.05) < 1e-9


def test_objective_ev_bonus():
    summ = {"coverage_initial": 0.20, "coverage_final": 0.20, "iterations": 10,
            "entropy_slope": 0.0, "explained_variance_final": 0.5}
    no_bonus, _ = compute_objective(summ, ent_penalty=0.5, ev_bonus=0.0)
    bonus, bd = compute_objective(summ, ent_penalty=0.5, ev_bonus=0.1)
    assert bonus > no_bonus
    assert abs(bd["ev_term"] - 0.05) < 1e-9


def test_objective_missing_coverage_is_failed():
    score, bd = compute_objective({"iterations": 5}, ent_penalty=0.5)
    assert score is None
    assert "reason" in bd


def test_progress_gain():
    assert abs(progress_gain({"coverage_initial": 0.2, "coverage_final": 0.35}) - 0.15) < 1e-9
    assert progress_gain({"coverage_final": 0.35}) is None


def test_runsummary_roundtrip_and_derived():
    path = os.path.join(tempfile.mkdtemp(), "summary.json")
    s = RunSummary(path)
    # before any update, the file exists and marks not done
    started = json.load(open(path))
    assert started["done"] is False
    s.update(0, {"rollout/mean_coverage": 0.20, "losses/entropy": 10.0,
                 "ppo/explained_variance": -0.1})
    s.update(1, {"rollout/mean_coverage": 0.26, "losses/entropy": 10.2,
                 "ppo/explained_variance": 0.1})
    s.update(2, {"rollout/mean_coverage": 0.31, "losses/entropy": 10.4,
                 "ppo/explained_variance": 0.25})
    s.finalize()
    d = json.load(open(path))
    assert d["done"] is True
    assert d["iterations"] == 3
    assert abs(d["coverage_initial"] - 0.20) < 1e-9
    assert abs(d["coverage_final"] - 0.31) < 1e-9
    assert abs(d["coverage_max"] - 0.31) < 1e-9
    assert d["entropy_slope"] > 0.0           # entropy rose
    assert abs(d["explained_variance_final"] - 0.25) < 1e-9
    # a full objective can be computed from the finalized summary
    score, bd = compute_objective(d, ent_penalty=0.5)
    assert score is not None


def test_runsummary_skips_nan():
    path = os.path.join(tempfile.mkdtemp(), "summary.json")
    s = RunSummary(path)
    s.update(0, {"rollout/mean_coverage": 0.2, "episodes/success_rate": float("nan")})
    s.finalize()
    row = json.load(open(path))["history"][0]
    assert "success_rate" not in row
    assert "coverage" in row


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all sweep tests passed")
