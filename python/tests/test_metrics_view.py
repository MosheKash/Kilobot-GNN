"""Tests for the curated TensorBoard view (docs/tuning.md phase 166).

merge_stats/build_stats return the FULL stats dict (RunSummary and the sweep
consume it); tb_view is the TensorBoard-only projection. The contract:
objective-first naming, clutter dropped, mode-inappropriate reward terms
dropped, and unknown keys pass through untouched.
"""

import math

from metrics import tb_view, Logger


def _full_stats():
    return {
        # objective (renamed)
        "rollout/mean_coverage": 0.512,
        "episodes/success_rate": 0.33,
        "episodes/mean_reward": 12.5,
        # optim (renamed)
        "losses/policy": -0.012,
        "losses/value": 0.11,
        "losses/entropy": -21.4,
        "ppo/approx_kl": 0.003,
        "ppo/clip_fraction": 0.02,
        "ppo/explained_variance": 0.87,
        # grads (renamed)
        "motor/grad_norm": 0.0034,
        "motor/msg_grad_norm": 0.0012,
        # policy (renamed)
        "policy/std_msg": 0.1,
        "policy/std_motor": 0.58,
        # clutter (dropped)
        "motor/saturation": 0.0,
        "motor/param_drift": 0.001,
        "motor/msg_param_drift": 0.002,
        "motor/preact_absmean": 0.4,
        "motor/out_mean": 0.5,
        "motor/preact_mean": -0.1,
        "motor/log_std": -0.5,
        "policy/log_std_mean": -0.6,
        "policy/std_mean": 0.55,
        "ppo/actor_grad_norm": 0.01,
        "ppo/critic_grad_norm": 0.02,
        "ppo/adv_mean": 0.0,
        "ppo/adv_std": 0.5,
        "ppo/mean_value": 1.0,
        "ppo/mean_return": 1.1,
        "ppo/return_std": 0.2,
        "time/env_step_sec": 0.3,
        "time/step_sec": 0.2,
        "time/parse_sec": 0.01,
        "time/getsteps_sec": 0.0,
        "time/parse_msgs": 100,
        "time/snapshots_sec": 0.1,
        "time/act_sec": 0.2,
        "episodes/count": 3.0,
        "episodes/mean_length": 300.0,
        "episodes/mean_final_coverage": 0.6,
        "rollout/mean_step_reward": 0.01,
        "rollout/success_rate": 0.33,
        "rollout/mean_final_coverage": 0.6,
        "rollout/mean_episode_length": 300.0,
        "rollout/decisions": 999.0,
        # reward terms
        "reward/on_fraction": 0.6,
        "reward/on_bonus_mean": 0.01,
        "reward/pack_mean": 0.005,
        "reward/off_pen_mean": -0.004,
        "reward/sep_mean": -0.001,
        "reward/accuracy_income_mean": 0.0084,
        "reward/time_cost_mean": -0.0012,
        "reward/target_shaping_mean": -0.4,
        "reward/global_shaping_mean": 0.05,
        "reward/stop_bonus_count": 312.0,
        "reward/stop_bonus_missed": 18.0,
        # diagnostics (renamed)
        "belief/mean_conf_pos": 0.4,
        "belief/frac_localized": 0.3,
        "rollout/split_seed_fraction": 0.2,
        "rollout/decision_rate": 0.5,
        "rollout/mean_displacement": 0.01,
        # unknown, must pass through
        "future/some_tag": 1.0,
    }


class _Cfg:
    reward_mode = "normal"


class _CfgTarget:
    reward_mode = "target"


def test_tb_view_renames_and_drops_clutter():
    v = tb_view(_full_stats(), _Cfg())
    assert v["objective/settled_fraction"] == 0.512
    assert v["objective/success_rate"] == 0.33
    assert v["optim/explained_variance"] == 0.87
    assert v["grads/message"] == 0.0012
    assert v["grads/motor"] == 0.0034
    assert v["policy/std_message"] == 0.1
    assert v["policy/std_motor"] == 0.58
    assert v["diag/on_shape_fraction_any_pixel"] == 0.6
    assert v["future/some_tag"] == 1.0
    for dropped in ("motor/saturation", "motor/param_drift", "motor/msg_param_drift",
                    "motor/preact_absmean", "motor/out_mean", "motor/preact_mean",
                    "motor/log_std", "policy/log_std_mean", "policy/std_mean",
                    "ppo/actor_grad_norm", "ppo/critic_grad_norm", "ppo/adv_mean",
                    "ppo/adv_std", "ppo/mean_value", "ppo/mean_return", "ppo/return_std",
                    "time/env_step_sec", "time/step_sec", "time/parse_sec",
                    "time/getsteps_sec", "time/parse_msgs", "time/snapshots_sec",
                    "time/act_sec", "episodes/count", "episodes/mean_length",
                    "episodes/mean_final_coverage", "rollout/mean_step_reward",
                    "rollout/success_rate", "rollout/mean_final_coverage",
                    "rollout/mean_episode_length", "rollout/decisions"):
        assert dropped not in v, dropped


def test_tb_view_target_mode_drops_dropped_income_terms():
    cfg = _CfgTarget()
    v = tb_view(_full_stats(), cfg)
    for dropped in ("reward/on_bonus_mean", "reward/pack_mean", "reward/off_pen_mean"):
        assert dropped not in v, dropped
    # the diagnostic any-pixel fraction and the live terms survive
    assert v["diag/on_shape_fraction_any_pixel"] == 0.6
    assert v["reward/accuracy_income_mean"] == 0.0084
    assert v["reward/sep_mean"] == -0.001


def test_tb_view_normal_mode_keeps_legacy_income_terms():
    v = tb_view(_full_stats(), _Cfg())
    assert v["reward/on_bonus_mean"] == 0.01
    assert v["reward/pack_mean"] == 0.005
    assert v["reward/off_pen_mean"] == -0.004


def test_console_line_formats_with_active_terms(capsys):
    stats = _full_stats()
    stats["time/steps_per_sec"] = 139.0
    Logger(None).console(42, 43008, stats)
    line = capsys.readouterr().out.strip()
    assert line.startswith("iter 42 step 43008 | settled 0.512")
    assert "stop 312(-18)" in line
    assert "acc 0.0084" in line
    assert "msg_g 0.0012" in line
    assert "std_m/i 0.10/0.58" in line
    assert "ev 0.87" in line


def test_console_line_nan_safe_without_episodes(capsys):
    stats = {"rollout/mean_coverage": float("nan"),
             "episodes/success_rate": float("nan"),
             "losses/policy": 0.0, "losses/value": 0.0, "losses/entropy": 0.0,
             "ppo/approx_kl": 0.0, "ppo/explained_variance": 0.0,
             "rollout/mean_step_reward": 0.0,
             "time/steps_per_sec": 100.0}
    Logger(None).console(0, 0, stats)
    line = capsys.readouterr().out.strip()
    assert line.startswith("iter 0 step 0 | cov nan")
    assert math.isnan(float(line.split("cov ")[1].split()[0]))
