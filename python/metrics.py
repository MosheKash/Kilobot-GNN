"""Logging and aggregation: numbers out of a run and into TensorBoard or JSON.

Logger writes scalars and histograms; rollout_stats turns a rollout payload
into the flat metric dict everything downstream expects; merge_stats and
aggregate_payloads combine those across parallel workers; RunSummary is the
JSON summary the sweep reads. Purely descriptive -- nothing here changes what a
run does.
"""

import os
import math
import json
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAVE_TENSORBOARD = True
except Exception:
    SummaryWriter = None
    HAVE_TENSORBOARD = False


class Logger:
    def __init__(self, logdir):
        self.writer = None
        if logdir is not None and HAVE_TENSORBOARD:
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(logdir)
            print("tensorboard logging to", logdir)
        elif logdir is not None and not HAVE_TENSORBOARD:
            print("tensorboard not installed; console logging only (pip install tensorboard)")

    def log_scalars(self, stats, step):
        if self.writer is None:
            return
        for key in stats:
            value = stats[key]
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def log_hparams(self, cfg_dict):
        if self.writer is None:
            return
        lines = ["| field | value |", "|---|---|"]
        for key in sorted(cfg_dict):
            lines.append("| %s | %s |" % (key, cfg_dict[key]))
        self.writer.add_text("run/config", "\n".join(lines), 0)
        self.writer.flush()

    def log_histograms(self, arrays, step):
        if self.writer is None:
            return
        for key in arrays:
            values = arrays[key]
            if values is None:
                continue
            if hasattr(values, "numel") and values.numel() == 0:
                continue
            self.writer.add_histogram(key, values, step)

    def console(self, it, step, stats):
        # The one line per iteration: objective first, then the two things
        # that predict whether it will keep working (critic fit, policy
        # health), then throughput. The full detail lives in TensorBoard.
        # Under success_metric=settled the settled value IS the repurposed
        # mean_coverage (phase 153/165 convention); labelled accordingly.
        label = "settled" if "reward/accuracy_income_mean" in stats else "cov"
        main = stats.get("objective/settled_fraction", stats.get("rollout/mean_coverage", float("nan")))
        succ = stats.get("objective/success_rate", stats.get("rollout/success_rate", float("nan")))
        stop_n = stats.get("reward/stop_bonus_count", None)
        stop_m = stats.get("reward/stop_bonus_missed", 0.0)
        stop = " stop %d(-%d)" % (stop_n, stop_m) if stop_n is not None else ""
        acc = stats.get("reward/accuracy_income_mean", None)
        accs = " acc %.4f" % acc if acc is not None else ""
        # Mean belief error (|filter mean - truth|, normalized): the
        # localization-health readout. Falling = the swarm is figuring out
        # where it is; flat-high = nobody localizes; rising = the filters are
        # being corrupted (garbage broadcasts fuse as wrong beacons).
        berr = stats.get("belief/mean_belief_err", None)
        berrs = " berr %.3f" % berr if berr is not None else ""
        rs = stats.get("rollout/mean_step_reward", float("nan"))
        pol = stats.get("optim/policy_loss", stats.get("losses/policy", float("nan")))
        val = stats.get("optim/value_loss", stats.get("losses/value", float("nan")))
        ent = stats.get("optim/entropy", stats.get("losses/entropy", float("nan")))
        kl = stats.get("optim/approx_kl", stats.get("ppo/approx_kl", float("nan")))
        ev = stats.get("optim/explained_variance", stats.get("ppo/explained_variance", float("nan")))
        sm = stats.get("policy/std_msg", None)
        st = stats.get("policy/std_motor", None)
        stds = " std_m/i %.2f/%.2f" % (sm, st) if sm is not None and st is not None else ""
        mg = stats.get("grads/message", stats.get("motor/msg_grad_norm", None))
        mgs = " msg_g %.4g" % mg if mg is not None else ""
        sps = stats.get("time/steps_per_sec", float("nan"))
        print("iter %d step %d | %s %.3f succ %.2f r_step %.4f%s%s%s | pol %.4f val %.4f ent %.3f kl %.4f ev %.2f%s | %.0f sps"
              % (it, step, label, main, succ, rs, stop, accs, berrs, pol, val, ent, kl, ev, mgs + stds, sps))

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def rollout_stats(payload):
    stats = {}
    if payload["reward_count"] > 0:
        stats["rollout/mean_step_reward"] = payload["reward_sum"] / payload["reward_count"]
        sfb = payload.get("seed_find_bonus_sum", 0.0)
        wfp = payload.get("wall_find_penalty_sum", 0.0)
        if sfb > 0.0 or wfp > 0.0:
            stats["reward/seed_find_bonus_mean"] = sfb / payload["reward_count"]
            stats["reward/wall_find_penalty_mean"] = wfp / payload["reward_count"]
        cb = payload.get("belief_conf_bonus_sum", 0.0)
        if cb != 0.0:
            stats["reward/belief_conf_bonus_mean"] = cb / payload["reward_count"]
        sh = payload.get("shaping_sum", 0.0)
        if sh != 0.0:
            stats["reward/shaping_mean"] = sh / payload["reward_count"]
        ts = payload.get("target_shaping_sum", 0.0)
        if ts != 0.0:
            stats["reward/target_shaping_mean"] = ts / payload["reward_count"]
        gs = payload.get("global_shaping_sum", 0.0)
        if gs != 0.0:
            stats["reward/global_shaping_mean"] = gs / payload["reward_count"]
        bs = payload.get("belief_shaping_sum", 0.0)
        if bs != 0.0:
            stats["reward/belief_shaping_mean"] = bs / payload["reward_count"]
        oc = payload.get("overclaim_sum", 0.0)
        if oc != 0.0:
            stats["reward/beacon_overclaim_mean"] = oc / payload["reward_count"]
        bec = payload.get("belief_err_count", 0)
        if bec > 0:
            stats["belief/mean_belief_err"] = payload.get("belief_err_sum", 0.0) / bec
        sb = payload.get("stop_bonus_count", 0)
        sm = payload.get("stop_bonus_missed", 0)
        if sb > 0 or sm > 0:
            stats["reward/stop_bonus_count"] = float(sb)
            stats["reward/stop_bonus_missed"] = float(sm)
        ai = payload.get("acc_income_sum", 0.0)
        if ai != 0.0:
            stats["reward/accuracy_income_mean"] = ai / payload["reward_count"]
        tc = payload.get("time_cost_sum", 0.0)
        if tc != 0.0:
            stats["reward/time_cost_mean"] = tc / payload["reward_count"]
    bc = payload.get("belief_count", 0)
    if bc > 0:
        stats["belief/mean_conf_pos"] = payload["conf_pos_sum"] / bc
        stats["belief/mean_conf_x"] = payload["conf_x_sum"] / bc
        stats["belief/mean_conf_y"] = payload["conf_y_sum"] / bc
        stats["belief/frac_localized"] = payload["localized_count"] / bc
    if payload["cov_count"] > 0:
        stats["rollout/mean_coverage"] = payload["cov_sum"] / payload["cov_count"]
    # ep_records (Trainer._ep_records) was already collected, per completed
    # episode, at the exact tick success-or-timeout fires -- present in the
    # payload but never previously surfaced here. mean_coverage above
    # is a per-tick average across the WHOLE rollout, diluted by however
    # much of it robots spent still travelling rather than settled; these
    # two are the direct, per-episode-outcome numbers underneath that
    # average, which is what actually tells apart "coverage is low
    # because robots are still working correctly, just not there yet" from
    # "coverage is low because episodes are ending in failure."
    ep_records = payload.get("ep_records")
    if ep_records:
        stats["rollout/success_rate"] = sum(1 for r in ep_records if r["success"]) / len(ep_records)
        stats["rollout/mean_final_coverage"] = sum(r["coverage"] for r in ep_records) / len(ep_records)
        stats["rollout/mean_episode_length"] = sum(r["length"] for r in ep_records) / len(ep_records)
        stats["rollout/episode_count"] = float(len(ep_records))
    if payload.get("disp_count", 0) > 0:
        stats["rollout/mean_displacement"] = payload["disp_sum"] / payload["disp_count"]
    if payload.get("split_total_events", 0) > 0:
        stats["rollout/split_seed_fraction"] = payload["split_seed_events"] / payload["split_total_events"]
    hb = payload.get("split_heartbeat_events", 0)
    if hb > 0:
        stats["rollout/split_heartbeat_fraction"] = hb / (hb + payload.get("split_total_events", 0))
    stats["rollout/decisions"] = float(payload["decisions"])
    if payload["agent_steps"] > 0:
        stats["rollout/decision_rate"] = payload["decisions"] / payload["agent_steps"]
    eps = payload["ep_records"]
    stats["episodes/count"] = float(len(eps))
    if len(eps) > 0:
        stats["episodes/mean_reward"] = sum(e["reward"] for e in eps) / len(eps)
        stats["episodes/mean_length"] = sum(e["length"] for e in eps) / len(eps)
        stats["episodes/success_rate"] = sum(1.0 for e in eps if e["success"]) / len(eps)
        stats["episodes/mean_final_coverage"] = sum(e["coverage"] for e in eps) / len(eps)
    comp = payload.get("comp")
    if comp and comp["count"] > 0:
        stats["reward/on_fraction"] = comp["on_count"] / comp["count"]
        stats["reward/on_bonus_mean"] = comp["on_bonus_sum"] / comp["count"]
        stats["reward/pack_mean"] = comp["pack_sum"] / comp["count"]
        stats["reward/off_pen_mean"] = comp["off_pen_sum"] / comp["count"]
        stats["reward/sep_mean"] = comp["sep_sum"] / comp["count"]
    return stats


def aggregate_payloads(payloads):
    agg = {"reward_sum": 0.0, "reward_count": 0, "cov_sum": 0.0, "cov_count": 0,
           "disp_sum": 0.0, "disp_count": 0,
           "split_seed_events": 0, "split_total_events": 0, "split_heartbeat_events": 0,
           "seed_find_bonus_sum": 0.0, "wall_find_penalty_sum": 0.0,
           "target_shaping_sum": 0.0, "target_shaping_count": 0,
           "global_shaping_sum": 0.0, "global_shaping_count": 0,
           "belief_shaping_sum": 0.0, "belief_err_sum": 0.0, "belief_err_count": 0,
           "overclaim_sum": 0.0,
           "stop_bonus_count": 0, "stop_bonus_missed": 0,
           "acc_income_sum": 0.0, "time_cost_sum": 0.0,
           "belief_conf_bonus_sum": 0.0, "shaping_sum": 0.0,
           "conf_pos_sum": 0.0, "conf_x_sum": 0.0, "conf_y_sum": 0.0,
           "localized_count": 0, "belief_count": 0,
           "decisions": 0, "agent_steps": 0, "ep_records": [],
           "comp": {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                    "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}}
    for p in payloads:
        agg["reward_sum"] = agg["reward_sum"] + p["reward_sum"]
        agg["reward_count"] = agg["reward_count"] + p["reward_count"]
        agg["cov_sum"] = agg["cov_sum"] + p["cov_sum"]
        agg["cov_count"] = agg["cov_count"] + p["cov_count"]
        agg["disp_sum"] = agg["disp_sum"] + p.get("disp_sum", 0.0)
        agg["disp_count"] = agg["disp_count"] + p.get("disp_count", 0)
        agg["split_seed_events"] = agg["split_seed_events"] + p.get("split_seed_events", 0)
        agg["split_total_events"] = agg["split_total_events"] + p.get("split_total_events", 0)
        agg["split_heartbeat_events"] = agg["split_heartbeat_events"] + p.get("split_heartbeat_events", 0)
        agg["seed_find_bonus_sum"] = agg["seed_find_bonus_sum"] + p.get("seed_find_bonus_sum", 0.0)
        agg["wall_find_penalty_sum"] = agg["wall_find_penalty_sum"] + p.get("wall_find_penalty_sum", 0.0)
        agg["target_shaping_sum"] = agg["target_shaping_sum"] + p.get("target_shaping_sum", 0.0)
        agg["target_shaping_count"] = agg["target_shaping_count"] + p.get("target_shaping_count", 0)
        agg["global_shaping_sum"] = agg["global_shaping_sum"] + p.get("global_shaping_sum", 0.0)
        agg["global_shaping_count"] = agg["global_shaping_count"] + p.get("global_shaping_count", 0)
        agg["belief_shaping_sum"] = agg["belief_shaping_sum"] + p.get("belief_shaping_sum", 0.0)
        agg["belief_err_sum"] = agg["belief_err_sum"] + p.get("belief_err_sum", 0.0)
        agg["belief_err_count"] = agg["belief_err_count"] + p.get("belief_err_count", 0)
        agg["overclaim_sum"] = agg["overclaim_sum"] + p.get("overclaim_sum", 0.0)
        agg["stop_bonus_count"] = agg["stop_bonus_count"] + p.get("stop_bonus_count", 0)
        agg["stop_bonus_missed"] = agg["stop_bonus_missed"] + p.get("stop_bonus_missed", 0)
        agg["acc_income_sum"] = agg["acc_income_sum"] + p.get("acc_income_sum", 0.0)
        agg["time_cost_sum"] = agg["time_cost_sum"] + p.get("time_cost_sum", 0.0)
        agg["belief_conf_bonus_sum"] = agg["belief_conf_bonus_sum"] + p.get("belief_conf_bonus_sum", 0.0)
        agg["shaping_sum"] = agg["shaping_sum"] + p.get("shaping_sum", 0.0)
        agg["conf_pos_sum"] = agg["conf_pos_sum"] + p.get("conf_pos_sum", 0.0)
        agg["conf_x_sum"] = agg["conf_x_sum"] + p.get("conf_x_sum", 0.0)
        agg["conf_y_sum"] = agg["conf_y_sum"] + p.get("conf_y_sum", 0.0)
        agg["localized_count"] = agg["localized_count"] + p.get("localized_count", 0)
        agg["belief_count"] = agg["belief_count"] + p.get("belief_count", 0)
        agg["decisions"] = agg["decisions"] + p["decisions"]
        agg["agent_steps"] = agg["agent_steps"] + p["agent_steps"]
        agg["ep_records"].extend(p["ep_records"])
        pc = p.get("comp")
        if pc:
            for ck in agg["comp"]:
                agg["comp"][ck] = agg["comp"][ck] + pc[ck]
    return agg


def build_histograms(ep_records, buffer, policy, message_size):
    # Deliberately three, not seven: the action distributions (what the policy
    # is actually sampling, motors and messages separately -- the message
    # histogram is how you watch the quiet start open up) and the raw log-std
    # parameter. Advantage/return/value histograms were dropped: the same
    # information is in optim/explained_variance, and episodes/* scalars
    # already carry the per-episode numbers.
    out = {}
    if len(buffer.decisions) > 0:
        actions = torch.stack([d["action"] for d in buffer.decisions])
        out["dist/action_transmission"] = actions[:, :message_size].reshape(-1)
        out["dist/action_motors"] = actions[:, message_size:].reshape(-1)
    out["policy/log_std"] = policy.log_std.detach()
    return {key: value.detach().cpu() for key, value in out.items()}


# ── the TensorBoard view ──────────────────────────────────────────────────────
# merge_stats returns the FULL stats dict, which RunSummary (and therefore the
# sweep) consumes. TensorBoard gets a curated projection of the same dict --
# objective first, the four health charts, and only the reward terms actually
# active in this run's mode. Investigation-era diagnostics (pre-activation
# means, parameter drifts, per-part timings) stay in the full dict for the
# summary but do not chart.

TB_RENAME = {
    "rollout/mean_coverage": "objective/settled_fraction",
    "episodes/success_rate": "objective/success_rate",
    "episodes/mean_reward": "objective/mean_episode_reward",
    "losses/policy": "optim/policy_loss",
    "losses/value": "optim/value_loss",
    "losses/entropy": "optim/entropy",
    "ppo/approx_kl": "optim/approx_kl",
    "ppo/clip_fraction": "optim/clip_fraction",
    "ppo/explained_variance": "optim/explained_variance",
    "motor/grad_norm": "grads/motor",
    "motor/msg_grad_norm": "grads/message",
    "policy/std_msg": "policy/std_message",
    "reward/on_fraction": "diag/on_shape_fraction_any_pixel",
    "belief/mean_conf_pos": "diag/belief_conf_pos",
    "belief/mean_belief_err": "diag/belief_err",
    "belief/frac_localized": "diag/belief_frac_localized",
    "rollout/split_seed_fraction": "diag/split_seed_fraction",
    "rollout/decision_rate": "diag/decision_rate",
    "rollout/mean_displacement": "diag/mean_displacement",
}

# Dropped unconditionally: investigation-era diagnostics whose questions are
# already answered by the charts above (explained_variance covers value fit;
# the settle numbers cover behaviour; drift/saturation/pre-activation charts
# were the phase-6..9 debugging instruments, not run monitors).
TB_DROP = {
    "motor/saturation", "motor/preact_absmean", "motor/out_mean",
    "motor/preact_mean", "motor/param_drift", "motor/msg_param_drift",
    "motor/log_std", "policy/log_std_mean", "policy/std_mean",
    "ppo/actor_grad_norm", "ppo/critic_grad_norm",
    "ppo/adv_mean", "ppo/adv_std", "ppo/mean_value", "ppo/mean_return",
    "ppo/return_std",
    "time/env_step_sec", "time/step_sec", "time/parse_sec",
    "time/getsteps_sec", "time/parse_msgs", "time/snapshots_sec",
    "time/act_sec", "time/collect_sec", "time/ppo_sec",
    "episodes/count", "episodes/mean_length", "episodes/mean_final_coverage",
    "rollout/mean_step_reward", "rollout/success_rate",
    "rollout/mean_final_coverage", "rollout/mean_episode_length",
    "rollout/decisions",
}

# In reward_mode="target" the any-pixel income is not part of the reward
# (only its diagnostic on_fraction survives, renamed above), so its component
# charts would chart a term that pays nothing.
TB_TARGET_MODE_DROP = {"reward/on_bonus_mean", "reward/pack_mean", "reward/off_pen_mean"}


def tb_view(stats, cfg):
    """The curated TensorBoard projection of a full stats dict.

    Unknown keys PASS THROUGH (so bc/*, oracle/* and any future tag stays
    visible by default); only the explicit drop sets above are removed, and
    the rename map puts the run's charts into one reading order: objective,
    reward, optim, grads, policy, diag, time.
    """
    mode_drop = TB_TARGET_MODE_DROP if getattr(cfg, "reward_mode", "normal") == "target" else set()
    out = {}
    for key, value in stats.items():
        if key in TB_DROP or key in mode_drop:
            continue
        out[TB_RENAME.get(key, key)] = value
    return out


def merge_stats(ppo_logs, timing, rollout_metrics):
    stats = {"losses/policy": ppo_logs["actor_loss"],
             "losses/value": ppo_logs["critic_loss"],
             "losses/entropy": ppo_logs["entropy"],
             "ppo/approx_kl": ppo_logs["approx_kl"],
             "ppo/clip_fraction": ppo_logs["clip_fraction"],
             "ppo/explained_variance": ppo_logs["explained_variance"],
             "ppo/actor_grad_norm": ppo_logs["actor_grad_norm"],
             "ppo/critic_grad_norm": ppo_logs["critic_grad_norm"],
             "ppo/adv_mean": ppo_logs["adv_mean"],
             "ppo/adv_std": ppo_logs["adv_std"],
             "ppo/mean_value": ppo_logs["mean_value"],
             "ppo/mean_return": ppo_logs["mean_return"],
             "ppo/return_std": ppo_logs["return_std"],
             "motor/grad_norm": ppo_logs["motor_grad_norm"],
             "motor/msg_grad_norm": ppo_logs["msg_grad_norm"],
             "motor/saturation": ppo_logs["motor_saturation"],
             "motor/preact_absmean": ppo_logs["motor_preact_absmean"],
             "motor/out_mean": ppo_logs["motor_out_mean"],
             "motor/preact_mean": ppo_logs["motor_preact_mean"],
             "motor/param_drift": ppo_logs["motor_param_drift"],
             "motor/msg_param_drift": ppo_logs["msg_param_drift"],
             "motor/log_std": ppo_logs["log_std_motor"],
             "policy/log_std_mean": ppo_logs["log_std_mean"],
             "policy/std_mean": ppo_logs["std_mean"],
             "policy/std_msg": ppo_logs.get("std_msg", 0.0),
             "policy/std_motor": ppo_logs.get("std_motor_eff", 0.0),
             "time/iter_seconds": timing["iter_seconds"],
             "time/steps_per_sec": timing["rollout_steps"] / timing["iter_seconds"] if timing["iter_seconds"] > 0 else 0.0,
             "time/env_step_sec": timing["env_step_sec"],
             "time/step_sec": timing["step_sec"],
             "time/parse_sec": timing["parse_sec"],
             "time/getsteps_sec": timing["getsteps_sec"],
             "time/parse_msgs": timing["parse_msgs"],
             "time/snapshots_sec": timing["snapshots_sec"],
             "time/act_sec": timing["act_sec"]}
    stats.update(rollout_metrics)
    return stats


def _slope(values):
    # Least-squares slope of values against their index. Used to measure whether
    # a quantity (entropy) is trending up or down over a run, which is more
    # robust than comparing only the first and last point.
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - mean_x
        num = num + dx * (y - mean_y)
        den = den + dx * dx
    if den == 0.0:
        return 0.0
    return num / den


# Keys pulled from the per-iteration stats dict into the run summary. The summary
# is what the hyperparameter sweep reads, so it is deliberately small and stable.
SUMMARY_KEYS = {
    "coverage": "rollout/mean_coverage",
    "final_coverage": "episodes/mean_final_coverage",
    "success_rate": "episodes/success_rate",
    "entropy": "losses/entropy",
    "explained_variance": "ppo/explained_variance",
    "adv_std": "ppo/adv_std",
    "adv_mean": "ppo/adv_mean",
    "mean_return": "ppo/mean_return",
    "return_std": "ppo/return_std",
    "motor_grad_norm": "motor/grad_norm",
    "msg_grad_norm": "motor/msg_grad_norm",
    "motor_saturation": "motor/saturation",
    "motor_preact_absmean": "motor/preact_absmean",
    "motor_out_mean": "motor/out_mean",
    "motor_preact_mean": "motor/preact_mean",
    "motor_param_drift": "motor/param_drift",
    "msg_param_drift": "motor/msg_param_drift",
    "log_std_motor": "motor/log_std",
    "mean_displacement": "rollout/mean_displacement",
    "split_seed_fraction": "rollout/split_seed_fraction",
    "policy_loss": "losses/policy",
    "value_loss": "losses/value",
    "log_std_mean": "policy/log_std_mean",
    "std_mean": "policy/std_mean",
    "approx_kl": "ppo/approx_kl",
    "on_fraction": "reward/on_fraction",
    "on_bonus_mean": "reward/on_bonus_mean",
    "pack_mean": "reward/pack_mean",
    "off_pen_mean": "reward/off_pen_mean",
}


class RunSummary:
    # Writes a small JSON file describing how a run is going, updated every
    # iteration so an outside process (the sweep) can poll it for early-stopping
    # and read the final result. The write is atomic so a reader never sees a
    # half-written file.
    def __init__(self, path):
        self.path = path
        self.history = []
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._write(done=False, status="starting")

    def update(self, it, stats):
        row = {"iter": int(it)}
        for name, key in SUMMARY_KEYS.items():
            v = stats.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                row[name] = float(v)
        self.history.append(row)
        self._write(done=False, status="running")

    def finalize(self, status="ok"):
        self._write(done=True, status=status)

    def _derived(self):
        out = {}
        if not self.history:
            return out
        cov = [r["coverage"] for r in self.history if "coverage" in r]
        ent = [r["entropy"] for r in self.history if "entropy" in r]
        ev = [r["explained_variance"] for r in self.history if "explained_variance" in r]
        if cov:
            out["coverage_initial"] = cov[0]
            out["coverage_final"] = cov[-1]
            out["coverage_max"] = max(cov)
            out["coverage_mean"] = sum(cov) / len(cov)
        if ent:
            out["entropy_initial"] = ent[0]
            out["entropy_final"] = ent[-1]
            out["entropy_slope"] = _slope(ent)
        if ev:
            out["explained_variance_final"] = ev[-1]
            out["explained_variance_max"] = max(ev)
        return out

    def _write(self, done, status):
        payload = {"done": done, "status": status, "iterations": len(self.history)}
        payload.update(self._derived())
        payload["history"] = self.history
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)
