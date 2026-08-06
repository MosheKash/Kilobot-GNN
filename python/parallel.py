"""ParallelTrainer: several worker processes collecting into one update.

Each worker owns its own player and runs its own rollout; this module packs a
RolloutBuffer for transport over a pipe, merges what comes back, and combines
per-worker stats. The learning is unchanged and single-process -- only
collection is parallel.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
import copy
import pickle
import multiprocessing as mp
import torch

from buffer import RolloutBuffer
from ppo import ppo_update
from metrics import Logger
from trainer import check_startup_coverage, conf_bonus_schedule
from metrics import rollout_stats, aggregate_payloads
from checkpoint import save_checkpoint, export_actor
from kilobot_gnn import DB_ROW_SIZE


def _pack_decisions(decisions):
    if len(decisions) == 0:
        return {"D": 0}
    gru = decisions[0].get("prev_hidden") is not None
    db_sizes = [0 if gru else d["prev_db"].shape[0] for d in decisions]
    out = {"D": len(decisions),
           "step_index": torch.tensor([d["step_index"] for d in decisions]),
           "local": torch.tensor([d["local"] for d in decisions]),
           "z": torch.stack([d["z"] for d in decisions]),
           "seed": torch.stack([d["seed"] for d in decisions]),
           "transmission": torch.stack([d["transmission"] for d in decisions]),
           "action": torch.stack([d["action"] for d in decisions]),
           "old_log_prob": torch.stack([d["old_log_prob"] for d in decisions]),
           "db_sizes": torch.tensor(db_sizes),
           "prev_db": torch.zeros(0, DB_ROW_SIZE) if gru else (
               torch.cat([d["prev_db"] for d in decisions], dim=0) if sum(db_sizes) > 0 else torch.zeros(0, DB_ROW_SIZE))}
    if gru:
        out["prev_hidden"] = torch.stack([d["prev_hidden"] for d in decisions])
        out["prop"] = torch.stack([d["prop"] for d in decisions])
    return out


def pack_buffer(buffer):
    steps = buffer.steps
    s = len(steps)
    if s == 0:
        return {"S": 0, "decisions": _pack_decisions(buffer.decisions)}
    return {"S": s,
            "m_sizes": torch.tensor([st["node"].shape[0] for st in steps]),
            "e_sizes": torch.tensor([st["edge_index"].shape[1] for st in steps]),
            "node": torch.cat([st["node"] for st in steps], dim=0),
            "edge_index": torch.cat([st["edge_index"] for st in steps], dim=1),
            "edge_attr": torch.cat([st["edge_attr"] for st in steps], dim=0),
            "z": torch.stack([st["z"] for st in steps]),
            "traj_id": torch.cat([st["traj_id"] for st in steps]),
            "reward": torch.cat([st["reward"] for st in steps]),
            "term": torch.cat([st["term"] for st in steps]),
            "cut": torch.cat([st["cut"] for st in steps]),
            "is_decision": torch.cat([st["is_decision"] for st in steps]),
            "arena_id": torch.tensor([st["arena_id"] for st in steps]),
            "env_step": torch.tensor([st["env_step"] for st in steps]),
            "decisions": _pack_decisions(buffer.decisions)}


def unpack_buffer(packed, cfg):
    buf = RolloutBuffer(cfg)
    s = packed["S"]
    if s > 0:
        m_sizes = packed["m_sizes"].tolist()
        e_sizes = packed["e_sizes"].tolist()
        node = torch.split(packed["node"], m_sizes, dim=0)
        ei = torch.split(packed["edge_index"], e_sizes, dim=1)
        ea = torch.split(packed["edge_attr"], e_sizes, dim=0)
        traj = torch.split(packed["traj_id"], m_sizes)
        rew = torch.split(packed["reward"], m_sizes)
        term = torch.split(packed["term"], m_sizes)
        cut = torch.split(packed["cut"], m_sizes)
        dec = torch.split(packed["is_decision"], m_sizes)
        z = packed["z"]
        arena = packed["arena_id"].tolist()
        env_step = packed["env_step"].tolist()
        for i in range(s):
            buf.steps.append({"arena_id": arena[i], "env_step": env_step[i],
                              "node": node[i], "edge_index": ei[i], "edge_attr": ea[i],
                              "z": z[i], "traj_id": traj[i], "reward": rew[i],
                              "term": term[i], "cut": cut[i], "is_decision": dec[i]})
    pd = packed["decisions"]
    if pd["D"] > 0:
        db_parts = torch.split(pd["prev_db"], pd["db_sizes"].tolist(), dim=0)
        si = pd["step_index"].tolist()
        loc = pd["local"].tolist()
        gru = "prev_hidden" in pd
        for j in range(pd["D"]):
            dec = {"step_index": si[j], "local": loc[j],
                   "z": pd["z"][j], "seed": pd["seed"][j],
                   "transmission": pd["transmission"][j],
                   "prev_db": None if gru else db_parts[j],
                   "action": pd["action"][j], "old_log_prob": pd["old_log_prob"][j]}
            if gru:
                dec["prev_hidden"] = pd["prev_hidden"][j]
                dec["prop"] = pd["prop"][j]
            buf.decisions.append(dec)
    return buf


def merge_buffers(parts, cfg):
    merged = RolloutBuffer(cfg)
    step_offset = 0
    traj_offset = 0
    for steps, decisions in parts:
        local_max = 0
        for s in steps:
            s2 = dict(s)
            s2["traj_id"] = s["traj_id"] + traj_offset
            if s["traj_id"].numel() > 0:
                local_max = max(local_max, int(s["traj_id"].max()))
            merged.steps.append(s2)
        for d in decisions:
            d2 = dict(d)
            d2["step_index"] = d["step_index"] + step_offset
            merged.decisions.append(d2)
        step_offset = step_offset + len(steps)
        traj_offset = traj_offset + local_max
    return merged


def build_stats(ppo_logs, payloads, timings, dt, t_collect, t_ppo, total_steps):
    stats = {"losses/policy": ppo_logs["actor_loss"],
             "losses/value": ppo_logs["critic_loss"],
             "losses/entropy": ppo_logs["entropy"],
             "ppo/approx_kl": ppo_logs["approx_kl"],
             "ppo/clip_fraction": ppo_logs["clip_fraction"],
             "ppo/explained_variance": ppo_logs["explained_variance"],
             "ppo/adv_mean": ppo_logs["adv_mean"],
             "ppo/adv_std": ppo_logs["adv_std"],
             "ppo/mean_return": ppo_logs["mean_return"],
             "ppo/return_std": ppo_logs["return_std"],
             "ppo/mean_value": ppo_logs["mean_value"],
             "ppo/actor_grad_norm": ppo_logs["actor_grad_norm"],
             "motor/grad_norm": ppo_logs["motor_grad_norm"],
             "motor/msg_grad_norm": ppo_logs["msg_grad_norm"],
             "motor/saturation": ppo_logs["motor_saturation"],
             "motor/preact_absmean": ppo_logs["motor_preact_absmean"],
             "motor/out_mean": ppo_logs["motor_out_mean"],
             "motor/preact_mean": ppo_logs["motor_preact_mean"],
             "motor/param_drift": ppo_logs["motor_param_drift"],
             "motor/msg_param_drift": ppo_logs["msg_param_drift"],
             "motor/log_std": ppo_logs["log_std_motor"],
             "ppo/critic_grad_norm": ppo_logs["critic_grad_norm"],
             "policy/log_std_mean": ppo_logs["log_std_mean"],
             "policy/std_mean": ppo_logs["std_mean"],
             "time/iter_seconds": dt,
             "time/steps_per_sec": total_steps / dt if dt > 0 else 0.0,
             "time/collect_sec": t_collect,
             "time/ppo_sec": t_ppo}
    stats.update(rollout_stats(aggregate_payloads(payloads)))
    if len(timings) > 0:
        stats["time/worker_step_max"] = max(t["step"] for t in timings)
        stats["time/worker_act_max"] = max(t["act"] for t in timings)
        stats["time/worker_parse_max"] = max(t["parse"] for t in timings)
    return stats


def _worker_cfg(cfg):
    # workers always run the actor and the environment on CPU, regardless of what
    # device the learner trains on -- only the actor/policy objects in worker_loop
    # were being forced to CPU, but self.cfg.device is read directly all over
    # trainer.py to build the actor's own input tensors (_act, _gather_split_state),
    # so a worker with cfg.device left at "cuda" builds cuda-resident inputs for a
    # cpu-resident actor and crashes on the first forward pass. Returns a copy so
    # this cannot affect the learner's own (correctly cuda) config or other workers.
    cfg = copy.copy(cfg)
    cfg.device = "cpu"
    return cfg


def worker_loop(worker_id, cfg, in_q, out_q):
    torch.set_num_threads(1)
    import launch
    from trainer import Trainer
    from policy import GaussianPolicy

    cfg = _worker_cfg(cfg)
    env, cc, pc = launch.make_env(worker_id)
    try:
        env.reset()
        behavior_name = launch.resolve_behavior(env)
        encoder = launch.load_encoder(launch.ENCODER_PATH, "cpu", expected_dim=launch.Z)
        pool = launch.build_image_pool(launch.FORMATIONS_DIR, launch.preprocess, limit=launch.MAX_FORMATIONS)

        actor = launch.build_actor(cfg).to("cpu")
        policy = GaussianPolicy(actor, cfg.log_std_init).to("cpu")
        trainer = Trainer(env, cc, pc, cfg, encoder, pool, behavior_name)
        trainer.setup()
        out_q.put(("READY", worker_id))

        while True:
            msg = in_q.get()
            if msg == "STOP":
                break
            payload = pickle.loads(msg)
            if isinstance(payload, dict) and "policy" in payload:
                policy.load_state_dict(payload["policy"])
                cfg.belief_conf_bonus = payload.get("belief_conf_bonus", getattr(cfg, "belief_conf_bonus", 0.0))
            else:
                policy.load_state_dict(payload)
            buffer = trainer.collect(policy, None)
            packed = pack_buffer(buffer)
            del buffer
            blob = pickle.dumps((worker_id, packed, trainer.rollout_payload(), trainer.collect_timing()))
            del packed
            out_q.put(("DATA", blob))
    finally:
        env.close()


class ParallelTrainer:
    def __init__(self, cfg, num_workers, worker_entry=worker_loop):
        self.cfg = cfg
        self.num_workers = num_workers
        self.get_timeout = 30.0
        self.collect_max_wait = float(getattr(cfg, "collect_max_wait", 1200.0))
        self.ctx = mp.get_context("spawn")
        self.out_q = self.ctx.Queue()
        self.in_qs = []
        self.procs = []
        for i in range(num_workers):
            in_q = self.ctx.Queue()
            p = self.ctx.Process(target=worker_entry, args=(i, cfg, in_q, self.out_q))
            p.daemon = True
            p.start()
            self.in_qs.append(in_q)
            self.procs.append(p)
        self._await_ready()

    def _await_ready(self):
        ready = 0
        while ready < self.num_workers:
            if not any(p.is_alive() for p in self.procs) and self.out_q.empty():
                raise RuntimeError("a worker process died during startup")
            try:
                msg = self.out_q.get(timeout=1.0)
            except Exception:
                continue
            if msg[0] == "READY":
                ready = ready + 1
        print("all %d workers ready" % self.num_workers)

    def run(self, policy, critic, actor_opt, critic_opt, iterations, logger=None,
            ckpt_path=None, ckpt_every=0, start_iter=0, min_start_cov=0.0, summary=None):
        if logger is None:
            logger = Logger(None)
        logger.log_hparams(vars(self.cfg))
        total_steps = self.cfg.rollout_steps * self.num_workers
        step = start_iter * total_steps
        bonus0 = getattr(self.cfg, "belief_conf_bonus", 0.0)
        bonus_iters = getattr(self.cfg, "belief_conf_bonus_iters", 0)
        summary_status = "error"
        try:
            for it in range(start_iter, iterations):
                t0 = time.time()
                self.cfg.belief_conf_bonus = conf_bonus_schedule(bonus0, bonus_iters, it)
                state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                blob = pickle.dumps({"policy": state, "belief_conf_bonus": self.cfg.belief_conf_bonus})
                for q in self.in_qs:
                    q.put(blob)

                parts = []
                payloads = []
                timings = []
                got = 0
                waited = 0.0
                while got < self.num_workers:
                    try:
                        msg = self.out_q.get(timeout=self.get_timeout)
                    except Exception:
                        dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
                        if dead:
                            raise RuntimeError("worker(s) %s died mid-run at iter %d; aborting "
                                               "(latest checkpoint retained)" % (dead, it))
                        waited = waited + self.get_timeout
                        if waited >= self.collect_max_wait:
                            raise RuntimeError("workers stalled at iter %d: no data for %.0fs; aborting "
                                               "(latest checkpoint retained)" % (it, waited))
                        continue
                    if msg[0] != "DATA":
                        continue
                    wid, packed, payload, timing = pickle.loads(msg[1])
                    buf = unpack_buffer(packed, self.cfg)
                    parts.append((buf.steps, buf.decisions))
                    payloads.append(payload)
                    timings.append(timing)
                    got = got + 1
                t_collect = time.time() - t0

                buffer = merge_buffers(parts, self.cfg)
                del parts
                t1 = time.time()
                ppo_logs = ppo_update(policy, critic, actor_opt, critic_opt, buffer, self.cfg)
                del buffer
                t_ppo = time.time() - t1
                dt = time.time() - t0
                step = step + total_steps

                stats = build_stats(ppo_logs, payloads, timings, dt, t_collect, t_ppo, total_steps)
                logger.log_scalars(stats, step)
                logger.console(it, step, stats)
                if summary is not None:
                    summary.update(it, stats)
                print("  time: collect %.2fs (worker step max %.2f) merge+ppo %.2fs | %d workers x %d steps"
                      % (t_collect, stats.get("time/worker_step_max", 0.0), t_ppo, self.num_workers, self.cfg.rollout_steps))

                if it == start_iter:
                    check_startup_coverage(stats, min_start_cov)

                if ckpt_path is not None and ckpt_every > 0 and (it + 1) % ckpt_every == 0:
                    save_checkpoint(ckpt_path, it + 1, policy, critic, actor_opt, critic_opt)
                    print("  checkpoint saved: %s (iter %d)" % (ckpt_path, it + 1))

            if ckpt_path is not None:
                save_checkpoint(ckpt_path, iterations, policy, critic, actor_opt, critic_opt)
                actor_path = os.path.join(os.path.dirname(ckpt_path) or ".", "actor_final.pt")
                export_actor(actor_path, policy)
                print("  final actor exported: %s" % actor_path)
            summary_status = "ok"
        finally:
            if summary is not None:
                summary.finalize(status=summary_status)
            for q in self.in_qs:
                q.put("STOP")
            # a worker only checks for STOP between rollouts, not during one,
            # and a single rollout can legitimately take as long as
            # collect_max_wait already assumes elsewhere in this same class
            # (1200s / 20 minutes by default) -- a much shorter join timeout
            # here would force-terminate a worker that's simply still mid-
            # rollout, killing it before its own env.close() (parallel.py's
            # worker_loop, in its own finally block) ever gets a chance to
            # run, orphaning the Unity subprocess underneath it
            join_timeout = getattr(self.cfg, "collect_max_wait", 1200.0) + 60.0
            for p in self.procs:
                p.join(timeout=join_timeout)
                if p.is_alive():
                    p.terminate()
            logger.close()
