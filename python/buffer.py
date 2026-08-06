"""RolloutBuffer: the container one iteration of experience is collected into.

Storage and batching only -- appends per-step graph snapshots and per-decision
records, then yields minibatches. It computes nothing about them; rewards are
reward.py's job and advantages are gae.py's.
"""

import torch
from graph_batch import build_critic_batch
from gae import compute_gae


class RolloutBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = []
        self.decisions = []
        self.values = []
        self.advantages = []
        self.returns = []

    def add_step(self, arena_id, env_step, node, edge_index, edge_attr, z,
                 traj_id, reward, term, cut, is_decision):
        self.steps.append({
            "arena_id": arena_id,
            "env_step": env_step,
            "node": node,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "z": z,
            "traj_id": traj_id,
            "reward": reward,
            "term": term,
            "cut": cut,
            "is_decision": is_decision
        })
        return len(self.steps) - 1

    def add_decision(self, step_index, local, z, seed, transmission, prev_db, action, old_log_prob, bc_target=None, prev_hidden=None, prop=None, arrived_target=None, was_turning=False, oracle_state=None):
        self.decisions.append({
            "step_index": step_index,
            "local": local,
            "z": z,
            "seed": seed,
            "transmission": transmission,
            "prev_db": prev_db,
            "action": action,
            "old_log_prob": old_log_prob,
            "bc_target": bc_target,
            "prev_hidden": prev_hidden,
            "prop": prop,
            "arrived_target": arrived_target,
            "was_turning": was_turning,
            "oracle_state": oracle_state
        })

    def _old_values(self, critic):
        device = next(critic.parameters()).device
        chunk = getattr(self.cfg, "critic_chunk_steps", 0)
        n = len(self.steps)
        if not chunk or chunk <= 0:
            chunk = n if n > 0 else 1
        flats = []
        with torch.no_grad():
            for i in range(0, n, chunk):
                sl = self.steps[i:i + chunk]
                nodes = [s["node"] for s in sl]
                edge_indices = [s["edge_index"] for s in sl]
                edge_attrs = [s["edge_attr"] for s in sl]
                zs = [s["z"] for s in sl]
                x, edge_attr, edge_index, z, batch = build_critic_batch(nodes, edge_indices, edge_attrs, zs)
                f = critic(x.to(device), edge_attr.to(device), edge_index.to(device), z.to(device), batch.to(device)).squeeze(-1).cpu()
                flats.append(f)
        flat = torch.cat(flats) if flats else torch.zeros(0)

        values = []
        offset = 0
        for s in self.steps:
            m = s["node"].shape[0]
            values.append(flat[offset:offset + m])
            offset += m
        return values

    def critic_chunks(self, chunk_size):
        # Yield batched critic inputs and matching targets, chunk_size steps at a
        # time, so the critic forward and backward stay within GPU memory.
        # Requires self.returns to be populated (call compute_returns first).
        n = len(self.steps)
        if not chunk_size or chunk_size <= 0:
            chunk_size = n if n > 0 else 1
        for i in range(0, n, chunk_size):
            sl = self.steps[i:i + chunk_size]
            nodes = [s["node"] for s in sl]
            edge_indices = [s["edge_index"] for s in sl]
            edge_attrs = [s["edge_attr"] for s in sl]
            zs = [s["z"] for s in sl]
            x, edge_attr, edge_index, z, batch = build_critic_batch(nodes, edge_indices, edge_attrs, zs)
            targets = torch.cat(self.returns[i:i + chunk_size], dim=0)
            yield x, edge_attr, edge_index, z, batch, targets

    def compute_returns(self, critic):
        cfg = self.cfg
        self.values = self._old_values(critic)
        self.advantages = [torch.zeros_like(v) for v in self.values]
        self.returns = [torch.zeros_like(v) for v in self.values]

        groups = {}
        for si in range(len(self.steps)):
            traj = self.steps[si]["traj_id"]
            for local in range(traj.shape[0]):
                key = int(traj[local])
                groups.setdefault(key, []).append((self.steps[si]["env_step"], si, local))

        for key in groups:
            items = sorted(groups[key])
            rewards = torch.stack([self.steps[si]["reward"][local] for _, si, local in items])
            values = torch.stack([self.values[si][local] for _, si, local in items])
            term = torch.stack([self.steps[si]["term"][local] for _, si, local in items])
            cut = torch.stack([self.steps[si]["cut"][local] for _, si, local in items])
            boot = values

            adv, ret = compute_gae(rewards, values, term, cut, boot, cfg.gamma, cfg.gae_lambda)

            for idx in range(len(items)):
                _, si, local = items[idx]
                self.advantages[si][local] = adv[idx]
                self.returns[si][local] = ret[idx]

    def decision_advantages(self):
        out = []
        for d in self.decisions:
            out.append(self.advantages[d["step_index"]][d["local"]])
        return torch.stack(out) if out else torch.zeros(0)
