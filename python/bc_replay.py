"""BCReservoir: the per-oracle-state sample store behaviour cloning fits from.

A BC rollout window sits inside one phase of a very long episode, so fitting it
alone teaches the current phase and unlearns the rest. This keeps a bounded
reservoir PER oracle state and serves balanced minibatches from all of them.
Storage and sampling policy only; the fit itself is bc.py.
"""

import math
import os
import time
import torch

# config.py's own bc_replay_capacity has the full rationale for why this
# module exists at all. The five states are simple_oracle.py's own, in its
# own order; UNLABELLED collects anything a teacher other than simple_oracle
# produced, so a non-simple_oracle BC run still trains on all of its data
# instead of silently dropping it.
BC_STATES = ("go_north", "turning", "wall_following", "navigating", "arrived")
ARRIVED = "arrived"
UNLABELLED = "_unlabelled"
_KEYS = ("tc", "prop", "h", "tgt", "arrived", "arrived_valid", "born")
RESERVOIR_VERSION = 1


class BCReservoir:
    # A per-state store of oracle-labelled BC samples that persists across
    # iterations, so a fit is no longer restricted to whichever single
    # episode phase the current rollout window happens to sit in.
    def __init__(self, cfg):
        self.capacity = int(getattr(cfg, "bc_replay_capacity", 0))
        self.max_age = int(getattr(cfg, "bc_replay_max_age", 0))
        self.balanced = bool(getattr(cfg, "bc_replay_balanced", True))
        self.evict = str(getattr(cfg, "bc_replay_evict", "random"))
        self.min_samples = int(getattr(cfg, "bc_replay_min_samples", 512))
        self.skip_arrived_motor = bool(getattr(cfg, "bc_motor_skip_arrived", False))
        self.store = {}
        self.iteration = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(int(getattr(cfg, "seed", 0)) + 90210)

    @property
    def enabled(self):
        return self.capacity > 0

    def counts(self):
        return {k: int(v["tgt"].shape[0]) for k, v in self.store.items()}

    def total(self):
        return sum(int(v["tgt"].shape[0]) for v in self.store.values())

    def add(self, decisions, iteration):
        # Everything is kept on cpu regardless of cfg.device: the store is
        # large by design and a minibatch is a few hundred KB, so holding it
        # in system RAM and moving one batch at a time costs nothing
        # measurable while leaving VRAM entirely for the fit itself.
        self.iteration = int(iteration)
        groups = {}
        for d in decisions:
            if d.get("bc_target") is None or d.get("prev_hidden") is None:
                continue
            key = d.get("oracle_state") or UNLABELLED
            groups.setdefault(key, []).append(d)
        for key in groups:
            batch = self._pack(groups[key], iteration)
            cur = self.store.get(key)
            if cur is None:
                merged = batch
            else:
                merged = {k: torch.cat([cur[k], batch[k]]) for k in _KEYS}
            self.store[key] = self._trim(merged)

    def _pack(self, ds, iteration):
        n = len(ds)
        arrived = torch.zeros(n, 1)
        arrived_valid = torch.zeros(n, dtype = torch.bool)
        for i, d in enumerate(ds):
            at = d.get("arrived_target")
            if at is not None:
                arrived[i] = at.detach().reshape(1).cpu()
                arrived_valid[i] = True
        return {
            "tc": torch.stack([d["transmission"] for d in ds]).detach().cpu().clone(),
            "prop": torch.stack([d["prop"] for d in ds]).detach().cpu().clone(),
            "h": torch.stack([d["prev_hidden"] for d in ds]).detach().cpu().clone(),
            "tgt": torch.stack([d["bc_target"] for d in ds]).detach().cpu().clone(),
            "arrived": arrived,
            "arrived_valid": arrived_valid,
            "born": torch.full((n,), float(iteration))
        }

    def _trim(self, s):
        n = int(s["tgt"].shape[0])
        if self.max_age > 0:
            keep = (s["born"] >= float(self.iteration - self.max_age)).nonzero().squeeze(-1)
            if int(keep.numel()) < n:
                s = {k: s[k][keep] for k in _KEYS}
                n = int(keep.numel())
        if n <= self.capacity:
            return s
        if self.evict == "fifo":
            keep = torch.arange(n - self.capacity, n)
        else:
            keep = torch.randperm(n, generator = self.rng)[:self.capacity]
        return {k: s[k][keep] for k in _KEYS}

    def sample(self, size, device):
        # Balanced draws an equal share from every non-empty state, so a
        # state holding 0.4% of all collected samples still occupies its
        # full share of every minibatch. Remainder goes to the states with
        # the most stored samples, which only ever adds data, never skews a
        # rare state down.
        keys = [k for k in self.store if int(self.store[k]["tgt"].shape[0]) > 0]
        if not keys:
            return None
        if self.balanced:
            # A state that has only just appeared holds a handful of samples,
            # and hooking it straight up to a full equal share means a
            # minibatch built almost entirely out of copies of those few
            # samples. Measured directly: the first time "arrived" entered a
            # real run it had 32 stored samples, took 1/5 of every minibatch
            # (roughly 200 copies each), and both the fit loss and the
            # cold-start error got worse for the next fifteen iterations
            # before recovering. A state under min_samples therefore ramps in
            # linearly instead of arriving at full weight, and whatever share
            # it does not take is redistributed across the states that do
            # have enough data to use it.
            weights = {}
            for k in keys:
                n = self._size(k)
                weights[k] = min(1.0, n / float(max(1, self.min_samples)))
                if self.skip_arrived_motor and k == ARRIVED:
                    # config.py's own bc_motor_skip_arrived has the rationale.
                    # arrived rows contribute nothing to the motor loss, so
                    # they get only the share the arrived head needs rather
                    # than a full fifth of the batch.
                    weights[k] = weights[k] * float(getattr(self, "arrived_batch_share", 0.2))
            total_w = sum(weights.values())
            if total_w <= 0.0:
                return None
            keys = sorted(keys, key = self._size, reverse = True)
            picks = []
            assigned = 0
            for i, k in enumerate(keys):
                want = int(round(size * weights[k] / total_w))
                want = max(1, min(want, size - assigned - (len(keys) - i - 1)))
                assigned = assigned + want
                picks.append(self._draw(k, want))
        else:
            sizes = torch.tensor([float(self._size(k)) for k in keys])
            share = (sizes / sizes.sum() * size).round().to(torch.long)
            picks = []
            for i, k in enumerate(keys):
                want = max(1, int(share[i]))
                picks.append(self._draw(k, want))
        out = {}
        for key in ("tc", "prop", "h", "tgt", "arrived", "arrived_valid"):
            out[key] = torch.cat([p[key] for p in picks]).to(device)
        # Which state each row came from, so the caller can treat "arrived"
        # rows differently from the rest without a second sampling pass.
        out["is_arrived"] = torch.cat([p["is_arrived"] for p in picks]).to(device)
        return out

    def save(self, path):
        # Persisted next to the checkpoint because the reservoir IS training
        # state, not a cache. Measured on a real restart: an interrupted run
        # came back with arrived at 416 samples instead of 11515 and turning
        # at 2841 instead of 9664, and the arrived head's own held-out recall
        # collapsed from 0.414 to 0.039 within ten iterations -- rare states
        # take a whole run to accumulate, so losing them undoes the thing this
        # class exists to do. Written atomically (temp + os.replace, the same
        # pattern checkpoint.py already uses) so an interruption during the
        # write itself can never leave a corrupt file at the real path.
        meta = {"tc": None, "prop": None, "h": None}
        for k in self.store:
            meta = {"tc": int(self.store[k]["tc"].shape[1]),
                    "prop": int(self.store[k]["prop"].shape[1]),
                    "h": int(self.store[k]["h"].shape[1])}
            break
        payload = {"version": RESERVOIR_VERSION, "store": self.store,
                   "iteration": self.iteration, "meta": meta}
        tmp = path + ".tmp"
        t0 = time.time()
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return os.path.getsize(path) / (1024.0 * 1024.0), time.time() - t0

    def load(self, path, expect_input = None, expect_h = None):
        # Refuses a reservoir whose observation widths do not match the actor
        # being trained. use_turn_anchor alone changes the combined tc+prop
        # width, so a silently-loaded mismatched store would either crash deep
        # inside a forward pass or, worse, train on misaligned columns.
        # tc and prop are checked as their sum, since that is exactly what
        # up1 consumes and the split between them is not separately
        # recoverable from the layer.
        if not os.path.exists(path):
            return False, "no file"
        try:
            payload = torch.load(path, map_location = "cpu", weights_only = False)
        except Exception as exc:
            return False, "unreadable (%s)" % exc
        if not isinstance(payload, dict) or payload.get("version") != RESERVOIR_VERSION:
            return False, "version mismatch"
        meta = payload.get("meta") or {}
        if expect_input is not None and meta.get("tc") is not None and meta.get("prop") is not None:
            stored = int(meta["tc"]) + int(meta["prop"])
            if stored != int(expect_input):
                return False, ("shape mismatch: stored tc+prop=%d, actor expects %d"
                               % (stored, int(expect_input)))
        if expect_h is not None and meta.get("h") is not None and int(meta["h"]) != int(expect_h):
            return False, "shape mismatch: stored hidden=%s, actor expects %s" % (meta["h"], expect_h)
        self.store = payload["store"]
        self.iteration = int(payload.get("iteration", 0))
        return True, "ok"

    def natural_arrived_rate(self):
        # The share of stored samples whose oracle state is "arrived". This is
        # the class prior the arrived head should be calibrated to, and it is
        # NOT the rate a balanced minibatch has -- balancing hands every state
        # an equal share regardless of how common it really is.
        total = self.total()
        if total <= 0:
            return 0.0
        return self._size(ARRIVED) / float(total) if ARRIVED in self.store else 0.0

    def _size(self, key):
        return int(self.store[key]["tgt"].shape[0])

    def _draw(self, key, want):
        s = self.store[key]
        n = self._size(key)
        idx = torch.randint(0, n, (int(want),), generator = self.rng)
        out = {k: s[k][idx] for k in ("tc", "prop", "h", "tgt", "arrived", "arrived_valid")}
        out["is_arrived"] = torch.full((int(want),), key == ARRIVED, dtype = torch.bool)
        return out
