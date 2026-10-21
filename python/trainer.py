"""The rollout/update loop: orchestration only.

Trainer owns the cycle -- step the players, gather observations, ask the policy
for actions, score them, and hand a full RolloutBuffer to ppo_update. It does
not implement any of those steps: observations come from observation.py, the
per-decision action path from actor_io.act, rewards from reward.py, the update
from ppo.py. Anything here that starts computing rather than sequencing belongs
in one of those.
"""

import time
import os
import torch
from concurrent.futures import ThreadPoolExecutor
import belief as belief_mod
import actor_io
from belief import belief_read, belief_population_stats, _matched_generator
from kilobot_gnn import MESSAGE_SIZE
from buffer import RolloutBuffer
from reward import (compute_rewards, coverage, reward_components, belief_confidence_bonus,
                    seed_wall_find_reward, assigned_target_dists, separation_reward,
                    accuracy_income, target_kernel,
                    belief_errors, arena_potential, honest_confidences,
                    beacon_overclaim_penalty)
from ppo import ppo_update
from metrics import Logger, rollout_stats, build_histograms, merge_stats, tb_view
from env_worker import EnvWorker
from checkpoint import save_checkpoint, export_actor

try:
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
    HAVE_MLAGENTS = True
except Exception:
    ActionTuple = None
    EnvironmentParametersChannel = None
    HAVE_MLAGENTS = False


STRENGTH_COL = MESSAGE_SIZE + 1


def _sorted_png_position(folder, name):
    """Sorted position of a formation filename within a folder, matching Unity's
    ImageLibrary.files[] ordering (Directory.GetFiles(dir, "*.png") sorted with
    StringComparer.Ordinal). This is the index Unity uses for the floor
    background texture AND the baked distance field, so it is the only safe
    integer to send back over send_image/send_reset. Returns the folder's
    position of `name`, or -1 if it is not present (so a caller can fall back).
    """
    full = sorted(n for n in os.listdir(folder) if n.endswith(".png"))
    try:
        return full.index(name)
    except ValueError:
        return -1


def sample_message(messages, generator=None):
    n = messages.shape[0]
    if n == 1:
        return messages[0]
    weights = messages[:, STRENGTH_COL].clamp(min=0.0)
    if float(weights.sum()) <= 0.0:
        weights = torch.ones(n, device = messages.device)
    index = int(torch.multinomial(weights, 1, generator=_matched_generator(generator, weights.device)))
    return messages[index]


def critic_extra_features(cfg):
    if getattr(cfg, "critic_belief_features", False) and cfg.actor_type == "gru_split_observation":
        from belief import BELIEF_FEATURES
        return BELIEF_FEATURES
    return 0


def make_critic(cfg):
    from kilobot_gnn import Critic, NODE_FEATURES
    return Critic(node_features = NODE_FEATURES + critic_extra_features(cfg))


def conf_bonus_schedule(base, horizon, it):
    if base <= 0.0 or horizon <= 0:
        return base
    frac = 1.0 - it / float(horizon)
    if frac < 0.0:
        frac = 0.0
    return base * frac


def shaping_schedule(base, iterations, it):
    """The proposal's shaping curriculum: k for the first 1/6, k*0.4 for the
    middle 1/3, 0 for the final half. With the proposal's 300 iterations that
    lands on exactly 5.0 (0-50), 2.0 (50-150), 0.0 (150-300), which are the
    plan's stated phase boundaries.
    """
    if base <= 0.0:
        return 0.0
    if iterations <= 0:
        return base
    if it < iterations / 6.0:
        return base
    if it < iterations / 2.0:
        return base * 0.4
    return 0.0


def check_startup_coverage(stats, min_start_cov, success_metric="coverage"):
    cov0 = stats.get("rollout/mean_coverage", 0.0)
    # Under success_metric=settled the mean_coverage value IS the settled
    # fraction, and a spawn value near zero is EXPECTED -- a robot dropped at
    # random is rarely within stop_radius of ITS OWN assigned point (the
    # 0.286 chance floor belongs to any-pixel coverage, which this is not).
    # The "encoder or formations may not have loaded" warning does not apply
    # to it; only the explicit min_start_cov abort is honoured, and that is
    # the operator's own chosen bar.
    if success_metric == "settled":
        if min_start_cov > 0.0 and cov0 < min_start_cov:
            raise RuntimeError("iter-0 settled fraction %.3f below KILOBOT_MIN_START_COV %.3f; aborting"
                               % (cov0, min_start_cov))
        return
    if cov0 < 0.05:
        print("  WARNING: iter-0 mean coverage %.3f is near zero - the encoder or "
              "formations may not have loaded correctly" % cov0)
    if min_start_cov > 0.0 and cov0 < min_start_cov:
        raise RuntimeError("iter-0 coverage %.3f below KILOBOT_MIN_START_COV %.3f; aborting"
                           % (cov0, min_start_cov))


class Trainer:
    def __init__(self, env, critic_channel, params_channel, cfg, encoder, image_pool, behavior_name, image_names = None, formations_dir = None):
        self.cfg = cfg
        self.encoder = encoder
        self.image_pool = image_pool
        self.image_names = image_names
        self.formations_dir = formations_dir
        self.workers = [EnvWorker(env, critic_channel, params_channel, behavior_name)]
        self._init_globals()

    @classmethod
    def from_workers(cls, workers, cfg, encoder, image_pool, image_names = None, formations_dir = None):
        self = cls.__new__(cls)
        self.cfg = cfg
        self.encoder = encoder
        self.image_pool = image_pool
        self.image_names = image_names
        self.formations_dir = formations_dir
        self.workers = list(workers)
        self._init_globals()
        return self

    def _init_globals(self):
        self.traj_counter = 0
        # The generator every belief.py generator=... call site shares. Built
        # from a real torch.device, and preceded by an actual tensor allocation
        # on that device: torch.Generator accepts a device argument before a
        # CUDA context is established, so a generator constructed too early can
        # succeed here and mismatch at first use. belief._matched_generator is
        # the second line of defence. See docs/code-history.md.
        target_device = torch.device(self.cfg.device)
        if target_device.type == "cuda":
            torch.zeros(1, device = target_device)
        self.sample_rng = torch.Generator(device = target_device)
        self.sample_rng.manual_seed(int(self.cfg.seed))
        self._pool = None
        self._critic_extra = critic_extra_features(self.cfg)
        belief_mod.set_layout(getattr(self.cfg, "seed_layout", "corners"))
        belief_mod.set_peer_conf_min(getattr(self.cfg, "peer_conf_min", 0.85))
        if getattr(self.cfg, "actor_type", None) == "gru_split_observation" and getattr(self.cfg, "heartbeat_ticks", 0) <= 0:
            print("WARNING: gru_split_observation with heartbeat_ticks=0 (the default). A robot that "
                  "never comes into range of a seed, wall seed, or neighbor never gets a single decision "
                  "and its motor command stays at zero for the entire episode. Measured directly "
                  "(temp_test_material/probe_stationary2.py): ~26% of robots stuck the whole rollout "
                  "without heartbeat, 0% with heartbeat_ticks=48. Set KILOBOT_HEARTBEAT_TICKS "
                  "(or --heartbeat on rl_driver.py) unless this is intentional.")
        if getattr(self.cfg, "belief_conf_bonus", 0.0) > 0.0 and getattr(self.cfg, "belief_conf_bonus_iters", 0) <= 0:
            print(("WARNING: belief_conf_bonus=%.3g with belief_conf_bonus_iters=0 (the default). This "
                  "bonus is scaffolding meant to be annealed to zero over the run (see reward.py's "
                  "belief_confidence_bonus docstring); with the anneal horizon at 0 it stays constant for "
                  "the entire run instead, unscaled by dt_fixed unlike every other reward term, and is not "
                  "contingent on reaching the target shape. Set KILOBOT_BELIEF_CONF_BONUS_ITERS (or "
                  "--conf-bonus-iters on rl_driver.py) to a positive number of iterations unless a constant "
                  "bonus is intentional.") % self.cfg.belief_conf_bonus)

    def setup(self):
        for worker in self.workers:
            worker.set_num_arenas(self.cfg.num_arenas)
            worker.reset_env()
            for k in range(self.cfg.num_arenas):
                self._reset_arena(worker, k, send_reset=False)

    def _pick_image(self):
        index = int(torch.randint(0, len(self.image_pool), (1,)))
        return index, self.image_pool[index]

    def _absolute_image_index(self, index):
        # Unity receives only a raw integer over
        # send_image/send_reset and does its own, independent lookup of
        # which formation file that corresponds to -- confirmed directly as
        # a real, reported bug: once Python's own pool became a
        # RANDOM sample of the full data/formations folder rather than its
        # first N files, index 0 within that sample no longer means "the
        # first file, alphabetically" -- it means whichever specific file
        # happened to land at position 0 in this run's particular random
        # draw. If Unity's own lookup still treats the raw integer as a
        # direct index into the full, sorted folder listing (unaware of
        # Python's sampling at all), sending the local pool index sends the
        # wrong file every time a limit narrower than the full folder is in
        # effect.
        #
        # The integer Unity receives is a POSITION into ImageLibrary's own
        # sorted listing of the formations folder (files[imageId], and the
        # same index into the baked distance field used for node[:,4]). So
        # the only correct value to send is the target file's sorted position
        # within that same folder.
        #
        # The historical implementation returned the numeric stem of the file
        # name (e.g. 54 for 000054.png) on the assumption that the folder is a
        # CONTIGUOUS %06d set where sorted position == numeric name -- true
        # for data/formations, FALSE for a non-contiguous subsample like
        # results/bc_v2/val_formations (position 54 there is 004025.png, not
        # 000054.png). Sending val_formations' stems sent Unity far out of its
        # sorted array, so ImageLibrary.GetTexture returned null (the target-
        # formation background image never appears on the floor) and
        # ImageLibrary.Sample fell back to distance 1.0 everywhere, silently
        # corrupting every coverage number. Send the true sorted position
        # instead, which is correct for contiguous and non-contiguous folders
        # alike. Falls back to the local index unchanged whenever the folder
        # listing or a name lookup is unavailable, so a non-standard formations
        # folder degrades to the old behaviour rather than crashing.
        image_names = getattr(self, "image_names", None)
        if image_names is None or index >= len(image_names):
            return index
        name = image_names[index]
        formations = getattr(self, "formations_dir", None)
        if not formations:
            return index
        try:
            pos = _sorted_png_position(formations, name)
            return pos if pos >= 0 else index
        except Exception:
            return index

    def _reset_arena(self, worker, k, send_reset=True):
        index, image = self._pick_image()
        image_names = getattr(self, "image_names", None)
        if image_names is not None and getattr(self.cfg, "log_formations", False):
            name = image_names[index] if index < len(image_names) else "?"
            print("arena %d: formation %d (%s)" % (k, index, name))
        with torch.no_grad():
            worker.z[k] = self.encoder(image).view(-1)
        worker.image_id[k] = index
        worker.databases[k] = {}
        worker.hidden[k] = {}
        worker.last_motor[k] = {}
        worker.last_dec_step[k] = {}
        worker.odometer[k] = {}
        worker.track_neighbor[k] = {}
        worker.track_seed[k] = {}
        worker.belief[k] = {}
        worker.pending_find_reward[k] = {}
        worker.step_count[k] = 0
        worker.ep_reward[k] = 0.0
        if hasattr(worker, "prev_dist"):
            worker.prev_dist.pop(k, None)
        if hasattr(worker, "prev_pos"):
            worker.prev_pos.pop(k, None)
        if hasattr(worker, "prev_target_dist"):
            worker.prev_target_dist.pop(k, None)
        if hasattr(worker, "prev_global_pot"):
            worker.prev_global_pot.pop(k, None)
        if hasattr(worker, "prev_belief_err"):
            worker.prev_belief_err.pop(k, None)
        if hasattr(worker, "pending_stop_bonus"):
            worker.pending_stop_bonus[k] = {}
        if hasattr(worker, "stop_bonus_paid"):
            worker.stop_bonus_paid[k] = {}
        if hasattr(worker, "oracle_ever_localized"):
            worker.oracle_ever_localized[k] = {}
        if hasattr(worker, "oracle_facing"):
            worker.oracle_facing[k] = {}
        if hasattr(worker, "oracle_lock"):
            worker.oracle_lock[k] = {}
        if hasattr(worker, "oracle_stuck_ref_strength"):
            worker.oracle_stuck_ref_strength[k] = {}
        if hasattr(worker, "oracle_stuck_ref_distance"):
            worker.oracle_stuck_ref_distance[k] = {}
        if hasattr(worker, "oracle_lock_confirmed_tick"):
            worker.oracle_lock_confirmed_tick[k] = {}
        if hasattr(worker, "oracle_current_target_idx"):
            worker.oracle_current_target_idx[k] = {}
        if hasattr(worker, "oracle_tried_occupied"):
            worker.oracle_tried_occupied[k] = {}
        # simple_oracle.py's own per-robot state --
        # mirrors the pattern immediately above exactly
        if hasattr(worker, "simple_heading"):
            worker.simple_heading[k] = {}
        if hasattr(worker, "simple_state"):
            worker.simple_state[k] = {}
        if hasattr(worker, "simple_turn_accum"):
            worker.simple_turn_accum[k] = {}
        if hasattr(worker, "simple_wall_name"):
            worker.simple_wall_name[k] = {}
        if hasattr(worker, "simple_target"):
            worker.simple_target[k] = {}
        # Direct, confirmed bug: never cleared here, unlike every other
        # per-robot dict above. arrived_switched_off persists across ticks
        # (set in actor_io.py, and with arrived_release_threshold <= 0 it is
        # never cleared there at all, so this stays required either way) -- a genuine episode reset
        # reuses the same local indices for brand-new robots, so without
        # this, a fresh robot silently inherits "already off" from whichever
        # robot used to hold its index last episode, forcing motor to zero
        # and freezing its hidden state before it ever makes a real
        # decision. Confirmed directly: a real smoke-test run's own
        # arrived_agreement collapsed to actor_only≈100% (both and
        # shadow_only both ≈0) at the exact iteration oracle_cov/
        # actor_eval_cov jumped together, the same signature already
        # established elsewhere as a genuine, swarm-wide reset.
        if hasattr(worker, "arrived_switched_off"):
            worker.arrived_switched_off[k] = {}
        # The per-robot caches that accumulated AFTER this list was written.
        # A genuine, confirmed leak (the episode-2 collapse of the first
        # transmission run: belief error ABOVE spawn level from tick one of
        # every episode after the first): _reset_arena reuses local indices
        # for brand-new robots, so a stale entry hands the new robot the OLD
        # robot's spawn heading (with oracle_known_start_heading the filter
        # is then TOLD a wrong heading and can never localize -- measured
        # berr 1.1 vs episode 1's 0.20), the previous robot's turn anchor
        # (the network steers turns by it), or the previous robot's
        # fit/anchor EMA state. Eval harnesses never reset arenas, which is
        # why only training runs ever saw this.
        for attr in ("_spawn_heading_cache", "turn_anchor", "turn_anchor_set_tick",
                     "belief_fit", "belief_anchor", "_stable_belief_heading",
                     "_last_ac_injection_tick", "_predict_update_debug_seen",
                     "_wall_debug_seen",
                     # arrival-gate protection state (actor_io._arrival_gate_
                     # protection / belief_ever_anchored): same leak family as
                     # _spawn_heading_cache above -- a stale "ever anchored" or
                     # a stale persistence count is the PREVIOUS robot's
                     # evidence, and would let a fresh robot latch (or be
                     # refused) on it.
                     "belief_ever_anchored", "_arrival_persist",
                     # the peer fix's per-robot injection cooldown (belief.
                     # add_peer_fix): a stale tick stamp is the previous
                     # robot's cooldown, suppressing or mis-timing the new
                     # robot's first fix hypothesis.
                     "_peer_inject_tick",
                     # the hop-count flood's per-robot counters
                     # (observation.hop_flood_update / config.hop_comms): a
                     # stale entry is the previous robot's graph distances,
                     # handing a fresh robot a confidence it never earned.
                     "hop",
                     # the learned receiver's trust gate (config.recv_gate):
                     # a stale entry is the previous robot's fusion trust.
                     "recv_gate",
                     # the Gate-1 encounter probe's per-robot counters
                     # (config.encounter_probe): a stale entry is the
                     # previous robot's reception history.
                     "_encounter_probe",
                     # the evidence channel's per-robot state (evidence.py):
                     # stale claims/referrals/homing tracks/last sightings
                     # are the PREVIOUS robot's evidence -- a fresh robot
                     # could inherit a referral that was never its own, and
                     # homing tracks would carry a dead robot's odometry
                     # frame.
                     "claim_store", "referral_store", "homing_state",
                     "last_sighting"):
            d = getattr(worker, attr, None)
            if isinstance(d, dict):
                d.pop(k, None)
        if hasattr(worker, "simple_hilbert_order"):
            worker.simple_hilbert_order.pop(k, None)
        # The coordinator keeps this
        # same bookkeeping on itself (cfg._oracle_coordinator), not on
        # worker -- the checks above never reached it, letting stale target
        # indices, occupancy history, and claimed-position data leak across
        # episode resets into what could be a different formation with a
        # different robot count entirely.
        coordinator = getattr(getattr(self, "cfg", None), "_oracle_coordinator", None)
        if coordinator is not None:
            if hasattr(coordinator, "oracle_current_target_idx"):
                coordinator.oracle_current_target_idx[k] = {}
            if hasattr(coordinator, "oracle_tried_occupied"):
                coordinator.oracle_tried_occupied[k] = {}
            if hasattr(coordinator, "oracle_claimed_pos"):
                coordinator.oracle_claimed_pos[k] = {}
        unity_index = self._absolute_image_index(index)
        if send_reset:
            worker.send_reset(k, unity_index)
        else:
            worker.send_image(k, unity_index)

    def _new_traj(self, worker, k, local):
        self.traj_counter = self.traj_counter + 1
        worker.traj_id[(k, local)] = self.traj_counter
        return self.traj_counter

    def _traj_for(self, worker, k, local):
        if (k, local) not in worker.traj_id:
            return self._new_traj(worker, k, local)
        return worker.traj_id[(k, local)]

    def _step_one(self, worker):
        worker.step()
        ds, ts = worker.get_steps()
        worker.idle_other_behaviors()
        return ds, ts

    def _run_step_phase(self):
        if len(self.workers) == 1:
            w = self.workers[0]
            t = time.time()
            w.step()
            self._t_step = self._t_step + (time.time() - t)
            t = time.time()
            ds, ts = w.get_steps()
            self._t_getsteps = self._t_getsteps + (time.time() - t)
            return [(ds, ts)]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=len(self.workers))
        t = time.time()
        results = list(self._pool.map(self._step_one, self.workers))
        self._t_step = self._t_step + (time.time() - t)
        return results

    def collect(self, policy, critic, deterministic=False):
        buffer = RolloutBuffer(self.cfg)
        self._deterministic = deterministic
        self._ep_records = []
        self._roll_reward_sum = 0.0
        self._roll_reward_count = 0
        self._roll_comp = {"on_count": 0.0, "on_bonus_sum": 0.0, "pack_sum": 0.0,
                           "off_pen_sum": 0.0, "sep_sum": 0.0, "count": 0.0}
        self._roll_cov_sum = 0.0
        self._roll_cov_count = 0
        self._roll_disp_sum = 0.0
        self._roll_disp_count = 0
        self._roll_split_seed_events = 0
        self._roll_split_total_events = 0
        self._roll_split_heartbeat_events = 0
        self._roll_seed_find_bonus_sum = 0.0
        self._roll_wall_find_penalty_sum = 0.0
        self._roll_target_shaping_sum = 0.0
        self._roll_target_shaping_count = 0
        self._roll_global_shaping_sum = 0.0
        self._roll_global_shaping_count = 0
        self._roll_belief_shaping_sum = 0.0
        self._roll_belief_err_sum = 0.0
        self._roll_belief_err_count = 0
        self._roll_overclaim_sum = 0.0
        self._roll_stop_bonus_count = 0
        self._roll_stop_bonus_missed = 0
        self._roll_acc_income_sum = 0.0
        self._roll_time_cost_sum = 0.0
        self._roll_belief_conf_bonus_sum = 0.0
        self._roll_shaping_sum = 0.0
        self._roll_conf_pos_sum = 0.0
        self._roll_conf_x_sum = 0.0
        self._roll_conf_y_sum = 0.0
        self._roll_localized_count = 0
        self._roll_belief_count = 0
        self._roll_decisions = 0
        self._roll_agent_steps = 0
        self._t_step = 0.0
        self._t_getsteps = 0.0
        self._t_snap = 0.0
        self._t_act = 0.0
        for worker in self.workers:
            worker.pop_timing()
        single = len(self.workers) == 1
        for _ in range(self.cfg.rollout_steps):
            results = self._run_step_phase()
            for worker, dsts in zip(self.workers, results):
                decision_steps, terminal_steps = dsts
                t = time.time()
                self._record_snapshots(buffer, worker, decision_steps)
                self._t_snap = self._t_snap + (time.time() - t)
                t = time.time()
                self._act(buffer, policy, worker, decision_steps)
                if getattr(self.cfg, "oracle_send_visual_state", False):
                    self._send_visual_states(worker)
                if single:
                    worker.idle_other_behaviors()
                self._t_act = self._t_act + (time.time() - t)
        self._t_parse = 0.0
        self._n_msgs = 0
        for worker in self.workers:
            s, c = worker.pop_timing()
            self._t_parse = self._t_parse + s
            self._n_msgs = self._n_msgs + c
        self._t_env = self._t_step + self._t_getsteps
        return buffer

    def _record_snapshots(self, buffer, worker, decision_steps):
        acting = {}
        if len(decision_steps) > 0:
            vector, _ = actor_io.split_obs(decision_steps.obs, self.cfg.device)
            for i in range(vector.shape[0]):
                acting.setdefault(int(vector[i, 0]), set()).add(int(vector[i, 1]))

        for k in range(self.cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None:
                continue
            node = snap["node"]
            m = node.shape[0]
            edge_index = snap["edge_index"]
            # reward_mode "target" (docs/tuning.md phase 165): the any-pixel
            # shape income (r_on/off_penalty/pack) is dropped -- it rewards
            # occupying SOMEONE's pixel, and the objective is the robot's OWN.
            # The separation term alone survives (anti-clumping along the
            # stroke); the accuracy income below replaces the rest.
            if self.cfg.reward_mode == "target":
                reward = separation_reward(node, self.cfg)
            else:
                reward = compute_rewards(node, self.cfg, edge_index)
            # While the ORACLE is driving, success/reset and
            # the cov_sum/cov_count-derived reporting come from the oracle's
            # own "arrived" state label -- worker.simple_state, the same
            # belief-based decision arrived_agreement already compares the
            # actor against -- not coverage()'s ground-truth position check.
            # That treats the oracle's own belief-based arrival as ground
            # truth for this system's purposes, consistent with the BC phase
            # being pure imitation of the oracle rather than independent task
            # success.
            #
            # It is ONLY valid while the oracle is actually driving. Nothing
            # writes worker.simple_state under any other motor_override, so
            # reading it during an actor-driven rollout does not measure the
            # actor -- it either reports 0 forever (a validation worker, which
            # never runs the oracle at all: confirmed directly against a real
            # 300-iteration Unity run, where val_cov was structurally
            # 0.0000 on all 60 evaluations while the actor was in fact
            # holding 13-30% ground-truth coverage on held-out formations), or
            # it silently re-reports the ORACLE's own labels left over from
            # the collect immediately before it (the actor-eval rollout inside
            # bc_train, where train_eval_cov consequently tracked oracle_cov
            # to within 0.012 mean -- not because the clone was faithful, but
            # because both numbers were the same dict read twice).
            #
            # So: oracle label when the oracle is driving, ground-truth
            # position otherwise. Both are per-robot fractions on [0, 1] and
            # feed the same accumulator, but they answer different questions
            # and must not be confused -- see rollout_stats' own consumers.
            # simple_state is shared by both oracle flavors (renewal_oracle.py
            # writes the same attr so this label and everything downstream of
            # it work unchanged); see renewal_oracle.py's module docstring.
            if self.cfg.motor_override in ("simple_oracle", "renewal_oracle"):
                arrived_here = getattr(worker, "simple_state", {}).get(k, {})
                arrived_count = sum(1 for l in range(m) if arrived_here.get(l) == "arrived")
                cov = arrived_count / m if m > 0 else 0.0
            else:
                cov = float(coverage(node, self.cfg))
            # Computed here so it can serve as both a reward input (below) and,
            # when cfg.success_metric == "settled", the success/reset value in
            # place of coverage -- the same cov-variable-repurpose pattern
            # phase 153 used for the oracle's arrived fraction. NaN targets
            # are excluded from the mean and from the reward entirely.
            dists = None
            valid_t = None
            dv = None
            errs = None
            if (self.cfg.reward_mode == "target"
                    or getattr(self.cfg, "target_shaping", 0.0) > 0.0
                    or getattr(self.cfg, "global_shaping", 0.0) > 0.0
                    or getattr(self.cfg, "stop_bonus", 0.0) > 0.0):
                dists = assigned_target_dists(worker, k, node, self.cfg)
                valid_t = ~torch.isnan(dists)
                dv = torch.where(valid_t, dists, torch.zeros_like(dists))
                if self.cfg.success_metric == "settled" and m > 0:
                    settled = bool(valid_t.any())
                    cov = (float(((dv < self.cfg.stop_radius) & valid_t).float().mean())
                           if settled else 0.0)
            if (getattr(self.cfg, "belief_shaping", 0.0) > 0.0
                    or getattr(self.cfg, "global_shaping", 0.0) > 0.0
                    or getattr(self.cfg, "beacon_honesty", 0.0) > 0.0):
                # Belief error is independent of target resolution, so it is
                # computed under its own gate (a run with belief shaping but
                # no formation pool still pays for correct localization).
                errs = belief_errors(worker, k, node)
            hconf = None
            if getattr(self.cfg, "beacon_honesty", 0.0) > 0.0:
                # The honesty constraint on the learned beacon: the claimed
                # conf is read from the transmission node column, the honest
                # one from the sender's own particles. Penalized per tick in
                # the reward -- the one channel whose gradient does not
                # vanish at tanh saturation (reward.beacon_overclaim_penalty).
                hconf = honest_confidences(worker, k, node)
                pen = beacon_overclaim_penalty(node, hconf,
                                               self.cfg.beacon_honesty, self.cfg.dt_fixed)
                reward = reward + pen
                self._roll_overclaim_sum = self._roll_overclaim_sum + float(pen.sum())

            # Potential-based shaping (optional): F = k * (prev_dist - gamma*dist), with
            # potential -dist. Rewards reducing distance to the stroke every step, giving
            # a strong navigation gradient in the open arena. Theory-safe: leaves the
            # optimal policy unchanged. No shaping on the first step after a reset.
            if self.cfg.reward_shaping > 0.0:
                if not hasattr(worker, "prev_dist"):
                    worker.prev_dist = {}
                curr_dist = node[:, 4]
                prev = worker.prev_dist.get(k)
                if prev is not None and prev.shape[0] == curr_dist.shape[0]:
                    shaping = self.cfg.reward_shaping * (prev - self.cfg.gamma * curr_dist)
                    reward = reward + shaping
                    self._roll_shaping_sum = self._roll_shaping_sum + float(shaping.sum())
                worker.prev_dist[k] = curr_dist.detach().clone()

            # Per-robot displacement (for the speed reward and the control-authority
            # probe). Tracks position across steps; first step after a reset has none.
            if not hasattr(worker, "prev_pos"):
                worker.prev_pos = {}
            pos = node[:, 0:2]
            prev_p = worker.prev_pos.get(k)
            disp = None
            proj = None
            if prev_p is not None and prev_p.shape[0] == pos.shape[0]:
                delta = pos - prev_p
                disp = delta.norm(dim=1)
                self._roll_disp_sum = self._roll_disp_sum + float(disp.sum())
                self._roll_disp_count = self._roll_disp_count + disp.shape[0]
                # Signed motion toward the target: project the step onto the unit direction
                # to the nearest stroke pixel (node cols 5:7). Positive means the robot moved
                # toward the shape this step. On-shape the direction is ~zero, so proj is ~0.
                dvec = node[:, 5:7]
                dunit = dvec / dvec.norm(dim=1, keepdim=True).clamp_min(1e-8)
                proj = (delta * dunit).sum(dim=1)
            worker.prev_pos[k] = pos.detach().clone()

            # Pure-speed isolation reward: reward only how far the robot moved this step,
            # so the optimal policy is simply to drive. Tests whether RL can move the motor
            # parameters at all, with no navigation, credit assignment, or messages.
            if self.cfg.reward_mode == "speed" and disp is not None:
                reward = self.cfg.speed_weight * disp

            # Steering reward: the navigation analogue of the speed reward. Reward directed
            # motion toward the target (signed displacement along the direction), so the
            # optimal policy must turn the wheels as a function of direction, not just drive.
            # "steer" isolates it (replaces the reward); "steer_blend" adds it on top of the
            # normal shape reward and any shaping.
            elif self.cfg.reward_mode == "steer" and proj is not None:
                reward = self.cfg.steer_weight * proj
            elif self.cfg.reward_mode == "steer_blend" and proj is not None:
                reward = reward + self.cfg.steer_weight * proj

            # Localization scaffolding for the split actor, and a nudge toward seeking
            # landmark seeds over wall seeds specifically -- both take their state
            # explicitly rather than reaching into worker/self, see reward.py
            bonus = getattr(self.cfg, "belief_conf_bonus", 0.0)
            if bonus > 0.0 and self.cfg.actor_type == "gru_split_observation":
                conf_bonus = belief_confidence_bonus(node, worker.belief.get(k), bonus)
                reward = reward + conf_bonus
                self._roll_belief_conf_bonus_sum = self._roll_belief_conf_bonus_sum + float(conf_bonus.sum())

            # Diagnostics only, no effect on reward -- tracked whether or not the bonus
            # above is active, so localization quality is visible even in a run that
            # deliberately keeps belief_conf_bonus at 0.
            if self.cfg.actor_type == "gru_split_observation":
                cps, cxs, cys, loc = belief_population_stats(worker.belief.get(k), node.shape[0], node.device)
                self._roll_conf_pos_sum = self._roll_conf_pos_sum + cps
                self._roll_conf_x_sum = self._roll_conf_x_sum + cxs
                self._roll_conf_y_sum = self._roll_conf_y_sum + cys
                self._roll_localized_count = self._roll_localized_count + loc
                self._roll_belief_count = self._roll_belief_count + node.shape[0]

            seed_bonus = getattr(self.cfg, "seed_find_bonus", 0.0)
            wall_penalty = getattr(self.cfg, "wall_find_penalty", 0.0)
            if (seed_bonus > 0.0 or wall_penalty > 0.0) and self.cfg.actor_type == "gru_split_observation":
                find_reward = seed_wall_find_reward(node, worker.pending_find_reward.get(k),
                                                    seed_bonus, wall_penalty)
                reward = reward + find_reward
                self._roll_seed_find_bonus_sum = self._roll_seed_find_bonus_sum + float(find_reward.clamp(min = 0.0).sum())
                self._roll_wall_find_penalty_sum = self._roll_wall_find_penalty_sum + float((-find_reward.clamp(max = 0.0)).sum())

            # ── RL fine-tune reward (docs/tuning.md phases 164-165) ───────────
            # Built around each robot's OWN assigned pixel: accuracy income
            # (smooth kernel, pays for being there -- tightest arrivals most),
            # potential-based shaping (the long-horizon approach gradient),
            # the shared smooth arena potential (sender credit for learned
            # messaging), a graded stop bonus on the frozen gate's latch, and
            # an explicit per-step time cost (speed). dists/valid_t/dv were
            # computed above for the settled success metric.
            if dists is not None:
                # Accuracy income -- replaces the any-pixel r_on income when
                # reward_mode == "target"; additive scaffolding otherwise.
                w_acc = getattr(self.cfg, "target_acc_weight", 0.0)
                sigma_acc = getattr(self.cfg, "target_acc_sigma", 0.05)
                if self.cfg.reward_mode == "target" and w_acc > 0.0:
                    inc = accuracy_income(dists, valid_t, w_acc, sigma_acc, self.cfg.dt_fixed)
                    reward = reward + inc
                    self._roll_acc_income_sum = self._roll_acc_income_sum + float(inc.sum())
                # Speed: a carrying cost per step, charged ONLY while the
                # robot is not yet settled at its own pixel (true distance
                # past stop_radius). Two properties this term needs to
                # actually work, both easy to get wrong:
                #   * It must be conditional. An unconditional per-step
                #     constant is a policy-gradient no-op: every trajectory
                #     of equal length pays the identical total, the value
                #     baseline absorbs it exactly, and no advantage changes.
                #     Charged only while unsettled, settling at tick 100 vs
                #     tick 1000 differs by 900 charged ticks, which IS a
                #     gradient -- and gamma discounting makes it prefer
                #     earlier still.
                #   * The relief bar must be TRUE distance, not the latch
                #     flag. A robot the gate falsely latched far from its
                #     target keeps paying -- wrong parking buys no cost
                #     relief, so the term cannot be mined by racing the
                #     filter into a premature latch.
                w_time = getattr(self.cfg, "target_time_cost", 0.0)
                if w_time > 0.0:
                    unsettled = valid_t & (dv >= self.cfg.stop_radius)
                    cost = torch.where(unsettled, torch.full_like(dv, w_time * self.cfg.dt_fixed),
                                       torch.zeros_like(dv))
                    reward = reward - cost
                    self._roll_time_cost_sum = self._roll_time_cost_sum + float(cost.sum())

                # Potential-based shaping on the robot's own assigned-target
                # distance. Same form as the existing reward_shaping, but in
                # normalized units against the per-robot target rather than
                # the arena's nearest-on-pixel; not annealed (policy-invariant
                # by construction). No shaping on the first step after a
                # reset, matching reward_shaping above.
                if getattr(self.cfg, "target_shaping", 0.0) > 0.0:
                    if not hasattr(worker, "prev_target_dist"):
                        worker.prev_target_dist = {}
                    prev = worker.prev_target_dist.get(k)
                    if prev is not None and prev.shape == dists.shape:
                        shaping = self.cfg.target_shaping * (prev - self.cfg.gamma * dv)
                        shaping = torch.where(valid_t, shaping, torch.zeros_like(shaping))
                        reward = reward + shaping
                        self._roll_target_shaping_sum = self._roll_target_shaping_sum + float(shaping.sum())
                        self._roll_target_shaping_count = self._roll_target_shaping_count + int(valid_t.sum())
                    worker.prev_target_dist[k] = dists.detach().clone()

                # Graded stop bonus, paid the tick after the arrival gate
                # latches (pending_stop_bonus set in act(), actor-driving
                # modes only): bonus * exp(-(d/stop_radius)^2). Full bonus
                # only for a tight arrival; a gate false positive far from
                # the pixel earns ~nothing (exp(-64)), so the bonus cannot
                # reward parking in the wrong place, and the smooth kernel
                # keeps a gradient between the tolerance bands instead of a
                # cliff. Paid ONCE per robot per episode: with
                # arrived_release_threshold > 0 a latched robot can release
                # and re-latch, and an arrived logit oscillating across the
                # hysteresis band would otherwise collect the bonus every
                # cycle -- benign for the settle metric (the robot IS at its
                # pixel) but a standing incentive for confidence
                # oscillation (docs/tuning.md phase 166).
                if getattr(self.cfg, "stop_bonus", 0.0) > 0.0:
                    if not hasattr(worker, "stop_bonus_paid"):
                        worker.stop_bonus_paid = {}
                    paid = worker.stop_bonus_paid.setdefault(k, {})
                    pending = worker.pending_stop_bonus.get(k, {})
                    for l, _flag in list(pending.items()):
                        if l < m and bool(valid_t[l]) and l not in paid:
                            b = self.cfg.stop_bonus * float(target_kernel(dv[l], self.cfg.stop_radius))
                            if b > 1e-6:
                                reward[l] = reward[l] + b
                                paid[l] = True
                                self._roll_stop_bonus_count = self._roll_stop_bonus_count + 1
                            else:
                                self._roll_stop_bonus_missed = self._roll_stop_bonus_missed + 1
                        else:
                            self._roll_stop_bonus_missed = self._roll_stop_bonus_missed + 1
                    if pending:
                        pending.clear()

            # ── belief-error terms (docs/tuning.md, the transmission phase) ──
            # Deliberately OUTSIDE the `dists is not None` block: these pay
            # for correct localization, which is defined whether or not the
            # arena resolved anyone's assigned pixel, and errs was computed
            # under its own gate above for exactly that reason.
            if errs is not None:
                finite = ~torch.isnan(errs)
                # Per-robot potential-based shaping on the robot's OWN belief
                # error (reward.belief_errors): F = k*(prev_e - gamma*e),
                # positive exactly when the filter's estimate moves toward
                # the truth. Correctness-grounded, not confidence-grounded
                # (see config.belief_shaping) -- a confidently wrong cloud
                # keeps e large and earns nothing. Dense across the
                # localization phase where the own-target kernels are flat.
                # NaN (no belief yet) means no gradient for that robot,
                # never zero -- same convention as the target shaping.
                if getattr(self.cfg, "belief_shaping", 0.0) > 0.0:
                    if not hasattr(worker, "prev_belief_err"):
                        worker.prev_belief_err = {}
                    # The mask must cover BOTH sides: a robot's belief entry
                    # appears at its first decision, so prev_e can be NaN on
                    # a row where the current errs is finite (measured: 7/46
                    # robots NaN'd the reward at the tick their belief first
                    # existed -- the mask from errs alone let prev_e's NaN
                    # through the where()). Robots gaining a belief pay no
                    # shaping on that tick, matching the reset convention.
                    valid_e = ~torch.isnan(errs)
                    prev_e = worker.prev_belief_err.get(k)
                    if prev_e is not None and prev_e.shape == errs.shape:
                        both = valid_e & ~torch.isnan(prev_e)
                        ev = torch.where(both, errs, torch.zeros_like(errs))
                        pv = torch.where(both, prev_e, torch.zeros_like(prev_e))
                        b_shaping = self.cfg.belief_shaping * (pv - self.cfg.gamma * ev)
                        b_shaping = torch.where(both, b_shaping, torch.zeros_like(b_shaping))
                        reward = reward + b_shaping
                        self._roll_belief_shaping_sum = self._roll_belief_shaping_sum + float(b_shaping.sum())
                    worker.prev_belief_err[k] = errs.detach().clone()
                self._roll_belief_err_sum = self._roll_belief_err_sum + float(errs[finite].sum())
                self._roll_belief_err_count = self._roll_belief_err_count + int(finite.sum())

            # Shared arena-level potential (sender credit): a SMOOTH mean
            # kernel over every robot's own-target distance, added equally
            # to all. Potential-based, so the optimal policy is unchanged;
            # a sender's return now improves whenever the swarm gets
            # closer, anywhere in the approach -- the gradient that makes
            # broadcasting learnable under per-robot GAE. A threshold
            # potential was rejected for exactly this: it is flat almost
            # everywhere and only credits the tick a neighbour crosses.
            # Two bands (reward.arena_potential): the settled kernel
            # above, plus a belief-error kernel that moves during the
            # localization phase -- where a helpful broadcast actually
            # does its work and where the settled kernel alone is flat
            # (phase 163's curve: settled<5u is 0.016 at tick 2000; the
            # localization phase is the whole first half of an episode).
            if getattr(self.cfg, "global_shaping", 0.0) > 0.0:
                if not hasattr(worker, "prev_global_pot"):
                    worker.prev_global_pot = {}
                pot = arena_potential(dists, valid_t, errs, None, self.cfg)
                prev_pot = worker.prev_global_pot.get(k)
                if prev_pot is not None:
                    # SIGN: pot is a higher-is-better kernel, so the PBRS form
                    # of the (t-1 -> t) transition is F = k*(gamma*pot_t -
                    # pot_{t-1}) -- positive when the arena IMPROVES. The
                    # phase-164/165 code computed k*(prev_pot - gamma*pot),
                    # the exact negative, paying the swarm for the potential
                    # FALLING and penalizing every improvement -- an inverted
                    # sender credit (a helpful broadcast that raised the
                    # potential penalized its own sender). Caught by
                    # test_global_shaping_belief_band_credits_without_target_
                    # resolution; the distance-shaped terms (reward_shaping,
                    # target_shaping, belief_shaping) were never affected --
                    # for a lower-is-better quantity, (prev - gamma*curr) IS
                    # the correct form, which is how the inversion went
                    # unnoticed for two phases.
                    g = self.cfg.global_shaping * (self.cfg.gamma * pot - prev_pot)
                    reward = reward + g
                    self._roll_global_shaping_sum = self._roll_global_shaping_sum + g * m
                self._roll_global_shaping_count = self._roll_global_shaping_count + m
                worker.prev_global_pot[k] = pot

            self._roll_reward_sum = self._roll_reward_sum + float(reward.sum())
            self._roll_reward_count = self._roll_reward_count + m
            comp = reward_components(node, self.cfg, edge_index)
            for ck in self._roll_comp:
                self._roll_comp[ck] = self._roll_comp[ck] + comp[ck]
            self._roll_cov_sum = self._roll_cov_sum + cov
            self._roll_cov_count = self._roll_cov_count + 1
            worker.ep_reward[k] = worker.ep_reward.get(k, 0.0) + float(reward.mean())

            term = torch.zeros(m)
            cut = torch.zeros(m)
            is_decision = torch.zeros(m, dtype=torch.bool)
            for local in acting.get(k, set()):
                if local < m:
                    is_decision[local] = True
            self._roll_decisions = self._roll_decisions + int(is_decision.sum())
            self._roll_agent_steps = self._roll_agent_steps + m

            worker.step_count[k] = worker.step_count[k] + 1
            success = cov >= self.cfg.success_threshold
            timeout = worker.step_count[k] >= self.cfg.max_episode_steps
            if success or timeout:
                cut[:] = 1.0
                if success:
                    term[:] = 1.0

            node_store = node
            if self._critic_extra > 0:
                extra = torch.zeros(m, self._critic_extra)
                bel = worker.belief.get(k)
                if bel:
                    idx = [l for l in bel if l < m]
                    if idx:
                        parts = torch.stack([bel[l] for l in idx])
                        extra[torch.tensor(idx, dtype = torch.long)] = belief_read(parts).cpu()
                node_store = torch.cat([node, extra], dim = 1)

            traj = torch.tensor([self._traj_for(worker, k, local) for local in range(m)], dtype=torch.long)
            si = buffer.add_step(k, snap["env_step"], node_store, snap["edge_index"], snap["edge_attr"],
                                 worker.z[k], traj, reward, term, cut, is_decision)
            snap["step_index"] = si

            if success or timeout:
                self._ep_records.append({"reward": worker.ep_reward.get(k, 0.0),
                                         "length": worker.step_count[k],
                                         "success": bool(success),
                                         "coverage": cov})
                self._reset_arena(worker, k, send_reset=True)
                for local in range(m):
                    worker.traj_id.pop((k, local), None)

    def _act(self, buffer, policy, worker, decision_steps):
        counts = actor_io.act(buffer, policy, worker, decision_steps, self.cfg, self.sample_rng,
                              deterministic = getattr(self, "_deterministic", False),
                              bc_capture = getattr(self, "_bc_capture", False),
                              probe = getattr(self, "_probe", False),
                              probe_log = getattr(self, "_probe_log", None),
                              audit = getattr(self, "_audit", False),
                              audit_log = getattr(self, "_audit_log", None),
                              pos_track = getattr(self, "_pos_track", False),
                              pos_log = getattr(self, "_pos_log", None))
        self._roll_split_total_events = self._roll_split_total_events + counts["total_events"]
        self._roll_split_seed_events = self._roll_split_seed_events + counts["seed_events"]
        self._roll_split_heartbeat_events = self._roll_split_heartbeat_events + counts["heartbeat_events"]

    def _send_visual_states(self, worker):
        # Purely observational -- only ever called
        # when cfg.oracle_send_visual_state is explicitly on, which nothing
        # in real training ever sets. Builds a complete, ordered list per
        # arena (defaulting to 0/straight for any robot that hasn't had a
        # decision yet this episode, rather than skipping it and leaving
        # SwarmManager's list misaligned with the actual robot count).
        visual_state = getattr(worker, "oracle_visual_state", None)
        if visual_state is None:
            return
        for k in range(self.cfg.num_arenas):
            snap = worker.snapshot(k)
            if snap is None:
                continue
            m = snap["node"].shape[0]
            per_robot = visual_state.get(k, {})
            states = [per_robot.get(l, 0) for l in range(m)]
            worker.send_robot_states(k, states)

    def rollout_payload(self):
        return {"reward_sum": self._roll_reward_sum,
                "reward_count": self._roll_reward_count,
                "cov_sum": self._roll_cov_sum,
                "cov_count": self._roll_cov_count,
                "disp_sum": self._roll_disp_sum,
                "disp_count": self._roll_disp_count,
                "split_seed_events": self._roll_split_seed_events,
                "split_total_events": self._roll_split_total_events,
                "split_heartbeat_events": self._roll_split_heartbeat_events,
                "seed_find_bonus_sum": self._roll_seed_find_bonus_sum,
                "wall_find_penalty_sum": self._roll_wall_find_penalty_sum,
                "target_shaping_sum": self._roll_target_shaping_sum,
                "target_shaping_count": self._roll_target_shaping_count,
                "global_shaping_sum": self._roll_global_shaping_sum,
                "global_shaping_count": self._roll_global_shaping_count,
                "belief_shaping_sum": self._roll_belief_shaping_sum,
                "belief_err_sum": self._roll_belief_err_sum,
                "belief_err_count": self._roll_belief_err_count,
                "overclaim_sum": self._roll_overclaim_sum,
                "stop_bonus_count": self._roll_stop_bonus_count,
                "stop_bonus_missed": self._roll_stop_bonus_missed,
                "acc_income_sum": self._roll_acc_income_sum,
                "time_cost_sum": self._roll_time_cost_sum,
                "belief_conf_bonus_sum": self._roll_belief_conf_bonus_sum,
                "shaping_sum": self._roll_shaping_sum,
                "conf_pos_sum": self._roll_conf_pos_sum,
                "conf_x_sum": self._roll_conf_x_sum,
                "conf_y_sum": self._roll_conf_y_sum,
                "localized_count": self._roll_localized_count,
                "belief_count": self._roll_belief_count,
                "decisions": self._roll_decisions,
                "agent_steps": self._roll_agent_steps,
                "ep_records": self._ep_records,
                "comp": self._roll_comp}

    def collect_timing(self):
        return {"step": self._t_step, "parse": self._t_parse, "getsteps": self._t_getsteps,
                "snap": self._t_snap, "act": self._t_act, "msgs": self._n_msgs}

    def _rollout_metrics(self):
        return rollout_stats(self.rollout_payload())

    def _histograms(self, buffer, policy):
        return build_histograms(self._ep_records, buffer, policy, MESSAGE_SIZE)

    def _merge_stats(self, ppo_logs, dt):
        timing = {"iter_seconds": dt, "rollout_steps": self.cfg.rollout_steps,
                  "env_step_sec": self._t_env, "step_sec": self._t_step,
                  "parse_sec": self._t_parse, "getsteps_sec": self._t_getsteps,
                  "parse_msgs": self._n_msgs, "snapshots_sec": self._t_snap,
                  "act_sec": self._t_act}
        return merge_stats(ppo_logs, timing, self._rollout_metrics())

    def run(self, policy, critic, actor_opt, critic_opt, iterations, logger=None,
            ckpt_path=None, ckpt_every=0, start_iter=0, min_start_cov=0.0, summary=None):
        if logger is None:
            logger = Logger(None)
        self.setup()
        logger.log_hparams(vars(self.cfg))
        step = start_iter * self.cfg.rollout_steps
        bonus0 = getattr(self.cfg, "belief_conf_bonus", 0.0)
        bonus_iters = getattr(self.cfg, "belief_conf_bonus_iters", 0)
        shaping0 = getattr(self.cfg, "reward_shaping", 0.0)
        summary_status = "error"
        try:
            for it in range(start_iter, iterations):
                self.cfg.belief_conf_bonus = conf_bonus_schedule(bonus0, bonus_iters, it)
                self.cfg.reward_shaping = shaping_schedule(shaping0, iterations, it)
                t0 = time.time()
                buffer = self.collect(policy, critic)
                t_collect = time.time() - t0
                t1 = time.time()
                ppo_logs = ppo_update(policy, critic, actor_opt, critic_opt, buffer, self.cfg)
                t_ppo = time.time() - t1
                dt = time.time() - t0
                step = step + self.cfg.rollout_steps

                stats = self._merge_stats(ppo_logs, dt)
                stats["time/collect_sec"] = t_collect
                stats["time/ppo_sec"] = t_ppo
                logger.log_scalars(tb_view(stats, self.cfg), step)
                logger.log_histograms(self._histograms(buffer, policy), step)
                logger.console(it, step, stats)
                if summary is not None:
                    summary.update(it, stats)
                print("  time: collect %.2fs (step %.2f [parse %.2f] getsteps %.2f snap %.2f act %.2f)  ppo %.2fs"
                      % (t_collect, self._t_step, self._t_parse, self._t_getsteps, self._t_snap, self._t_act, t_ppo))

                if it == start_iter:
                    check_startup_coverage(stats, min_start_cov,
                                           getattr(self.cfg, "success_metric", "coverage"))

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
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
            logger.close()
