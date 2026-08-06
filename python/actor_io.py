"""The act() loop: one decision batch in, one action batch out.

This is the seam between a worker and the actor. act() unpacks a decision
batch, asks observation.py for the actor's input tensors, optionally lets a
scripted controller override the motors, runs the policy, and writes both the
action and the buffer record. The scripted controller is simple_oracle.py, which act() short-circuits to
under cfg.motor_override = "simple_oracle".

The pipeline this file sits in the middle of:

    worker  ->  observation.py  ->  act()  ->  policy / simple_oracle.py  ->  worker
                (what the robot   (this      (what it decides)
                 can perceive)     file)
"""

import math
import numpy as np
import torch

from belief import (belief_read, belief_conf, belief_track_anchor, ANCHOR_STATE_SIZE,
                    ARENA_HALF, LOCALIZED_CONF_THRESHOLD, SEED_LAYOUTS, HEADING_NOISE_SCALE,
                    _matched_generator)
from kilobot_gnn import SEED_SIZE, WALL_SIZE, MESSAGE_SIZE, MOTOR_SIZE, NODE_FEATURES, priv_cols, SPLIT_SEED_OFFSET, SPLIT_WALL_OFFSET
from kinematics import dead_reckon, split_tick_motion, split_track_update, split_track_read
import observation
# re-exported: these used to be defined in this module and other modules still
# reach for them at actor_io's own path
from belief import IR_RANGE, WALL_SPACING, WALL_NAMES
from observation import (STRENGTH_COL, TARGET_POOL_SIZE, ensure_target, split_obs,
                         gather_databases, gather_nodes, gather_gru_state,
                         gather_split_state, sample_split_event, _resolve_wall_seed_xy)


def scripted_motors(node_b, mode, force_motor, assigned_dir = None, stopped_mask = None, belief_heading = None):
    # Privileged scripted controllers for the physics/measurement probe.
    n = node_b.shape[0]
    if mode == "forward":
        return torch.ones((n, 2))
    if mode == "fixed":
        if not force_motor:
            return None
        return torch.tensor([float(force_motor[0]), float(force_motor[1])]).repeat(n, 1)
    if mode == "oracle":
        # turn toward the direction-to-shape vector, drive forward. When
        # assigned_dir is provided, steers toward this robot's own per-episode
        # assigned target point instead of node_b's shared nearest-point
        # direction, so robots spread across the shape instead of every
        # robot converging on whichever point happens to be nearest to each
        # of them individually -- the uncoordinated version deliberately
        # left in place below, unchanged, as the fallback
        #
        # Heading comes from belief_heading -- the particle filter's estimate,
        # belief_read's sin_m/r, cos_m/r columns -- not true heading. It is
        # REQUIRED, with no fallback to node_b, so that a caller forgetting to
        # pass it raises instead of quietly reintroducing a privileged read.
        # This path only runs for robots the caller has determined are
        # exploring-committed, by which point triangulation has had a real
        # chance to resolve heading; wall contact alone cannot, because of the
        # east/west symmetry of the wall-following phase.
        if belief_heading is None:
            raise ValueError("scripted_motors('oracle') requires belief_heading -- "
                            "true heading (node_b[:,2:4]) is no longer read here at all")
        h = belief_heading
        if assigned_dir is not None:
            g = assigned_dir
        else:
            g = node_b[:, 5:7]
        hn = h / (h.norm(dim=1, keepdim=True) + 1e-6)
        gn = g / (g.norm(dim=1, keepdim=True) + 1e-6)
        # Un-negated: verified against real Unity, not a Python model of
        # KilobotMovement's turnRate formula. See docs/code-history.md.
        cross = hn[:, 0] * gn[:, 1] - hn[:, 1] * gn[:, 0]
        dot = (hn * gn).sum(dim=1)
        # This reacquire magnitude and the proportional gain k below are
        # calibrated together for the ballistic held-command architecture: a
        # command is held for up to a full heartbeat interval (48 ticks) before
        # anything re-evaluates it. A gain tuned for frequent re-evaluation
        # overshoots and settles into bounded oscillation instead of converging.
        # Both were swept together over many starting headings at the real held
        # interval. See docs/code-history.md.
        REACQUIRE_TURN = 0.45
        turn = torch.where(dot < 0,
                           torch.where(cross >= 0, torch.full_like(cross, REACQUIRE_TURN), torch.full_like(cross, -REACQUIRE_TURN)),
                           cross)
        base = torch.full((n,), 0.9, device=node_b.device)
        k = 0.35
        left = torch.clamp(base - k * turn, 0.0, 1.0)
        right = torch.clamp(base + k * turn, 0.0, 1.0)
        # The steering law has no notion of "arrived": turn=0 means drive
        # straight, not stop, and a zero direction vector produces turn=0, i.e.
        # full throttle ahead. So stopping needs this hard override, applied
        # after the steering computation so it always wins.
        if stopped_mask is not None:
            left = torch.where(stopped_mask, torch.zeros_like(left), left)
            right = torch.where(stopped_mask, torch.zeros_like(right), right)
        return torch.stack([left, right], dim=1).cpu()
    return None


def executed_motors(env_action, scripted):
    # per-robot motor command that Unity will actually execute this tick:
    # the override when one is active, the policy's squashed sample otherwise
    if scripted is not None:
        return scripted.to(env_action.device)
    return env_action[:, MESSAGE_SIZE:]


def _extract_wall_seed_rows(rows, device):
    """Pull real-Unity wall-seed entries out of the neighbour-message rows.

    Wall-seed position entries are marked with a negative
    sender id (see SwarmManager.cs's comment for the full rationale -- added
    via the existing message channel's slack capacity, not a new or widened
    observation, so no ml-agents shape renegotiation is needed). Extracted into
    an (n, WALL_SIZE, 2) table, then removed from rows entirely so they never
    compete as an ordinary
    neighbor message in sample_split_event's draw -- they were never a real
    robot, only piggybacking on the same channel's spare row capacity.

    Returns (rows, wall_seed_xy_unity); wall_seed_xy_unity is None when this
    observation carries no such entries (the replica never sends them).
    """
    if rows.shape[1] == 0:
        return rows, None
    sender_col = rows[:, :, MESSAGE_SIZE]
    is_wall_seed_row = sender_col < 0
    if not bool(is_wall_seed_row.any()):
        return rows, None
    wall_seed_xy_unity = torch.zeros(rows.shape[0], WALL_SIZE, 2, device = device)
    for i_idx, r_idx in is_wall_seed_row.nonzero(as_tuple = False).tolist():
        band = int(round(-float(sender_col[i_idx, r_idx]))) - 1
        if 0 <= band < WALL_SIZE:
            wall_seed_xy_unity[i_idx, band, 0] = rows[i_idx, r_idx, 0]
            wall_seed_xy_unity[i_idx, band, 1] = rows[i_idx, r_idx, 1]
    rows = torch.where(is_wall_seed_row.unsqueeze(-1), torch.zeros_like(rows), rows)
    return rows, wall_seed_xy_unity


def _cache_spawn_heading(worker, cfg, vector, arena_ids, locals_, n):
    """Remember each robot's spawn rotation the first tick it is ever seen.

    KilobotAgent.cs's CollectObservations
    appends this as the very last fixed-length column, after wallObs, so every
    other index is untouched -- a genuinely older, not-yet-rebuilt real-Unity
    player simply won't have this column at all (vector one narrower than
    expected), which the width check below treats as gracefully absent rather
    than a crash.

    decision_steps only ever contains robots requesting a decision this
    specific tick, not every robot every tick -- so this value, though sent
    unconditionally every tick by CollectObservations, is only ever visible
    here on the ticks a given robot happens to appear on. Cached once, the
    first time each robot is seen, into the same kind of per-arena/per-robot
    dict this pipeline already uses elsewhere (worker.image_id,
    worker.simple_target) -- read back later via EnvWorker.spawn_heading(a, l),
    regardless of which of simple_oracle.py's or gather_split_state's own
    initialization this robot's first real decision happens to go through.
    """
    if vector.shape[1] <= 2 + SEED_SIZE + WALL_SIZE:
        _warn_missing_spawn_heading_column(worker, cfg, vector)
        return
    spawn_heading_unity = vector[:, 2 + SEED_SIZE + WALL_SIZE]
    if not hasattr(worker, "_spawn_heading_cache"):
        worker._spawn_heading_cache = {}
    for i in range(n):
        cache_a = worker._spawn_heading_cache.setdefault(int(arena_ids[i]), {})
        l = int(locals_[i])
        if l not in cache_a:
            cache_a[l] = float(spawn_heading_unity[i])


def _warn_missing_spawn_heading_column(worker, cfg, vector):
    """A setup-correctness check, not a diagnostic -- hence not debug-gated.

    A real case: a
    debug.log showed every SIMPLE_ORACLE_SPAWN_CHECK line reading
    assumed_deg=90.0 regardless of true_deg (which correctly varied across all
    four cardinals, confirming SwarmManager.cs's own new rotation logic WAS
    active), with WALL_DEBUG_SHAPE confirming vector.shape stuck at the
    pre-phase-108 width. Root cause: Vector Observation Space Size on the
    Kilobot prefab's own Behavior Parameters wasn't incremented in the Editor
    (an explicitly flagged manual step) -- ml-agents silently
    truncates the extra CollectObservations value rather than erroring, so
    nothing crashes; every robot that didn't happen to spawn facing north just
    gets systematically wrong turns and wall-following from a wrong assumed
    starting heading, which is exactly what was reported (failure to converge,
    an unexplained "180 flip" specific to certain walls).

    Restricted to hasattr(worker, "channel"), i.e. a real player, since the
    observation width this checks is an Editor-configured property of the
    build.
    """
    if not getattr(cfg, "oracle_known_start_heading", False):
        return
    if not hasattr(worker, "channel") or hasattr(worker, "_spawn_heading_width_warned"):
        return
    worker._spawn_heading_width_warned = True
    print("WARNING: oracle_known_start_heading is on but this observation is only %d wide -- "
          "the spawnHeading column isn't present, so every robot will be assumed "
          "to start facing north regardless of its real spawn rotation. Most likely cause: "
          "Vector Observation Space Size on the Kilobot prefab's Behavior Parameters needs "
          "incrementing by one in the Unity Editor." % vector.shape[1])


def _drive_simple_oracle(worker, arena_ids, locals_, walls, cfg, rng, device,
                         wall_seed_xy_unity, actions, n):
    """The simple_oracle short-circuit: command motors and nothing else.

    A genuinely separate path, not touching anything else in act()
    (scripted_motors, the trained-policy observation-building and buffer
    machinery) at all -- it returns
    as soon as arena_ids/locals_/walls exist, before the rest of act() runs.
    See simple_oracle.py's own module docstring for the full state machine.

    act() gates this on "not bc_capture". Behaviour cloning needs the
    actor's own observation gathered every tick (z/tc/prop/prev_hidden, stored
    alongside this oracle's action as the bc_target label), which this
    short-circuit skips entirely by design. When bc_capture is True, act()
    falls through to its normal flow instead, where the scripted_motors call
    site substitutes this oracle in as the teacher -- see that call site for why
    reusing the existing scripted/executed_motor/bc_target machinery there,
    rather than duplicating it here, is both less code and the only way the
    actor's own observation-gathering and this oracle's action ever end up
    computed together for the same tick.
    """
    import simple_oracle
    # wall_seed_xy_unity, extracted from the player's message rows.
    wall_seed_xy = _resolve_wall_seed_xy(worker, arena_ids, locals_, cfg, device, wall_seed_xy_unity)
    motor = simple_oracle.simple_oracle_motors(worker, arena_ids, locals_, walls,
                                               wall_seed_xy, cfg, rng,
                                               getattr(cfg, "_oracle_formation_pool", None))
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        worker.last_motor.setdefault(a, {})[l] = motor[i].detach().clone()
        worker.last_dec_step.setdefault(a, {})[l] = worker.step_count.get(a, 0)
    actions[:, MESSAGE_SIZE:] = motor


def _draw_neighbour_message(rows, valid, has_msg, rng, n):
    """Pick the one neighbour message a single-receiver robot actually heard.

    Weighted by signal strength when any is positive, uniform over valid rows
    otherwise, and uniform over all rows for a robot that heard nothing (whose
    draw is then ignored downstream, but must still be a legal index).
    """
    strength = (rows[:, :, STRENGTH_COL] * valid).clamp(min = 0.0)
    ssum = strength.sum(dim = 1, keepdim = True)
    weights = torch.where(ssum > 0, strength, valid.float())
    weights = torch.where(has_msg.unsqueeze(1), weights, torch.ones_like(weights))
    chosen_idx = torch.multinomial(weights, 1, generator = _matched_generator(rng, weights.device)).squeeze(1)
    chosen = rows[torch.arange(n, device = rows.device), chosen_idx]
    return chosen, chosen[:, :MESSAGE_SIZE], chosen[:, MESSAGE_SIZE]


def _select_arrived_claim(worker, cfg, arena_ids, locals_, n, device):
    """The single most-confident arrived claim per robot, if any.

    Picked from worker.oracle_received_arrived_claims
    (already gated to genuine, actual reception -- see that extraction's own
    comment) -- the same "one sender per update" pattern already used for
    peer_pos, so a clique of correlated senders can't multiply into a
    false-sharp likelihood.

    Run in a full, real system (24 robots,
    thousands of ticks) and found to have a real, serious problem -- the
    cooldown check below is the direct result, not decoration. Without it,
    oracle_received_arrived_claims' own scan of the whole retained message
    history means a single arrived neighbor's broadcast can keep satisfying "a
    valid claim exists" on nearly every subsequent decision a nearby cold robot
    makes, not just the tick it was actually, freshly received -- confirmed
    directly (71% of all belief_update calls in one run) to repeatedly discard a
    cold robot's progress before it could ever converge, dropping real arrivals
    from 4/24 to 1/24. The cooldown closes that specific defect but is not
    itself a confirmed net improvement over not having this feature at all -- a
    second seed showed fewer arrivals with it on (though more accurate ones).
    Still off by default for exactly this reason -- see config.py's own comment
    on oracle_arrived_claim_injection for the full numbers from both seeds.
    """
    if not getattr(cfg, "oracle_arrived_claim_injection", False):
        return None
    if not hasattr(worker, "_last_ac_injection_tick"):
        worker._last_ac_injection_tick = {}
    ac_cooldown = getattr(cfg, "oracle_arrived_claim_cooldown_ticks", 400)
    ac_pos_t = torch.zeros(n, 2, device = device)
    ac_conf_t = torch.zeros(n, device = device)
    ac_valid_t = torch.zeros(n, dtype = torch.bool, device = device)
    ac_strength_t = torch.ones(n, device = device)
    for i, (a, l) in enumerate(zip(arena_ids.tolist(), locals_.tolist())):
        candidates = getattr(worker, "oracle_received_arrived_claims", {}).get(a, {}).get(l, [])
        if not candidates:
            continue
        cur_tick = int(worker.step_count[a])
        last_tick = worker._last_ac_injection_tick.get(a, {}).get(l)
        if ac_cooldown > 0 and last_tick is not None and (cur_tick - last_tick) < ac_cooldown:
            continue
        bx, by, bconf, bstrength = max(candidates, key = lambda c: c[2])
        ac_pos_t[i, 0] = bx / ARENA_HALF
        ac_pos_t[i, 1] = by / ARENA_HALF
        ac_conf_t[i] = bconf
        ac_valid_t[i] = True
        ac_strength_t[i] = bstrength
        worker._last_ac_injection_tick.setdefault(a, {})[l] = cur_tick
    return ac_pos_t, ac_conf_t, ac_valid_t, ac_strength_t


def _inject_cold_starts(worker, cfg, arena_ids, locals_, rng):
    """Randomly drop cached hidden state, to train the actor's cold start.

    config.py's own cold_start_injection_prob has the full rationale. Only ever
    pops worker.hidden's own cached entry -- everything else about the robot
    (real position, belief-filter state, sensor/track history) stays exactly as
    it is, so the oracle's own target motor computed for this same tick is
    still the real, correct one for this robot's real, current physical
    situation, not synthetic.

    torch.rand's own output defaults to CPU regardless of what device rng lives
    on -- on a real --device cuda run, rng is CUDA, and torch does not silently
    allow a CUDA generator to drive a CPU tensor (confirmed directly: this
    crashed a real run with "Expected a 'cpu' device type for generator but
    found 'cuda'"). Fixed using this pipeline's own already-established
    _matched_generator pattern rather than a new, untested fix -- deliberately
    keeping the draw itself on CPU (not cfg.device) since it's only ever read
    one Python-side .item() at a time, never used in GPU tensor math.
    """
    cold_start_prob = getattr(cfg, "cold_start_injection_prob", 0.0)
    if cold_start_prob <= 0.0 or cfg.motor_override != "simple_oracle":
        return
    draws = torch.rand(len(arena_ids), generator = _matched_generator(rng, "cpu"))
    for i in range(len(arena_ids)):
        if draws[i].item() < cold_start_prob:
            worker.hidden[int(arena_ids[i])].pop(int(locals_[i]), None)


def _extract_received_claims(worker, rows, valid, arena_ids, locals_, n):
    """Record which target claims each robot actually received this tick.

    Extracted from rows/valid -- the actor's own received-message
    database, not privileged simulator state. A claim only ever appears here if
    this specific robot actually received a message conveying it, through the
    same range-limited, competitive reception every other message type already
    goes through.

    Two separate extractions, deliberately not merged. oracle_received_claims
    is every committed robot's claim (slot 5), which is what occupancy
    avoidance wants. oracle_received_arrived_claims is the stricter
    arrived-only subset (slots 5 and 6), carrying the sender's own belief_conf
    (slot 7) so belief_update's injection can reuse the exact same
    confidence-derived sigma-inflation logic already validated for
    peer_pos/peer_conf/peer_strength.

    Both scan the WHOLE retained rows/valid history,
    appropriate for occupancy checking (which wants recent history, not just
    this instant) but confirmed directly to be the root cause of a real,
    serious problem at scale for the arrived variant: a claim stays "valid"
    here on every tick it remains in the retained history, not just the tick it
    was freshly received, so a single arrived neighbor could keep satisfying
    this for many consecutive decisions. _select_arrived_claim throttles
    consumption with a cooldown instead; this extraction is unchanged, and is
    not gated on fresh, single-tick reception the way wall_seed_xy is. That
    would likely be the more precise fix, closing the defect at its source
    rather than rate-limiting its symptom -- not attempted yet.
    """
    if not hasattr(worker, "oracle_received_claims"):
        worker.oracle_received_claims = {}
    if not hasattr(worker, "oracle_received_arrived_claims"):
        worker.oracle_received_arrived_claims = {}
    claim_valid_mask = valid & (rows[:, :, 5] > 0.5)
    arrived_valid_mask = claim_valid_mask & (rows[:, :, 6] > 0.5)
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        rows_i = claim_valid_mask[i].nonzero(as_tuple = True)[0]
        worker.oracle_received_claims.setdefault(a, {})[l] = [
            (float(rows[i, r, 3]) * ARENA_HALF, float(rows[i, r, 4]) * ARENA_HALF)
            for r in rows_i.tolist()]
        arrived_i = arrived_valid_mask[i].nonzero(as_tuple = True)[0]
        worker.oracle_received_arrived_claims.setdefault(a, {})[l] = [
            (float(rows[i, r, 3]) * ARENA_HALF, float(rows[i, r, 4]) * ARENA_HALF,
             float(rows[i, r, 7]), float(rows[i, r, STRENGTH_COL]))
            for r in arrived_i.tolist()]


def _belief_heading(worker, arena_ids, locals_, n, device):
    """Each robot's own believed heading, held steady when the estimate is weak.

    The particle filter's own estimate, not true heading. Computed
    after gather_split_state has already run belief_update for this tick, so it
    reflects this tick's freshest estimate. Required by scripted_motors('oracle');
    see that function's own comment for why there is no silent fallback.

    belief_read's r is the circular mean's
    concentration -- confirmed directly (see HEADING_CONCENTRATION_MIN's own
    comment) that dividing by a small r turns ordinary particle-cloud noise into
    a direction that can flip nearly at random tick to tick. Below the bar, keep
    steering off the last reading that WAS trustworthy rather than a freshly
    renormalized one -- still this robot's own belief, never privileged, just
    not re-derived from a division that is currently amplifying noise more than
    signal.
    """
    import belief as belief_mod
    if not hasattr(worker, "_stable_belief_heading"):
        worker._stable_belief_heading = {}
    belief_heading = torch.zeros(n, 2, device = device)
    for i in range(n):
        a = int(arena_ids[i]); l = int(locals_[i])
        bp = worker.belief.get(a, {}).get(l)
        sbh = worker._stable_belief_heading.setdefault(a, {})
        if bp is None:
            # no belief yet -- arbitrary but harmless, and matches
            # scripted_motors' own "aligned" fallback shape. Never privileged:
            # a genuinely uninitialized filter has no heading information yet
            # regardless of source.
            belief_heading[i, 0] = 1.0
            continue
        br = belief_mod.belief_read(bp.unsqueeze(0).to(device))
        if float(br[0, 5]) >= belief_mod.HEADING_CONCENTRATION_MIN or l not in sbh:
            sbh[l] = (float(br[0, 3]), float(br[0, 2]))
        belief_heading[i, 0] = sbh[l][0]
        belief_heading[i, 1] = sbh[l][1]
    return belief_heading


def _write_claim_broadcast(action_row, worker, a, l, arrived):
    """Broadcast this robot's own committed target and arrival, slots 3-7.

    This robot's committed local-navigation target (its own
    choice, not privileged simulator state), so a nearby robot can only ever
    learn of it by actually receiving this message -- see config.py's comment
    for the full rationale. Slots 3-5, distinct from belief_comms's 0-2, so the
    two features never collide. Explicitly zeroed when uncommitted rather than
    left as whatever the actor network happened to output there, or an
    uncommitted robot could look like a false-positive claim.

    Slots 6-7 are a separate signal from the claim.
    "Arrived" (stop_on_arrival, which already requires both distance
    AND belief_conf >= LOCALIZED_CONF_THRESHOLD) is a strictly stronger, and
    crucially different, condition than "committed": a merely committed robot
    is still moving and still accumulating heading-divergence-driven drift tick
    to tick, so its confidence is a stale snapshot by the time a neighbor could
    act on it. An arrived robot has stopped (its own motor commands go to zero
    once stopped_mask fires), so it isn't accumulating any further drift --
    the dominant mechanism (the
    noise-tuning test): heading divergence compounds with distance traveled, and
    a stopped robot travels no further distance. That is what makes an arrived
    robot usable as a genuinely stable, landmark-like broadcast source, not just
    a confident one. The sender's own belief_conf rides alongside (not just a
    binary flag) so the receiving side can reuse the exact same
    confidence-derived sigma-inflation logic already validated for the
    peer_pos/peer_conf/peer_strength pathway.
    """
    committed = getattr(worker, "oracle_ever_localized", {}).get(a, {}).get(l, False)
    claimed = getattr(worker, "oracle_claimed_pos", {}).get(a, {}).get(l)
    if committed and claimed is not None:
        action_row[3] = float(claimed[0]) / ARENA_HALF
        action_row[4] = float(claimed[1]) / ARENA_HALF
        action_row[5] = 1.0
    else:
        action_row[3] = 0.0
        action_row[4] = 0.0
        action_row[5] = 0.0
    if arrived:
        p_self = getattr(worker, "belief", {}).get(a, {}).get(l)
        action_row[6] = 1.0
        action_row[7] = float(belief_conf(p_self.unsqueeze(0))[0]) if p_self is not None else 0.0
    else:
        action_row[6] = 0.0
        action_row[7] = 0.0


def _arrived_head_gate(worker, policy, cfg, a, l, idx):
    """Whether the actor's arrived head has switched this robot off.

    config.py's own use_arrived_head has the full rationale. Only ever applies
    when motor_override == "none" -- during BC's own oracle-driven collection
    simple_oracle.py already handles its own arrived-stop; this is specifically
    for the actor's own, actually-deployed behavior.

    arrived_release_threshold turns this into hysteresis rather than a plain
    re-test: a robot switches off above arrived_confidence_threshold and only
    switches back on once confidence falls below this strictly lower bar, so it
    cannot chatter on and off around a single threshold. release <= 0 keeps the
    original, permanent behaviour.
    """
    if policy.actor.head_arrived is None or cfg.motor_override != "none":
        return False
    if not hasattr(worker, "arrived_switched_off"):
        worker.arrived_switched_off = {}
    already_off = worker.arrived_switched_off.get(a, {}).get(l, False)
    p_arrived = torch.sigmoid(policy.actor._arrived_logit[idx]).item()
    release = float(getattr(cfg, "arrived_release_threshold", 0.0))
    if already_off:
        still_off = True if release <= 0.0 else (p_arrived >= release)
    else:
        still_off = p_arrived > cfg.arrived_confidence_threshold
    worker.arrived_switched_off.setdefault(a, {})[l] = still_off

    # Purely visual, and off unless cfg.oracle_send_visual_state is set. Reuses
    # the existing visual-state channel: state 4 is already "arrived, stopped ->
    # green" in KilobotAgent.SetVisualState, so no Unity-side change is needed.
    # 0 (warm ivory) rather than leaving the previous value, so a robot that
    # RELEASES the gate under arrived_release_threshold visibly goes back to
    # moving -- a stuck green would hide exactly the false positives this is
    # meant to reveal.
    if getattr(cfg, "oracle_send_visual_state", False):
        if not hasattr(worker, "oracle_visual_state"):
            worker.oracle_visual_state = {}
        worker.oracle_visual_state.setdefault(a, {})[l] = 4 if still_off else 0
    return still_off


def act(buffer, policy, worker, decision_steps, cfg, rng, deterministic = False, bc_capture = False,
       probe = False, probe_log = None, audit = False, audit_log = None,
       pos_track = False, pos_log = None):
    """One decision batch: perceive, decide, record, command.

    The phases, in order: unpack the observation; optionally hand the whole
    tick to simple_oracle and return; gather the actor's inputs and run it;
    let the scripted oracle override the motors; record one buffer entry per
    active decider; write the actions back to the worker.
    """
    device = cfg.device
    n = len(decision_steps)
    actions = torch.zeros((n, MESSAGE_SIZE + MOTOR_SIZE))
    event_counts = {"total_events": 0, "seed_events": 0, "heartbeat_events": 0}
    if n == 0:
        worker.set_actions(actions.numpy())
        return event_counts

    # ── unpack the observation ────────────────────────────────────────────
    vector, rows = split_obs(decision_steps.obs, device)
    rows, wall_seed_xy_unity = _extract_wall_seed_rows(rows, device)
    if getattr(cfg, "oracle_debug_wall_log", False) and not hasattr(worker, "_wall_debug_shape_printed"):
        worker._wall_debug_shape_printed = True
        print("WALL_DEBUG_SHAPE vector.shape=%s" % str(list(vector.shape)))
    arena_ids = vector[:, 0].long()
    locals_ = vector[:, 1].long()
    seeds = vector[:, 2:2 + SEED_SIZE]
    walls = vector[:, 2 + SEED_SIZE:2 + SEED_SIZE + WALL_SIZE]
    _cache_spawn_heading(worker, cfg, vector, arena_ids, locals_, n)

    if cfg.motor_override == "simple_oracle" and not bc_capture:
        _drive_simple_oracle(worker, arena_ids, locals_, walls, cfg, rng, device,
                             wall_seed_xy_unity, actions, n)
        worker.set_actions(actions.numpy())
        return event_counts

    node_b = gather_nodes(worker, arena_ids, locals_, n, device)
    if getattr(cfg, "oracle_debug_wall_log", False):
        _log_wall_observations(worker, vector, walls, arena_ids, locals_)

    actor_type = cfg.actor_type
    split_obs_actor = actor_type == "gru_split_observation"
    gru = actor_type == "gru"

    # seed visibility is read from the raw seed slice for every actor, before any
    # privileged columns are appended, so a seed-only sighting counts as an event
    # for all of them (the seed is a typed input to each actor). a wall
    # sighting counts the same way: it is a real environment event even for
    # actors that do not consume that channel themselves
    has_seed = (seeds.sum(dim=1) > 0) | (walls.sum(dim=1) > 0)

    if not split_obs_actor:
        cols = priv_cols(cfg.actor_priv_mode)
        if cols:
            seeds = torch.cat([seeds, node_b[:, cols]], dim=1)

    valid = rows.abs().sum(dim=2) > 0
    has_msg = valid.any(dim=1)
    chosen, msg_b, sender_b = _draw_neighbour_message(rows, valid, has_msg, rng, n)
    z_b = torch.stack([worker.z[int(a)] for a in arena_ids])

    # ── gather the actor's own inputs, and run it ─────────────────────────
    if split_obs_actor:
        # wall_seed_xy uses the per-tick
        # snapshot (_scan_and_snapshot) when present; falls back to
        # wall_seed_xy_unity (extracted above from the real-Unity message
        # channel) otherwise. None if neither source has anything for this
        # call, or if oracle_wall_seed_position is off AND this specific point
        # isn't one of the near-corner exceptions -- both of which
        # sample_split_event already handles as "this signal doesn't exist here."
        wall_seed_xy = _resolve_wall_seed_xy(worker, arena_ids, locals_, cfg, device, wall_seed_xy_unity)
        tc_b, seed_narrowed, wall_narrowed, wall_seed_xy_narrowed = sample_split_event(
            seeds, walls, rows, valid, cfg, rng, wall_seed_xy = wall_seed_xy)
        # True heading is no longer read here at all
        # -- confirmed directly as a genuine, unintentional privileged-
        # information leak into the actor's own observations, not just the
        # oracle's: it ran unconditionally, for every decision regardless of
        # who was driving, feeding the actor's own belief-heading observation
        # field an exact, zero-noise readout of ground-truth simulator state a
        # real, deployed Kilobot could never have. This removal is
        # unconditional and final; what (if anything) should estimate heading
        # instead is still being worked out.
        arrived_claim = _select_arrived_claim(worker, cfg, arena_ids, locals_, n, device)
        _inject_cold_starts(worker, cfg, arena_ids, locals_, rng)
        h_prev, prop_b = gather_split_state(worker, arena_ids, locals_, seed_narrowed, wall_narrowed,
                                            cfg, rng, rows = rows, valid = valid, wall_seed_xy = wall_seed_xy_narrowed,
                                            arrived_claim = arrived_claim)
        action, env_action, log_prob, h_new = policy.act_batch_split(
            tc_b, prop_b, h_prev, deterministic=deterministic)
    elif gru:
        h_prev, prop_b, raw_path = gather_gru_state(worker, arena_ids, locals_, cfg)
        action, env_action, log_prob, h_new = policy.act_batch_gru(
            z_b, seeds, msg_b, prop_b, h_prev, deterministic=deterministic)
    else:
        prev_rows, prev_valid = gather_databases(worker, arena_ids, locals_, device)
        action, env_action, log_prob, new_rows, new_valid = policy.act_batch(
            z_b, seeds, msg_b, sender_b, prev_rows, prev_valid, deterministic=deterministic)

    # ── let the scripted oracle decide the motors instead, if it is driving ──
    # "oracle" here is scripted_motors' own belief-steered controller, which is
    # independent of the removed oracle.py -- see scripted_motors below.
    claim_broadcast = cfg.motor_override == "oracle" and getattr(cfg, "oracle_claim_broadcast", True)
    if claim_broadcast:
        _extract_received_claims(worker, rows, valid, arena_ids, locals_, n)

    # assigned_dir/stopped_mask came from the removed oracle.py coordinator.
    # scripted_motors treats a None assigned_dir as "steer to the nearest
    # on-shape point", which is the original, uncoordinated oracle behaviour.
    assigned_dir = stopped_mask = exploring_motor = exploring_mask = None
    belief_heading = None
    if cfg.motor_override == "oracle":
        belief_heading = _belief_heading(worker, arena_ids, locals_, n, device)

    if cfg.motor_override == "simple_oracle":
        # Only reached when bc_capture is True; otherwise the short-circuit at
        # the top of act() already returned. Calling the oracle here, rather
        # than duplicating what scripted/executed_motor/bc_target do below, is
        # what puts the actor's observation (gathered just above into
        # z_b/tc_b/prop_b/h_prev) and the oracle's action in the same buffer
        # entry as a matched (input, label) pair.
        #
        # belief_attr differs from worker.belief because gather_split_state,
        # called just above for this same tick, reads and writes worker.belief
        # itself -- sharing would corrupt whichever ran second.
        import simple_oracle
        assert split_obs_actor, \
            "simple_oracle is only wired for actor_type='gru_split_observation', not %r" % actor_type
        scripted = simple_oracle.simple_oracle_motors(worker, arena_ids, locals_, walls, wall_seed_xy, cfg, rng,
                                                      getattr(cfg, "_oracle_formation_pool", None),
                                                      belief_attr = "simple_belief")
    else:
        scripted = scripted_motors(node_b, cfg.motor_override, cfg.force_motor, assigned_dir = assigned_dir,
                                   stopped_mask = stopped_mask, belief_heading = belief_heading)
    if scripted is not None and exploring_motor is not None:
        scripted = torch.where(exploring_mask.unsqueeze(1), exploring_motor, scripted)
    if getattr(cfg, "oracle_debug_wall_log", False) and assigned_dir is not None:
        _log_wall_motor(worker, arena_ids, locals_, n, assigned_dir, scripted)

    active = has_msg | has_seed
    # Heartbeat deciders arrive with no event at all; they must still be
    # commanded (a zero action would stop the robot) and their state advanced.
    if getattr(cfg, "heartbeat_ticks", 0) > 0 and actor_type in ("gru", "gru_split_observation"):
        active = torch.ones_like(active)
    # what the robot will actually EXECUTE this tick. Under a motor override
    # (oracle BC, control probe) that is the scripted command, not the policy's
    # sample; the odometry trackers dead-reckon from last_motor, so recording the
    # policy's unexecuted sample there corrupts every proprioception input in
    # those modes (see docs/tuning.md).
    executed_motor = executed_motors(env_action, scripted)

    # ── record one buffer entry per active decider ────────────────────────
    for idx in range(n):
        if not bool(active[idx]):
            continue
        a = int(arena_ids[idx])
        l = int(locals_[idx])
        actions[idx] = env_action[idx].detach().cpu()
        if split_obs_actor and getattr(cfg, "belief_comms", False):
            actions[idx][0] = float(prop_b[idx][8])
            actions[idx][1] = float(prop_b[idx][9])
            actions[idx][2] = float(prop_b[idx][12] * prop_b[idx][13])
        if split_obs_actor and claim_broadcast:
            if stopped_mask is not None:
                arrived_now = bool(stopped_mask[idx])
            else:
                # simple_oracle.py's own terminal state is the arrival signal now
                # that the coordinator which used to supply stopped_mask is gone
                arrived_now = getattr(worker, "simple_state", {}).get(a, {}).get(l) == "arrived"
            _write_claim_broadcast(actions[idx], worker, a, l, arrived_now)
        snap = worker.snapshot(a)
        valid_snap = snap is not None and "step_index" in snap and l < snap["node"].shape[0]
        bc_t = scripted[idx].detach() if (bc_capture and scripted is not None) else None
        # config.py's own use_arrived_head has the full rationale. Reuses
        # worker.simple_state exactly as-is -- the same, already-verified
        # ground truth this whole project already trusts for "has this
        # robot arrived" everywhere else (state census, shadow tracking) --
        # not a new notion of arrival, just a new training target for it.
        arrived_t = (torch.tensor([1.0 if worker.simple_state.get(a, {}).get(l) == "arrived" else 0.0])
                    if (bc_capture and hasattr(worker, "simple_state")) else None)
        # Which state actually produced the motor command now in bc_t, not
        # whichever state the robot ended the tick in -- the two differ on
        # transition ticks. None outside BC collection and for any teacher
        # other than simple_oracle, which the reservoir treats as "no label"
        # rather than guessing one.
        oracle_state = (getattr(worker, "simple_motor_state", {}).get(a, {}).get(l)
                       if bc_capture else None)
        # Derived from oracle_state above, not from worker.simple_state, which
        # at this point reflects the state the oracle has already transitioned
        # INTO: a tick leaving go_north for turning would be labelled turning
        # while its bc_target was still go_north's [1.0, 1.0].
        was_turning = bool(oracle_state == "turning")
        if split_obs_actor:
            freeze_hidden = _arrived_head_gate(worker, policy, cfg, a, l, idx)
            if not freeze_hidden:
                # Skipped when freeze_hidden: keeps h_prev at whatever it was
                # when the robot switched off, rather than letting the GRU keep
                # evolving over a long run of near-identical arrived
                # observations, which drives its recurrent state into a drift
                # that does not self-correct.
                worker.hidden[a][l] = h_new[idx].detach().clone()
            # Zero, not executed_motor, once switched off -- the real,
            # sent motor is forced to zero further below (right before
            # worker.set_actions), and this value feeds dead-reckoning on
            # the NEXT tick's own proprioception. Recording the actor's
            # own raw, pre-override output here would tell the odometry
            # tracker this robot moved when it genuinely, physically
            # didn't -- the same corruption this function's own, existing
            # comment already warns about for the motor_override case.
            worker.last_motor[a][l] = (torch.zeros(2) if freeze_hidden
                                       else executed_motor[idx].detach().clone())
            worker.last_dec_step[a][l] = worker.step_count[a]
            is_seed_event = bool((tc_b[idx, SPLIT_SEED_OFFSET:] != 0).any())
            is_neighbor_event = bool((tc_b[idx, :SPLIT_SEED_OFFSET] != 0).any())
            # landmark vs wall specifically, for seed_find_bonus/wall_find_penalty
            # below. applied the tick after this one (_record_snapshots for this tick
            # already ran before act does), so it is a one-time nudge, not a persisting
            # bonus like belief_conf_bonus
            is_landmark_event = bool((tc_b[idx, SPLIT_SEED_OFFSET:SPLIT_WALL_OFFSET] != 0).any())
            is_wall_event = bool((tc_b[idx, SPLIT_WALL_OFFSET:SPLIT_WALL_OFFSET + WALL_SIZE] != 0).any())
            if is_landmark_event:
                worker.pending_find_reward[a][l] = 1.0
            elif is_wall_event:
                worker.pending_find_reward[a][l] = -1.0
            if is_seed_event:
                worker.track_seed[a][l] = torch.zeros(4, device=device)
            elif is_neighbor_event:
                worker.track_neighbor[a][l] = torch.zeros(4, device=device)
            if valid_snap:
                if is_seed_event or is_neighbor_event:
                    event_counts["total_events"] = event_counts["total_events"] + 1
                    if is_seed_event:
                        event_counts["seed_events"] = event_counts["seed_events"] + 1
                else:
                    event_counts["heartbeat_events"] = event_counts["heartbeat_events"] + 1
                buffer.add_decision(snap["step_index"], l, z_b[idx].detach(), tc_b[idx].detach(),
                                    tc_b[idx].detach(), None, action[idx].detach(), log_prob[idx].detach(),
                                    bc_target=bc_t, prev_hidden=h_prev[idx].detach().clone(),
                                    prop=prop_b[idx].detach().clone(), arrived_target=arrived_t,
                                    was_turning=was_turning, oracle_state=oracle_state)
        elif gru:
            worker.hidden[a][l] = h_new[idx].detach().clone()
            worker.last_motor[a][l] = executed_motor[idx].detach().clone()
            worker.last_dec_step[a][l] = worker.step_count[a]
            worker.odometer[a][l] = worker.odometer[a].get(l, 0.0) + float(raw_path[idx])
            if valid_snap:
                buffer.add_decision(snap["step_index"], l, z_b[idx].detach(), seeds[idx].detach(),
                                    chosen[idx].detach(), None, action[idx].detach(), log_prob[idx].detach(),
                                    bc_target=bc_t, prev_hidden=h_prev[idx].detach().clone(),
                                    prop=prop_b[idx].detach().clone())
        else:
            worker.databases[a][l] = (new_rows[idx].detach().clone(), new_valid[idx].detach().clone())
            if valid_snap:
                prev_db = prev_rows[idx][prev_valid[idx]].detach().clone()
                buffer.add_decision(snap["step_index"], l, z_b[idx].detach(), seeds[idx].detach(),
                                    chosen[idx].detach(), prev_db, action[idx].detach(), log_prob[idx].detach(),
                                    bc_target=bc_t)

    # ── diagnostics, then hand the actions back to the worker ─────────────
    if probe and probe_log is not None:
        mot = action[:, MESSAGE_SIZE:].detach().cpu().numpy()
        probe_log.append((node_b.detach().cpu().numpy(), mot))

    if audit and audit_log is not None:
        audit_log.append((
            getattr(policy, "_last_mean").cpu().numpy().copy(),
            action.detach().cpu().numpy().copy(),
            env_action.detach().cpu().numpy().copy()))

    if pos_track and pos_log is not None:
        _log_positions(worker, arena_ids, locals_, node_b, pos_log)

    if scripted is not None:
        actions[:, MESSAGE_SIZE:] = scripted

    # config.py's own use_arrived_head has the full rationale. Applied to
    # every row, not just this tick's active deciders, since a
    # switched-off robot should stay at exactly zero on every tick, not
    # only the ones where it happens to make a fresh decision.
    if split_obs_actor and hasattr(worker, "arrived_switched_off") and worker.arrived_switched_off:
        for i in range(len(arena_ids)):
            a = int(arena_ids[i]); l = int(locals_[i])
            if worker.arrived_switched_off.get(a, {}).get(l, False):
                actions[i, MESSAGE_SIZE:] = 0.0

    worker.set_actions(actions.numpy())
    return event_counts


# ─── debug logging, all gated on cfg.oracle_debug_wall_log ────────────────────

def _log_wall_observations(worker, vector, walls, arena_ids, locals_):
    """Which wall channel is strongest, when, for which robot.

    The geometric-mismatch check (comparing received wall signal
    against what the wall's true position geometrically implies) is removed --
    it required true position to compute, which is no longer read here at all,
    even for this diagnostic-only, off-by-default path. What remains is fully
    non-privileged.
    """
    if not hasattr(worker, "_wall_debug_seen"):
        worker._wall_debug_seen = {}
    has_wall_signal = walls.sum(dim = 1) > 0
    for i in torch.nonzero(has_wall_signal).flatten().tolist():
        a, l = int(arena_ids[i]), int(locals_[i])
        wall_vals = walls[i].cpu().numpy().tolist()
        argmax_name = WALL_NAMES[int(walls[i].argmax())]
        tick = worker.step_count.get(a, -1) if hasattr(worker, "step_count") else -1
        state = worker._wall_debug_seen.get((a, l))
        if state is not None and state[1] == argmax_name and (tick - state[0]) < 25:
            continue
        worker._wall_debug_seen[(a, l)] = (tick, argmax_name)
        print("WALL_DEBUG tick=%s arena=%d robot=%d "
              "raw_wallObs=[N=%.3f, E=%.3f, S=%.3f, W=%.3f] argmax=%s full_row=%s" % (
                  tick, a, l, wall_vals[0], wall_vals[1], wall_vals[2], wall_vals[3],
                  argmax_name, ["%.3f" % v for v in vector[i].cpu().numpy().tolist()]))


def _log_wall_motor(worker, arena_ids, locals_, n, assigned_dir, scripted):
    """The motor command a wall-locked or committed robot actually got."""
    if not hasattr(worker, "_wall_debug_motor_seen"):
        worker._wall_debug_motor_seen = {}
    lock_map = getattr(worker, "oracle_lock", {})
    loc_map = getattr(worker, "oracle_ever_localized", {})
    for i in range(n):
        a, l = int(arena_ids[i]), int(locals_[i])
        lock = lock_map.get(a, {}).get(l)
        committed = loc_map.get(a, {}).get(l, False)
        if (lock is None or lock[0] != "wall") and not committed:
            continue
        tick = worker.step_count.get(a, -1) if hasattr(worker, "step_count") else -1
        last = worker._wall_debug_motor_seen.get((a, l))
        if last is not None and (tick - last) < 25:
            continue
        worker._wall_debug_motor_seen[(a, l)] = tick
        ad = assigned_dir[i].cpu().numpy().tolist()
        lr = scripted[i].cpu().numpy().tolist()
        # No longer reads true heading (node_b[:, 2:4]) here
        print("WALL_DEBUG_MOTOR tick=%s arena=%d robot=%d lock=%s committed=%s "
              "assigned_dir=[%.3f, %.3f] motor_left=%.3f motor_right=%.3f" % (
                  tick, a, l, lock, committed, ad[0], ad[1], lr[0], lr[1]))


def _log_positions(worker, arena_ids, locals_, node_b, pos_log):
    """Identity-tagged true positions, for a human to read.

    So a specific robot's own true
    position (node_b's own P columns, [0:2] -- see kilobot_gnn.NODE_FEATURES's
    own comment for the full column layout) can be tracked across separate
    decision ticks, not just observed once per act() call the way probe_log
    already is. This is real, ground-truth position -- fine here specifically
    because it is printed for a human to read, never fed back into the network
    or any decision logic; the project's own no-privileged-information rule is
    about what the policy can condition on, not what a debug tool may show a
    person.

    Each robot's own current image_id (its arena's formation assignment, set
    only in _reset_arena) rides alongside position -- direct follow-up request:
    a robot's own displacement should never be measured across an episode
    boundary as if it were continuous movement. image_id changing between two
    consecutive observations for the same (arena, local) is exactly what a
    genuine reset looks like, and is the one direct, reliable signal available
    here for it -- not a guess from position size.
    """
    image_ids = np.array([worker.image_id.get(int(a), -1) for a in arena_ids.cpu().numpy()])
    pos_log.append((int(worker.step_count.get(int(arena_ids[0]), -1)) if len(arena_ids) > 0 else -1,
                    arena_ids.cpu().numpy().copy(), locals_.cpu().numpy().copy(),
                    node_b[:, 0:2].detach().cpu().numpy().copy(), image_ids))
