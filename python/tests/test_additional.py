"""
Additional tests, written to close coverage gaps found while auditing the
existing suite (test_kilobot.py / test_parallel.py / test_checkpoint.py /
test_sweep.py). Nothing here needs Unity; everything runs on CPU, matching
the project's existing test conventions.

Sections:
  - Actor database semantics (capacity eviction, single-vs-batched divergence)
  - RecurrentActor single-robot vs batched consistency
  - Critic privileged-column blinding (KILOBOT_CRITIC_BLIND) and actor priv_cols
  - The direct-motor diagnostic head
  - Behavior cloning (bc_update)
  - GRU decisions through the multiprocess pack/unpack path
  - Trainer wiring: scripted motors, arena reset bookkeeping
  - graph_batch with a mix of edge-free and edged graphs
  - RolloutBuffer.compute_returns is insertion-order independent
  - launch.py's pure helpers (env parsing, actor construction, behavior
    resolution, latent-dim check, image preprocessing), imported behind a
    lightweight mlagents_envs stub since the real package needs Python 3.10
    and is not installable in this environment.
"""
import os
import sys
import types
import tempfile

import torch

from tests.conftest import requires_unity
import pytest

import actor_io
import diagnostics
import bc as bc_mod
from kilobot_gnn import Actor, Critic, RecurrentActor, SplitObservationActor, empty_database, Z, SEED_SIZE, MESSAGE_SIZE, MOTOR_SIZE, NODE_FEATURES, DB_ROW_SIZE, DB_CAPACITY, GRU_HIDDEN, SPLIT_TC_SIZE, SPLIT_ODOM_SIZE, actor_forward_batch, recurrent_forward_batch, priv_cols, PRIV_COLS
from kinematics import PROP_SIZE
from config import Config
from policy import GaussianPolicy
from buffer import RolloutBuffer
from graph_batch import build_critic_batch
from reward import compute_rewards, DIST_COL
import ppo
import bc


# ---------------------------------------------------------------------------
# Actor database semantics
# ---------------------------------------------------------------------------

def _row(msg_fill, sender, age):
    return torch.cat([torch.full((MESSAGE_SIZE,), float(msg_fill)),
                       torch.tensor([float(sender)]), torch.tensor([float(age)])])


def test_batched_db_evicts_oldest_when_full():
    # A full, fixed-capacity DB (as used during training) must evict the
    # single oldest row -- not the lowest-indexed row, not a random one --
    # when a message from a never-before-seen sender arrives.
    torch.manual_seed(0)
    actor = Actor()
    actor.eval()
    K = 4
    db = torch.stack([_row(0.1, 10, 1), _row(0.2, 11, 2), _row(0.3, 12, 3), _row(0.4, 13, 4)])
    rows = torch.zeros(1, K, DB_ROW_SIZE)
    rows[0, :K] = db
    valid = torch.zeros(1, K, dtype=torch.bool)
    valid[0, :K] = True
    z = torch.randn(1, Z)
    seed = torch.randn(1, SEED_SIZE)

    with torch.no_grad():
        _, new_rows, new_valid = actor_forward_batch(
            actor, z, seed, torch.randn(1, MESSAGE_SIZE), torch.tensor([99.0]),
            rows.clone(), valid.clone())

    senders_after = sorted(int(s) for s in new_rows[0][new_valid[0]][:, MESSAGE_SIZE])
    assert senders_after == [11, 12, 13, 99]      # sender 10 (age 1, oldest) evicted
    assert int(new_valid[0].sum()) == K           # capacity is not exceeded


def test_batched_db_same_sender_replaces_in_place():
    # A repeat message from a sender already in the DB must overwrite that
    # sender's row (refreshing its age) rather than appending a new row.
    torch.manual_seed(0)
    actor = Actor()
    actor.eval()
    K = 4
    db = torch.stack([_row(0.1, 10, 1), _row(0.2, 11, 2), _row(0.3, 12, 3), _row(0.4, 13, 4)])
    rows = torch.zeros(1, K, DB_ROW_SIZE)
    rows[0, :K] = db
    valid = torch.zeros(1, K, dtype=torch.bool)
    valid[0, :K] = True
    z = torch.randn(1, Z)
    seed = torch.randn(1, SEED_SIZE)

    with torch.no_grad():
        _, new_rows, new_valid = actor_forward_batch(
            actor, z, seed, torch.full((1, MESSAGE_SIZE), 9.9), torch.tensor([11.0]),
            rows.clone(), valid.clone())

    assert int(new_valid[0].sum()) == K  # count unchanged: replace, not append
    slot = (new_rows[0, :, MESSAGE_SIZE] == 11).nonzero().flatten()
    assert slot.numel() == 1
    assert float(new_rows[0, slot[0], -1]) == 5.0          # newest age
    # the stored row is the parser's embedding of the new message (tanh-bounded),
    # not the raw input -- it must differ from the old embedding (0.2-filled)
    assert not torch.allclose(new_rows[0, slot[0], :MESSAGE_SIZE],
                              torch.full((MESSAGE_SIZE,), 0.2), atol=1e-4)
    assert float(new_rows[0, slot[0], :MESSAGE_SIZE].abs().max()) <= 1.0 + 1e-5


def test_single_robot_db_path_has_no_capacity_limit():
    # The single-robot forward path (TransmissionParser._update_database, used
    # in the deployment convenience path Actor.forward) only ever evicts a row
    # on a repeat sender; it never enforces DB_CAPACITY. This is a genuine
    # divergence from the batched training path, which is fixed-capacity and
    # evicts the oldest row once full (see test_batched_db_evicts_oldest_when_full).
    # DB_CAPACITY is documented as a padding bound ("must be >= maxKilobots"),
    # so in normal operation distinct-sender count should stay under it; this
    # test simply pins down what the code actually does if that assumption is
    # ever violated, so a future change to either path doesn't silently start
    # disagreeing with the other in a new way.
    torch.manual_seed(0)
    actor = Actor()
    db = empty_database()
    z = torch.randn(Z)
    seed = torch.randn(SEED_SIZE)
    n_senders = DB_CAPACITY + 5
    for i in range(n_senders):
        tx = torch.randn(MESSAGE_SIZE + 3)
        tx[MESSAGE_SIZE] = float(i)
        with torch.no_grad():
            _, _, db = actor(z, seed, tx, db)
    assert db.shape[0] == n_senders  # grew past DB_CAPACITY; batched path would have capped at DB_CAPACITY


# ---------------------------------------------------------------------------
# RecurrentActor: single-robot convenience path vs batched path
# ---------------------------------------------------------------------------

def test_recurrent_actor_single_matches_batched():
    torch.manual_seed(0)
    actor = RecurrentActor()
    actor.eval()
    z = torch.randn(Z)
    seed = torch.randn(SEED_SIZE)
    tx = torch.randn(MESSAGE_SIZE)
    prop = torch.randn(PROP_SIZE)
    h = torch.randn(GRU_HIDDEN)

    with torch.no_grad():
        out_msg, motor_pre, h_new = actor(z, seed, tx, prop, h)
        mean_b, h_new_b = recurrent_forward_batch(
            actor, z.unsqueeze(0), seed.unsqueeze(0), tx.unsqueeze(0),
            prop.unsqueeze(0), h.unsqueeze(0))

    assert torch.allclose(out_msg, mean_b[0, :MESSAGE_SIZE])
    assert torch.allclose(motor_pre, mean_b[0, MESSAGE_SIZE:])
    assert torch.allclose(h_new, h_new_b[0])


def test_recurrent_actor_batched_cold_start_is_finite():
    # First-ever tick: hidden state is all zeros (RecurrentActor.initial_hidden).
    torch.manual_seed(1)
    actor = RecurrentActor()
    n = 6
    h0 = actor.initial_hidden(n)
    assert h0.shape == (n, GRU_HIDDEN)
    assert torch.count_nonzero(h0) == 0
    z = torch.randn(n, Z)
    seed = torch.randn(n, SEED_SIZE)
    msg = torch.randn(n, MESSAGE_SIZE)
    prop = torch.randn(n, PROP_SIZE)
    mean, h_new = recurrent_forward_batch(actor, z, seed, msg, prop, h0)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(h_new).all()


# ---------------------------------------------------------------------------
# Critic privileged-column blinding and actor priv_cols
# ---------------------------------------------------------------------------

def test_critic_blind_none_is_identity_mask():
    os.environ.pop("KILOBOT_CRITIC_BLIND", None)
    critic = Critic()
    assert float(critic.in_mask.sum()) == NODE_FEATURES
    assert critic.blind_cols == []


@pytest.mark.parametrize("mode,expected_zero_cols", [
    ("dist", [4]),
    ("dir", [5, 6]),
    ("distdir", [4, 5, 6]),
    ("spatial", [0, 1, 2, 3, 4, 5, 6]),
])
def test_critic_blind_named_groups_zero_expected_columns(mode, expected_zero_cols):
    os.environ["KILOBOT_CRITIC_BLIND"] = mode
    try:
        critic = Critic()
        mask = critic.in_mask
        for c in range(NODE_FEATURES):
            expected = 0.0 if c in expected_zero_cols else 1.0
            assert float(mask[c]) == expected, "col %d wrong for mode %s" % (c, mode)
    finally:
        del os.environ["KILOBOT_CRITIC_BLIND"]


def test_critic_blind_custom_csv_columns():
    os.environ["KILOBOT_CRITIC_BLIND"] = "2,9"
    try:
        critic = Critic()
        for c in range(NODE_FEATURES):
            expected = 0.0 if c in (2, 9) else 1.0
            assert float(critic.in_mask[c]) == expected
    finally:
        del os.environ["KILOBOT_CRITIC_BLIND"]


def test_critic_blind_masking_actually_zeroes_the_forward_input():
    # The mask must be applied to x before the encoder, not just stored.
    os.environ["KILOBOT_CRITIC_BLIND"] = "dist"
    try:
        torch.manual_seed(0)
        critic = Critic()
        critic.eval()
        node = torch.randn(3, NODE_FEATURES)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
        z = torch.randn(Z)
        with torch.no_grad():
            out_a = critic(node, edge_attr, edge_index, z)
            node2 = node.clone()
            node2[:, DIST_COL] = 12345.0   # should be invisible to the critic
            out_b = critic(node2, edge_attr, edge_index, z)
        assert torch.allclose(out_a, out_b, atol=1e-5)
    finally:
        del os.environ["KILOBOT_CRITIC_BLIND"]


def test_priv_cols_known_modes():
    assert priv_cols("none") == []
    assert priv_cols("dir") == [5, 6]
    assert priv_cols("heading") == [2, 3]
    assert priv_cols("dir_heading") == [2, 3, 5, 6]
    assert priv_cols("pose") == [0, 1, 2, 3]
    assert priv_cols("full") == [0, 1, 2, 3, 5, 6]
    assert set(PRIV_COLS.keys()) == {"none", "dir", "heading", "dir_heading", "pose", "full"}


def test_priv_cols_unknown_mode_defaults_to_empty():
    assert priv_cols("some_typo_mode") == []


# ---------------------------------------------------------------------------
# Direct-motor diagnostic head
# ---------------------------------------------------------------------------

def test_direct_motor_actor_batched_and_diagnostics():
    torch.manual_seed(0)
    actor = Actor(direct=True)
    policy = GaussianPolicy(actor)
    n = 4
    z = torch.randn(n, Z)
    seed = torch.randn(n, SEED_SIZE)
    msg = torch.randn(n, MESSAGE_SIZE)
    sender = torch.arange(n).float()
    rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE)
    valid = torch.zeros(n, DB_CAPACITY, dtype=torch.bool)

    mean, new_rows, new_valid = actor_forward_batch(actor, z, seed, msg, sender, rows, valid)
    assert mean.shape == (n, MESSAGE_SIZE + MOTOR_SIZE)
    # motor pre-activation is produced by direct_head, not rho_2's motor rows
    assert hasattr(actor, "direct_head")
    assert actor._motor_preact.shape == (n, MOTOR_SIZE)

    motor_before, msg_before = ppo._motor_param_vec(policy)
    assert motor_before.numel() == actor.direct_head.weight.numel() + actor.direct_head.bias.numel()

    loss = mean.sum()
    policy.zero_grad()
    loss.backward()
    mgn, sgn, sat, absmean = ppo._motor_diagnostics(policy)
    assert mgn > 0.0   # direct_head received gradient
    assert sgn > 0.0
    assert sat >= 0.0 and absmean >= 0.0


# ---------------------------------------------------------------------------
# Behavior cloning (bc_update)
# ---------------------------------------------------------------------------

def _bc_buffer(cfg, n=12, with_targets=True):
    buffer = RolloutBuffer(cfg)
    for i in range(n):
        z = torch.randn(Z)
        seed = torch.randn(SEED_SIZE)
        tx = torch.cat([torch.randn(MESSAGE_SIZE), torch.tensor([float(i)]), torch.rand(1)])
        action = torch.randn(MESSAGE_SIZE + MOTOR_SIZE)
        buffer.decisions.append({
            "z": z, "seed": seed, "transmission": tx, "prev_db": empty_database(),
            "action": action, "old_log_prob": torch.tensor(0.0),
            "bc_target": torch.rand(MOTOR_SIZE) if with_targets else None,
        })
    return buffer


def test_bc_update_moves_params_and_reduces_loss():
    torch.manual_seed(0)
    cfg = Config()
    cfg.minibatch = 4
    policy = GaussianPolicy(Actor())
    buffer = _bc_buffer(cfg, n=16, with_targets=True)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    before = policy.actor.deepset.rho_2.weight.detach().clone()
    first_loss = bc.bc_update(policy, opt, buffer, cfg, epochs=1)
    after_one = policy.actor.deepset.rho_2.weight.detach().clone()
    assert not torch.allclose(before, after_one)

    # Continuing to fit the same fixed targets should not blow up the loss.
    last_loss = bc.bc_update(policy, opt, buffer, cfg, epochs=20)
    assert last_loss < first_loss


def test_bc_update_no_targets_is_a_noop():
    cfg = Config()
    policy = GaussianPolicy(Actor())
    buffer = _bc_buffer(cfg, n=5, with_targets=False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    before = policy.actor.deepset.rho_2.weight.detach().clone()
    loss = bc.bc_update(policy, opt, buffer, cfg, epochs=3)
    after = policy.actor.deepset.rho_2.weight.detach().clone()
    assert loss == 0.0
    assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# GRU decisions through the multiprocess pack/unpack path
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip_gru_decisions():
    from parallel import pack_buffer, unpack_buffer
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    buffer = RolloutBuffer(cfg)
    buffer.steps.append({
        "arena_id": 0, "env_step": 0, "node": torch.randn(2, NODE_FEATURES),
        "edge_index": torch.zeros(2, 0, dtype=torch.long), "edge_attr": torch.zeros(0, 1),
        "z": torch.randn(Z), "traj_id": torch.tensor([1, 2]), "reward": torch.randn(2),
        "term": torch.zeros(2), "cut": torch.zeros(2), "is_decision": torch.tensor([True, True]),
    })
    buffer.add_decision(0, 0, torch.randn(Z), torch.randn(SEED_SIZE), torch.randn(MESSAGE_SIZE),
                        None, torch.randn(MESSAGE_SIZE + MOTOR_SIZE), torch.tensor(0.3),
                        prev_hidden=torch.randn(GRU_HIDDEN), prop=torch.randn(PROP_SIZE))
    buffer.add_decision(0, 1, torch.randn(Z), torch.randn(SEED_SIZE), torch.randn(MESSAGE_SIZE),
                        None, torch.randn(MESSAGE_SIZE + MOTOR_SIZE), torch.tensor(-0.1),
                        prev_hidden=torch.randn(GRU_HIDDEN), prop=torch.randn(PROP_SIZE))

    packed = pack_buffer(buffer)
    assert "prev_hidden" in packed["decisions"]
    assert "prop" in packed["decisions"]

    out = unpack_buffer(packed, cfg)
    assert len(out.decisions) == 2
    for orig, rt in zip(buffer.decisions, out.decisions):
        assert rt["prev_db"] is None
        assert torch.allclose(rt["prev_hidden"], orig["prev_hidden"])
        assert torch.allclose(rt["prop"], orig["prop"])
        assert torch.allclose(rt["old_log_prob"], orig["old_log_prob"])
        assert torch.allclose(rt["action"], orig["action"])


def test_pack_buffer_empty_decisions_gru_or_not_is_harmless():
    from parallel import pack_buffer, unpack_buffer
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    buffer = RolloutBuffer(cfg)
    packed = pack_buffer(buffer)
    assert packed["S"] == 0
    assert packed["decisions"]["D"] == 0
    out = unpack_buffer(packed, cfg)
    assert out.steps == []
    assert out.decisions == []


# ---------------------------------------------------------------------------
# Trainer wiring: scripted motors and arena reset bookkeeping
# ---------------------------------------------------------------------------

def _mk_trainer_shell(cfg):
    import trainer as T
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    return tr


def test_scripted_motors_forward_is_full_throttle():
    from types import SimpleNamespace
    tr = _mk_trainer_shell(SimpleNamespace())
    node = torch.zeros(3, NODE_FEATURES)
    out = actor_io.scripted_motors(node, "forward", getattr(tr.cfg, "force_motor", None))
    assert torch.allclose(out, torch.ones(3, 2))


def test_scripted_motors_fixed_uses_config_and_none_when_unset():
    from types import SimpleNamespace
    cfg = SimpleNamespace(force_motor=(0.3, 0.7))
    tr = _mk_trainer_shell(cfg)
    node = torch.zeros(2, NODE_FEATURES)
    out = actor_io.scripted_motors(node, "fixed", getattr(tr.cfg, "force_motor", None))
    assert torch.allclose(out, torch.tensor([[0.3, 0.7], [0.3, 0.7]]))

    cfg.force_motor = ()
    assert actor_io.scripted_motors(node, "fixed", getattr(tr.cfg, "force_motor", None)) is None


def test_scripted_motors_none_mode_returns_none():
    from types import SimpleNamespace
    tr = _mk_trainer_shell(SimpleNamespace())
    node = torch.zeros(2, NODE_FEATURES)
    assert actor_io.scripted_motors(node, "none", getattr(tr.cfg, "force_motor", None)) is None


def test_scripted_motors_oracle_aligned_drives_forward_without_turning():
    from types import SimpleNamespace
    tr = _mk_trainer_shell(SimpleNamespace())
    node = torch.zeros(1, NODE_FEATURES)
    node[0, 2:4] = torch.tensor([1.0, 0.0])   # heading
    node[0, 5:7] = torch.tensor([1.0, 0.0])   # direction to shape: identical to heading
    out = actor_io.scripted_motors(node, "oracle", getattr(tr.cfg, "force_motor", None), belief_heading = node[:, 2:4])
    # perfectly aligned: cross==0 -> turn==0 -> left == right == base (0.9)
    assert abs(float(out[0, 0]) - float(out[0, 1])) < 1e-5
    assert float(out[0, 0]) > 0.5   # driving forward, not stalled


def test_scripted_motors_oracle_target_behind_forces_bounded_hard_turn():
    # reacquire magnitude and gain k recalibrated
    # together (0.45, 0.35) once the underlying sign bug was found and
    # fixed -- see that phase's comment in actor_io.py for the full
    # rationale. Still a strong, decisive turn -- one wheel well below its
    # resting value, the other at its ceiling -- just calibrated against
    # the confirmed, deliberately-ballistic held-command duration rather
    # than left unbounded.
    from types import SimpleNamespace
    tr = _mk_trainer_shell(SimpleNamespace())
    node = torch.zeros(1, NODE_FEATURES)
    node[0, 2:4] = torch.tensor([1.0, 0.0])    # heading: +x
    node[0, 5:7] = torch.tensor([-1.0, 0.0])   # target directly behind
    out = actor_io.scripted_motors(node, "oracle", getattr(tr.cfg, "force_motor", None), belief_heading = node[:, 2:4])
    # dot<0 (target behind) forces a strong, bounded turn: one wheel at
    # 0.9 - 0.35*0.45 = 0.7425, the other clamped at its ceiling
    assert (abs(float(out[0, 0]) - 0.7425) < 1e-2 and float(out[0, 1]) > 1.0 - 1e-3) or \
           (abs(float(out[0, 1]) - 0.7425) < 1e-2 and float(out[0, 0]) > 1.0 - 1e-3)


def test_scripted_motors_oracle_perpendicular_target_turns_toward_it():
    from types import SimpleNamespace
    tr = _mk_trainer_shell(SimpleNamespace())
    node = torch.zeros(1, NODE_FEATURES)
    node[0, 2:4] = torch.tensor([1.0, 0.0])   # heading: +x
    node[0, 5:7] = torch.tensor([0.0, 1.0])   # target 90 degrees to one side
    out = actor_io.scripted_motors(node, "oracle", getattr(tr.cfg, "force_motor", None), belief_heading = node[:, 2:4])
    assert abs(float(out[0, 0]) - float(out[0, 1])) > 1e-3   # not driving straight
    assert out[0, 0] >= 0.0 and out[0, 0] <= 1.0
    assert out[0, 1] >= 0.0 and out[0, 1] <= 1.0


class _FakeResetWorker:
    def __init__(self):
        self.z, self.image_id, self.databases, self.hidden = {}, {}, {}, {}
        self.last_motor, self.last_dec_step, self.odometer = {}, {}, {}
        self.track_neighbor, self.track_seed = {}, {}
        self.belief = {}
        self.pending_find_reward = {}
        self.step_count, self.ep_reward = {}, {}
        self.sent_reset, self.sent_image = [], []

    def send_reset(self, k, idx):
        self.sent_reset.append((k, idx))

    def send_image(self, k, idx):
        self.sent_image.append((k, idx))


def _fixed_encoder_3(img):
    return torch.tensor([1.0, 2.0, 3.0])


def test_reset_arena_reinitializes_state_and_sends_reset():
    from types import SimpleNamespace
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer_shell(cfg)
    tr.encoder = _fixed_encoder_3
    tr.image_pool = ["a", "b", "c"]
    w = _FakeResetWorker()
    w.ep_reward[0] = 99.0
    w.step_count[0] = 50
    w.prev_dist = {0: torch.tensor([1.0])}
    w.prev_pos = {0: torch.tensor([1.0, 2.0])}
    w.databases[0] = {1: "stale"}

    torch.manual_seed(0)
    tr._reset_arena(w, 0, send_reset=True)

    assert torch.equal(w.z[0], torch.tensor([1.0, 2.0, 3.0]))
    assert 0 <= w.image_id[0] < len(tr.image_pool)
    assert w.databases[0] == {}
    assert w.hidden[0] == {} and w.last_motor[0] == {} and w.odometer[0] == {}
    assert w.track_neighbor[0] == {} and w.track_seed[0] == {}
    assert w.step_count[0] == 0
    assert w.ep_reward[0] == 0.0
    assert 0 not in w.prev_dist
    assert 0 not in w.prev_pos
    assert w.sent_reset == [(0, w.image_id[0])]
    assert w.sent_image == []


def _fixed_encoder_zero3(img):
    return torch.zeros(3)


def test_reset_arena_send_image_when_not_sending_reset():
    from types import SimpleNamespace
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer_shell(cfg)
    tr.encoder = _fixed_encoder_zero3
    tr.image_pool = ["only_one"]
    w = _FakeResetWorker()
    tr._reset_arena(w, 2, send_reset=False)
    assert w.sent_image == [(2, 0)]
    assert w.sent_reset == []


def test_pick_image_uses_full_pool_range():
    from types import SimpleNamespace
    cfg = SimpleNamespace(seed=0)
    tr = _mk_trainer_shell(cfg)
    tr.image_pool = list(range(7))
    torch.manual_seed(0)
    seen = set()
    for _ in range(200):
        idx, img = tr._pick_image()
        assert img == idx
        seen.add(idx)
    assert seen == set(range(7))  # every image reachable


# ---------------------------------------------------------------------------
# graph_batch: mixed edge-free / edged graphs, and critic forward stays finite
# ---------------------------------------------------------------------------

def test_build_critic_batch_handles_edge_free_graph_in_the_mix():
    torch.manual_seed(0)
    nodes = [torch.randn(3, NODE_FEATURES), torch.randn(2, NODE_FEATURES), torch.randn(4, NODE_FEATURES)]
    edges = [torch.zeros(2, 0, dtype=torch.long), torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, 0, dtype=torch.long)]
    attrs = [torch.zeros(0, 1), torch.rand(2, 1), torch.zeros(0, 1)]
    zs = [torch.randn(Z), torch.randn(Z), torch.randn(Z)]

    x, edge_attr, edge_index, z, batch = build_critic_batch(nodes, edges, attrs, zs)
    assert x.shape == (9, NODE_FEATURES)
    assert batch.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # the edged graph's indices are offset by the first graph's node count (3)
    assert edge_index.tolist() == [[3, 4], [4, 3]]

    critic = Critic()
    critic.eval()
    with torch.no_grad():
        out = critic(x, edge_attr, edge_index, z, batch)
    assert torch.isfinite(out).all()
    assert out.shape == (9, 1)


# ---------------------------------------------------------------------------
# RolloutBuffer.compute_returns: correctness must not depend on insertion order
# ---------------------------------------------------------------------------

def test_compute_returns_independent_of_step_insertion_order():
    torch.manual_seed(0)
    cfg = Config()
    critic = Critic()
    critic.eval()
    m = 2

    def make_step_data(step):
        node = torch.randn(m, NODE_FEATURES)
        edge_index = torch.randint(0, m, (2, m))
        edge_attr = torch.rand(m, 1)
        reward = compute_rewards(node, cfg)
        term = torch.zeros(m)
        cut = torch.zeros(m)
        if step == 3:
            cut[:] = 1.0
        traj = torch.arange(m, dtype=torch.long)
        is_decision = torch.ones(m, dtype=torch.bool)
        return dict(node=node, edge_index=edge_index, edge_attr=edge_attr,
                    reward=reward, term=term, cut=cut, traj=traj, is_decision=is_decision)

    z = torch.randn(Z)
    steps_data = [make_step_data(s) for s in range(4)]

    def build_buffer(order):
        buffer = RolloutBuffer(cfg)
        for step in order:
            d = steps_data[step]
            buffer.add_step(0, step, d["node"], d["edge_index"], d["edge_attr"], z,
                            d["traj"], d["reward"], d["term"], d["cut"], d["is_decision"])
        buffer.compute_returns(critic)
        return buffer

    forward = build_buffer([0, 1, 2, 3])
    shuffled = build_buffer([2, 0, 3, 1])

    # Look up each buffer's returns for (env_step, local) pairs and compare.
    def returns_by_env_step(buffer):
        out = {}
        for si, step in enumerate(buffer.steps):
            for local in range(step["node"].shape[0]):
                out[(step["env_step"], local)] = float(buffer.returns[si][local])
        return out

    r1 = returns_by_env_step(forward)
    r2 = returns_by_env_step(shuffled)
    assert r1.keys() == r2.keys()
    for k in r1:
        assert abs(r1[k] - r2[k]) < 1e-4, k


# ---------------------------------------------------------------------------
# launch.py: pure helpers, imported behind a mlagents_envs stub
# ---------------------------------------------------------------------------

def _stub_mlagents_envs():
    # launch.py imports mlagents_envs unconditionally at module scope (unlike
    # trainer.py/channels.py/env_worker.py, which guard the import). The real
    # package needs Python 3.10 and is not installable here, so we register
    # minimal stand-ins in sys.modules before importing launch, purely so the
    # module body executes and its pure helper functions become testable.
    if "mlagents_envs" in sys.modules and getattr(sys.modules["mlagents_envs"], "__kilobot_stub__", False):
        return
    if "mlagents_envs.environment" in sys.modules and not getattr(sys.modules.get("mlagents_envs"), "__kilobot_stub__", False):
        # a real mlagents_envs is present; nothing to stub
        return

    def _mk(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    pkg = _mk("mlagents_envs")
    pkg.__kilobot_stub__ = True
    env_mod = _mk("mlagents_envs.environment")
    env_mod.UnityEnvironment = object
    base_env_mod = _mk("mlagents_envs.base_env")
    base_env_mod.ActionTuple = object
    _mk("mlagents_envs.side_channel")
    sc_mod = _mk("mlagents_envs.side_channel.side_channel")
    sc_mod.SideChannel = object
    im_mod = _mk("mlagents_envs.side_channel.incoming_message")
    im_mod.IncomingMessage = object
    om_mod = _mk("mlagents_envs.side_channel.outgoing_message")
    om_mod.OutgoingMessage = object
    epc_mod = _mk("mlagents_envs.side_channel.environment_parameters_channel")
    epc_mod.EnvironmentParametersChannel = object
    ecc_mod = _mk("mlagents_envs.side_channel.engine_configuration_channel")
    ecc_mod.EngineConfigurationChannel = object


def _import_launch():
    try:
        import mlagents_envs  # noqa: F401
        HAVE_REAL = True
    except Exception:
        HAVE_REAL = False
    if not HAVE_REAL:
        _stub_mlagents_envs()
    import launch
    return launch


def test_init_actor_env_var_defaults_to_none_and_reads_path():
    launch = _import_launch()
    os.environ.pop("KILOBOT_INIT_ACTOR", None)
    assert launch._env("KILOBOT_INIT_ACTOR", None) is None
    os.environ["KILOBOT_INIT_ACTOR"] = "../results/bc_gru_actor.pt"
    assert launch._env("KILOBOT_INIT_ACTOR", None) == "../results/bc_gru_actor.pt"
    os.environ.pop("KILOBOT_INIT_ACTOR", None)


def test_success_threshold_env_var_overrides_default():
    # regression test: success_threshold had no env var override at all before
    # this -- the same class of gap this project has formalized before for
    # other real settings (collect_max_wait, seed_find_bonus/wall_find_penalty).
    # Added specifically to let watch_oracle.sh disable success-triggered
    # episode resets (threshold above 1.0, which coverage can never reach)
    # for genuinely unbounded-episode observation.
    launch = _import_launch()
    os.environ.pop("KILOBOT_SUCCESS_THRESHOLD", None)
    assert launch._env_float("KILOBOT_SUCCESS_THRESHOLD", 0.85) == 0.85
    os.environ["KILOBOT_SUCCESS_THRESHOLD"] = "1.1"
    assert launch._env_float("KILOBOT_SUCCESS_THRESHOLD", 0.85) == 1.1
    os.environ.pop("KILOBOT_SUCCESS_THRESHOLD", None)



    launch = _import_launch()
    from kilobot_gnn import RecurrentActor
    from policy import GaussianPolicy
    from checkpoint import export_actor, load_for_eval
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bc_actor.pt")
        source = GaussianPolicy(RecurrentActor())
        export_actor(path, source)

        target = GaussianPolicy(RecurrentActor())
        before = target.actor.head_motor.weight.detach().clone()
        load_for_eval(path, target, "cpu")
        after = target.actor.head_motor.weight.detach().clone()
        assert not torch.allclose(before, after)
        assert torch.allclose(after, source.actor.head_motor.weight.detach())

    assert launch._env("KILOBOT_TESTVAR", "default") == "default"
    os.environ["KILOBOT_TESTVAR"] = "hello"
    assert launch._env("KILOBOT_TESTVAR", "default") == "hello"

    os.environ["KILOBOT_TESTVAR"] = "7"
    assert launch._env_int("KILOBOT_TESTVAR", 0) == 7
    assert launch._env_float("KILOBOT_TESTVAR", 0.0) == 7.0

    for truthy in ("1", "true", "True", "yes", "on"):
        os.environ["KILOBOT_TESTVAR"] = truthy
        assert launch._env_bool("KILOBOT_TESTVAR", False) is True
    for falsy in ("0", "false", "no", "off", "garbage"):
        os.environ["KILOBOT_TESTVAR"] = falsy
        assert launch._env_bool("KILOBOT_TESTVAR", True) is False
    os.environ.pop("KILOBOT_TESTVAR", None)
    assert launch._env_bool("KILOBOT_TESTVAR", True) is True

    os.environ["KILOBOT_TESTVAR"] = "none"
    assert launch._env_opt_int("KILOBOT_TESTVAR", 5) is None
    os.environ["KILOBOT_TESTVAR"] = "all"
    assert launch._env_opt_int("KILOBOT_TESTVAR", 5) is None
    os.environ["KILOBOT_TESTVAR"] = "42"
    assert launch._env_opt_int("KILOBOT_TESTVAR", 5) == 42
    os.environ.pop("KILOBOT_TESTVAR", None)
    assert launch._env_opt_int("KILOBOT_TESTVAR", 5) == 5


def test_build_actor_selects_deepset_by_default_and_gru_when_configured():
    launch = _import_launch()
    cfg = Config()
    cfg.actor_type = "deepset"
    a = launch.build_actor(cfg)
    assert isinstance(a, Actor)
    assert a.direct_motor is False

    cfg2 = Config()
    cfg2.actor_type = "gru"
    cfg2.gru_hidden = 48
    a2 = launch.build_actor(cfg2)
    assert isinstance(a2, RecurrentActor)
    assert a2.hidden_size == 48


def test_build_actor_wires_priv_mode_extra_and_direct_motor():
    launch = _import_launch()
    cfg = Config()
    cfg.actor_priv_mode = "dir_heading"   # 4 extra cols
    cfg.direct_motor = True
    a = launch.build_actor(cfg)
    assert isinstance(a, Actor)
    assert a.direct_motor is True
    # extra cols are folded into the parser's input width
    expected_in = SEED_SIZE + 4 + MESSAGE_SIZE + Z
    assert a.parser.linear1.in_features == expected_in


def test_resolve_behavior_finds_matching_action_spec():
    launch = _import_launch()

    class Spec:
        def __init__(self, n):
            self.action_spec = types.SimpleNamespace(continuous_size=n)

    class FakeEnv:
        behavior_specs = {"Other": Spec(3), "Kilobot?team=0": Spec(launch.ACTION_SIZE)}

    old = launch.BEHAVIOR_NAME
    launch.BEHAVIOR_NAME = None
    try:
        name = launch.resolve_behavior(FakeEnv())
        assert name == "Kilobot?team=0"
    finally:
        launch.BEHAVIOR_NAME = old


def test_resolve_behavior_raises_when_no_match():
    launch = _import_launch()

    class Spec:
        def __init__(self, n):
            self.action_spec = types.SimpleNamespace(continuous_size=n)

    class FakeEnv:
        behavior_specs = {"Other": Spec(3)}

    old = launch.BEHAVIOR_NAME
    launch.BEHAVIOR_NAME = None
    try:
        with pytest.raises(RuntimeError):
            launch.resolve_behavior(FakeEnv())
    finally:
        launch.BEHAVIOR_NAME = old


def test_resolve_behavior_honors_explicit_override():
    launch = _import_launch()
    old = launch.BEHAVIOR_NAME
    launch.BEHAVIOR_NAME = "ExplicitName"
    try:
        assert launch.resolve_behavior(object()) == "ExplicitName"
    finally:
        launch.BEHAVIOR_NAME = old


def _encoder_returning_z(x):
    return torch.zeros(Z)


def _encoder_returning_z_plus_one(x):
    return torch.zeros(Z + 1)


def test_check_latent_dim_raises_on_empty_pool():
    launch = _import_launch()
    with pytest.raises(RuntimeError):
        launch.check_latent_dim(_encoder_returning_z, [], "cpu")


def test_check_latent_dim_raises_on_mismatched_width():
    launch = _import_launch()
    with pytest.raises(RuntimeError):
        launch.check_latent_dim(_encoder_returning_z_plus_one, [torch.zeros(1, 1, 4, 4)], "cpu")


def test_check_latent_dim_passes_on_matching_width():
    launch = _import_launch()
    launch.check_latent_dim(_encoder_returning_z, [torch.zeros(1, 1, 4, 4)], "cpu")


def test_uses_parallel_trainer_only_for_rl_mode_with_multiple_workers():
    launch = _import_launch()
    assert launch.uses_parallel_trainer(2, "rl") is True
    assert launch.uses_parallel_trainer(4, "rl") is True
    for mode in ("bc", "probe", "reward_probe", "audit", "control"):
        assert launch.uses_parallel_trainer(2, mode) is False, mode


def test_uses_parallel_trainer_false_for_single_worker_any_mode():
    launch = _import_launch()
    for mode in ("rl", "bc", "probe", "reward_probe", "audit", "control"):
        assert launch.uses_parallel_trainer(1, mode) is False, mode


def test_preprocess_grayscale_image_shape_and_range():
    launch = _import_launch()
    from PIL import Image
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shape.png")
        arr = (np.random.rand(28, 28) * 255).astype("uint8")
        Image.fromarray(arr, mode = "L").save(path)
        tensor = launch.preprocess(path)
        assert tensor.shape == (1, 1, launch.IMAGE_SIZE, launch.IMAGE_SIZE)
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


def _gru_bc_buffer(cfg, n = 16, with_targets = True):
    buffer = RolloutBuffer(cfg)
    for i in range(n):
        z = torch.randn(Z)
        seed = torch.randn(SEED_SIZE)
        tx = torch.randn(MESSAGE_SIZE + 2)
        action = torch.randn(MESSAGE_SIZE + MOTOR_SIZE)
        buffer.add_decision(0, i, z, seed, tx, None, action, torch.tensor(0.0),
                            bc_target = torch.rand(MOTOR_SIZE) if with_targets else None,
                            prev_hidden = torch.randn(GRU_HIDDEN), prop = torch.randn(PROP_SIZE))
    return buffer


def _split_bc_buffer(cfg, hidden, n = 16, with_targets = True):
    buffer = RolloutBuffer(cfg)
    for i in range(n):
        z = torch.randn(Z)
        tc = torch.randn(SPLIT_TC_SIZE)
        action = torch.randn(MESSAGE_SIZE + MOTOR_SIZE)
        buffer.add_decision(0, i, z, tc, tc, None, action, torch.tensor(0.0),
                            bc_target = torch.rand(MOTOR_SIZE) if with_targets else None,
                            prev_hidden = torch.randn(hidden), prop = torch.randn(SPLIT_ODOM_SIZE))
    return buffer


def test_bc_update_trains_gru_actor():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru"
    cfg.minibatch = 4
    policy = GaussianPolicy(RecurrentActor())
    buffer = _gru_bc_buffer(cfg, n = 16, with_targets = True)
    opt = torch.optim.Adam(policy.parameters(), lr = 1e-2)

    before = policy.actor.head_motor.weight.detach().clone()
    first_loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 1)
    after_one = policy.actor.head_motor.weight.detach().clone()
    assert not torch.allclose(before, after_one)

    last_loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 20)
    assert last_loss < first_loss


def test_bc_update_trains_split_actor():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.minibatch = 4
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    buffer = _split_bc_buffer(cfg, cfg.split_gru_hidden, n = 16, with_targets = True)
    opt = torch.optim.Adam(policy.parameters(), lr = 1e-2)

    before = policy.actor.head_motor.weight.detach().clone()
    first_loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 1)
    after_one = policy.actor.head_motor.weight.detach().clone()
    assert not torch.allclose(before, after_one)

    last_loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 20)
    assert last_loss < first_loss


def test_bc_update_extra_stats_are_populated_and_backward_compatible():
    # regression test for the extra parameter added to support richer
    # per-iteration logging (docs/tuning.md): must be fully additive --
    # every existing caller that doesn't pass extra keeps getting exactly
    # the same single float return it always did.
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.minibatch = 4
    policy = GaussianPolicy(SplitObservationActor(gru_hidden = cfg.split_gru_hidden))
    buffer = _split_bc_buffer(cfg, cfg.split_gru_hidden, n = 16, with_targets = True)
    opt = torch.optim.Adam(policy.parameters(), lr = 1e-2)

    # without extra: identical to the pre-existing behavior, just a float
    loss_only = bc.bc_update(policy, opt, buffer, cfg, epochs = 3)
    assert isinstance(loss_only, float)

    # with extra: same return value contract, plus the dict gets populated
    extra = {}
    last_loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 3, extra = extra)
    assert isinstance(last_loss, float)
    assert "mean_loss" in extra and "grad_norm" in extra and "n_decisions" in extra
    assert extra["n_decisions"] == 16, "expected n_decisions to match the buffer's actual decision count"
    assert extra["grad_norm"] >= 0.0, "a norm can never be negative"
    # mean_loss is averaged over every minibatch of the fit; last_loss is
    # only the very last minibatch -- they need not be equal, but both
    # should be finite, sane loss values, not e.g. accidentally zero
    assert extra["mean_loss"] > 0.0


def test_bc_update_extra_with_no_targets_gets_sane_defaults():
    # the early-return (no bc_target decisions at all) path must also
    # populate extra sensibly, not leave it empty or raise
    cfg = Config()
    cfg.actor_type = "gru"
    policy = GaussianPolicy(RecurrentActor())
    buffer = _gru_bc_buffer(cfg, n = 5, with_targets = False)
    opt = torch.optim.Adam(policy.parameters(), lr = 1e-2)

    extra = {}
    loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 3, extra = extra)
    assert loss == 0.0
    assert extra == {"mean_loss": 0.0, "grad_norm": 0.0, "n_decisions": 0}


def test_bc_update_gru_no_targets_is_a_noop():
    cfg = Config()
    cfg.actor_type = "gru"
    policy = GaussianPolicy(RecurrentActor())
    buffer = _gru_bc_buffer(cfg, n = 5, with_targets = False)
    opt = torch.optim.Adam(policy.parameters(), lr = 1e-2)
    before = policy.actor.head_motor.weight.detach().clone()
    loss = bc.bc_update(policy, opt, buffer, cfg, epochs = 3)
    after = policy.actor.head_motor.weight.detach().clone()
    assert loss == 0.0
    assert torch.allclose(before, after)


def test_audit_style_replay_consistency_gru():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru"
    policy = GaussianPolicy(RecurrentActor())
    n = 10
    z = torch.randn(n, Z)
    seed = torch.randn(n, SEED_SIZE)
    msg = torch.randn(n, MESSAGE_SIZE)
    prop = torch.randn(n, PROP_SIZE)
    h0 = torch.zeros(n, policy.actor.hidden_size)
    u, env_action, log_prob, h_new = policy.act_batch_gru(z, seed, msg, prop, h0, deterministic = False)

    buffer = RolloutBuffer(cfg)
    for i in range(n):
        tx = torch.cat([msg[i], torch.zeros(2)])
        buffer.add_decision(0, i, z[i], seed[i], tx, None, u[i], log_prob[i],
                            prev_hidden = h0[i], prop = prop[i])

    ratio = diagnostics.audit_replay_ratio(policy, cfg, buffer)
    assert abs(float(ratio.mean()) - 1.0) <= 0.02
    assert float(ratio.std()) <= 0.05


def test_audit_style_replay_consistency_split_observation():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    policy = GaussianPolicy(SplitObservationActor())
    n = 10
    z = torch.randn(n, Z)
    tc = torch.randn(n, SPLIT_TC_SIZE)
    prop = torch.randn(n, SPLIT_ODOM_SIZE)
    h0 = torch.zeros(n, policy.actor.hidden_size)
    u, env_action, log_prob, h_new = policy.act_batch_split(tc, prop, h0, deterministic = False)

    buffer = RolloutBuffer(cfg)
    for i in range(n):
        buffer.add_decision(0, i, z[i], tc[i], tc[i], None, u[i], log_prob[i],
                            prev_hidden = h0[i], prop = prop[i])

    ratio = diagnostics.audit_replay_ratio(policy, cfg, buffer)
    assert abs(float(ratio.mean()) - 1.0) <= 0.02
    assert float(ratio.std()) <= 0.05


def test_audit_replay_ratio_returns_none_for_empty_buffer():
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    policy = GaussianPolicy(SplitObservationActor())
    buffer = RolloutBuffer(cfg)
    assert diagnostics.audit_replay_ratio(policy, cfg, buffer) is None


def test_audit_replay_ratio_deepset_still_works():
    torch.manual_seed(0)
    cfg = Config()
    cfg.actor_type = "deepset"
    policy = GaussianPolicy(Actor())
    n = 10
    z = torch.randn(n, Z)
    seed = torch.randn(n, SEED_SIZE)
    msg = torch.randn(n, MESSAGE_SIZE)
    sender = torch.arange(n).float()
    prev_rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE)
    prev_valid = torch.zeros(n, DB_CAPACITY, dtype = torch.bool)
    u, env_action, log_prob, new_rows, new_valid = policy.act_batch(z, seed, msg, sender, prev_rows, prev_valid,
                                                                     deterministic = False)

    buffer = RolloutBuffer(cfg)
    for i in range(n):
        tx = torch.cat([msg[i], sender[i:i + 1], torch.zeros(1)])
        buffer.add_decision(0, i, z[i], seed[i], tx, empty_database(), u[i], log_prob[i])

    ratio = diagnostics.audit_replay_ratio(policy, cfg, buffer)
    assert abs(float(ratio.mean()) - 1.0) <= 0.02
    assert float(ratio.std()) <= 0.05



class _FakeLogger:
    # records log_scalars calls directly, rather than parsing real
    # TensorBoard event file binary format -- tests the actual contract
    # bc_train needs (call log_scalars with the right data, call close at
    # the end), independent of Logger's own, separately-established
    # SummaryWriter implementation
    def __init__(self):
        self.calls = []
        self.closed = False

    def log_scalars(self, stats, step):
        self.calls.append((dict(stats), step))

    def close(self):
        self.closed = True


@requires_unity
def test_bc_train_writes_to_the_tensorboard_logger(unity_players):
    # regression test: bc_train previously took no logger parameter at all,
    # so TensorBoard could never show anything for a BC run no matter how
    # long it ran -- not a timing issue, launch.py constructed a real
    # Logger(run_dir) (which prints "tensorboard logging to ..." at
    # construction, independent of whether it's ever used) but never passed
    # it into bc_train, and bc_train had no parameter to receive it even if
    # it had.
    import conftest
    import trainer as T
    from policy import GaussianPolicy
    from kilobot_gnn import build_actor
    from config import Config

    cfg = conftest.unity_cfg(arenas = 1, rollout = 100)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 4, max_bots = 4)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)

    fake_logger = _FakeLogger()
    bc_mod.bc_train(tr, policy, actor_opt, cfg, 2, None, 4, None, logger = fake_logger)

    assert len(fake_logger.calls) == 2, "expected one log_scalars call per iteration, got %d" % len(fake_logger.calls)
    stats0, step0 = fake_logger.calls[0]
    assert step0 == 0
    assert "bc/motor_mse" in stats0 and "rollout/mean_coverage" in stats0
    stats1, step1 = fake_logger.calls[1]
    assert step1 == 1
    assert fake_logger.closed, "bc_train should close the logger when it's done, matching the RL training path's own convention"


@requires_unity
def test_bc_train_oracle_ceiling_is_captured_before_being_overwritten(unity_players):
    # regression test: Trainer.collect() resets its own coverage
    # accumulators (_roll_cov_sum, _roll_cov_count, etc.) to zero at the
    # start of every call, so the oracle-driven rollout's own coverage --
    # computed every tick regardless of who's driving -- was previously
    # discarded the moment the second, actor-eval collect() call started.
    # bc_train now calls trainer.rollout_payload() immediately after the
    # first (oracle) collect() call, before that reset happens. This test
    # verifies the captured oracle stats are genuinely the oracle's own
    # rollout, not an accidental duplicate of the actor's -- by
    # independently tracking every rollout_payload() call and confirming
    # bc_train's logged "oracle/..." keys match the first call specifically,
    # not the second.
    import conftest
    import trainer as T
    from policy import GaussianPolicy
    from kilobot_gnn import build_actor
    from config import Config
    from metrics import rollout_stats

    cfg = conftest.unity_cfg(arenas = 1, rollout = 200)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 8, max_bots = 12)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)

    captured_payloads = []
    orig_payload = T.Trainer.rollout_payload
    def tracking_payload(self):
        pay = orig_payload(self)
        captured_payloads.append(dict(pay))
        return pay
    T.Trainer.rollout_payload = tracking_payload

    fake_logger = _FakeLogger()
    bc_mod.bc_train(tr, policy, actor_opt, cfg, 1, None, 4, None, logger = fake_logger)

    T.Trainer.rollout_payload = orig_payload

    assert len(captured_payloads) == 2, "expected exactly two rollout_payload() calls per iteration (oracle, then actor), got %d" % len(captured_payloads)
    independently_computed_oracle_stats = rollout_stats(captured_payloads[0])
    independently_computed_actor_stats = rollout_stats(captured_payloads[1])

    logged_stats, logged_step = fake_logger.calls[0]
    assert logged_step == 0
    for key, val in independently_computed_oracle_stats.items():
        assert "oracle/" + key in logged_stats, "expected oracle/%s to be logged" % key
        assert abs(logged_stats["oracle/" + key] - val) < 1e-6, "oracle/%s mismatch: logged %r vs independently recomputed %r" % (key, logged_stats["oracle/" + key], val)
    for key, val in independently_computed_actor_stats.items():
        assert key in logged_stats, "expected %s (the actor's own, unprefixed key) to be logged" % key
        assert abs(logged_stats[key] - val) < 1e-6, "%s mismatch: logged %r vs independently recomputed %r" % (key, logged_stats[key], val)


@requires_unity
def test_bc_train_checkpoint_survives_a_mid_run_interruption(unity_players):
    # regression test, added directly after
    # this exact gap cost a real, unattended run several hours of
    # unrecoverable progress: bc_train previously called export_actor
    # exactly once, after the full loop finished, so an interruption at any
    # point before that -- a Ctrl+C, a crash -- meant nothing was ever
    # saved. This test simulates precisely that: forces an exception
    # partway through a run (after at least one periodic checkpoint should
    # have fired) and confirms a real, loadable checkpoint exists anyway,
    # rather than trusting that periodic saving "should" work from reading
    # the loop structure alone.
    import os
    import tempfile
    import conftest
    import trainer as T
    from policy import GaussianPolicy
    from kilobot_gnn import build_actor
    from config import Config
    from checkpoint import load_for_eval

    cfg = conftest.unity_cfg(arenas = 1, rollout = 100)
    tr, worker = conftest.unity_trainer(cfg, min_bots = 6, max_bots = 10)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)

    call_count = [0]
    orig_collect = T.Trainer.collect
    def collect_that_crashes_on_iteration_7(self, policy, critic, deterministic = False):
        call_count[0] += 1
        # each bc_train iteration calls collect() twice (oracle, then actor
        # eval); crash partway through iteration 7's oracle collect, well
        # after checkpoint_every=5 should have already fired once at the
        # end of iteration 4 (0-indexed: iterations 0-4 complete = 5 done)
        if call_count[0] == 15:
            raise RuntimeError("simulated crash / Ctrl+C mid-run")
        return orig_collect(self, policy, critic, deterministic = deterministic)
    T.Trainer.collect = collect_that_crashes_on_iteration_7

    with tempfile.TemporaryDirectory() as d:
        bc_out = os.path.join(d, "checkpoint.pt")
        try:
            bc_mod.bc_train(tr, policy, actor_opt, cfg, 20, None, 4, bc_out, checkpoint_every = 5)
            assert False, "expected the simulated crash to propagate"
        except RuntimeError as e:
            assert "simulated crash" in str(e)
        finally:
            T.Trainer.collect = orig_collect

        assert os.path.exists(bc_out), "a periodic checkpoint should have survived the crash, but no file exists at bc_out"
        # confirm it's a real, loadable checkpoint, not a corrupted partial write
        fresh_policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
        load_for_eval(bc_out, fresh_policy, "cpu")


def test_scripted_motors_stopped_mask_produces_a_hard_zero_motor_command():
    # direct test of scripted_motors' own override: a robot flagged stopped
    # must get exactly (0, 0), regardless of what the normal steering math
    # would otherwise compute -- and confirms a naive zero direction vector
    # would NOT have stopped it (turn=0 means drive straight, not stop),
    # which is exactly why this explicit mask exists
    import actor_io
    import torch

    node = torch.zeros(2, 19)
    node[:, 2:4] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])   # both facing +x
    assigned_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0]])   # both target is straight ahead -- would drive full speed
    stopped_mask = torch.tensor([True, False])

    motors = actor_io.scripted_motors(node, "oracle", None, assigned_dir=assigned_dir, stopped_mask=stopped_mask, belief_heading = node[:, 2:4])

    assert torch.allclose(motors[0], torch.zeros(2)), "a robot flagged stopped must get an exact (0,0) motor command"
    assert not torch.allclose(motors[1], torch.zeros(2)), "a robot NOT flagged stopped must still drive normally"


def test_wall_tangent_is_clockwise_wall_on_the_left():
    # regression test for the CW correction: north wall tangent is +X (a
    # right/CW turn onto the east wall's -Y tangent), not the earlier,
    # incorrect -X counterclockwise version
    # simple_oracle.py owns WALL_TANGENT now -- it is the oracle, and the
    # deprecated lock-based one this used to be read from is gone
    import simple_oracle
    import numpy as np
    assert np.allclose(simple_oracle.WALL_TANGENT["north"], [1.0, 0.0])
    assert np.allclose(simple_oracle.WALL_TANGENT["east"], [0.0, -1.0])
    assert np.allclose(simple_oracle.WALL_TANGENT["south"], [-1.0, 0.0])
    assert np.allclose(simple_oracle.WALL_TANGENT["west"], [0.0, 1.0])
    # confirm the turn at a corner is genuinely clockwise (right): rotating
    # the north tangent -90 degrees (CW) must equal the east tangent
    rotate_cw = lambda v: np.array([v[1], -v[0]])
    assert np.allclose(rotate_cw(simple_oracle.WALL_TANGENT["north"]), simple_oracle.WALL_TANGENT["east"])


def test_every_cfg_field_referenced_in_launch_py_is_actually_declared():
    # regression test for a real, severe bug: oracle_orbit_axis_trust_threshold
    # was referenced directly in launch.py (cfg.oracle_orbit_axis_trust_threshold,
    # not via getattr with a fallback) but its dataclass field declaration was
    # never actually written -- only its comment block was. Every other
    # consumption of this field went through getattr(cfg, "...", default),
    # which silently tolerates a missing attribute, and no test ever
    # instantiates Config() and reads the field directly the way launch.py's
    # own env-var wiring does -- so 273 passing tests never caught this.
    # launch.py's main() itself is never exercised by pytest at all (it
    # requires a real Unity connection), so this was the one code path
    # capable of exposing the gap, and nothing was checking it. Found only
    # when the user's actual run crashed with AttributeError on this exact
    # line. This test closes that gap structurally: every cfg.FIELD
    # reference in launch.py must correspond to an actual declared field,
    # not just be assumed present.
    import ast
    import re
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.py")) as f:
        config_src = f.read()
    tree = ast.parse(config_src)
    declared = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    declared.add(item.target.id)

    # Every entry point, not just launch.py: run_bc_monitored.py had drifted
    # the same way (cfg.val_tape_ticks / cfg.val_probe_ticks written but never
    # declared or read), which this guard did not cover.
    referenced = set()
    for entry in ("launch.py", "run_bc_monitored.py", "run_bc_simple_oracle.py"):
        with open(os.path.join(here, entry)) as f:
            referenced |= set(re.findall(r'\bcfg\.([a-zA-Z_][a-zA-Z0-9_]*)', f.read()))

    # legitimate dynamic attributes: written onto cfg at runtime (not a
    # declared configuration field with a default), always read elsewhere
    # via getattr with a fallback, never read directly in launch.py itself
    known_dynamic = {"_oracle_coordinator", "_oracle_formation_pool"}

    missing = referenced - declared - known_dynamic
    assert not missing, "an entry point references cfg.%s but Config has no such declared field -- this will crash the moment launch.py actually runs, exactly as it did for oracle_orbit_axis_trust_threshold" % missing


def test_turn_sign_matches_unity_right_minus_left_convention():
    # replaces the previous version of this test
    # (from the original dtheta sign investigation), which asserted the
    # opposite sign convention based on an assumed Unity
    # turnRate=(left-right)*turnSpeed formula. That assumption was never
    # actually confirmed against the live, running KilobotMovement.cs --
    # only assumed and simulated in Python -- and was later proven backwards
    # by direct, real-Unity evidence when the identically-assumed formula
    # broke the steering law itself (real robot headings
    # measured opposite to their target direction under that same
    # assumption; reverting the sign there fixed it). Reverting the dead-
    # reckoning sign to match is the direct fix for the resulting belief-
    # filter drift: every particle sharing the same wrong-signed dtheta
    # formula drifted together in the same wrong direction every tick,
    # staying tightly clustered (confident) while walking away from the
    # truth -- confirmed directly from real Unity logs (median ~90-unit
    # belief-position error, near-zero correlation with reported
    # confidence). This test asserts the corrected convention: omega
    # proportional to (right - left), matching turnRate = (right - left) *
    # turnSpeed.
    from kinematics import split_tick_motion, dead_reckon

    left_faster = torch.tensor([[0.9, 0.1]])   # left > right
    steps = torch.tensor([20.0])

    _, _, dtheta, _ = split_tick_motion(left_faster, steps, 0.02, 0.10, 0.05)
    assert float(dtheta) < 0.0, "left > right must give NEGATIVE dtheta in split_tick_motion, matching unity's turnRate = (right - left) * turnSpeed"

    dead_out = dead_reckon(left_faster, steps, torch.tensor([0.0]), 0.02, 0.10, 0.05, 1.0, 1.0, 1.0)
    sin_dtheta = float(dead_out[0, 2])
    assert sin_dtheta < 0.0, "left > right must give NEGATIVE sin(dtheta) in dead_reckon, matching unity's turnRate = (right - left) * turnSpeed"

    right_faster = torch.tensor([[0.1, 0.9]])   # right > left, opposite case
    _, _, dtheta2, _ = split_tick_motion(right_faster, steps, 0.02, 0.10, 0.05)
    assert float(dtheta2) > 0.0, "right > left must give POSITIVE dtheta, the mirror-image case"


def test_steering_law_symmetric_properties_hold():
    # replaces the previous version of this test
    # , which claimed to verify convergence against "unity's own,
    # real turnRate=(left-right)*turnSpeed formula" -- but that formula was
    # never actually confirmed against the live, running KilobotMovement.cs,
    # only assumed and simulated in Python. Direct evidence from real Unity
    # (WALL_DEBUG_MOTOR logging, 2390 committed-robot samples) later showed
    # the negated sign that test was "verifying" produced heading converging
    # to the OPPOSITE of the target (median dot(heading, assigned_dir) =
    # -0.982) -- i.e. the earlier test was checking a model of Unity's
    # physics that did not match the real thing, and happened to produce a
    # self-consistent but wrong answer. The actual physical rotational
    # direction (which way left/right maps to clockwise vs counterclockwise
    # turning) is a fact about the real, compiled KilobotMovement.cs that
    # cannot be reliably established by simulating an assumed formula in
    # Python -- it can only be verified against real Unity data, which is
    # what actually caught and fixed this. What CAN be verified honestly
    # from this side, without assuming that formula's sign: the steering
    # law drives straight when already facing the target, and turns in a
    # consistent, non-trivial way when it is not -- properties any correct
    # sign convention must have, regardless of which physical direction
    # left-right maps to.
    from types import SimpleNamespace
    import numpy as np
    tr = _mk_trainer_shell(SimpleNamespace())

    def motors_for(heading_deg, target_deg):
        node = torch.zeros(1, NODE_FEATURES)
        h = np.radians(heading_deg)
        g = np.radians(target_deg)
        node[0, 2] = np.cos(h)
        node[0, 3] = np.sin(h)
        node[0, 5] = np.cos(g)
        node[0, 6] = np.sin(g)
        out = actor_io.scripted_motors(node, "oracle", getattr(tr.cfg, "force_motor", None), belief_heading = node[:, 2:4])
        return float(out[0, 0]), float(out[0, 1])

    # already facing the target: drive straight, no turn
    left, right = motors_for(30.0, 30.0)
    assert abs(left - right) < 1e-4, "facing the target exactly should produce zero turn"

    # not facing the target: some turn must happen, consistently, for a
    # range of angular offsets -- not asserting which physical direction
    for offset in [10, 45, 90, 135, 170]:
        left, right = motors_for(0.0, float(offset))
        assert abs(left - right) > 1e-3, "a %d degree heading error must produce a nonzero turn" % offset
        left_neg, right_neg = motors_for(0.0, float(-offset))
        # the two offsets are mirror images of each other; the turn produced
        # must also mirror (equal magnitude, opposite sign of left-right)
        assert abs((left - right) + (left_neg - right_neg)) < 1e-3, \
            "mirrored heading errors must produce mirrored turns"


def test_formation_paths_limit_is_not_always_the_first_n_alphabetically():
    # regression test: confirmed directly as a
    # real, reported bug -- names[:limit] always took the first `limit`
    # names after an alphabetical sort, so any limit smaller than the full
    # folder always produced the exact same subset (e.g. 000000.png,
    # 000001.png, ...) no matter how many times the script ran. This test
    # builds a temp folder of 200 distinctly-named files and confirms a
    # limited sample is NOT simply the alphabetically-first N -- true with
    # overwhelming probability for a genuine random sample of 10 out of
    # 200, and false with certainty for the old, unfixed behavior.
    import tempfile
    import os
    import images

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(200):
            open(os.path.join(tmp, "%06d.png" % i), "w").close()
        images._formation_paths_cache.clear()
        paths = images.formation_paths(tmp, limit = 10)
        names = sorted(os.path.basename(p) for p in paths)
        first_ten = sorted("%06d.png" % i for i in range(10))
        assert names != first_ten, \
            "a limited sample must not simply be the alphabetically-first N files"


def test_formation_paths_same_args_agree_within_one_run():
    # build_image_pool (the encoder's tensor pool) and build_formation_pool
    # (the reward/oracle's geometry pool) are separate calls with identical
    # arguments that must agree index-for-index on which formation each
    # index refers to -- confirmed here directly: two calls with the same
    # (folder, pattern, limit) must return the identical subset, in the
    # identical order, within the same run.
    import tempfile
    import os
    import images

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(150):
            open(os.path.join(tmp, "%06d.png" % i), "w").close()
        images._formation_paths_cache.clear()
        paths1 = images.formation_paths(tmp, limit = 20)
        paths2 = images.formation_paths(tmp, limit = 20)
        assert paths1 == paths2, "repeated calls with identical arguments must agree within the same run"


def test_formation_paths_varies_across_separate_cache_states():
    # confirms the sampled subset genuinely differs run to run (a fresh
    # process has an empty cache, same as clearing it here) rather than
    # being pinned to one fixed subset forever -- the original bug's
    # defining symptom was that it never varied at all, across any number
    # of reruns.
    import tempfile
    import os
    import images

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(300):
            open(os.path.join(tmp, "%06d.png" % i), "w").close()
        seen = set()
        for _ in range(5):
            images._formation_paths_cache.clear()
            paths = images.formation_paths(tmp, limit = 8)
            seen.add(tuple(sorted(os.path.basename(p) for p in paths)))
        assert len(seen) > 1, "the sampled subset must vary across separate runs, not stay fixed forever"


def test_build_image_pool_and_build_formation_pool_stay_paired():
    # end-to-end check of the actual correctness property this all exists
    # for: index i must mean the same real formation file in both the
    # encoder's tensor pool and the geometry pool, even after the phase-63
    # fix changed the pool from a deterministic prefix to a random sample.
    import tempfile
    import os
    import torch
    import numpy as np
    from PIL import Image
    import images
    from formations import build_formation_pool

    def preprocess(path):
        return torch.zeros(1, 4, 4)

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(80):
            arr = np.zeros((8, 8), dtype = np.uint8)
            arr[3:5, 3:5] = 255   # a small on-pixel block, present in every image
            Image.fromarray(arr, mode = "L").save(os.path.join(tmp, "%06d.png" % i))
        images._formation_paths_cache.clear()
        image_pool = images.build_image_pool(tmp, preprocess, limit = 15)
        formation_pool = build_formation_pool(tmp, limit = 15)
        assert len(image_pool) == len(formation_pool) == 15


def test_watch_oracle_preserves_simple_oracle_override():
    # diagnostics.watch_oracle used to force
    # cfg.motor_override = "oracle" unconditionally, which silently
    # defeated scripts/watch_oracle.sh's own KILOBOT_MOTOR_OVERRIDE=
    # simple_oracle -- both scripts dispatch through this same
    # KILOBOT_MODE=watch_oracle entry point, so the old, unconditional
    # force meant scripts/watch_oracle.sh always actually watched the old
    # oracle. A sentinel exception on the first trainer.collect call
    # stands in for Ctrl+C, since this function otherwise loops forever.
    class _Stop(Exception):
        pass

    class _FakeTrainer:
        def setup(self):
            pass

        def collect(self, policy, x, deterministic):
            raise _Stop()

    trainer = _FakeTrainer()

    from types import SimpleNamespace
    cfg = SimpleNamespace(motor_override = "simple_oracle")
    try:
        diagnostics.watch_oracle(trainer, None, cfg)
    except _Stop:
        pass
    assert cfg.motor_override == "simple_oracle", \
        "an explicitly-chosen simple_oracle override must survive watch_oracle, not be overwritten"

    # the same must hold for an explicit "oracle" -- the deprecated controller
    # is still selectable on purpose, it just isn't the default any more
    cfg_old = SimpleNamespace(motor_override = "oracle")
    try:
        diagnostics.watch_oracle(trainer, None, cfg_old)
    except _Stop:
        pass
    assert cfg_old.motor_override == "oracle", \
        "an explicitly-chosen oracle override must survive watch_oracle too"

    cfg2 = SimpleNamespace(motor_override = "none")
    try:
        diagnostics.watch_oracle(trainer, None, cfg2)
    except _Stop:
        pass
    assert cfg2.motor_override == "simple_oracle", \
        "with no controller chosen, watch_oracle defaults to simple_oracle -- the old " \
        "lock-based oracle is deprecated (see oracle.py's module docstring), so a bare " \
        "KILOBOT_MODE=watch_oracle should show the controller this project actually uses"
