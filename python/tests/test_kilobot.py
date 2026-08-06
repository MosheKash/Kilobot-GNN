import pytest
import torch

from kilobot_gnn import Actor, Critic, empty_database, Z, SEED_SIZE, TRANSMISSION_SIZE, MESSAGE_SIZE, MOTOR_SIZE, NODE_FEATURES, actor_forward_batch, DB_ROW_SIZE, DB_CAPACITY
from config import Config
from reward import compute_rewards, DIST_COL, NEAREST_COL
from gae import compute_gae
from graph_batch import build_critic_batch
from policy import GaussianPolicy
from buffer import RolloutBuffer
from ppo import ppo_update
from trainer import sample_message, STRENGTH_COL


def make_transmission(robot_id):
    tx = torch.randn(TRANSMISSION_SIZE)
    tx[MESSAGE_SIZE] = float(robot_id)
    return tx


def test_actor_output_ranges():
    from policy import squash_action
    actor = Actor()
    db = empty_database()
    z = torch.randn(Z)
    seed = torch.randn(SEED_SIZE)
    out_tx, motors, db = actor(z, seed, make_transmission(1), db)
    assert out_tx.shape == (MESSAGE_SIZE,)
    assert motors.shape == (2,)
    # The actor now emits an unbounded mean; the policy's squash maps it into range.
    mean = torch.cat([out_tx, motors], dim=0).detach()
    a = squash_action(mean)
    msg = a[:MESSAGE_SIZE]
    mot = a[MESSAGE_SIZE:]
    assert float(msg.min()) >= -1.0 and float(msg.max()) <= 1.0
    assert float(mot.min()) >= 0.0 and float(mot.max()) <= 1.0


def test_database_eviction():
    actor = Actor()
    db = empty_database()
    z = torch.randn(Z)
    seed = torch.randn(SEED_SIZE)
    _, _, db = actor(z, seed, make_transmission(1), db)
    _, _, db = actor(z, seed, make_transmission(2), db)
    assert db.shape[0] == 2
    _, _, db = actor(z, seed, make_transmission(1), db)
    assert db.shape[0] == 2


def test_critic_batched_equals_single():
    torch.manual_seed(1)
    critic = Critic()
    critic.eval()
    sizes = [4, 6, 3]
    nodes, edges, attrs, zs = [], [], [], []
    for m in sizes:
        nodes.append(torch.randn(m, NODE_FEATURES))
        e = m * 2
        edges.append(torch.randint(0, m, (2, e)))
        attrs.append(torch.rand(e, 1))
        zs.append(torch.randn(Z))

    x, edge_attr, edge_index, z, batch = build_critic_batch(nodes, edges, attrs, zs)
    with torch.no_grad():
        batched = critic(x, edge_attr, edge_index, z, batch).squeeze(-1)

    offset = 0
    for i in range(len(sizes)):
        with torch.no_grad():
            single = critic(nodes[i], attrs[i], edges[i], zs[i]).squeeze(-1)
        chunk = batched[offset:offset + sizes[i]]
        assert torch.allclose(single, chunk, atol=1e-5)
        offset += sizes[i]


def test_reward_signs():
    cfg = Config()
    node = torch.zeros(2, NODE_FEATURES)
    node[0, DIST_COL] = 0.0
    node[0, NEAREST_COL] = 1.0
    node[1, DIST_COL] = 0.9
    node[1, NEAREST_COL] = 1.0
    r = compute_rewards(node, cfg)
    assert float(r[0]) > 0.0
    assert float(r[1]) < 0.0


def test_gae_truncation_vs_terminal():
    rewards = torch.ones(3)
    values = torch.zeros(3)
    cut = torch.tensor([0.0, 0.0, 1.0])

    boot = torch.tensor([0.0, 0.0, 5.0])
    term_trunc = torch.tensor([0.0, 0.0, 0.0])
    term_succ = torch.tensor([0.0, 0.0, 1.0])

    adv_trunc, _ = compute_gae(rewards, values, term_trunc, cut, boot, 1.0, 1.0)
    adv_succ, _ = compute_gae(rewards, values, term_succ, cut, boot, 1.0, 1.0)

    assert torch.allclose(adv_succ, torch.tensor([3.0, 2.0, 1.0]))
    assert float(adv_trunc[2]) == 6.0
    assert float(adv_trunc[2] - adv_succ[2]) == 5.0


def test_actor_batch_matches_per_agent():
    torch.manual_seed(0)
    actor = Actor()
    actor.eval()
    n = 50

    def pad(dbs):
        rows = torch.zeros(len(dbs), DB_CAPACITY, DB_ROW_SIZE)
        valid = torch.zeros(len(dbs), DB_CAPACITY, dtype=torch.bool)
        for i, db in enumerate(dbs):
            if db.shape[0] > 0:
                rows[i, :db.shape[0]] = db
                valid[i, :db.shape[0]] = True
        return rows, valid

    zs = [torch.randn(Z) for _ in range(n)]
    seeds = [torch.randn(SEED_SIZE) for _ in range(n)]
    dbs = []
    for i in range(n):
        m = i % 5
        if m == 0:
            dbs.append(empty_database())
        else:
            rows = []
            for j in range(m):
                rows.append(torch.cat([torch.randn(MESSAGE_SIZE),
                                       torch.tensor([float(j + 1)]),
                                       torch.tensor([float(j + 1)])]))
            dbs.append(torch.stack(rows))

    for _ in range(3):
        tx = []
        for i in range(n):
            sender = float(dbs[i][0, MESSAGE_SIZE]) if (dbs[i].shape[0] > 0 and i % 2 == 0) else float(100 + i)
            tx.append(torch.cat([torch.randn(MESSAGE_SIZE), torch.tensor([sender]), torch.rand(1)]))

        ref, new_dbs = [], []
        for i in range(n):
            with torch.no_grad():
                ot, mc, db2 = actor(zs[i], seeds[i], tx[i], dbs[i])
            ref.append(torch.cat([ot, mc]))
            new_dbs.append(db2)
        ref = torch.stack(ref)

        tx_b = torch.stack(tx)
        rows, valid = pad(dbs)
        with torch.no_grad():
            mean_b, rows, valid = actor_forward_batch(actor, torch.stack(zs), torch.stack(seeds),
                                                      tx_b[:, :MESSAGE_SIZE], tx_b[:, MESSAGE_SIZE], rows, valid)

        assert torch.allclose(mean_b, ref, atol=1e-5)
        for i in range(n):
            got = rows[i][valid[i]]
            assert got.shape[0] == new_dbs[i].shape[0]
        dbs = new_dbs


def test_ppo_actor_loss_batched_matches_per_decision():
    import ppo
    torch.manual_seed(1)
    policy = GaussianPolicy(Actor())
    n = 24
    decisions = []
    for i in range(n):
        z = torch.randn(Z)
        seed = torch.randn(SEED_SIZE)
        tx = torch.cat([torch.randn(MESSAGE_SIZE), torch.tensor([float(10 + i % 4)]), torch.rand(1)])
        m = i % 3
        if m == 0:
            prev = empty_database()
        else:
            prev = torch.stack([torch.cat([torch.randn(MESSAGE_SIZE), torch.tensor([float(30 + j)]),
                                           torch.tensor([float(j + 1)])]) for j in range(m)])
        action = torch.randn(MESSAGE_SIZE + 2)
        with torch.no_grad():
            lp, _ = policy.evaluate(z, seed, tx, prev, action)
        decisions.append({"z": z, "seed": seed, "transmission": tx, "prev_db": prev,
                          "action": action, "old_log_prob": lp.detach()})

    advantages = torch.randn(n)

    class C:
        clip = 0.2
        entropy_coef = 0.01

    cfg = C()
    chunk = torch.arange(n)

    total = torch.zeros(())
    ent = torch.zeros(())
    for idx in chunk.tolist():
        d = decisions[idx]
        lp, e = policy.evaluate(d["z"], d["seed"], d["transmission"], d["prev_db"], d["action"])
        ratio = (lp - d["old_log_prob"]).exp()
        total = total - torch.min(ratio * advantages[idx],
                                  ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[idx])
        ent = ent + e
    ref_loss = total / n - cfg.entropy_coef * (ent / n)
    policy.zero_grad()
    ref_loss.backward()
    ref_grad = torch.cat([p.grad.flatten() for p in policy.parameters()]).clone()

    data = ppo._stack_decisions(decisions)
    policy.zero_grad()
    loss, _, _, _ = ppo._actor_loss(policy, data, advantages, chunk, cfg)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in policy.parameters()]).clone()

    assert abs(float(ref_loss) - float(loss)) < 1e-4
    assert (ref_grad - grad).abs().max() < 1e-4


def test_gae_rollout_boundary_no_cut():
    # Rollout ends mid-episode: the last step has cut=0 and there is no values[t+1].
    # It must bootstrap from boot instead of indexing past the end.
    rewards = torch.ones(4)
    values = torch.zeros(4)
    term = torch.zeros(4)
    cut = torch.zeros(4)
    boot = torch.tensor([0.0, 0.0, 0.0, 5.0])

    adv, ret = compute_gae(rewards, values, term, cut, boot, 1.0, 1.0)

    # Last step bootstraps: delta = 1 + 5 - 0 = 6, and earlier steps accumulate it.
    assert float(adv[3]) == 6.0
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.tensor([9.0, 8.0, 7.0, 6.0]))


def test_policy_evaluate_matches_act():
    torch.manual_seed(2)
    policy = GaussianPolicy(Actor())
    z = torch.randn(Z)
    seed = torch.randn(SEED_SIZE)
    prev_db = empty_database()
    last = make_transmission(4)
    action, env_action, log_prob, _ = policy.act(z, seed, last, prev_db)
    log_prob2, entropy = policy.evaluate(z, seed, last, prev_db, action)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    assert torch.isfinite(entropy)


def synth_buffer(cfg, policy):
    buffer = RolloutBuffer(cfg)
    z = torch.randn(Z)
    arena = 0
    m = 3
    db = {local: empty_database() for local in range(m)}
    for step in range(4):
        node = torch.randn(m, NODE_FEATURES)
        edge_index = torch.randint(0, m, (2, m * 2))
        edge_attr = torch.rand(m * 2, 1)
        reward = compute_rewards(node, cfg)
        term = torch.zeros(m)
        cut = torch.zeros(m)
        is_decision = torch.ones(m, dtype=torch.bool)
        if step == 3:
            cut[:] = 1.0
        traj = torch.arange(m, dtype=torch.long)
        si = buffer.add_step(arena, step, node, edge_index, edge_attr, z,
                             traj, reward, term, cut, is_decision)
        for local in range(m):
            seed = torch.randn(SEED_SIZE)
            last = make_transmission(local + 1)
            prev_db = db[local].detach().clone()
            action, _, log_prob, db[local] = policy.act(z, seed, last, prev_db)
            buffer.add_decision(si, local, z, seed, last, prev_db,
                                action.detach(), log_prob.detach())
    return buffer


def test_buffer_returns_match_direct_gae():
    cfg = Config()
    policy = GaussianPolicy(Actor())
    critic = Critic()
    buffer = synth_buffer(cfg, policy)
    buffer.compute_returns(critic)

    rewards = torch.stack([buffer.steps[si]["reward"][0] for si in range(4)])
    values = torch.stack([buffer.values[si][0] for si in range(4)])
    term = torch.stack([buffer.steps[si]["term"][0] for si in range(4)])
    cut = torch.stack([buffer.steps[si]["cut"][0] for si in range(4)])
    adv, ret = compute_gae(rewards, values, term, cut, values, cfg.gamma, cfg.gae_lambda)

    got = torch.stack([buffer.returns[si][0] for si in range(4)])
    assert torch.allclose(got, ret, atol=1e-5)


def test_critic_chunking_matches_full():
    import copy
    from ppo import _explained_variance, _critic_update
    torch.manual_seed(0)
    cfg = Config()
    policy = GaussianPolicy(Actor())
    critic = Critic()
    critic.eval()
    buffer = synth_buffer(cfg, policy)
    buffer.compute_returns(critic)

    # old values: full batch vs one graph at a time
    cfg.critic_chunk_steps = 0
    full_vals = torch.cat(buffer._old_values(critic))
    cfg.critic_chunk_steps = 1
    chunk_vals = torch.cat(buffer._old_values(critic))
    assert torch.allclose(full_vals, chunk_vals, atol=1e-5)

    # explained variance is independent of chunk size
    cfg.critic_chunk_steps = 0
    ev_full = _explained_variance(critic, buffer)
    cfg.critic_chunk_steps = 2
    ev_chunk = _explained_variance(critic, buffer)
    assert abs(ev_full - ev_chunk) < 1e-5

    # critic gradients match between full and chunked backward (lr 0, compare .grad)
    c1 = copy.deepcopy(critic)
    c2 = copy.deepcopy(critic)
    opt1 = torch.optim.SGD(c1.parameters(), lr=0.0)
    opt2 = torch.optim.SGD(c2.parameters(), lr=0.0)
    cfg.critic_chunk_steps = 0
    _critic_update(c1, opt1, buffer, cfg)
    cfg.critic_chunk_steps = 1
    _critic_update(c2, opt2, buffer, cfg)
    for p1, p2 in zip(c1.parameters(), c2.parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4)


def test_ppo_update_runs_and_separates():
    cfg = Config()
    cfg.minibatch = 4
    cfg.ppo_epochs = 2
    policy = GaussianPolicy(Actor())
    critic = Critic()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    before_actor = policy.log_std.detach().clone()
    before_net = policy.actor.deepset.rho_2.weight.detach().clone()
    before_critic = next(critic.parameters()).detach().clone()

    buffer = synth_buffer(cfg, policy)
    logs = ppo_update(policy, critic, actor_opt, critic_opt, buffer, cfg)

    assert "actor_loss" in logs and "critic_loss" in logs
    assert not torch.allclose(before_actor, policy.log_std.detach())
    assert not torch.allclose(before_net, policy.actor.deepset.rho_2.weight.detach())
    assert not torch.allclose(before_critic, next(critic.parameters()).detach())


def test_guarded_imports():
    import channels
    import trainer
    assert hasattr(channels, "CriticChannel")
    assert hasattr(trainer, "Trainer")


def make_messages(strengths, senders=None):
    n = len(strengths)
    rows = torch.zeros((n, MESSAGE_SIZE + 2))
    rows[:, :MESSAGE_SIZE] = torch.randn(n, MESSAGE_SIZE)
    for i in range(n):
        sid = i + 1 if senders is None else senders[i]
        rows[i, MESSAGE_SIZE] = float(sid)
        rows[i, STRENGTH_COL] = float(strengths[i])
    return rows


def test_sample_message_single():
    rows = make_messages([0.3])
    chosen = sample_message(rows, None)
    assert torch.equal(chosen, rows[0])


def test_sample_message_weighted():
    gen = torch.Generator()
    gen.manual_seed(0)
    rows = make_messages([0.001, 0.001, 10.0, 0.001])
    counts = torch.zeros(rows.shape[0])
    for _ in range(2000):
        chosen = sample_message(rows, gen)
        match = (rows == chosen).all(dim=1).nonzero()
        counts[int(match[0])] = counts[int(match[0])] + 1
    assert int(counts.argmax()) == 2
    assert float(counts[2]) > float(counts.sum()) * 0.8


def test_sample_message_degenerate_weights():
    gen = torch.Generator()
    gen.manual_seed(1)
    rows = make_messages([0.0, 0.0, 0.0])
    seen = set()
    for _ in range(200):
        chosen = sample_message(rows, gen)
        match = (rows == chosen).all(dim=1).nonzero()
        seen.add(int(match[0]))
    assert seen == {0, 1, 2}


def test_env_worker_state_isolation():
    from env_worker import EnvWorker
    w0 = EnvWorker(None, None, None, "Kilobot")
    w1 = EnvWorker(None, None, None, "Kilobot")
    w0.z[0] = torch.ones(4)
    w1.z[0] = torch.zeros(4)
    w0.databases[0] = {1: "a"}
    assert float(w0.z[0].sum()) == 4.0
    assert float(w1.z[0].sum()) == 0.0
    assert 0 not in w1.databases


def test_global_traj_ids_unique_across_workers():
    from types import SimpleNamespace
    from env_worker import EnvWorker
    import trainer as T
    cfg = SimpleNamespace(seed=0, device="cpu")
    tr = T.Trainer(None, None, None, cfg, None, None, "Kilobot")
    tr.workers.append(EnvWorker(None, None, None, "Kilobot"))
    a = tr._traj_for(tr.workers[0], 0, 0)
    b = tr._traj_for(tr.workers[1], 0, 0)
    c = tr._traj_for(tr.workers[0], 0, 0)
    assert a != b
    assert c == a
    assert tr.workers[0].traj_id[(0, 0)] == a
    assert tr.workers[1].traj_id[(0, 0)] == b


import time as _time


class _FakeWorker:
    def __init__(self, tag, delay=0.0):
        self.tag = tag
        self.delay = delay
        self.calls = []

    def step(self):
        if self.delay:
            _time.sleep(self.delay)
        self.calls.append("step")

    def get_steps(self):
        self.calls.append("get_steps")
        return ("ds_" + self.tag, "ts_" + self.tag)

    def idle_other_behaviors(self):
        self.calls.append("idle")

    def pop_timing(self):
        return 0.0, 0


def _make_trainer(workers):
    from types import SimpleNamespace
    import trainer as T
    cfg = SimpleNamespace(seed=0, device="cpu")
    tr = T.Trainer.from_workers(workers, cfg, None, None)
    tr._t_step = 0.0
    tr._t_getsteps = 0.0
    return tr


def test_step_phase_single_no_pool_no_idle():
    w = _FakeWorker("a")
    tr = _make_trainer([w])
    results = tr._run_step_phase()
    assert results == [("ds_a", "ts_a")]
    assert w.calls == ["step", "get_steps"]
    assert tr._pool is None


def test_step_phase_threaded_order_and_calls():
    workers = [_FakeWorker("a"), _FakeWorker("b"), _FakeWorker("c")]
    tr = _make_trainer(workers)
    results = tr._run_step_phase()
    assert results == [("ds_a", "ts_a"), ("ds_b", "ts_b"), ("ds_c", "ts_c")]
    for w in workers:
        assert w.calls == ["step", "get_steps", "idle"]
    tr._pool.shutdown()


def test_step_phase_runs_concurrently():
    workers = [_FakeWorker(str(i), delay=0.1) for i in range(4)]
    tr = _make_trainer(workers)
    t = _time.time()
    tr._run_step_phase()
    elapsed = _time.time() - t
    tr._pool.shutdown()
    assert elapsed < 0.3




def test_act_batch_deterministic():
    from policy import GaussianPolicy, squash_action
    torch.manual_seed(0)
    policy = GaussianPolicy(Actor(), -0.5)
    policy.eval()
    n = 8

    def fresh():
        z = torch.randn(n, Z, generator=torch.Generator().manual_seed(1))
        seed = torch.randn(n, SEED_SIZE, generator=torch.Generator().manual_seed(2))
        msg = torch.randn(n, MESSAGE_SIZE, generator=torch.Generator().manual_seed(3))
        sender = torch.full((n,), 100.0)
        rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE)
        valid = torch.zeros(n, DB_CAPACITY, dtype=torch.bool)
        return z, seed, msg, sender, rows, valid

    with torch.no_grad():
        a1, e1, lp1, _, _ = policy.act_batch(*fresh(), deterministic=True)
        a2, e2, lp2, _, _ = policy.act_batch(*fresh(), deterministic=True)
        z, seed, msg, sender, rows, valid = fresh()
        mean, _, _ = actor_forward_batch(policy.actor, z, seed, msg, sender, rows, valid)

    assert torch.allclose(e1, e2)
    assert torch.allclose(a1, mean)
    # deterministic executed action is the squashed mean, and it is in range
    assert torch.allclose(e1, squash_action(mean))
    assert e1[:, :MESSAGE_SIZE].abs().max() <= 1.0 + 1e-6
    assert e1[:, MESSAGE_SIZE:].min() >= -1e-6 and e1[:, MESSAGE_SIZE:].max() <= 1.0 + 1e-6
    assert torch.allclose(lp1, torch.zeros(n))


def test_check_startup_coverage():
    from trainer import check_startup_coverage
    raised = False
    try:
        check_startup_coverage({"rollout/mean_coverage": 0.01}, 0.05)
    except RuntimeError:
        raised = True
    assert raised
    check_startup_coverage({"rollout/mean_coverage": 0.01}, 0.0)
    check_startup_coverage({"rollout/mean_coverage": 0.5}, 0.05)


def test_log_std_clamp_bounds_runaway():
    from policy import GaussianPolicy, LOG_STD_MIN, LOG_STD_MAX
    p = GaussianPolicy(Actor(), log_std_init=0.0)
    with torch.no_grad():
        p.log_std.fill_(10.0)  # simulate the entropy runaway
    assert float(p._std().max()) <= float(torch.tensor(float(LOG_STD_MAX)).exp()) + 1e-6
    p.clamp_log_std()
    assert float(p.log_std.max()) <= LOG_STD_MAX + 1e-6
    with torch.no_grad():
        p.log_std.fill_(-10.0)
    p.clamp_log_std()
    assert float(p.log_std.min()) >= LOG_STD_MIN - 1e-6


def test_off_shape_penalty_is_bounded():
    cfg = Config()
    node = torch.zeros(2, NODE_FEATURES)
    node[0, DIST_COL] = 1.0 + cfg.tau_v + cfg.l_scale   # well beyond saturation
    node[1, DIST_COL] = 100.0                            # absurdly far
    node[:, NEAREST_COL] = 1.0
    r = compute_rewards(node, cfg)
    floor = -cfg.k_pos * cfg.dt_fixed
    assert abs(float(r[0]) - floor) < 1e-6
    assert abs(float(r[1]) - floor) < 1e-6   # no further than the floor


def test_packing_rewards_density_on_shape():
    cfg = Config()
    node = torch.zeros(2, NODE_FEATURES)
    node[0, DIST_COL] = 0.0
    node[0, NEAREST_COL] = cfg.tau_sep   # on-shape, densely packed
    node[1, DIST_COL] = 0.0
    node[1, NEAREST_COL] = 1.0           # on-shape, isolated
    r = compute_rewards(node, cfg)
    assert float(r[0]) > float(r[1])     # packing pays for density
    off = torch.zeros(1, NODE_FEATURES)
    off[0, DIST_COL] = 0.9
    off[0, NEAREST_COL] = 1.0
    assert float(r[1]) > float(compute_rewards(off, cfg)[0])  # on-shape still beats off


def test_privileged_packing_requires_on_shape_neighbor():
    cfg = Config()
    node = torch.zeros(4, NODE_FEATURES)
    node[0, 0] = 0.0;          node[0, 1] = 0.0
    node[1, 0] = cfg.tau_sep;  node[1, 1] = 0.0
    node[2, 0] = 0.5;          node[2, 1] = 0.5
    node[3, 0] = 0.55;         node[3, 1] = 0.5
    node[:, DIST_COL] = torch.tensor([0.0, 0.0, 0.0, 0.9])  # 0,1,2 on-shape; 3 off
    node[:, NEAREST_COL] = 1.0
    edge_index = torch.tensor([[0, 2], [1, 3]])             # edges 0-1 and 2-3
    r = compute_rewards(node, cfg, edge_index)
    base = cfg.r_on * cfg.dt_fixed
    assert float(r[0]) > base + 1e-6          # on-shape with on-shape neighbor: bonus
    assert abs(float(r[2]) - base) < 1e-6     # on-shape, neighbor off-shape: base only
    assert float(r[3]) < 0.0                  # off-shape: penalty
    # fallback without edges still works (plain nearest-neighbor spacing)
    r2 = compute_rewards(node, cfg)
    assert r2.shape[0] == 4


def test_squashed_gaussian_jacobian():
    # The Jacobian correction must equal sum_dims log|d action / d u| (finite-diff check),
    # so the policy log-prob is the true density of the executed (squashed) action.
    # Done in float64: float32 central differences cannot resolve the large log-dets where
    # tanh is near saturation and the derivative is tiny.
    from policy import squash_action, squash_log_det
    torch.manual_seed(0)
    n, D = 16, MESSAGE_SIZE + 2
    u = torch.randn(n, D, dtype=torch.float64)
    eps = 1e-4
    logdet_fd = torch.zeros(n, dtype=torch.float64)
    for j in range(D):
        up = u.clone(); up[:, j] += eps
        um = u.clone(); um[:, j] -= eps
        deriv = (squash_action(up)[:, j] - squash_action(um)[:, j]) / (2 * eps)
        logdet_fd = logdet_fd + deriv.abs().clamp_min(1e-12).log()
    assert torch.allclose(squash_log_det(u), logdet_fd, atol=1e-4)


def test_squashed_gaussian_in_range_and_logprob_consistent():
    # Squash always lands in range, and re-evaluating a stored sample reproduces the
    # collection log-prob exactly (PPO ratio 1.0 on the first eval).
    from policy import GaussianPolicy, squash_action
    torch.manual_seed(0)
    policy = GaussianPolicy(Actor(), -0.5)
    n = 12
    z = torch.randn(n, Z); seed = torch.randn(n, SEED_SIZE)
    msg = torch.randn(n, MESSAGE_SIZE); sender = torch.full((n,), 100.0)
    rows = torch.zeros(n, DB_CAPACITY, DB_ROW_SIZE)
    valid = torch.zeros(n, DB_CAPACITY, dtype=torch.bool)
    torch.manual_seed(1)
    u, env_action, lp, r2, v2 = policy.act_batch(z, seed, msg, sender, rows, valid, deterministic=False)
    # env_action is in range and equals squash(stored u)
    assert torch.allclose(env_action, squash_action(u), atol=1e-6)
    assert env_action[:, :MESSAGE_SIZE].abs().max() <= 1.0 + 1e-6
    assert env_action[:, MESSAGE_SIZE:].min() >= -1e-6
    assert env_action[:, MESSAGE_SIZE:].max() <= 1.0 + 1e-6
    # replay reproduces the log-prob
    lp2, _ = policy.evaluate_batch(z, seed, msg, sender, rows, valid, u)
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_gru_actor_replay_consistent():
    # The GRU actor's act/evaluate must be consistent: executed action equals squash of
    # the stored sample, actions are in range, and replay reproduces the log-prob (PPO
    # ratio 1.0 on the first eval), exactly as required for the recurrent PPO path.
    from policy import GaussianPolicy, squash_action
    from kilobot_gnn import RecurrentActor, GRU_HIDDEN
    from kinematics import PROP_SIZE
    torch.manual_seed(0)
    actor = RecurrentActor(extra=2)          # dir_heading appends 2 cols to the seed slot
    policy = GaussianPolicy(actor, -0.5)
    n = 6
    z = torch.randn(n, Z)
    seed = torch.randn(n, SEED_SIZE + 2)
    msg = torch.randn(n, MESSAGE_SIZE)
    prop = torch.randn(n, PROP_SIZE)
    h = torch.zeros(n, GRU_HIDDEN)
    torch.manual_seed(1)
    u, env_action, lp, h_new = policy.act_batch_gru(z, seed, msg, prop, h, deterministic=False)
    assert env_action.shape == (n, MESSAGE_SIZE + MOTOR_SIZE)
    assert h_new.shape == (n, GRU_HIDDEN)
    assert torch.allclose(env_action, squash_action(u), atol=1e-6)
    assert env_action[:, MESSAGE_SIZE:].min() >= -1e-6
    assert env_action[:, MESSAGE_SIZE:].max() <= 1.0 + 1e-6
    lp2, _ = policy.evaluate_batch_gru(z, seed, msg, prop, h, u)
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_gru_hidden_state_evolves():
    # The hidden state must actually change tick to tick and carry information.
    from policy import GaussianPolicy
    from kilobot_gnn import RecurrentActor, GRU_HIDDEN
    from kinematics import PROP_SIZE
    torch.manual_seed(0)
    policy = GaussianPolicy(RecurrentActor(extra=0), -0.5)
    n = 4
    z = torch.randn(n, Z); seed = torch.randn(n, SEED_SIZE); msg = torch.randn(n, MESSAGE_SIZE)
    prop = torch.randn(n, PROP_SIZE)
    h0 = torch.zeros(n, GRU_HIDDEN)
    _, _, _, h1 = policy.act_batch_gru(z, seed, msg, prop, h0, deterministic=True)
    _, _, _, h2 = policy.act_batch_gru(z, seed, msg, prop, h1, deterministic=True)
    assert not torch.allclose(h0, h1)
    assert not torch.allclose(h1, h2)


def test_dead_reckon_carries_time_and_distance():
    # The proprioception must expose elapsed TIME and cumulative DISTANCE as their own
    # channels, not fold them into displacement. In particular a stationary robot moves
    # zero distance but different amounts of time must remain distinguishable.
    from kinematics import dead_reckon, PROP_SIZE
    ms, wb, dt, sc, ts, cs = 0.02, 0.10, 0.05, 50.0, 2.0, 5.0

    # stationary (no wheels), 1 step vs 50 steps
    stay = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    p = dead_reckon(stay, torch.tensor([1.0, 50.0]), torch.tensor([0.0, 0.0]), ms, wb, dt, sc, ts, cs)
    assert p.shape == (2, PROP_SIZE)
    assert torch.allclose(p[:, 0:2], torch.zeros(2, 2))      # no distance either way
    assert p[1, 4] > p[0, 4] + 1e-6                          # but time channel distinguishes them

    # cumulative-distance channel reflects the running odometer
    drive = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    p2 = dead_reckon(drive, torch.tensor([10.0, 10.0]), torch.tensor([0.0, 0.5]), ms, wb, dt, sc, ts, cs)
    assert p2[1, 5] > p2[0, 5] + 1e-6
    assert torch.allclose(p2[:, 0], p2[:, 1], atol=1e-5)     # straight: path == euclid

    # turning: arc chord (euclid) is shorter than path length
    turn = torch.tensor([[0.9, 0.1]])
    p3 = dead_reckon(turn, torch.tensor([20.0]), torch.tensor([0.0]), ms, wb, dt, sc, ts, cs)
    assert p3[0, 1] < p3[0, 0]
    assert p3[0, 2].abs() > 1e-3                             # heading actually changed


def test_calibrated_kinematics_are_the_defaults():
    # KilobotMovement.cs does not read these, so nothing else keeps a plain
    # Config() in sync with the simulator. They are DERIVED from Unity's own
    # constants rather than measured, so this asserts the derivation rather
    # than magic numbers -- if someone changes moveSpeed, turnSpeed,
    # framesPerStep or the Fixed Timestep in Unity without mirroring it in
    # config.py, this fails.
    import math
    from config import (_UNITY_MOVE_SPEED, _UNITY_TURN_SPEED_DEG,
                        _UNITY_FRAMES_PER_STEP, _UNITY_FIXED_DT)
    cfg = Config()
    assert cfg.dt_fixed == 0.02
    per_step_disp = _UNITY_MOVE_SPEED * _UNITY_FIXED_DT * _UNITY_FRAMES_PER_STEP
    per_step_turn = math.radians(_UNITY_TURN_SPEED_DEG) * _UNITY_FIXED_DT * _UNITY_FRAMES_PER_STEP
    assert cfg.prop_max_speed == pytest.approx(per_step_disp / cfg.dt_fixed)
    assert cfg.prop_max_speed == pytest.approx(4.0)
    assert cfg.prop_max_speed / cfg.prop_wheelbase == pytest.approx(per_step_turn / cfg.dt_fixed)
    assert cfg.prop_max_speed / cfg.prop_wheelbase == pytest.approx(math.pi)
    assert cfg.prop_wheelbase == pytest.approx(4.0 / math.pi)


def test_dead_reckoning_reproduces_unity_motion_exactly():
    # The simulator rotates and THEN translates once per FixedUpdate, so an
    # interval is a polygon of steps*framesPerStep equal segments, not a smooth
    # arc. split_tick_motion models that exactly; this pins it against a direct
    # replay of KilobotMovement.FixedUpdate.
    #
    # The arc approximation this replaced put the travel direction at
    # dtheta/2 instead of (M+1)/(2M)*dtheta -- 0.45 degrees per step at full
    # spin, measured against a live player, which walks the belief filter's
    # position sideways while its magnitude and heading stay exact.
    # tools/check_dead_reckoning.py is the live-player version of this.
    import math
    from kinematics import split_tick_motion, FRAMES_PER_STEP
    cfg = Config()

    def unity(L, R, steps):
        x = y = h = 0.0
        for _ in range(steps * FRAMES_PER_STEP):
            h += -math.radians((L - R) * 45.0 * 0.02)
            s = (L + R) * 0.5 * 1.0 * 0.02
            x += s * math.cos(h)
            y += s * math.sin(h)
        return math.hypot(x, y), math.atan2(y, x), h

    for L, R in [(1.0, 0.0), (0.9, 0.15), (0.75, 0.25), (1.0, 1.0), (0.0, 1.0)]:
        for steps in (1, 4, 12, 48):
            ud, ua, uh = unity(L, R, steps)
            x, y, dth, _t = split_tick_motion(
                torch.tensor([[L, R]]), torch.tensor([float(steps)]),
                cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed)
            assert float(torch.hypot(x, y)) == pytest.approx(ud, rel = 1e-5, abs = 1e-9)
            assert float(dth) == pytest.approx(uh, rel = 1e-5, abs = 1e-9)
            if ud > 1e-9:
                assert float(torch.atan2(y, x)) == pytest.approx(ua, rel = 1e-5, abs = 1e-7)


def test_split_tracker_scale_matches_calibrated_speed():
    # A full-throttle interval, from one tick up through several heartbeat windows
    # (an isolated robot going a long time between events of one kind), must stay
    # in a range an untrained network can actually use. Guards split_prop_scale and
    # split_prop_time_scale against drifting out of sync with prop_max_speed again.
    from kinematics import split_tick_motion, split_track_update, split_track_read
    cfg = Config()
    for k in (1.0, 48.0, 400.0):
        motor = torch.tensor([[1.0, 1.0]])
        steps = torch.tensor([k])
        x, y, dth, t = split_tick_motion(motor, steps, cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed)
        track = split_track_update(torch.zeros(1, 4), x, y, dth, t)
        read = split_track_read(track, cfg.split_prop_scale, cfg.split_prop_time_scale)
        assert float(read[0, 0].abs()) < 5.0
        assert float(read[0, 3].abs()) < 5.0


def test_gru_prop_scale_matches_calibrated_speed():
    # The gru actor decides almost every tick in a dense swarm (eligibility fires
    # on any neighbor message), so its typical interval is one tick, unlike the
    # split actor's anchored-since-event trackers; prop_scale/prop_time_scale
    # should land THAT near unit range.
    from kinematics import dead_reckon
    cfg = Config()
    motor = torch.tensor([[1.0, 1.0]])
    steps = torch.tensor([1.0])
    p = dead_reckon(motor, steps, torch.tensor([0.0]), cfg.prop_max_speed, cfg.prop_wheelbase,
                    cfg.dt_fixed, cfg.prop_scale, cfg.prop_time_scale, cfg.prop_cum_scale)
    assert float(p[0, 0].abs()) < 3.0
    assert float(p[0, 1].abs()) < 3.0
    assert float(p[0, 4].abs()) < 3.0
