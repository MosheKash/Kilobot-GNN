import torch
from types import SimpleNamespace


def test_worker_cfg_forces_cpu_without_mutating_the_original():
    from config import Config
    import parallel

    cfg = Config()
    cfg.device = "cuda"
    worker_cfg = parallel._worker_cfg(cfg)

    assert worker_cfg.device == "cpu"
    # the learner's own config, and any other worker's, must be unaffected
    assert cfg.device == "cuda"
    assert worker_cfg is not cfg


def _mk_step(traj_ids):
    n = len(traj_ids)
    return {"arena_id": 0, "env_step": 0,
            "node": torch.zeros(n, 19),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_attr": torch.zeros(0, 1),
            "z": torch.zeros(64),
            "traj_id": torch.tensor(traj_ids),
            "reward": torch.zeros(n), "term": torch.zeros(n),
            "cut": torch.ones(n), "is_decision": torch.zeros(n, dtype=torch.bool)}


def test_merge_buffers_step_index_and_traj_offset():
    from parallel import merge_buffers
    cfg = SimpleNamespace()
    w0_steps = [_mk_step([1]), _mk_step([2])]
    w0_decs = [{"step_index": 0, "local": 0}, {"step_index": 1, "local": 0}]
    w1_steps = [_mk_step([1])]
    w1_decs = [{"step_index": 0, "local": 0}]
    merged = merge_buffers([(w0_steps, w0_decs), (w1_steps, w1_decs)], cfg)
    assert len(merged.steps) == 3
    assert len(merged.decisions) == 3
    assert merged.decisions[2]["step_index"] == 2
    assert int(merged.steps[0]["traj_id"][0]) == 1
    assert int(merged.steps[1]["traj_id"][0]) == 2
    assert int(merged.steps[2]["traj_id"][0]) == 3
    allids = [int(s["traj_id"][j]) for s in merged.steps for j in range(s["traj_id"].numel())]
    assert len(allids) == len(set(allids))


def test_merge_does_not_mutate_inputs():
    from parallel import merge_buffers
    cfg = SimpleNamespace()
    steps = [_mk_step([5])]
    decs = [{"step_index": 0, "local": 0}]
    merge_buffers([(steps, decs), (steps, decs)], cfg)
    assert int(steps[0]["traj_id"][0]) == 5
    assert decs[0]["step_index"] == 0


def test_aggregate_payloads_and_stats():
    from metrics import aggregate_payloads, rollout_stats
    p1 = {"reward_sum": 2.0, "reward_count": 4, "cov_sum": 0.4, "cov_count": 2,
          "decisions": 3, "agent_steps": 10,
          "ep_records": [{"reward": 1.0, "length": 5, "success": True, "coverage": 0.3}]}
    p2 = {"reward_sum": 1.0, "reward_count": 1, "cov_sum": 0.2, "cov_count": 1,
          "decisions": 1, "agent_steps": 5,
          "ep_records": [{"reward": -1.0, "length": 3, "success": False, "coverage": 0.1}]}
    agg = aggregate_payloads([p1, p2])
    assert agg["reward_count"] == 5
    assert abs(agg["reward_sum"] - 3.0) < 1e-9
    assert len(agg["ep_records"]) == 2
    st = rollout_stats(agg)
    assert abs(st["rollout/mean_step_reward"] - 0.6) < 1e-9
    assert abs(st["episodes/success_rate"] - 0.5) < 1e-9


def _fake_worker_entry(worker_id, cfg, in_q, out_q):
    import torch
    import pickle
    from parallel import pack_buffer
    from buffer import RolloutBuffer
    out_q.put(("READY", worker_id))
    while True:
        msg = in_q.get()
        if msg == "STOP":
            break
        buf = RolloutBuffer(cfg)
        buf.steps.append({"arena_id": 0, "env_step": 0,
                          "node": torch.zeros(1, 19), "edge_index": torch.zeros(2, 0, dtype=torch.long),
                          "edge_attr": torch.zeros(0, 1), "z": torch.zeros(64),
                          "traj_id": torch.tensor([1]), "reward": torch.zeros(1),
                          "term": torch.zeros(1), "cut": torch.ones(1),
                          "is_decision": torch.zeros(1, dtype=torch.bool)})
        payload = {"reward_sum": 1.0, "reward_count": 1, "cov_sum": 0.2, "cov_count": 1,
                   "decisions": 0, "agent_steps": 1, "ep_records": []}
        timing = {"step": 0.1, "parse": 0.0, "getsteps": 0.0, "snap": 0.0, "act": 0.0, "msgs": 0}
        out_q.put(("DATA", pickle.dumps((worker_id, pack_buffer(buf), payload, timing))))


def test_parallel_orchestration_fake_workers():
    import parallel
    captured = {}

    def fake_ppo(policy, critic, actor_opt, critic_opt, buffer, cfg):
        captured["steps"] = len(buffer.steps)
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "clip_fraction": 0.0, "explained_variance": 0.0, "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0, "log_std_mean": 0.0, "std_mean": 0.0,
                "adv_mean": 0.0, "adv_std": 0.0, "mean_return": 0.0, "return_std": 0.0,
                "mean_value": 0.0, "motor_grad_norm": 0.0, "msg_grad_norm": 0.0,
                "motor_saturation": 0.0, "motor_preact_absmean": 0.0,
                "motor_out_mean": 0.0, "motor_preact_mean": 0.0, "motor_param_drift": 0.0, "msg_param_drift": 0.0, "log_std_motor": 0.0}

    orig = parallel.ppo_update
    parallel.ppo_update = fake_ppo
    try:
        cfg = SimpleNamespace(rollout_steps=4)
        policy = torch.nn.Linear(2, 2)
        pt = parallel.ParallelTrainer(cfg, 3, worker_entry=_fake_worker_entry)
        pt.run(policy, None, None, None, iterations=2)
        assert captured["steps"] == 3
    finally:
        parallel.ppo_update = orig


def test_shutdown_join_timeout_derived_from_collect_max_wait():
    # regression test: the shutdown path's p.join(timeout=...) previously used
    # a hardcoded 30s, wildly inconsistent with this same class's own
    # collect_max_wait (1200s default) for how long a single rollout can
    # legitimately take -- a worker still genuinely mid-rollout when the
    # final STOP was sent would get force-terminated before its own
    # env.close() (worker_loop's finally block) ever ran, orphaning the
    # Unity subprocess underneath it. Wraps join to record the timeout
    # actually passed, without waiting for it -- the fake worker exits
    # promptly on STOP regardless of what timeout value is given.
    import parallel

    def fake_ppo(policy, critic, actor_opt, critic_opt, buffer, cfg):
        return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "clip_fraction": 0.0, "explained_variance": 0.0, "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0, "log_std_mean": 0.0, "std_mean": 0.0,
                "adv_mean": 0.0, "adv_std": 0.0, "mean_return": 0.0, "return_std": 0.0,
                "mean_value": 0.0, "motor_grad_norm": 0.0, "msg_grad_norm": 0.0,
                "motor_saturation": 0.0, "motor_preact_absmean": 0.0,
                "motor_out_mean": 0.0, "motor_preact_mean": 0.0, "motor_param_drift": 0.0, "msg_param_drift": 0.0, "log_std_motor": 0.0}

    orig = parallel.ppo_update
    parallel.ppo_update = fake_ppo
    try:
        cfg = SimpleNamespace(rollout_steps=4, collect_max_wait=777.0)
        policy = torch.nn.Linear(2, 2)
        pt = parallel.ParallelTrainer(cfg, 1, worker_entry=_fake_worker_entry)

        recorded = []
        for p in pt.procs:
            orig_join = p.join
            def wrapped(timeout=None, _orig=orig_join):
                recorded.append(timeout)
                return _orig(timeout=timeout)
            p.join = wrapped

        pt.run(policy, None, None, None, iterations=1)
        assert recorded, "join was never called"
        assert all(t == 777.0 + 60.0 for t in recorded)
    finally:
        parallel.ppo_update = orig



    import pickle
    from parallel import pack_buffer, unpack_buffer
    from buffer import RolloutBuffer
    cfg = SimpleNamespace()
    buf = RolloutBuffer(cfg)
    buf.steps.append({"arena_id": 1, "env_step": 5, "node": torch.randn(2, 19),
                      "edge_index": torch.tensor([[0, 1, 0], [1, 0, 1]]), "edge_attr": torch.randn(3, 1),
                      "z": torch.randn(64), "traj_id": torch.tensor([7, 8]), "reward": torch.randn(2),
                      "term": torch.zeros(2), "cut": torch.tensor([0.0, 1.0]),
                      "is_decision": torch.tensor([True, False])})
    buf.steps.append({"arena_id": 2, "env_step": 6, "node": torch.randn(1, 19),
                      "edge_index": torch.zeros(2, 0, dtype=torch.long), "edge_attr": torch.zeros(0, 1),
                      "z": torch.randn(64), "traj_id": torch.tensor([9]), "reward": torch.randn(1),
                      "term": torch.zeros(1), "cut": torch.ones(1), "is_decision": torch.tensor([True])})
    buf.add_decision(0, 0, torch.randn(64), torch.randn(5), torch.randn(11),
                     torch.randn(2, 11), torch.randn(11), torch.tensor(0.5))

    packed = pickle.loads(pickle.dumps(pack_buffer(buf)))
    out = unpack_buffer(packed, cfg)

    assert len(out.steps) == 2
    assert torch.allclose(out.steps[0]["node"], buf.steps[0]["node"])
    assert torch.equal(out.steps[0]["edge_index"], buf.steps[0]["edge_index"])
    assert torch.equal(out.steps[0]["traj_id"], buf.steps[0]["traj_id"])
    assert torch.allclose(out.steps[0]["cut"], buf.steps[0]["cut"])
    assert out.steps[1]["edge_index"].shape == (2, 0)
    assert out.steps[0]["arena_id"] == 1 and out.steps[0]["env_step"] == 5
    assert len(out.decisions) == 1
    assert torch.allclose(out.decisions[0]["prev_db"], buf.decisions[0]["prev_db"])
    assert torch.allclose(out.decisions[0]["old_log_prob"], buf.decisions[0]["old_log_prob"])


def _dying_worker_entry(worker_id, cfg, in_q, out_q):
    import sys
    out_q.put(("READY", worker_id))
    msg = in_q.get()
    if msg == "STOP":
        return
    sys.exit(1)


def test_worker_death_aborts_run():
    import parallel
    cfg = SimpleNamespace(rollout_steps=4)
    policy = torch.nn.Linear(2, 2)
    pt = parallel.ParallelTrainer(cfg, 1, worker_entry=_dying_worker_entry)
    pt.get_timeout = 0.5
    raised = False
    try:
        pt.run(policy, None, None, None, iterations=1)
    except RuntimeError as e:
        raised = "died" in str(e)
    assert raised
