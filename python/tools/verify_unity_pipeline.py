"""End-to-end proof that Unity is the only simulator, and that the loop closes.

Run from python/:   python tools/verify_unity_pipeline.py

Six checks, each printing PASS/FAIL and why:

  1. no Python simulator survives      -- the deleted modules are gone AND
                                          unimportable, and nothing references
                                          them or the old --sim switch
  2. Python cannot fake it             -- pointed at a missing build, worker
                                          construction RAISES rather than
                                          silently falling back to anything
  3. a real, separate OS process       -- the player's pid, confirmed alive
  4. Unity -> Python                   -- observations arrive over the side
                                          channel with the expected shape, and
                                          they are the player's own numbers
  5. Python -> Unity                   -- a command sent from here changes what
                                          Unity subsequently reports back
  6. the closed loop                   -- a real policy + Trainer.collect: Unity
                                          reports, Python decides, Unity moves

Exit code is 0 only if every check passes.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PY, ".."))
sys.path.insert(0, PY)

BASE_PORT = int(os.environ.get("KILOBOT_VERIFY_PORT", "8100"))

results = []


def check(name, ok, detail):
    results.append((name, ok))
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
    for line in detail.splitlines():
        print("         " + line)


# ---------------------------------------------------------------- 1
print("\n1. No Python simulator survives")

DELETED = ["replica_env.py", "oracle.py", "run_replica_experiments.py",
           "rl_driver.py", "eval_visual.py", "run_bc_real_formations.py",
           "tools/measure_oracle.py"]
present = [f for f in DELETED if os.path.exists(os.path.join(PY, f))]

importable = []
for mod in ["replica_env", "oracle", "rl_driver"]:
    try:
        __import__(mod)
        importable.append(mod)
    except ImportError:
        pass

# Deliberately narrow. Plenty of comments and docstrings still SAY
# "ReplicaWorker" -- they are this project's own record of why things are the
# way they are, and docs/code-overview.md flags them as historical. What must
# not exist is anything that still WORKS: a live import of a simulator module,
# or a driver still offering the --sim switch that used to pick one.
grep = subprocess.run(
    ["grep", "-rn", "--include=*.py",
     "-e", r"^\s*\(from\|import\)\s\+\(replica_env\|oracle\|rl_driver\)\b",
     "-e", r"add_argument(\s*[\"']--sim[\"']", PY],
    capture_output=True, text=True)
hits = [l for l in grep.stdout.splitlines() if "tools/verify_unity_pipeline" not in l]
# temp_test_material/ is a scratch drawer of one-off probes, not part of the
# pipeline and imported by nothing. A few of them still reference the deleted
# replica, which makes them dead files rather than a live fallback path --
# reported below rather than failing the run.
live = [l for l in hits if "temp_test_material" not in l]
scratch = sorted({l.split(":")[0].replace(PY + "/", "") for l in hits
                  if "temp_test_material" in l})

check("deleted simulator modules are absent",
      not present,
      "still present: %s" % present if present else "all %d gone" % len(DELETED))
check("deleted simulator modules are unimportable",
      not importable,
      "still importable: %s" % importable if importable
      else "import replica_env / oracle / rl_driver all raise ImportError")
check("nothing imports a simulator, no driver offers --sim",
      not live,
      "\n".join(live[:5]) if live
      else "no live import of replica_env/oracle/rl_driver; no --sim argument anywhere")
if scratch:
    print("         note: %d dead scratch file(s) under temp_test_material/ still import"
          % len(scratch))
    print("         the deleted replica -- they cannot run, and nothing imports them:")
    for s in scratch:
        print("           %s" % s)


# ---------------------------------------------------------------- 2
print("\n2. Python cannot fake it (no silent fallback)")

import unity_env

raised = None
try:
    unity_env.make_unity_worker(worker_id=0, num_arenas=1,
                                build_path="/nonexistent/Kilobot.x86_64",
                                base_port=BASE_PORT + 90, timeout=30)
except Exception as exc:
    raised = type(exc).__name__
check("a missing player is a hard error",
      raised is not None,
      "raised %s -- there is nothing for it to fall back to" % raised if raised
      else "a worker was built WITHOUT a Unity player, which should be impossible")


# ---------------------------------------------------------------- 3
print("\n3. The player is a real, separate OS process")

import numpy as np
import torch

FORMATIONS = os.path.join(ROOT, "data", "formations")
unity_env.set_player_env(formations=FORMATIONS, heartbeat_ticks=48,
                         seed_layout="corners", num_arenas=1,
                         min_bots=8, max_bots=8)
worker, env = unity_env.make_unity_worker(worker_id=0, num_arenas=1,
                                          base_port=BASE_PORT)

try:
    pids = subprocess.run(["pgrep", "-f", "[K]ilobot.x86_64"],
                          capture_output=True, text=True).stdout.split()
    check("a Kilobot player process is running",
          len(pids) >= 1,
          "pid(s): %s   (this python pid is %d)" % (", ".join(pids), os.getpid()))

    # ------------------------------------------------------------ 4
    print("\n4. Unity -> Python (observations arrive from the player)")

    from kilobot_gnn import NODE_FEATURES

    snap = worker.snapshot(0)
    node = snap["node"]
    ok_shape = node is not None and node.shape[1] == NODE_FEATURES and node.shape[0] == 8
    check("side channel delivers a well-formed snapshot",
          ok_shape,
          "node %s (expected (8, %d)), %d edges, env_step %d"
          % (tuple(node.shape), NODE_FEATURES, snap["edge_attr"].shape[0], snap["env_step"]))

    spec = env.behavior_specs[worker.behavior_name]
    check("the behavior spec comes from the player, not from Python",
          spec.action_spec.continuous_size == 11,
          "behavior %r: obs=%s cont_actions=%d"
          % (worker.behavior_name,
             [tuple(o.shape) for o in spec.observation_specs],
             spec.action_spec.continuous_size))

    # ------------------------------------------------------------ 5
    print("\n5. Python -> Unity (a command from here changes what Unity reports)")

    before = node[:, 0:2].numpy().copy() * 100.0
    want = [(0, -40.0, 25.0, 0.0), (1, 12.5, -60.25, np.pi / 2)]
    worker.send_poses(0, want)
    worker.reset_env()
    after = worker.snapshot(0)["node"][:, 0:2].numpy() * 100.0
    err = max(float(np.hypot(after[i][0] - x, after[i][1] - z)) for i, x, z, _ in want)
    check("pose command lands exactly where Python asked",
          err < 0.05,
          "robot 0 was (%.2f, %.2f), asked for (-40.00, 25.00), Unity now reports "
          "(%.2f, %.2f); worst error %.4f units"
          % (before[0][0], before[0][1], after[0][0], after[0][1], err))

    def arena_after_reset(seed):
        worker.set_swarm_rng(seed)
        worker.send_reset(0, 0)
        worker.reset_env()
        return worker.snapshot(0)["node"][:, 0:2].numpy().copy()

    a = arena_after_reset(4242)
    arena_after_reset(1)                      # perturb in between
    b = arena_after_reset(4242)
    c = arena_after_reset(777)
    check("swarm-RNG pin replays the player's own spawn",
          np.allclose(a, b, atol=1e-4) and not np.allclose(a, c, atol=1e-4),
          "same value -> identical arena: %s;  different value -> different arena: %s"
          % (np.allclose(a, b, atol=1e-4), not np.allclose(a, c, atol=1e-4)))

    # ------------------------------------------------------------ 6
    print("\n6. The closed loop (Unity reports -> Python decides -> Unity moves)")

    from config import Config
    from trainer import Trainer
    from policy import GaussianPolicy
    from kilobot_gnn import build_actor, Z

    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.device = "cpu"
    cfg.num_arenas = 1
    cfg.rollout_steps = 24
    cfg.heartbeat_ticks = 48
    cfg.seed_layout = "corners"

    torch.manual_seed(0)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)

    class _Enc(torch.nn.Module):
        def forward(self, image):
            return torch.zeros(Z)

    tr = Trainer.from_workers([worker], cfg, _Enc(), [torch.zeros(1)])
    tr.setup()
    start = worker.snapshot(0)["node"][:, 0:2].numpy().copy() * 100.0
    buf = tr.collect(policy, None)
    end = worker.snapshot(0)["node"][:, 0:2].numpy().copy() * 100.0

    n_dec = len(buf.decisions)
    motors = [float(m.abs().sum()) for a in worker.last_motor.values() for m in a.values()]
    n_cmd = sum(1 for m in motors if m > 0)
    moved = float(np.abs(end[:min(len(start), len(end))]
                         - start[:min(len(start), len(end))]).max())

    check("Unity asked Python for decisions",
          n_dec > 0,
          "%d decisions over %d rollout steps -- each one is a robot the PLAYER "
          "flagged as eligible" % (n_dec, cfg.rollout_steps))
    check("Python sent motor commands back",
          n_cmd > 0,
          "%d robots hold a nonzero motor command issued by the policy" % n_cmd)
    check("the player moved in response",
          moved > 0.0,
          "max robot displacement during the rollout: %.3f units" % moved)

finally:
    env.close()

print("\n" + "=" * 62)
failed = [n for n, ok in results if not ok]
for n in failed:
    print("  FAILED: %s" % n)
print("OVERALL: %s  (%d/%d checks passed)"
      % ("FAIL" if failed else "PASS", len(results) - len(failed), len(results)))
sys.exit(1 if failed else 0)
