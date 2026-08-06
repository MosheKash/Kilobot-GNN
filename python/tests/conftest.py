import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Port range for test-owned Unity players. Deliberately far from the training
# defaults (5005) so a test run cannot collide with a training run in progress.
TEST_BASE_PORT = int(os.environ.get("KILOBOT_TEST_BASE_PORT", "7100"))


def _build_available():
    from unity_env import DEFAULT_BUILD_PATH
    return os.path.exists(os.environ.get("KILOBOT_BUILD_PATH", DEFAULT_BUILD_PATH))


NO_BUILD_REASON = (
    "no Unity player found -- build it with:\n"
    "    <unity-editor> -batchmode -quit -nographics -projectPath . "
    "-executeMethod BuildPlayer.BuildLinux -logFile -\n"
    "and generate the formation pool with data-prep/quickdraw_to_png.py. "
    "See the first-time setup in README.md.")

requires_unity = pytest.mark.skipif(not _build_available(), reason = NO_BUILD_REASON)


def skip_without_build():
    """Skip, rather than fail, when there is no player to talk to.

    Most Unity-backed tests reach the factory through helpers rather than the
    requires_unity marker, so without this a fresh clone reports a wall of
    failures where the real answer is "you have not built the player yet".
    """
    if not _build_available():
        pytest.skip(NO_BUILD_REASON)


class UnityWorkerFactory:
    """Hands out Unity-backed workers to tests, reusing players where possible.

    Booting a player costs ~0.7s and a reset ~0.03s, so the expensive thing is
    the process, not the episode. Players are therefore cached for the whole
    session, keyed by every setting fixed at launch: swarm size, heartbeat,
    seed layout, formations (SwarmManager reads those once in Awake) AND arena
    count -- SceneBootstrap reads num_arenas in Awake/Start too, at scene load,
    not on reset. Pushing a new value down the parameters channel and resetting
    does NOT restripe a running player's arenas, so a test asking for a
    different count gets its own process.

    A test that asks for settings some earlier test already used gets that same
    player back, freshly reset. One asking for a new combination pays a boot.
    """

    def __init__(self):
        self._players = {}
        self._next_worker_id = 0

    def get(self, num_arenas = 1, min_bots = 6, max_bots = 6, heartbeat_ticks = 48,
            seed_layout = "corners", formations = None, swarm_rng = None):
        skip_without_build()
        import unity_env
        if formations is None:
            formations = os.path.join(os.path.dirname(__file__), "..", "..", "data", "formations")
        key = (num_arenas, min_bots, max_bots, heartbeat_ticks, seed_layout,
               os.path.abspath(formations))
        if key not in self._players:
            unity_env.set_player_env(formations = formations, heartbeat_ticks = heartbeat_ticks,
                                     seed_layout = seed_layout, num_arenas = num_arenas,
                                     min_bots = min_bots, max_bots = max_bots)
            worker, env = unity_env.make_unity_worker(
                worker_id = self._next_worker_id, num_arenas = num_arenas,
                base_port = TEST_BASE_PORT, no_graphics = True,
                # its own subdir: ml-agents names the file from the worker id
                # alone, so a test run would otherwise overwrite a training
                # run's Player-0.log
                log_subdir = "tests")
            self._next_worker_id += 1
            self._players[key] = (worker, env)
        worker, env = self._players[key]
        # Always pushed, never left to carry over: the pin lives in the player's
        # environment parameters, so a test that pinned a seed would otherwise
        # leave every LATER test on this cached player replaying that one arena
        # on every episode reset.
        worker.set_swarm_rng(swarm_rng)
        env.reset()
        if swarm_rng is not None:
            # env.reset() on its own respawns nothing -- SwarmManager rebuilds
            # its swarm only on a KIND_RESET over CriticChannel -- so a pinned
            # seed needs an explicit arena reset before it means anything.
            for k in range(num_arenas):
                worker.send_reset(k, 0)
            env.reset()
        # per-robot state is keyed by (arena, local) and outlives a reset, so a
        # reused player would otherwise hand the next test another test's belief
        # clouds, trackers and hidden state
        for attr in ("z", "image_id", "databases", "hidden", "last_motor", "last_dec_step",
                     "odometer", "track_neighbor", "belief", "track_seed",
                     "pending_find_reward", "step_count", "traj_id", "ep_reward"):
            getattr(worker, attr).clear()
        for attr in ("simple_state", "simple_heading", "simple_target", "simple_motor_state",
                     "simple_turn_accum", "simple_wall_name", "_spawn_heading_cache"):
            if hasattr(worker, attr):
                getattr(worker, attr).clear()
        return worker

    def close(self):
        for _worker, env in self._players.values():
            try:
                env.close()
            except Exception as exc:
                print("warning: failed to close a test Unity env (%s)" % exc)
        self._players.clear()


# A single process-wide factory rather than only a fixture: test helpers
# (module-level _cfg()/_worker() functions that predate any of this) need to
# reach it without every caller threading a fixture argument down to them.
# The fixture below is a thin accessor so tests that prefer declaring it
# explicitly still read naturally.
_FACTORY = None


def players():
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = UnityWorkerFactory()
    return _FACTORY


def pytest_sessionfinish(session, exitstatus):
    # NOT atexit: UnityEnvironment.close() can block indefinitely when it first
    # runs during interpreter shutdown -- the same hang that left a finished BC
    # run sitting for minutes after its checkpoint was already written. pytest
    # calls this hook while the interpreter is still fully alive.
    if _FACTORY is not None:
        _FACTORY.close()


@pytest.fixture(scope = "session")
def unity_players():
    return players()


@pytest.fixture
def unity_worker(unity_players):
    """A freshly-reset Unity worker with a small, exactly-sized swarm.

    Default 6 robots in 1 arena -- small enough to reason about, and only
    possible at all since SwarmManager gained KILOBOT_MIN_BOTS/MAX_BOTS.
    """
    return unity_players.get(num_arenas = 1, min_bots = 6, max_bots = 6)


def unity_cfg(actor_type = "gru_split_observation", rollout = 8, arenas = 1,
              heartbeat_ticks = 48, **overrides):
    """A Config for a Unity-backed test.

    Starts from Config()'s defaults unmodified -- those are the values
    calibrated against the real player -- and overrides only what a test needs.
    """
    from config import Config
    cfg = Config()
    cfg.actor_type = actor_type
    cfg.device = "cpu"
    cfg.num_arenas = arenas
    cfg.rollout_steps = rollout
    cfg.heartbeat_ticks = heartbeat_ticks
    cfg.seed_layout = "corners"
    cfg.minibatch = 512
    cfg.ppo_epochs = 4
    cfg.entropy_coef = 0.0
    cfg.critic_chunk_steps = 64
    cfg.success_threshold = 2.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def place(worker, k, poses):
    """Teleport named robots in arena k to exact poses, and let it take effect.

    poses: an iterable of (local_index, x, z, heading). x/z are arena-local raw
    units (within +/- ARENA_HALF, the units belief.py works in, NOT the [-1, 1]
    pair node[:, 0:2] reports); heading is radians in python's convention,
    direction (cos h, sin h) in (x, z).

    Lets a test guarantee a geometric precondition -- two robots out of IR
    range of each other, a robot exactly on a target -- rather than spawn one
    and hope.

    The reset_env() is what flushes the queued side-channel message; without it
    the poses sit in the outgoing queue until something else happens to step.
    """
    worker.send_poses(k, poses)
    worker.reset_env()


def unity_trainer(cfg, min_bots = 6, max_bots = 6, formations = None,
                  image_pool = None, encoder = None, unity_players = None,
                  swarm_rng = None):
    """A Trainer driving one Unity worker.

    Returns (trainer, worker). The worker comes from the session-scoped factory,
    so it is reused across tests with matching launch settings.

    swarm_rng pins the arena exactly (see UnityWorkerFactory.get); pass one
    when a test needs two separate collects to see the same arena, which is the
    only way to compare rollouts against each other on a real player.
    """
    import torch
    from trainer import Trainer
    unity_players = unity_players or players()
    worker = unity_players.get(num_arenas = cfg.num_arenas, min_bots = min_bots,
                               max_bots = max_bots, heartbeat_ticks = cfg.heartbeat_ticks,
                               seed_layout = cfg.seed_layout, formations = formations,
                               swarm_rng = swarm_rng)
    if encoder is None:
        encoder = _fixed_encoder()
    if image_pool is None:
        image_pool = [torch.zeros(1)]
    tr = Trainer.from_workers([worker], cfg, encoder, image_pool)
    tr.setup()
    return tr, worker


def _fixed_encoder(seed = 0):
    """Deterministic stand-in encoder: one fixed latent, whatever the image."""
    import torch
    from kilobot_gnn import Z

    class FixedEncoder(torch.nn.Module):
        def __init__(self, seed = 0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("latent", torch.randn(Z, generator = g) * 0.3)

        def forward(self, image):
            return self.latent

    return FixedEncoder(seed)


MAX_ROWS = 16   # max neighbour rows in one observation


class FakeSteps:
    """A stand-in for ml-agents' DecisionSteps, for tests that build one by hand.

    Just the (vector, rows) obs pair plus a length, which is all act() reads.
    """

    def __init__(self, vector, rows):
        self.obs = [vector, rows]
        self.agent_id = list(range(len(vector)))

    def __len__(self):
        return len(self.obs[0])
