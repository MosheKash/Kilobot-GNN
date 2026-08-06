"""Plain behaviour cloning: clone the actor to simple_oracle.py and export it.

A CTDE warm start meant to be followed by RL fine-tuning, not a finished policy.
run_bc_monitored.py is the instrumented alternative -- held-out validation, a
val tape, per-state reporting -- and is what scripts/train_bc.sh runs. Use this
one when you want the smallest possible BC loop.

Sets oracle_known_start_heading, which simple_oracle.py's dead-reckoned heading
requires: every robot must genuinely spawn at a known heading for its tracking
to mean anything. There is no coordination flag -- simple_oracle has no
inter-robot coordination by design.

Usage: python run_bc_simple_oracle.py <out.pt> [--iterations 100]
    [--formations ../data/formations] [--encoder ../data/image_encoder.pt]
    [--limit N] [--instances 1] [--arenas 4]

--instances is how many separate headless Unity players to launch (each with
its own worker id, and its own --swarm-rng offset so their rollouts don't
correlate); --arenas is cfg.num_arenas, how many parallel arenas each of those
players runs. Total parallel arenas per iteration is instances * arenas.

Produces a bare actor-weights export (checkpoint.export_actor's format, which
KILOBOT_INIT_ACTOR loads) at <out.pt>, checkpointed every iteration so an
interrupted run keeps its progress. To continue into RL fine-tuning:

    KILOBOT_INIT_ACTOR=<out.pt> KILOBOT_MODE=rl python launch.py

which loads only the cloned actor's weights and starts a normal RL run with a
fresh critic and optimizers. No separate bridging step is needed.
"""
import argparse
import atexit

import torch

from config import Config
from policy import GaussianPolicy
from formations import build_formation_pool
from trainer import Trainer
from kilobot_gnn import build_actor
from bc import bc_train
from encoder import load_encoder
from images import build_image_pool
from kilobot_gnn import Z


def preprocess(path):
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.asarray(img, dtype = np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iterations", type = int, default = 100)
    ap.add_argument("--formations", default = "../data/formations")
    ap.add_argument("--encoder", default = "../data/image_encoder.pt")
    ap.add_argument("--limit", type = int, default = None)
    ap.add_argument("--min-bots", type = int, default = 40)
    ap.add_argument("--max-bots", type = int, default = 60)
    ap.add_argument("--instances", type = int, default = 1)
    ap.add_argument("--arenas", type = int, default = 4)
    ap.add_argument("--rollout", type = int, default = 4096)
    ap.add_argument("--heartbeat", type = int, default = 48)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--swarm-rng", type = int, default = None,
                    help = "seed the Unity player's spawn RNG, making the run's sequence of arenas "
                           "replayable (episodes still differ from one another). Distinct from "
                           "--seed, which only seeds torch on this side. Each instance gets this "
                           "plus its index, so parallel players stay diverse. Left unset, spawns "
                           "are unseeded, as they have always been.")
    ap.add_argument("--bc-epochs", type = int, default = 4)
    ap.add_argument("--device", default = "cpu")
    ap.add_argument("--build", default = None, help = "path to the Unity player")
    ap.add_argument("--base-port", type = int, default = 5005, help = "ml-agents derives each player's socket from base_port + worker_id")
    ap.add_argument("--time-scale", type = float, default = 20.0,
                    help = "Unity time scale for headless collection")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = Config()
    cfg.actor_type = "gru_split_observation"
    cfg.seed_layout = "corners"
    cfg.heartbeat_ticks = args.heartbeat
    cfg.rollout_steps = args.rollout
    cfg.motor_override = "simple_oracle"
    # simple_oracle.py's dead-reckoned heading assumes every robot genuinely
    # spawns at one of belief.CARDINAL_HEADINGS. Only true if the player was
    # built with SwarmManager's knownStartHeading on.
    cfg.oracle_known_start_heading = True
    # trainer.Trainer.setup() calls worker.set_num_arenas(cfg.num_arenas)
    # for every worker in the list below -- one shared value, not per-
    # instance, matching how a real multi-worker Unity run already works
    # (every KILOBOT_NUM_WORKERS process gets the same KILOBOT_NUM_ARENAS)
    cfg.num_arenas = args.arenas

    encoder = load_encoder(args.encoder, args.device, expected_dim = Z)
    image_pool = build_image_pool(args.formations, preprocess, limit = args.limit, device = args.device)
    formation_pool = build_formation_pool(args.formations, limit = args.limit)
    assert len(image_pool) == len(formation_pool), \
        "encoder pool and formation pool length mismatch -- should never happen, both are built from images.formation_paths"
    print("loaded %d real formations" % len(formation_pool))

    # simple_oracle_motors reads this directly for its own hash-based target
    # assignment (spatial_hash.py) -- the same formation_pool the reward and
    # the encoder's own image_pool are already built from, kept in sync by
    # construction since all three are built from images.formation_paths
    cfg._oracle_formation_pool = formation_pool

    # each instance is its own worker (rather than num_arenas alone), so
    # instances don't collect identical, correlated rollouts -- one worker
    # already runs cfg.num_arenas arenas internally (Trainer.setup, above), so
    # total parallel arenas collected per iteration is instances * arenas.
    # One headless player per instance.
    import unity_env
    envs = []
    workers = []
    for i in range(args.instances):
        # set before EACH player launches -- read once, in Awake, from the
        # environment the player inherits from this process. swarm_rng is
        # offset by the instance index because all of them inherit this one
        # environment: a single shared value would give every player the
        # identical sequence of spawns, which is the opposite of what running
        # several of them is for.
        unity_env.set_player_env(formations = args.formations, heartbeat_ticks = cfg.heartbeat_ticks,
                                 seed_layout = cfg.seed_layout, num_arenas = args.arenas,
                                 min_bots = args.min_bots, max_bots = args.max_bots,
                                 swarm_rng = None if args.swarm_rng is None
                                              else args.swarm_rng + i)
        worker, env = unity_env.make_unity_worker(
            worker_id = i, num_arenas = args.arenas, build_path = args.build,
            no_graphics = True, base_port = args.base_port, time_scale = args.time_scale)
        envs.append(env)
        workers.append(worker)

    def close_envs(_closed = []):
        # a Unity player outlives this process (and keeps its port) if never
        # closed, including on an exception or Ctrl+C partway through.
        # Called explicitly at the end of main AND registered with atexit:
        # the explicit call is what normally runs, because closing during
        # interpreter shutdown can block indefinitely inside ml-agents'
        # own communicator teardown (observed directly -- a finished run
        # sat for minutes after its checkpoint was already written).
        # atexit stays as the backstop for the paths that never reach the
        # explicit call.
        if _closed:
            return
        _closed.append(True)
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                print("warning: failed to close a Unity env cleanly (%s)" % exc)

    atexit.register(close_envs)
    print("%d instance(s) x %d arena(s) = %d parallel arenas per iteration" %
          (args.instances, args.arenas, args.instances * args.arenas))

    tr = Trainer.from_workers(workers, cfg, encoder, image_pool)
    policy = GaussianPolicy(build_actor(cfg), cfg.log_std_init)
    actor_opt = torch.optim.Adam(policy.parameters(), lr = cfg.actor_lr)

    bc_train(tr, policy, actor_opt, cfg, args.iterations, None, args.bc_epochs, args.out,
            teacher = "simple_oracle")

    close_envs()
    print("\ncloned actor saved to %s" % args.out)
    print("to continue into RL fine-tuning:")
    print("  KILOBOT_INIT_ACTOR=%s KILOBOT_MODE=rl python launch.py" % args.out)


if __name__ == "__main__":
    main()
