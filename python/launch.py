"""Entry point: environment variables in, a configured run out.

Three jobs, in order:

  1. resolve every KILOBOT_* variable into module constants and a Config
  2. connect to the Unity player(s) -- make_env, and the parallel variant
  3. dispatch on KILOBOT_MODE: rl, bc, eval, or one of the probes

The training loop itself is trainer.py, behaviour cloning is bc.py, the probes
are diagnostics.py. run_eval stays here because it is a dispatch mode and reads
this module's resolved configuration directly; its log formatting does not, and
lives in diagnostics.print_eval_log_summary.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time
import numpy as np
import torch
from PIL import Image

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from config import Config
import unity_env
from channels import CriticChannel
from trainer import Trainer
from env_worker import EnvWorker
from images import build_image_pool, formation_paths
from kilobot_gnn import Actor, RecurrentActor, SplitObservationActor, priv_cols, Critic, Z, MESSAGE_SIZE, build_actor
from policy import GaussianPolicy, ACTION_SIZE
from encoder import load_encoder
from metrics import Logger
from diagnostics import (control_probe, audit_run, reward_probe, probe_run, watch_oracle,
                         print_eval_log_summary)
from bc import bc_train



def _env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _env_int(name, default):
    v = _env(name, None)
    return default if v is None else int(v)


def _env_float(name, default):
    v = _env(name, None)
    return default if v is None else float(v)


def _env_bool(name, default):
    v = _env(name, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_opt_int(name, default):
    v = _env(name, None)
    if v is None:
        return default
    if v.strip().lower() in ("none", "all"):
        return None
    return int(v)


# Every path and knob can be overridden by an environment variable, so the same
# code runs locally and in a container with no edits. Defaults reproduce the
# current local behavior. KILOBOT_FORMATIONS is shared with the Unity build, so
# one variable can point both sides at the same folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_PATH = _env("KILOBOT_BUILD_PATH", os.path.join(HERE, "..", "Builds", "Kilobot.x86_64"))
ENCODER_PATH = _env("KILOBOT_ENCODER_PATH", os.path.join(HERE, "..", "data", "image_encoder.pt"))
FORMATIONS_DIR = _env("KILOBOT_FORMATIONS", os.path.join(HERE, "..", "data", "formations"))
LOGDIR = _env("KILOBOT_LOGDIR", os.path.join(HERE, "..", "results", "tb"))

BEHAVIOR_NAME = _env("KILOBOT_BEHAVIOR", None)
NUM_ARENAS = _env_opt_int("KILOBOT_NUM_ARENAS", None)
DEVICE = _env("KILOBOT_DEVICE", "cpu")
NO_GRAPHICS = _env_bool("KILOBOT_NO_GRAPHICS", True)
BASE_PORT = _env_int("KILOBOT_BASE_PORT", 5005)
NUM_ENVS = _env_int("KILOBOT_NUM_ENVS", 1)
NUM_WORKERS = _env_int("KILOBOT_NUM_WORKERS", 1)
# Seeds the Unity player's spawn RNG, making the run's sequence of arenas
# replayable; episodes still differ from one another. Unset (the default) leaves
# spawns unseeded, exactly as every run before this behaved. See make_env for
# why it is per-worker, and why it is not KILOBOT_SEED.
SWARM_RNG = _env_opt_int("KILOBOT_SWARM_RNG", None)
TORCH_THREADS = _env_int("KILOBOT_TORCH_THREADS", 1)
TIMEOUT = _env_int("KILOBOT_TIMEOUT", 600)
TIME_SCALE = _env_float("KILOBOT_TIME_SCALE", 20.0)
ITERATIONS = _env_int("KILOBOT_ITERATIONS", 100)
CKPT_EVERY = _env_int("KILOBOT_CKPT_EVERY", 10)
RESUME = _env("KILOBOT_RESUME", None)
MODE = _env("KILOBOT_MODE", "rl")               # "rl" or "bc"
INIT_ACTOR = _env("KILOBOT_INIT_ACTOR", None)   # warm-start: load actor weights only
BC_OUT = _env("KILOBOT_BC_OUT", None)           # where bc mode saves the cloned actor
BC_EPOCHS = _env_int("KILOBOT_BC_EPOCHS", 4)
# saves to BC_OUT every N iterations during the run, not just once at the
# end -- added after an interrupted, unattended
# run lost several hours of progress with no way to recover it, since a
# single end-of-loop save means an interruption before that point saves
# nothing at all, no matter how far the run had gotten
BC_CHECKPOINT_EVERY = _env_int("KILOBOT_BC_CHECKPOINT_EVERY", 1)
# bc_train's own signature defaults this to "oracle", a different teacher that
# steers from the belief filter's heading and never calls simple_oracle_motors.
# run_bc_monitored.py has always passed "simple_oracle" explicitly; launch.py
# never did, so KILOBOT_MODE=bc against real Unity was cloning a different
# expert from every replica measurement. Default left unchanged to avoid
# silently altering existing runs; the guard below refuses the combination that
# actually breaks.
BC_TEACHER = _env("KILOBOT_BC_TEACHER", "oracle")
EVAL = _env_bool("KILOBOT_EVAL", False)
EVAL_WEIGHTS = _env("KILOBOT_EVAL_WEIGHTS", None)
EVAL_ITERS = _env_int("KILOBOT_EVAL_ITERS", 5)
# Print what the actor actually computes and sends, per decision, in a live
# Unity scene -- exercising EnvWorker's own observation-gathering and
# set_actions call, not just act()/policy. Off by default.
EVAL_LOG = _env_bool("KILOBOT_EVAL_LOG", False)
MIN_START_COV = _env_float("KILOBOT_MIN_START_COV", 0.0)
ROLLOUT_STEPS = _env_opt_int("KILOBOT_ROLLOUT_STEPS", None)
MAX_EPISODE_STEPS = _env_opt_int("KILOBOT_MAX_EPISODE_STEPS", None)

# When set, write a small JSON summary of the run (per-iteration coverage,
# entropy, explained variance) to this path. The hyperparameter sweep reads it.
SUMMARY_PATH = _env("KILOBOT_SUMMARY", None)

# Cap how many formations Python loads/encodes. With 175k on disk, encoding all
# of them is pointless for a smoke run; None = use every image (slow).
MAX_FORMATIONS = _env_opt_int("KILOBOT_MAX_FORMATIONS", 256)

# Print which formation file each arena is showing whenever it's set (initial
# setup and every episode reset). Off by default -- a normal, many-arena
# training run resets constantly and this would flood the log; opt in
# specifically (watch_actor.sh/watch_oracle.sh do) rather than have it fire
# for every run unconditionally.
LOG_FORMATIONS = _env_bool("KILOBOT_LOG_FORMATIONS", False)

# Matches the autoencoder training pipeline: 1x28x28 grayscale, scaled to [0,1].
IMAGE_SIZE = _env_int("KILOBOT_IMAGE_SIZE", 28)
IMAGE_MODE = _env("KILOBOT_IMAGE_MODE", "L")

# Small first run to validate the Unity <-> Python connection end to end.
SMOKE = _env_bool("KILOBOT_SMOKE", True)


def preprocess(path):
    img = Image.open(path).convert(IMAGE_MODE).resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0)


def resolve_behavior(env):
    if BEHAVIOR_NAME is not None:
        return BEHAVIOR_NAME
    for name in env.behavior_specs:
        spec = env.behavior_specs[name]
        if spec.action_spec.continuous_size == ACTION_SIZE:
            return name
    raise RuntimeError(
        "No behavior with %d continuous actions was found. Set BEHAVIOR_NAME explicitly." % ACTION_SIZE
    )


def check_latent_dim(encoder, pool, device):
    if len(pool) == 0:
        raise RuntimeError("No formation images found in %s." % FORMATIONS_DIR)
    with torch.no_grad():
        z = encoder(pool[0].to(device)).view(-1)
    if z.shape[0] != Z:
        raise RuntimeError(
            "Encoder latent width is %d but kilobot_gnn.Z is %d. "
            "Fix preprocess / the encoder, or change Z." % (z.shape[0], Z)
        )


def dump_behaviors(env):
    print("behaviors seen by Python:")
    for name in env.behavior_specs:
        spec = env.behavior_specs[name]
        shapes = [tuple(o.shape) for o in spec.observation_specs]
        print("  %s: obs=%s cont_actions=%d" % (name, shapes, spec.action_spec.continuous_size))


def make_env(worker_id):
    if SWARM_RNG is not None:
        # Offset per worker, and set immediately before this player launches:
        # SwarmManager reads KILOBOT_SWARM_RNG once, in Awake, from the
        # environment the player inherits from this process, so every worker
        # would otherwise run the identical sequence of arenas.
        #
        # Deliberately not KILOBOT_SEED, which seeds torch and numpy above: a
        # reproducible network init and a reproducible arena are separate
        # requests, and the player would silently pick up the former.
        os.environ["KILOBOT_SWARM_RNG"] = str(SWARM_RNG + worker_id)
    critic_channel = CriticChannel()
    params_channel = EnvironmentParametersChannel()
    engine_channel = EngineConfigurationChannel()
    # The player's own output (engine banner, physics backend, every Debug.Log
    # from every arena) otherwise interleaves with this run's training numbers,
    # since the player inherits this process's stdout. Redirected per worker;
    # KILOBOT_UNITY_LOG_DIR="" puts it back on the console.
    log_folder = unity_env.configure_player_logging()
    env = UnityEnvironment(
        file_name=BUILD_PATH,
        side_channels=[critic_channel, params_channel, engine_channel],
        no_graphics=NO_GRAPHICS,
        base_port=BASE_PORT,
        worker_id=worker_id,
        timeout_wait=TIMEOUT,
        log_folder=log_folder,
    )
    if worker_id == 0 and log_folder is not None:
        print("unity player log: %s/Player-<worker>.log" % log_folder)
    engine_channel.set_configuration_parameters(time_scale=TIME_SCALE)
    return env, critic_channel, params_channel
def run_eval(cfg):
    from checkpoint import load_for_eval
    from metrics import aggregate_payloads, rollout_stats
    weights = EVAL_WEIGHTS or RESUME
    if weights is None:
        raise SystemExit("eval needs KILOBOT_EVAL_WEIGHTS (or KILOBOT_RESUME) "
                         "pointing at a ckpt.pt or actor_final.pt")

    if MAX_EPISODE_STEPS is not None:
        # a real, separate bug of the exact same class as the rollout_steps
        # one below: `if MAX_EPISODE_STEPS is not None: cfg.max_episode_steps
        # = MAX_EPISODE_STEPS` (main()'s own copy of this) sits after `if
        # EVAL: run_eval(cfg); return`, so it never actually runs for the
        # eval path either. Confirmed directly from a real log: episodes
        # were resetting every ~2048 ticks -- Config's own plain default --
        # not the 1000000000 watch_actor.sh actually sets via
        # KILOBOT_MAX_EPISODE_STEPS, which was silently never reaching cfg
        # at all. Applied here, unconditionally (not gated on EVAL_LOG --
        # this affects every eval invocation, logged or not).
        cfg.max_episode_steps = MAX_EPISODE_STEPS

    if EVAL_LOG:
        # cfg.rollout_steps is never touched for the eval path at all (the
        # only two places that set it, --smoke and an explicit override,
        # both sit after `if EVAL: run_eval(cfg); return` in main() and so
        # never run here) -- it stays at Config's own plain default, 4096.
        # At KILOBOT_TIME_SCALE=1 (what watch_actor.sh always sets) that is
        # not a short wait: measured directly, even the pure-Python
        # simulation with no rendering and no real-time lock at all takes
        # ~110 seconds for 4096 ticks with a real 40-60 robot arena, and a
        # live Unity scene is a floor of at least that, likely more once
        # rendering and the real-time lock are added. With EVAL_ITERS
        # itself defaulting sky-high too (watch_actor.sh's own EPISODES
        # default, 999999, maps straight to it) the practical effect was:
        # the very first collect() call, and therefore the very first
        # logged line, could take minutes to arrive -- not a bug, just the
        # wrong granularity for a live, watched session. Shortened here
        # specifically when EVAL_LOG is on, so each collect() call returns
        # in a few real seconds instead: enough decisions for every robot
        # to be seen, not the thousands of ticks a genuine, full-episode
        # success/coverage summary would need. That summary becomes
        # meaningless at this length -- expected and fine, since the point
        # of EVAL_LOG is fast, live diagnostic output, not final metrics.
        original_rollout_steps = cfg.rollout_steps
        cfg.rollout_steps = max(1, cfg.heartbeat_ticks * 4)
        print("EVAL_LOG: cfg.rollout_steps %d -> %d for fast, responsive per-collect() logging "
              "(the eval results printed at the end will not be meaningful at this length -- "
              "that trade is intentional here, not a bug)"
              % (original_rollout_steps, cfg.rollout_steps))

    # Eval is single-process over one env; it collects CPU observations, so the
    # policy must be on CPU regardless of KILOBOT_DEVICE (which is for the GPU
    # learner). A few deterministic rollouts on CPU are cheap.
    device = "cpu"
    env, cc, pc = make_env(0)
    try:
        try:
            env.reset()
        except Exception as e:
            print("EVAL_LOG: env.reset() itself failed -- %r  (this means Unity never accepted "
                  "the connection at all: confirm the scene is in Play mode, and that "
                  "KILOBOT_BUILD_PATH/the Editor's own listen port match what this script is "
                  "trying to connect to)" % e)
            raise
        behavior_name = resolve_behavior(env)
        if EVAL_LOG:
            spec = env.behavior_specs[behavior_name]
            obs_shapes = [tuple(o.shape) for o in spec.observation_specs]
            print("EVAL_LOG: connected -- behavior_name=%r  observation_specs shapes=%s  "
                  "action continuous_size=%d"
                  % (behavior_name, obs_shapes, spec.action_spec.continuous_size))
        encoder = load_encoder(ENCODER_PATH, device, expected_dim=Z)
        pool = build_image_pool(FORMATIONS_DIR, preprocess, limit=MAX_FORMATIONS, device=device)
        check_latent_dim(encoder, pool, device)
        print("loaded encoder and %d formations" % len(pool))
        # Always computed, not gated behind
        # LOG_FORMATIONS -- Trainer._absolute_image_index needs this to
        # translate the local pool index into the absolute folder index
        # Unity's own file lookup expects. Cheap (filename strings only),
        # and formation_paths is cached per (folder, pattern, limit) since
        # cached, so this reuses build_image_pool's own call above rather
        # than re-scanning the folder.
        image_names = [os.path.basename(p) for p in formation_paths(FORMATIONS_DIR, limit=MAX_FORMATIONS)]

        actor = build_actor(cfg).to(device)
        policy = GaussianPolicy(actor, cfg.log_std_init).to(device)
        trained_iter = load_for_eval(weights, policy, device)
        policy.eval()
        print("eval: loaded %s (trained to iter %d)" % (weights, trained_iter))
        if EVAL_LOG:
            all_params = torch.cat([p.detach().flatten() for p in actor.parameters()])
            nan_ct = int(torch.isnan(all_params).sum())
            print("EVAL_LOG: loaded actor weights -- %d params, mean=%.6f std=%.6f "
                  "min=%.4f max=%.4f nan_count=%d  (all near-zero, or any nan_count, "
                  "means the checkpoint itself is the problem, before Unity is even involved)"
                  % (all_params.numel(), float(all_params.mean()), float(all_params.std()),
                     float(all_params.min()), float(all_params.max()), nan_ct))

        trainer = Trainer(env, cc, pc, cfg, encoder, pool, behavior_name, image_names = image_names, formations_dir = FORMATIONS_DIR)
        trainer.setup()
        if EVAL_LOG:
            print("EVAL_LOG: cfg.motor_override = %r  (expect 'none' for the actor's own trained "
                  "output to actually drive the robots -- 'simple_oracle'/'oracle' would drive with "
                  "the oracle instead, which still moves, just not via the trained network)" % cfg.motor_override)
            print("EVAL_LOG: cfg.num_arenas=%r cfg.heartbeat_ticks=%r cfg.seed_layout=%r cfg.actor_type=%r"
                  % (cfg.num_arenas, cfg.heartbeat_ticks, cfg.seed_layout, cfg.actor_type))
            trainer._audit = True
            trainer._probe = True
            trainer._pos_track = True
        payloads = []
        for it in range(EVAL_ITERS):
            if EVAL_LOG:
                trainer._audit_log = []
                trainer._probe_log = []
                trainer._pos_log = []
            try:
                with torch.no_grad():
                    trainer.collect(policy, None, deterministic=True)
            except Exception as e:
                if EVAL_LOG:
                    print("EVAL_LOG iter %d: collect() itself raised -- %r  (this is a hard failure "
                          "in the Unity communication layer or the step loop, not a quiet symptom; "
                          "the traceback below is the real error)" % (it, e))
                raise
            payloads.append(trainer.rollout_payload())
            if EVAL_LOG:
                print_eval_log_summary(it, trainer._audit_log, trainer._probe_log, trainer._pos_log)

        agg = aggregate_payloads(payloads)
        stats = rollout_stats(agg)
        n_eps = len(agg["ep_records"])
        print("eval results over %d completed episodes (%d rollouts, deterministic actions):"
              % (n_eps, EVAL_ITERS))
        print("  success_rate        = %.3f" % stats.get("episodes/success_rate", float("nan")))
        print("  mean_final_coverage = %.3f" % stats.get("episodes/mean_final_coverage", float("nan")))
        print("  mean_step_coverage  = %.3f" % stats.get("rollout/mean_coverage", float("nan")))
    finally:
        env.close()


def uses_parallel_trainer(num_workers, mode):
    # ParallelTrainer only implements run(), not setup()/collect(), so every
    # diagnostic mode (bc, probe, reward_probe, audit, control) needs the
    # single-process Trainer regardless of worker count. Only "rl" can use it.
    return num_workers > 1 and mode == "rl"


def main():
    torch.set_num_threads(TORCH_THREADS)
    cfg = Config()
    cfg.device = DEVICE
    if NUM_ARENAS is not None:
        cfg.num_arenas = NUM_ARENAS

    if not NO_GRAPHICS and cfg.num_arenas > 2:
        show_radius = _env_bool("KILOBOT_SHOW_RADIUS", False)
        print("WARNING: KILOBOT_NO_GRAPHICS=false with KILOBOT_NUM_ARENAS=%d and "
              "KILOBOT_TIME_SCALE=%.1f. Rendering %d arenas (each up to a few hundred "
              "kilobots/seeds%s) in real time is far outside what graphics mode was tested "
              "at -- it was built for a single arena at a time to sanity-check spawns and "
              "behavior, not for full training runs. If Unity is slow to render this, the "
              "environment can go fully unresponsive and time out (KILOBOT_TIMEOUT=%ds), "
              "which looks identical to a crash from here since Unity gives no diagnostic "
              "back over the socket. For an actual training run, use KILOBOT_NO_GRAPHICS=true "
              "(the default). If you want to watch, use KILOBOT_NUM_ARENAS=1."
              % (cfg.num_arenas, TIME_SCALE, cfg.num_arenas,
                 " plus a radius disc per robot (KILOBOT_SHOW_RADIUS=true)" if show_radius else "",
                 TIMEOUT))

    # Each hyperparameter can be overridden by an environment variable so a sweep
    # can vary it without editing config.py. Unset variables keep the Config
    # default. These apply to training; eval also reads cfg.log_std_init.
    cfg.actor_lr = _env_float("KILOBOT_ACTOR_LR", cfg.actor_lr)
    cfg.critic_lr = _env_float("KILOBOT_CRITIC_LR", cfg.critic_lr)
    cfg.entropy_coef = _env_float("KILOBOT_ENTROPY_COEF", cfg.entropy_coef)
    cfg.clip = _env_float("KILOBOT_CLIP", cfg.clip)
    cfg.ppo_epochs = _env_int("KILOBOT_PPO_EPOCHS", cfg.ppo_epochs)
    cfg.minibatch = _env_int("KILOBOT_MINIBATCH", cfg.minibatch)
    cfg.gamma = _env_float("KILOBOT_GAMMA", cfg.gamma)
    cfg.gae_lambda = _env_float("KILOBOT_GAE_LAMBDA", cfg.gae_lambda)
    cfg.max_grad_norm = _env_float("KILOBOT_MAX_GRAD_NORM", cfg.max_grad_norm)
    cfg.log_std_init = _env_float("KILOBOT_LOG_STD_INIT", cfg.log_std_init)
    cfg.r_on = _env_float("KILOBOT_R_ON", cfg.r_on)
    cfg.k_pos = _env_float("KILOBOT_K_POS", cfg.k_pos)
    cfg.tau_v = _env_float("KILOBOT_TAU_V", cfg.tau_v)
    cfg.l_scale = _env_float("KILOBOT_L_SCALE", cfg.l_scale)
    cfg.reward_shaping = _env_float("KILOBOT_REWARD_SHAPING", cfg.reward_shaping)
    cfg.success_threshold = _env_float("KILOBOT_SUCCESS_THRESHOLD", cfg.success_threshold)
    cfg.actor_type = _env("KILOBOT_ACTOR", cfg.actor_type)
    cfg.gru_hidden = _env_int("KILOBOT_GRU_HIDDEN", cfg.gru_hidden)
    cfg.split_upscale_hidden = _env_int("KILOBOT_SPLIT_UPSCALE_HIDDEN", cfg.split_upscale_hidden)
    cfg.split_gru_hidden = _env_int("KILOBOT_SPLIT_GRU_HIDDEN", cfg.split_gru_hidden)
    cfg.split_head_hidden = _env_int("KILOBOT_SPLIT_HEAD_HIDDEN", cfg.split_head_hidden)
    # Direct, reported bug: watch_actor.sh had no way to pass either flag
    # through, so launch.py's own build_actor(cfg) (line ~31) always built
    # a plain actor -- no head_arrived, prop width 40 not 42 -- regardless
    # of what the checkpoint being loaded was actually trained with.
    # load_for_eval's own strict load_state_dict then correctly refused a
    # checkpoint saved with either flag on: "Unexpected key(s)...
    # head_arrived.weight" and a size mismatch on up1.weight (42 vs 40),
    # exactly the reported error. run_bc_monitored.py's own --use-arrived-
    # head/--use-turn-anchor never had an equivalent here since those two
    # flags didn't exist yet when this environment-variable block was
    # first written. Same _env_bool/default-preserving pattern as every
    # other flag on this block, so an unset variable changes nothing.
    cfg.use_arrived_head = _env_bool("KILOBOT_USE_ARRIVED_HEAD", cfg.use_arrived_head)
    cfg.use_turn_anchor = _env_bool("KILOBOT_USE_TURN_ANCHOR", cfg.use_turn_anchor)
    # config.py's own split_activation has the rationale. Same shape of switch
    # as the two flags above: a checkpoint trained under one activation is a
    # different function under another, and nothing in a state_dict records
    # which one it was, so evaluating or warm-starting from a bc_offline.py
    # checkpoint means setting this to what its meta says.
    cfg.split_activation = os.environ.get("KILOBOT_SPLIT_ACTIVATION", cfg.split_activation)
    # Everything run_bc_monitored.py exposes as a --flag, exposed here as an
    # environment variable, so KILOBOT_MODE=bc against real Unity trains under
    # the same pipeline as the replica rather than the pre-reservoir one.
    # Deliberately placed in this block, which runs BEFORE main()'s own
    # `if EVAL: run_eval(cfg); return` -- two separate bugs in this file
    # (rollout_steps, then max_episode_steps) came from
    # assignments sitting after that return and silently never applying.
    # Same _env_*/default-preserving pattern as every flag above, so an unset
    # variable changes nothing.
    # Config's own default is still "deepset", the original actor this file was
    # written around, and nothing here ever overrode it -- so KILOBOT_MODE=bc
    # against real Unity has only ever trained the deepset, never the
    # split-observation GRU that run_bc_monitored.py defaults to. Surfaced by a
    # real Unity run crashing on "'Actor' object has no attribute 'up1'".
    cfg.actor_type = _env("KILOBOT_ACTOR_TYPE", cfg.actor_type)
    cfg.val_tape_path = _env("KILOBOT_VAL_TAPE", getattr(cfg, "val_tape_path", ""))
    cfg.val_tape_interval = _env_int("KILOBOT_VAL_TAPE_INTERVAL", getattr(cfg, "val_tape_interval", 5))
    cfg.actor_recurrent = _env_bool("KILOBOT_ACTOR_RECURRENT", getattr(cfg, "actor_recurrent", True))
    cfg.bc_replay_capacity = _env_int("KILOBOT_BC_REPLAY_CAPACITY", getattr(cfg, "bc_replay_capacity", 0))
    cfg.bc_replay_balanced = _env_bool("KILOBOT_BC_REPLAY_BALANCED", getattr(cfg, "bc_replay_balanced", True))
    cfg.bc_replay_max_age = _env_int("KILOBOT_BC_REPLAY_MAX_AGE", getattr(cfg, "bc_replay_max_age", 0))
    cfg.bc_replay_evict = _env("KILOBOT_BC_REPLAY_EVICT", getattr(cfg, "bc_replay_evict", "random"))
    cfg.bc_replay_min_samples = _env_int("KILOBOT_BC_REPLAY_MIN_SAMPLES", getattr(cfg, "bc_replay_min_samples", 512))
    cfg.bc_replay_persist = _env_bool("KILOBOT_BC_REPLAY_PERSIST", getattr(cfg, "bc_replay_persist", False))
    cfg.bc_replay_save_interval = _env_int("KILOBOT_BC_REPLAY_SAVE_INTERVAL", getattr(cfg, "bc_replay_save_interval", 20))
    cfg.bc_motor_skip_arrived = _env_bool("KILOBOT_BC_MOTOR_SKIP_ARRIVED", getattr(cfg, "bc_motor_skip_arrived", False))
    cfg.bc_arrived_natural_prior = _env_bool("KILOBOT_BC_ARRIVED_NATURAL_PRIOR", getattr(cfg, "bc_arrived_natural_prior", False))
    cfg.bc_actor_eval_interval = _env_int("KILOBOT_BC_ACTOR_EVAL_INTERVAL", getattr(cfg, "bc_actor_eval_interval", 1))
    cfg.arrived_confidence_threshold = _env_float("KILOBOT_ARRIVED_CONFIDENCE_THRESHOLD", cfg.arrived_confidence_threshold)
    cfg.arrived_release_threshold = _env_float("KILOBOT_ARRIVED_RELEASE_THRESHOLD", getattr(cfg, "arrived_release_threshold", 0.0))
    cfg.turn_anchor_latch = _env_bool("KILOBOT_TURN_ANCHOR_LATCH", getattr(cfg, "turn_anchor_latch", True))
    # The arrival gate, deployed: swap (or OR) the learned arrived head for the
    # oracle's own closed-form rule (config.py's use_closed_form_arrived /
    # closed_form_arrival_dist / closed_form_hybrid). Same default-preserving
    # pattern as every flag above, so an unset variable changes nothing -- this
    # is what makes a "watching the hybrid" run actually run the hybrid instead
    # of silently the plain learned-head gate. 0 arrival distance (the Config
    # default) means cfg.tau_v, the oracle's own rule; the actor's own filter
    # under-reports closeness, so the hybrid's 0.08 is the measured value.
    cfg.use_closed_form_arrived = _env_bool("KILOBOT_USE_CLOSED_FORM_ARRIVED",
                                            getattr(cfg, "use_closed_form_arrived", False))
    cfg.closed_form_arrival_dist = _env_float("KILOBOT_CLOSED_FORM_ARRIVAL_DIST",
                                              getattr(cfg, "closed_form_arrival_dist", 0.0))
    cfg.closed_form_hybrid = _env_bool("KILOBOT_CLOSED_FORM_HYBRID",
                                       getattr(cfg, "closed_form_hybrid", False))
    # The three auxiliary heads (arrived, state, wall) plus the oracle head and
    # steer feature. Same direct bug the two flags above fixed: build_actor(cfg)
    # here has no access to what a checkpoint was trained with, and load_for_eval
    # strictly refuses a mismatch ("head_state.weight" unexpected, up1 width 40
    # vs 42) for any watching/eval run of a bc_offline checkpoint -- which ships
    # all of these on. _env_bool/default-preserving like the rest of the block.
    cfg.use_state_head = _env_bool("KILOBOT_USE_STATE_HEAD", cfg.use_state_head)
    cfg.use_wall_head = _env_bool("KILOBOT_USE_WALL_HEAD", cfg.use_wall_head)
    cfg.use_oracle_head = _env_bool("KILOBOT_USE_ORACLE_HEAD", cfg.use_oracle_head)
    cfg.use_steer_feature = _env_bool("KILOBOT_USE_STEER_FEATURE", cfg.use_steer_feature)
    # The reservoir persists next to whatever BC_OUT the run writes its actor
    # to, matching run_bc_monitored.py's own <out_dir>/bc_reservoir.pt.
    if BC_OUT:
        cfg.bc_replay_path = os.path.join(os.path.dirname(os.path.abspath(BC_OUT)) or ".",
                                           "bc_reservoir.pt")
    cfg.split_prop_scale = _env_float("KILOBOT_SPLIT_PROP_SCALE", cfg.split_prop_scale)
    cfg.split_prop_time_scale = _env_float("KILOBOT_SPLIT_PROP_TIME_SCALE", cfg.split_prop_time_scale)
    cfg.split_seed_weight_boost = _env_float("KILOBOT_SPLIT_SEED_WEIGHT_BOOST", cfg.split_seed_weight_boost)
    cfg.prop_max_speed = _env_float("KILOBOT_PROP_MAX_SPEED", cfg.prop_max_speed)
    cfg.prop_wheelbase = _env_float("KILOBOT_PROP_WHEELBASE", cfg.prop_wheelbase)
    cfg.prop_scale = _env_float("KILOBOT_PROP_SCALE", cfg.prop_scale)
    cfg.prop_time_scale = _env_float("KILOBOT_PROP_TIME_SCALE", cfg.prop_time_scale)
    cfg.prop_cum_scale = _env_float("KILOBOT_PROP_CUM_SCALE", cfg.prop_cum_scale)
    cfg.reward_mode = _env("KILOBOT_REWARD_MODE", cfg.reward_mode)
    cfg.speed_weight = _env_float("KILOBOT_SPEED_WEIGHT", cfg.speed_weight)
    cfg.steer_weight = _env_float("KILOBOT_STEER_WEIGHT", cfg.steer_weight)
    cfg.seed_layout = _env("KILOBOT_SEED_LAYOUT", cfg.seed_layout)
    cfg.belief_comms = _env_bool("KILOBOT_BELIEF_COMMS", cfg.belief_comms)
    cfg.belief_conf_bonus = _env_float("KILOBOT_BELIEF_CONF_BONUS", cfg.belief_conf_bonus)
    cfg.belief_conf_bonus_iters = _env_int("KILOBOT_BELIEF_CONF_BONUS_ITERS", cfg.belief_conf_bonus_iters)
    cfg.seed_find_bonus = _env_float("KILOBOT_SEED_FIND_BONUS", cfg.seed_find_bonus)
    cfg.wall_find_penalty = _env_float("KILOBOT_WALL_FIND_PENALTY", cfg.wall_find_penalty)
    cfg.heartbeat_ticks = _env_int("KILOBOT_HEARTBEAT_TICKS", cfg.heartbeat_ticks)
    if cfg.heartbeat_ticks > 0 and cfg.actor_type == "deepset":
        raise ValueError("KILOBOT_HEARTBEAT_TICKS requires the gru or gru_split_observation "
                         "actor; the deepset database has no representation for an "
                         "event-less decision")
    _fm = _env("KILOBOT_FORCE_MOTOR", "")
    if _fm:
        parts = _fm.split(",")
        cfg.force_motor = (float(parts[0]), float(parts[1]))
    cfg.k_sep = _env_float("KILOBOT_K_SEP", cfg.k_sep)
    cfg.tau_sep = _env_float("KILOBOT_TAU_SEP", cfg.tau_sep)
    cfg.r_pack = _env_float("KILOBOT_R_PACK", cfg.r_pack)
    cfg.pack_range = _env_float("KILOBOT_PACK_RANGE", cfg.pack_range)
    cfg.actor_priv_mode = os.environ.get("KILOBOT_ACTOR_PRIV_MODE", cfg.actor_priv_mode)
    cfg.motor_override = os.environ.get("KILOBOT_MOTOR_OVERRIDE", cfg.motor_override)

    # Deliberately placed AFTER every cfg field these guards read has been
    # assigned. Placed earlier, the heartbeat guard fired on a real Unity run
    # that had KILOBOT_HEARTBEAT_TICKS=48 set correctly, because
    # cfg.heartbeat_ticks is assigned further down this same block -- the same
    # read-before-assign ordering that produced the phase-137 rollout_steps and
    # the later max_episode_steps bugs in this file.
    if MODE == "bc" and cfg.actor_type == "gru_split_observation" and cfg.heartbeat_ticks <= 0:
        raise SystemExit("KILOBOT_MODE=bc with the split-observation actor requires "
                         "KILOBOT_HEARTBEAT_TICKS > 0 (48 is what every replica run used): "
                         "with heartbeat_ticks=0 a robot that never comes into range of a seed "
                         "or neighbour never makes a single decision, measured at ~26% of "
                         "robots stuck for a whole rollout")
    if cfg.bc_replay_capacity > 0 and BC_TEACHER != "simple_oracle":
        raise SystemExit("KILOBOT_BC_REPLAY_CAPACITY requires KILOBOT_BC_TEACHER=simple_oracle: "
                         "bc_train's own default teacher (%r) never calls simple_oracle_motors, "
                         "so no per-decision oracle state labels exist and every sample lands in "
                         "the reservoir as _unlabelled -- balanced sampling then has nothing to "
                         "balance and silently degrades to uniform" % BC_TEACHER)
    if cfg.actor_type != "gru_split_observation" and (cfg.bc_replay_capacity > 0
            or cfg.bc_motor_skip_arrived or not getattr(cfg, "actor_recurrent", True)):
        raise SystemExit("KILOBOT_BC_REPLAY_CAPACITY / KILOBOT_BC_MOTOR_SKIP_ARRIVED / "
                         "KILOBOT_ACTOR_RECURRENT=false require "
                         "KILOBOT_ACTOR_TYPE=gru_split_observation -- cfg.actor_type is "
                         "currently %r (Config's own default is \"deepset\")" % cfg.actor_type)
    if cfg.bc_motor_skip_arrived and not cfg.use_arrived_head:
        raise SystemExit("KILOBOT_BC_MOTOR_SKIP_ARRIVED requires KILOBOT_USE_ARRIVED_HEAD: "
                         "without the arrived head nothing stops a robot, since the motor head "
                         "is deliberately never trained to output [0,0]")
    if cfg.arrived_release_threshold > 0 and cfg.arrived_release_threshold >= cfg.arrived_confidence_threshold:
        raise SystemExit("KILOBOT_ARRIVED_RELEASE_THRESHOLD must be strictly below "
                         "KILOBOT_ARRIVED_CONFIDENCE_THRESHOLD (it is a hysteresis floor)")
    cfg.oracle_known_start_heading = _env_bool("KILOBOT_ORACLE_KNOWN_START_HEADING", cfg.oracle_known_start_heading)
    cfg.oracle_orbit_axis_trust_threshold = _env_float("KILOBOT_ORACLE_ORBIT_AXIS_TRUST_THRESHOLD", cfg.oracle_orbit_axis_trust_threshold)
    cfg.oracle_send_visual_state = _env_bool("KILOBOT_ORACLE_SEND_VISUAL_STATE", cfg.oracle_send_visual_state)
    cfg.oracle_debug_wall_log = _env_bool("KILOBOT_ORACLE_DEBUG_WALL_LOG", cfg.oracle_debug_wall_log)
    cfg.oracle_perfect_heading = _env_bool("KILOBOT_ORACLE_PERFECT_HEADING", cfg.oracle_perfect_heading)
    cfg.direct_motor = _env_bool("KILOBOT_DIRECT_MOTOR", cfg.direct_motor)
    cfg.critic_chunk_steps = _env_int("KILOBOT_CRITIC_CHUNK", cfg.critic_chunk_steps)
    cfg.collect_max_wait = _env_float("KILOBOT_COLLECT_MAX_WAIT", cfg.collect_max_wait)
    cfg.log_formations = LOG_FORMATIONS
    cfg.seed = _env_int("KILOBOT_SEED", cfg.seed)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if EVAL:
        run_eval(cfg)
        return

    iterations = ITERATIONS
    if SMOKE:
        cfg.rollout_steps = 128
        cfg.max_episode_steps = 256
        iterations = 3
    if ROLLOUT_STEPS is not None:
        cfg.rollout_steps = ROLLOUT_STEPS
    if MAX_EPISODE_STEPS is not None:
        cfg.max_episode_steps = MAX_EPISODE_STEPS

    summary = None
    if SUMMARY_PATH is not None:
        from metrics import RunSummary
        summary = RunSummary(SUMMARY_PATH)

    if uses_parallel_trainer(NUM_WORKERS, MODE):
        from parallel import ParallelTrainer
        actor = build_actor(cfg).to(cfg.device)
        policy = GaussianPolicy(actor, cfg.log_std_init).to(cfg.device)
        critic = Critic().to(cfg.device)
        actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        if INIT_ACTOR is not None:
            from checkpoint import load_for_eval
            load_for_eval(INIT_ACTOR, policy, cfg.device)
            print("warm-started actor from %s" % INIT_ACTOR)
        run_dir = os.path.join(LOGDIR, time.strftime("run_%Y%m%d_%H%M%S"))
        logger = Logger(run_dir)
        print("to view metrics: tensorboard --logdir", LOGDIR)
        print("launching %d worker processes" % NUM_WORKERS)
        start_iter = 0
        if RESUME is not None:
            from checkpoint import load_checkpoint
            start_iter = load_checkpoint(RESUME, policy, critic, actor_opt, critic_opt, cfg.device)
            print("resumed from %s at iteration %d" % (RESUME, start_iter))
        ckpt_path = os.path.join(run_dir, "ckpt.pt")
        trainer = ParallelTrainer(cfg, NUM_WORKERS)
        trainer.run(policy, critic, actor_opt, critic_opt, iterations, logger=logger,
                    ckpt_path=ckpt_path, ckpt_every=CKPT_EVERY, start_iter=start_iter,
                    min_start_cov=MIN_START_COV, summary=summary)
        print("done")
        return

    envs = []
    channels = []
    behavior_name = None

    try:
        for i in range(NUM_ENVS):
            env, cc, pc = make_env(i)
            envs.append(env)
            channels.append((cc, pc))
            try:
                env.reset()
            except Exception:
                dump_behaviors(env)
                raise
            if i == 0:
                dump_behaviors(env)
                behavior_name = resolve_behavior(env)
                print("kilobot behavior:", behavior_name)

        encoder = load_encoder(ENCODER_PATH, cfg.device, expected_dim=Z)
        pool = build_image_pool(FORMATIONS_DIR, preprocess, limit=MAX_FORMATIONS, device=cfg.device)
        check_latent_dim(encoder, pool, cfg.device)
        print("loaded encoder and %d formations" % len(pool))
        # See the identical comment on the eval
        # path above for the full rationale
        image_names = [os.path.basename(p) for p in formation_paths(FORMATIONS_DIR, limit=MAX_FORMATIONS)]

        # Mirrors the block immediately above,
        # for the new, from-scratch simple_oracle.py -- same formation-pool
        # source, same only-built-when-needed guard, entirely independent
        # of whether the old oracle's own coordinator also gets built
        # MODE=="bc" is checked alongside motor_override because bc_train sets
        # cfg.motor_override = teacher inside its own loop, long after this
        # block runs -- so in BC mode the override is still whatever
        # KILOBOT_MOTOR_OVERRIDE said (normally "none") and this guard was
        # never true. A real Unity BC run crashed on exactly that:
        # "object of type 'NoneType' has no len()" from ensure_target, because
        # simple_oracle_motors got formation_pool=None.
        if cfg.motor_override == "simple_oracle" or (MODE == "bc" and BC_TEACHER == "simple_oracle"):
            from formations import build_formation_pool
            formation_pool = build_formation_pool(FORMATIONS_DIR, limit=MAX_FORMATIONS)
            assert len(formation_pool) == len(pool), \
                "formation pool and image pool length mismatch -- should never happen, both built from images.formation_paths"
            cfg._oracle_formation_pool = formation_pool
            print("simple_oracle formation pool built (%d formations) -- see simple_oracle.py's own module docstring"
                  % len(formation_pool))

        actor = build_actor(cfg).to(cfg.device)
        policy = GaussianPolicy(actor, cfg.log_std_init).to(cfg.device)
        critic = Critic().to(cfg.device)
        actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

        if NUM_ENVS == 1:
            cc, pc = channels[0]
            trainer = Trainer(envs[0], cc, pc, cfg, encoder, pool, behavior_name, image_names = image_names, formations_dir = FORMATIONS_DIR)
        else:
            workers = []
            for i in range(NUM_ENVS):
                cc, pc = channels[i]
                workers.append(EnvWorker(envs[i], cc, pc, behavior_name))
            trainer = Trainer.from_workers(workers, cfg, encoder, pool, image_names = image_names, formations_dir = FORMATIONS_DIR)
            print("running %d parallel Unity instances" % NUM_ENVS)

        run_dir = os.path.join(LOGDIR, time.strftime("run_%Y%m%d_%H%M%S"))
        logger = Logger(run_dir)
        print("to view metrics: tensorboard --logdir", LOGDIR)

        if INIT_ACTOR is not None:
            from checkpoint import load_for_eval
            load_for_eval(INIT_ACTOR, policy, cfg.device)
            print("warm-started actor from %s" % INIT_ACTOR)

        if MODE == "bc":
            bc_train(trainer, policy, actor_opt, cfg, iterations, summary, BC_EPOCHS, BC_OUT, logger = logger, checkpoint_every = BC_CHECKPOINT_EVERY, teacher = BC_TEACHER)
            print("done")
            return

        if MODE == "watch_oracle":
            watch_oracle(trainer, policy, cfg)
            return

        if MODE == "probe":
            probe_run(trainer, policy, cfg, iterations)
            print("done")
            return

        if MODE == "reward_probe":
            reward_probe(trainer, policy, cfg, iterations)
            print("done")
            return

        if MODE == "audit":
            audit_run(trainer, policy, cfg, iterations)
            print("done")
            return

        if MODE == "control":
            control_probe(trainer, policy, cfg, iterations)
            print("done")
            return

        start_iter = 0
        if RESUME is not None:
            from checkpoint import load_checkpoint
            start_iter = load_checkpoint(RESUME, policy, critic, actor_opt, critic_opt, cfg.device)
            print("resumed from %s at iteration %d" % (RESUME, start_iter))
        ckpt_path = os.path.join(run_dir, "ckpt.pt")
        trainer.run(policy, critic, actor_opt, critic_opt, iterations, logger=logger,
                    ckpt_path=ckpt_path, ckpt_every=CKPT_EVERY, start_iter=start_iter,
                    min_start_cov=MIN_START_COV, summary=summary)
        print("done")
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
