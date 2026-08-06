"""Building a Unity-backed EnvWorker, standalone.

launch.py has always been able to do this, but only through module-level
globals resolved from KILOBOT_* environment variables at import time -- fine
for launch.py itself, useless for any other script that wants a Unity worker
with explicitly chosen settings. This module is that same wiring, as a
function, so the BC drivers can build Unity workers without importing
launch.py or inheriting its global state.

Two separate channels of configuration reach the Unity player, and the
distinction matters because one of them is set too late to change:

  environment variables   read once by SwarmManager.Awake / ImageLibrary in
                          the player process at startup (KILOBOT_FORMATIONS,
                          KILOBOT_HEARTBEAT_TICKS, KILOBOT_SEED_LAYOUT, ...).
                          The player inherits them from this process, so they
                          must be set BEFORE UnityEnvironment launches it --
                          setting them afterwards silently does nothing.

  side channels           EnvironmentParametersChannel ("num_arenas", read by
                          SceneBootstrap on every reset) and CriticChannel
                          (the per-arena node snapshots the trainer reads
                          back). These can change between resets.

"""

import os

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from channels import CriticChannel
from env_worker import EnvWorker
from policy import ACTION_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILD_PATH = os.path.join(HERE, "..", "Builds", "Kilobot.x86_64")
DEFAULT_LOG_DIR = os.path.abspath(os.path.join(HERE, "..", "results", "unity-logs"))


def configure_player_logging(log_folder = None, subdir = None):
    """Send a player's output to its own file instead of this process's console.

    Returns the absolute folder to hand UnityEnvironment(log_folder=...), or
    None to leave the player attached to the console.

    A Unity player inherits this process's stdout, so by default everything it
    prints -- engine banner, memory configuration, physics backend, every
    Debug.Log from every arena -- interleaves with the trainer's own output.
    Measured on a 1-iteration, 2-arena run: 339 lines, of which ~320 were the
    player's. ml-agents can redirect that to a file instead (-logFile), which is
    what this resolves the destination for; each player gets its own
    Player-<worker_id>.log inside, so parallel workers do not interleave either.

    Precedence: the explicit argument, then KILOBOT_UNITY_LOG_DIR, then
    results/unity-logs/. Setting either to the empty string means "do not
    redirect" -- the player prints to the console exactly as it used to, which
    is what you want when the player is failing to start and you need to see it
    happen live.

    subdir keeps unrelated groups of players out of each other's files (the test
    fixture uses one), since ml-agents names the file from the worker id alone
    and a test run would otherwise clobber a training run's Player-0.log.

    -logFile does not catch everything: the boot.config memory-setup echo (~30
    lines) reaches stdout before it takes effect. ml-agents can send the
    subprocess's stdout to DEVNULL instead, but the condition it uses reads
    logger.level, which stays 0 (NOTSET) until something sets it explicitly --
    so that branch never fires and the player stays attached to the terminal.
    Set here, and only when redirecting: with no log file to read afterwards, a
    player that dies during startup would otherwise die silently. WARNING rather
    than INFO because that is already ml-agents' effective level, so this
    changes what the subprocess does without changing what ml-agents prints.
    """
    if log_folder is None:
        log_folder = os.environ.get("KILOBOT_UNITY_LOG_DIR", DEFAULT_LOG_DIR)
    if log_folder == "":
        return None
    if subdir:
        log_folder = os.path.join(log_folder, subdir)
    log_folder = os.path.abspath(log_folder)   # ml-agents requires an absolute path
    os.makedirs(log_folder, exist_ok = True)
    from mlagents_envs import logging_util
    logging_util.set_log_level(logging_util.WARNING)
    return log_folder


def resolve_behavior(env, behavior_name = None, max_steps = 64):
    """The behavior whose action width matches the actor's.

    ml-agents only registers a behavior once one of its agents has actually
    requested a decision, so immediately after the first reset the specs may
    hold nothing but Heartbeat. With a large swarm that never shows: somebody is
    always in range of a seed or a neighbour on tick 0, so the Kilobot behavior
    appears straight away. With a small one -- a handful of robots spread across
    an empty arena, which is exactly what a focused test asks for -- nobody has
    an event, the first decision does not come until the heartbeat fires
    KILOBOT_HEARTBEAT_TICKS later, and resolving right after reset raises.

    So step (bounded) until it shows up rather than fail on swarm size.
    """
    if behavior_name is not None:
        return behavior_name
    for _ in range(max_steps + 1):
        for name in env.behavior_specs:
            if env.behavior_specs[name].action_spec.continuous_size == ACTION_SIZE:
                return name
        env.step()
    raise RuntimeError(
        "No behavior with %d continuous actions appeared within %d steps; saw %s. "
        "Pass behavior_name explicitly, or check KILOBOT_HEARTBEAT_TICKS is not larger "
        "than that budget." % (ACTION_SIZE, max_steps, list(env.behavior_specs)))


def set_player_env(formations = None, heartbeat_ticks = None, seed_layout = None,
                   num_arenas = None, show_radius = None, show_target_floor = None,
                   min_bots = None, max_bots = None, swarm_rng = None):
    """Set the environment variables the Unity player reads at startup.

    Must be called before make_unity_worker, since the player inherits this
    process's environment when UnityEnvironment launches it and reads these
    exactly once, in Awake.

    KILOBOT_FORMATIONS is resolved by the player against its OWN working
    directory (ImageLibrary.ResolvePath), not against python/ -- a relative
    path that resolves correctly here can still leave the player with an empty
    file list, or worse, a different directory, in which case imageId indexes
    two different sorted lists and Python and Unity disagree about which shape
    is showing. Passed through as an absolute path for that reason.
    """
    if formations is not None:
        os.environ["KILOBOT_FORMATIONS"] = os.path.abspath(formations)
    if heartbeat_ticks is not None:
        os.environ["KILOBOT_HEARTBEAT_TICKS"] = str(int(heartbeat_ticks))
    if seed_layout is not None:
        os.environ["KILOBOT_SEED_LAYOUT"] = str(seed_layout)
    if num_arenas is not None:
        os.environ["KILOBOT_NUM_ARENAS"] = str(int(num_arenas))
    if show_radius is not None:
        os.environ["KILOBOT_SHOW_RADIUS"] = "true" if show_radius else "false"
    if show_target_floor is not None:
        os.environ["KILOBOT_SHOW_TARGET_FLOOR"] = "true" if show_target_floor else "false"
    if min_bots is not None:
        os.environ["KILOBOT_MIN_BOTS"] = str(int(min_bots))
    if max_bots is not None:
        os.environ["KILOBOT_MAX_BOTS"] = str(int(max_bots))
    if swarm_rng is not None:
        # Makes the whole run replayable, first spawn included -- which the
        # parameters channel cannot do, since SpawnInitial runs before it has
        # necessarily delivered anything (see SwarmManager.swarmRngEnv).
        # Successive episodes still differ from one another; EnvWorker's
        # set_swarm_rng is the separate knob that pins one spawn exactly.
        os.environ["KILOBOT_SWARM_RNG"] = str(int(swarm_rng))


def make_unity_worker(worker_id = 0, num_arenas = 1, build_path = None, no_graphics = True,
                      base_port = 5005, time_scale = 20.0, timeout = 600, behavior_name = None,
                      swarm_rng = None, log_folder = None, log_subdir = None):
    """Launch one Unity player and wrap it in an EnvWorker.

    worker_id must be distinct per concurrent player -- ml-agents derives the
    socket port from base_port + worker_id, so reusing one silently collides
    with an already-running instance.

    Swarm size comes from KILOBOT_MIN_BOTS/KILOBOT_MAX_BOTS via set_player_env;
    SwarmManager falls back to its Inspector values when they are unset.

    swarm_rng makes the whole RUN replayable: the same seed replays the same
    sequence of spawns (population counts, positions, cardinal headings).
    Episodes still differ from one another -- to pin one spawn exactly, which is
    a test's problem and not a run's, use EnvWorker.set_swarm_rng. Leave None
    for the historical, unseeded behavior.

    The player's own output goes to <log_folder>/Player-<worker_id>.log rather
    than to this process's console -- see configure_player_logging, including how to
    turn that off when a player is failing to start.

    Returns (worker, env). The caller owns env and must close() it.
    """
    if swarm_rng is not None:
        set_player_env(swarm_rng = swarm_rng)
    critic_channel = CriticChannel()
    params_channel = EnvironmentParametersChannel()
    engine_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name = build_path or DEFAULT_BUILD_PATH,
        side_channels = [critic_channel, params_channel, engine_channel],
        no_graphics = no_graphics,
        base_port = base_port,
        worker_id = worker_id,
        timeout_wait = timeout,
        log_folder = configure_player_logging(log_folder, log_subdir),
    )
    engine_channel.set_configuration_parameters(time_scale = time_scale)
    params_channel.set_float_parameter("num_arenas", float(num_arenas))
    env.reset()
    worker = EnvWorker(env, critic_channel, params_channel, resolve_behavior(env, behavior_name))
    return worker, env
