"""Measure residual dead-reckoning error against a live Unity player.

Dead reckoning here is deterministic: the motor command is known, the physics
constants are known, and no noise is injected on either side. Any residual is
model mismatch, not measurement noise -- so the target is zero, not "small".

For each motor pair, holds that command and compares what split_tick_motion
predicts for one environment step against what the player actually did, read
from the snapshot channel.

Only robots that were commanded on two consecutive ticks are measured: a robot
that did not request a decision keeps coasting on its previous command, so
including it would compare a prediction against motion the command did not
produce. Exits non-zero if no such samples were collected, rather than
reporting a vacuous 0% error.

usage: python tools/check_dead_reckoning.py [--port N] [--steps N]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple

from config import Config
from kinematics import split_tick_motion

H = 100.0
PAIRS = [(1.0, 1.0), (1.0, 0.0), (0.9, 0.15), (0.5, 0.5), (0.75, 0.25), (0.0, 1.0)]


def snap(critic, arena):
    node = critic.latest[arena]["node"]
    return (node[:, 0:2].numpy() * H).copy(), np.arctan2(node[:, 3].numpy(), node[:, 2].numpy()).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None)
    ap.add_argument("--port", type=int, default=5400)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    from channels import CriticChannel
    import unity_env

    cfg = Config()
    # a heartbeat guarantees every robot is commanded regularly, so samples exist
    # even for a robot that never comes into range of anything
    unity_env.set_player_env(formations=os.path.join(os.path.dirname(__file__), "..", "..",
                                                     "data", "formations"),
                             heartbeat_ticks=1, seed_layout="corners", num_arenas=1,
                             min_bots=12, max_bots=12)
    critic, params, engine = CriticChannel(), EnvironmentParametersChannel(), EngineConfigurationChannel()
    env = UnityEnvironment(file_name=args.build or unity_env.DEFAULT_BUILD_PATH,
                           side_channels=[critic, params, engine], no_graphics=True,
                           base_port=args.port, worker_id=0, timeout_wait=120,
                           log_folder=unity_env.configure_player_logging())
    engine.set_configuration_parameters(time_scale=20.0)
    params.set_float_parameter("num_arenas", 1.0)
    env.reset()
    name = unity_env.resolve_behavior(env)

    print("prop_max_speed=%.6f  prop_wheelbase=%.6f  dt_fixed=%.4f"
          % (cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed))
    print()
    print("%-13s %-6s %-12s %-12s %-9s | %-12s %-12s %-9s"
          % ("motors", "n", "disp Unity", "disp model", "err", "dth Unity", "dth model", "err")
          + " | %-9s %-9s %s" % ("dir Unity", "dir model", "err(deg)"))

    worst, total, worst_dir = 0.0, 0, [0.0]
    for L, R in PAIRS:
        step = torch.tensor([1.0])
        m = torch.tensor([[L, R]], dtype=torch.float32)
        x, y, dth, _ = split_tick_motion(m, step, cfg.prop_max_speed, cfg.prop_wheelbase, cfg.dt_fixed)
        # split_tick_motion returns RAW arena units (belief_predict is what
        # divides by ARENA_HALF), so no rescaling here.
        pred_d, pred_t = float(torch.sqrt(x ** 2 + y ** 2)), float(dth)
        # direction of travel in the robot's frame at the START of the step.
        # belief_predict rotates the local displacement by the particle's
        # heading, so a bias here walks the position estimate sideways even
        # when the magnitude and the heading change are both exact.
        pred_dir = float(torch.atan2(y, x))

        ds_list, dt_list, dir_list, prev = [], [], [], set()
        for t in range(args.steps):
            ds, _ts = env.get_steps(name)
            vec = ds.obs[-1]
            n = vec.shape[0]
            act = np.zeros((n, 11), dtype=np.float32)
            act[:, 9], act[:, 10] = L, R
            env.set_actions(name, ActionTuple(continuous=act))
            decided = {(int(vec[i, 0]), int(vec[i, 1])) for i in range(n)}
            before = snap(critic, 0)
            env.step()
            after = snap(critic, 0)
            if before[0].shape == after[0].shape and t >= 2:
                for (a, l) in decided & prev:
                    if a != 0 or l >= after[0].shape[0]:
                        continue
                    step_vec = after[0][l] - before[0][l]
                    ds_list.append(float(np.linalg.norm(step_vec)))
                    raw = after[1][l] - before[1][l]
                    dt_list.append(float(np.arctan2(np.sin(raw), np.cos(raw))))
                    if np.linalg.norm(step_vec) > 1e-6:
                        world = np.arctan2(step_vec[1], step_vec[0])
                        rel = world - before[1][l]
                        dir_list.append(float(np.arctan2(np.sin(rel), np.cos(rel))))
            prev = decided

        if not ds_list:
            print("(%.2f,%.2f)     NO SAMPLES -- no robot was commanded on consecutive ticks" % (L, R))
            continue
        total += len(ds_list)
        real_d, real_t = float(np.mean(ds_list)), float(np.mean(dt_list))
        # relative where the quantity is real, absolute where it is ~zero: a
        # straight-line pair leaves float noise in Unity's heading, and a
        # relative error against that is meaningless
        de = abs(pred_d - real_d) / abs(real_d) * 100 if abs(real_d) > 1e-6 else 0.0
        te = abs(pred_t - real_t) / abs(real_t) * 100 if abs(real_t) > 1e-6 else 0.0
        if abs(real_t) <= 1e-6 and abs(pred_t) > 1e-6:
            te = 100.0
        real_dir = float(np.mean(dir_list)) if dir_list else 0.0
        dir_err_deg = abs(np.degrees(pred_dir - real_dir))
        worst = max(worst, de, te)
        worst_dir[0] = max(worst_dir[0], dir_err_deg)
        print("(%.2f,%.2f)     %-6d %12.7f %12.7f %+7.3f%% | %+12.7f %+12.7f %+7.3f%% | %+9.6f %+9.6f %7.4f"
              % (L, R, len(ds_list), real_d, pred_d, de, real_t, pred_t, te,
                 real_dir, pred_dir, dir_err_deg))

    env.close()
    print()
    if total == 0:
        print("FAILED: no samples collected -- the measurement, not the model, is broken")
        return 2
    print("samples: %d   worst magnitude/heading residual: %.4f%%   worst travel-direction error: %.4f deg"
          % (total, worst, worst_dir[0]))
    return 0 if (worst < 0.5 and worst_dir[0] < 0.05) else 1


if __name__ == "__main__":
    sys.exit(main())
