"""Measure prop_max_speed and prop_wheelbase against a live Unity player.

These two Config constants are the kinematics the belief filter dead-reckons
with; Unity's KilobotMovement.cs does not read them, so they are kept in sync by
hand and go stale silently. This is the tool that produced the current values --
config.py and docs/code-history.md both name it.

Drives fixed motor commands, reads the true pose back off the snapshot channel,
and solves for the speed and wheelbase that reproduce the observed motion.

usage: python tools/calibrate_kinematics.py [--build PATH] [--port N] [--graphics]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from channels import CriticChannel

H = 100.0


def snap_pose(critic, arena):
    node = critic.latest[arena]["node"]
    pos = node[:, 0:2].numpy() * H
    heading = np.arctan2(node[:, 3].numpy(), node[:, 2].numpy())
    return pos, heading


def main():
    critic = CriticChannel()
    params = EnvironmentParametersChannel()
    engine = EngineConfigurationChannel()
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default = None, help = "path to the Unity player")
    ap.add_argument("--port", type = int, default = 5350)
    ap.add_argument("--graphics", action = "store_true", help = "open a window (needs a display)")
    args = ap.parse_args()

    import unity_env
    env = UnityEnvironment(file_name = args.build or unity_env.DEFAULT_BUILD_PATH,
                           side_channels = [critic, params, engine],
                           no_graphics = not args.graphics, base_port = args.port,
                           worker_id = 0, timeout_wait = 120,
                           log_folder = unity_env.configure_player_logging())
    engine.set_configuration_parameters(time_scale = 50.0)
    env.reset()
    name = list(env.behavior_specs.keys())[0]

    def drive(left, right, steps, warm = 4):
        disp = []
        dth = []
        prev = {}
        for t in range(steps):
            ds, ts = env.get_steps(name)
            vec = ds.obs[-1]
            n = vec.shape[0]
            act = np.zeros((n, 11), dtype = np.float32)
            act[:, 9] = left
            act[:, 10] = right
            from mlagents_envs.base_env import ActionTuple
            env.set_actions(name, ActionTuple(continuous = act))
            decided = {}
            for i in range(n):
                decided[(int(vec[i, 0]), int(vec[i, 1]))] = True
            poses = {}
            for a in sorted(critic.latest.keys()):
                poses[a] = snap_pose(critic, a)
            env.step()
            for a in sorted(critic.latest.keys()):
                new_pos, new_head = snap_pose(critic, a)
                old = poses.get(a)
                if old is None or old[0].shape[0] != new_pos.shape[0]:
                    continue
                for l in range(new_pos.shape[0]):
                    if (a, l) in decided and (a, l) in prev and t >= warm:
                        disp.append(np.linalg.norm(new_pos[l] - old[0][l]))
                        d = new_head[l] - old[1][l]
                        dth.append(np.arctan2(np.sin(d), np.cos(d)))
            prev = decided
        return np.array(disp), np.array(dth)

    disp_f, dth_f = drive(1.0, 1.0, 16)
    disp_s, dth_s = drive(1.0, 0.0, 16)
    disp_z, dth_z = drive(0.0, 0.0, 10)

    print("forward: disp/step mean %.4f std %.4f  dtheta mean %.5f" %
          (disp_f.mean(), disp_f.std(), dth_f.mean()))
    print("spin(1,0): disp/step mean %.4f  dtheta/step mean %.5f std %.5f" %
          (disp_s.mean(), dth_s.mean(), dth_s.std()))
    print("stop: disp/step mean %.4f" % disp_z.mean())

    # was 0.05, independent of cfg.dt_fixed and
    # never actually matching Unity's real Fixed Timestep -- confirmed
    # directly from the project's own ProjectSettings/TimeManager.asset:
    # 0.02. This script's own measurements (disp_f, dth_s, the raw,
    # dt-independent per-step displacement/rotation Unity actually
    # produced) are unaffected by this constant; only the conversion to
    # per-second rates (prop_max_speed, prop_wheelbase) below depends on
    # it matching reality.
    dt = 0.02
    vmax = disp_f.mean() / dt
    omega = abs(dth_s.mean()) / dt
    wheelbase = vmax / omega
    print("fitted: prop_max_speed %.3f  prop_wheelbase %.3f  (dt_fixed %.2f)" % (vmax, wheelbase, dt))
    print("spin chord check: v = vmax/2 -> expected disp %.4f measured %.4f" %
          (0.5 * vmax * dt, disp_s.mean()))
    env.close()


if __name__ == "__main__":
    main()
