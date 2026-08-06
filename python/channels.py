"""The CriticChannel side channel: the Python end of the custom Unity protocol.

Inbound, the per-arena graph snapshot (node features, edge index, edge attrs)
that the reward and critic are computed from. Outbound, four commands:
set-image, reset-arena, set-visual-state (observation only), and set-poses
(tests only). Message kinds must stay in sync with Assets/Scripts/CriticChannel.cs.
"""

import uuid
import time
import torch

from kilobot_gnn import NODE_FEATURES

try:
    from mlagents_envs.side_channel.side_channel import SideChannel
    from mlagents_envs.side_channel.incoming_message import IncomingMessage
    from mlagents_envs.side_channel.outgoing_message import OutgoingMessage
    HAVE_MLAGENTS = True
except Exception:
    SideChannel = object
    IncomingMessage = object
    OutgoingMessage = object
    HAVE_MLAGENTS = False

CHANNEL_GUID = uuid.UUID("d3f1a2b4-1c2d-4e5f-8a9b-0c1d2e3f4a5b")
KIND_IMAGE = 0
KIND_RESET = 1
KIND_ROBOT_STATES = 2
KIND_SET_POSES = 3


class CriticChannel(SideChannel):
    def __init__(self):
        super().__init__(CHANNEL_GUID)
        self.latest = {}
        self.parse_seconds = 0.0
        self.message_count = 0

    def on_message_received(self, msg):
        t = time.perf_counter()
        arena_id = msg.read_int32()
        env_step = msg.read_int32()
        m = msg.read_int32()
        e = msg.read_int32()

        node = torch.tensor(msg.read_float32_list(), dtype=torch.float32).reshape(m, NODE_FEATURES)
        edge_index = torch.tensor(msg.read_float32_list(), dtype=torch.float32).reshape(2, e).long()
        edge_attr = torch.tensor(msg.read_float32_list(), dtype=torch.float32).reshape(e, 1)

        self.latest[arena_id] = {
            "env_step": env_step,
            "node": node,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }
        self.parse_seconds = self.parse_seconds + (time.perf_counter() - t)
        self.message_count = self.message_count + 1

    def pop_timing(self):
        s = self.parse_seconds
        c = self.message_count
        self.parse_seconds = 0.0
        self.message_count = 0
        return s, c

    def send_image(self, arena_id, image_id):
        out = OutgoingMessage()
        out.write_int32(KIND_IMAGE)
        out.write_int32(arena_id)
        out.write_int32(image_id)
        super().queue_message_to_send(out)

    def send_reset(self, arena_id, image_id):
        out = OutgoingMessage()
        out.write_int32(KIND_RESET)
        out.write_int32(arena_id)
        out.write_int32(image_id)
        super().queue_message_to_send(out)

    def send_robot_states(self, arena_id, states):
        # states: one int visual-state code per kilobot in this arena, same
        # order as SwarmManager's own kilobots list (localIndex). Purely
        # observational -- never read back by on_message_received, never
        # touches training in any way; only ever sent when
        # cfg.oracle_send_visual_state is explicitly on (trainer.py).
        out = OutgoingMessage()
        out.write_int32(KIND_ROBOT_STATES)
        out.write_int32(arena_id)
        out.write_int32(len(states))
        for s in states:
            out.write_int32(int(s))
        super().queue_message_to_send(out)

    def send_poses(self, arena_id, poses):
        # poses: an iterable of (local_index, x, z, heading).
        #
        # x/z are ARENA-LOCAL and unnormalized -- the same units belief.py works
        # in, i.e. within [-ARENA_HALF, ARENA_HALF] -- NOT the [-1, 1] pair that
        # comes back in node[:, 0:2], which is those coordinates divided by
        # ARENA_HALF. heading is radians, python's convention: direction
        # (cos h, sin h) in (x, z). SwarmManager.ApplyPendingPoses converts.
        #
        # Test-only, like send_robot_states: nothing in training calls this. A
        # pose sent in the same packet as send_reset lands AFTER the respawn, so
        # local_index refers to the newly spawned robots.
        out = OutgoingMessage()
        out.write_int32(KIND_SET_POSES)
        out.write_int32(arena_id)
        poses = list(poses)
        out.write_int32(len(poses))
        for local, x, z, heading in poses:
            out.write_int32(int(local))
            out.write_float32(float(x))
            out.write_float32(float(z))
            out.write_float32(float(heading))
        super().queue_message_to_send(out)

    def snapshot(self, arena_id):
        return self.latest.get(arena_id)
