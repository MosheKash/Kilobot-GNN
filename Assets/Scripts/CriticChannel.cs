using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;

public class CriticChannel : SideChannel
{
    const int KIND_IMAGE = 0;
    const int KIND_RESET = 1;
    const int KIND_ROBOT_STATES = 2;
    const int KIND_SET_POSES = 3;

    Dictionary<int, SwarmManager> arenas = new Dictionary<int, SwarmManager>();

    public CriticChannel()
    {
        ChannelId = new Guid("d3f1a2b4-1c2d-4e5f-8a9b-0c1d2e3f4a5b");
    }

    public void Register(int arenaId, SwarmManager manager)
    {
        arenas[arenaId] = manager;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        int kind = msg.ReadInt32();
        int arenaId = msg.ReadInt32();

        if (kind == KIND_ROBOT_STATES)
        {
            int count = msg.ReadInt32();
            List<int> states = new List<int>(count);
            for (int i = 0; i < count; i++)
            {
                states.Add(msg.ReadInt32());
            }
            if (arenas.ContainsKey(arenaId))
            {
                arenas[arenaId].SetRobotStates(states);
            }
            return;
        }

        if (kind == KIND_SET_POSES)
        {
            // Teleports named robots to exact poses. Exists for tests: the
            // python-side replica let a test write positions straight into its
            // arena array, and a real player has no equivalent, so geometry-
            // dependent assertions (this robot is out of IR range of that one,
            // this robot sits exactly on the target) had nothing to stand on.
            // Not used by training.
            int count = msg.ReadInt32();
            List<KilobotPose> poses = new List<KilobotPose>(count);
            for (int i = 0; i < count; i++)
            {
                KilobotPose p;
                p.localIndex = msg.ReadInt32();
                p.x = msg.ReadFloat32();
                p.z = msg.ReadFloat32();
                p.heading = msg.ReadFloat32();
                poses.Add(p);
            }
            if (arenas.ContainsKey(arenaId))
            {
                arenas[arenaId].SetPosesCommand(poses);
            }
            return;
        }

        int imageId = msg.ReadInt32();

        if (!arenas.ContainsKey(arenaId))
        {
            return;
        }

        if (kind == KIND_RESET)
        {
            arenas[arenaId].ResetCommand(imageId);
        }
        else
        {
            arenas[arenaId].SetImageCommand(imageId);
        }
    }

    public void SendSnapshot(int arenaId, int envStep, int m, int e,
                             List<float> node, List<float> edgeIndex, List<float> edgeAttr)
    {
        OutgoingMessage msg = new OutgoingMessage();
        msg.WriteInt32(arenaId);
        msg.WriteInt32(envStep);
        msg.WriteInt32(m);
        msg.WriteInt32(e);
        msg.WriteFloatList(node);
        msg.WriteFloatList(edgeIndex);
        msg.WriteFloatList(edgeAttr);
        QueueMessageToSend(msg);
        msg.Dispose();
    }
}
