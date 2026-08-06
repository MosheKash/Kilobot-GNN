using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class KilobotAgent : Agent
{
    public int arenaId;
    public int localIndex;
    public int senderId;
    // The setup fact "which of the four cardinal
    // headings was I actually placed at" -- set once by SwarmManager.cs's
    // spawn loop, the same call that also physically rotates this robot to
    // match, so the two can never disagree. Radians, this project's own
    // Python-side convention (belief.CARDINAL_HEADINGS), not Unity degrees
    // -- computed in SwarmManager.cs from the exact same random draw that
    // picks the physical rotation, not derived here.
    public float spawnHeading;

    [HideInInspector] public float nearestRobotDist;
    [HideInInspector] public int lastDecisionStep;
    [HideInInspector] public float[] seedObs = new float[4];
    [HideInInspector] public float[] wallObs = new float[4];
    [HideInInspector] public float[] transmission = new float[9];
    [HideInInspector] public List<float[]> receivedMessages = new List<float[]>();
    [HideInInspector] public KilobotMovement movement;

    BufferSensorComponent bufferSensor;
    Renderer bodyRenderer;
    [HideInInspector] public Renderer ringRenderer;

    public override void Initialize()
    {
        movement = GetComponent<KilobotMovement>();
        bufferSensor = GetComponent<BufferSensorComponent>();
        // GetComponentInChildren, not GetComponent: robust regardless of
        // whether the visible mesh sits on this object or a child, which is
        // the more common layout for a robot body + wheels/parts hierarchy
        bodyRenderer = GetComponentInChildren<Renderer>();
    }

    // Two deliberate colour families:
    //
    // Every seed (this project's static, unmoving infrastructure -- see
    // SeedRobot.cs/WallSeedRobot.cs) lives in cool blue/teal, so at a glance
    // anything cool-hued in the scene is a landmark, never a robot. A
    // kilobot's own body color, by contrast, is a warm progression tracking
    // how far along its own task is -- ivory (nothing known yet) through
    // gold, amber, and a deep red (actively working, escalating urgency),
    // deliberately breaking to green -- the one cool-adjacent, universally
    // "done" color in the whole system -- only at genuine completion. No
    // hue is reused between the two families, so body and ring colors
    // together read as one system, not independently-chosen accents.
    //
    // The ring (CommRadiusIndicator, if KILOBOT_SHOW_RADIUS attached one --
    // ringRenderer is null otherwise, guarded below) takes the same colour as
    // the body on every call. This
    // phase simply weighs that trade-off the other way.
    //
    // Called from SwarmManager.SetRobotStates, itself only ever invoked
    // when the python side has KILOBOT_ORACLE_SEND_VISUAL_STATE on --
    // never touches observations, actions, or reward.
    //   0 = straight-line exploring, no lock yet    -> warm ivory
    //   1 = following a wall                        -> amber
    //   2 = orbiting a corner / turning              -> gold
    //   3 = committed, heading to its assigned point -> deep red
    //   4 = arrived, stopped                          -> green
    public void SetVisualState(int state)
    {
        if (bodyRenderer == null)
        {
            return;
        }
        Color c;
        switch (state)
        {
            case 1: c = new Color(0.95f, 0.55f, 0.12f); break;   // amber: sustained, active search
            case 2: c = new Color(1.00f, 0.82f, 0.35f); break;   // gold: brief, lighter than amber -- a quick transition, not a destination
            case 3: c = new Color(0.86f, 0.20f, 0.25f); break;   // deep red: committed, escalating urgency
            case 4: c = new Color(0.20f, 0.72f, 0.38f); break;   // green: the one deliberate break in the warm progression -- done
            default: c = new Color(0.96f, 0.94f, 0.88f); break;  // warm ivory: nothing known yet, not stark white
        }
        // .material, not .sharedMaterial: creates a per-instance copy the
        // first time it's accessed on this renderer, same reasoning
        // SwarmManager.cs's own floorRenderer comment already documents --
        // using .sharedMaterial here would recolor every other object
        // sharing the same material asset, not just this one kilobot.
        bodyRenderer.material.color = c;
        // Preserve the ring's existing alpha and change only its RGB. Every
        // Color literal in the switch above omits the alpha argument, which C#
        // defaults to 1.0 -- correct for a solid body, but writing c straight
        // onto the ring would overwrite CommRadiusIndicator.ALPHA with full
        // opacity on every call, making the ring permanently opaque no matter
        // what that constant said.
        if (ringRenderer != null)
        {
            float ringAlpha = ringRenderer.material.color.a;
            ringRenderer.material.color = new Color(c.r, c.g, c.b, ringAlpha);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation((float)arenaId);
        sensor.AddObservation((float)localIndex);
        for (int k = 0; k < seedObs.Length; k++)
        {
            sensor.AddObservation(seedObs[k]);
        }
        for (int k = 0; k < wallObs.Length; k++)
        {
            sensor.AddObservation(wallObs[k]);
        }
        // APPENDED, not inserted: every existing index into this fixed-length
        // part (Python's vector[:, 0]=arenaId, [:, 1]=localIndex,
        // [:, 2:2+SEED_SIZE]=seedObs, then the wallObs slice) must stay where
        // it is. Only this column exists past the end.
        sensor.AddObservation(spawnHeading);
        for (int m = 0; m < receivedMessages.Count; m++)
        {
            bufferSensor.AppendObservation(receivedMessages[m]);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        ActionSegment<float> continuous = actions.ContinuousActions;
        for (int j = 0; j < 9; j++)
        {
            transmission[j] = Mathf.Clamp(continuous[j], -1f, 1f);
        }
        movement.leftMotor = Mathf.Clamp01(continuous[9]);
        movement.rightMotor = Mathf.Clamp01(continuous[10]);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuous = actionsOut.ContinuousActions;
        for (int j = 0; j < continuous.Length; j++)
        {
            continuous[j] = 0f;
        }
    }
}
