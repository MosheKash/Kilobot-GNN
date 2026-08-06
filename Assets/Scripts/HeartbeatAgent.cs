using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class HeartbeatAgent : Agent
{
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(0f);
    }
}
