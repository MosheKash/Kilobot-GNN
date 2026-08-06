using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class StepDriver : MonoBehaviour
{
    public int framesPerStep = 4;

    List<SwarmManager> managers = new List<SwarmManager>();
    HeartbeatAgent heartbeat;

    public void Configure(List<SwarmManager> swarmManagers, HeartbeatAgent beat)
    {
        managers = swarmManagers;
        heartbeat = beat;
    }

    void Start()
    {
        Academy.Instance.AgentPreStep += OnAgentPreStep;
    }

    void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            Academy.Instance.AgentPreStep -= OnAgentPreStep;
        }
    }

    void OnAgentPreStep(int academyStep)
    {
        if (academyStep % framesPerStep != 0)
        {
            return;
        }
        for (int i = 0; i < managers.Count; i++)
        {
            managers[i].Tick();
        }
        heartbeat.RequestDecision();
    }
}
