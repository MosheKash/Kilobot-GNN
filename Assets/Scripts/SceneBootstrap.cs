using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class SceneBootstrap : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject swarmManagerPrefab;
    public GameObject heartbeatPrefab;

    [Header("Shared")]
    public ImageLibrary imageLibrary;

    [Header("Layout")]
    public int defaultArenas = 4;
    public float arenaSpacing = 260f;
    public int framesPerStep = 4;

    CriticChannel channel;
    List<SwarmManager> managers = new List<SwarmManager>();

    void Awake()
    {
        channel = new CriticChannel();
        SideChannelManager.RegisterSideChannel(channel);
    }

    void Start()
    {
        int n = ResolveArenaCount();
        Debug.Log("SceneBootstrap: spawning " + n + " arenas");
        SpawnArenas(n);

        GameObject beatGo = Instantiate(heartbeatPrefab);
        HeartbeatAgent beat = beatGo.GetComponent<HeartbeatAgent>();

        StepDriver driver = gameObject.AddComponent<StepDriver>();
        driver.framesPerStep = framesPerStep;
        driver.Configure(managers, beat);
    }

    int ResolveArenaCount()
    {
        string env = System.Environment.GetEnvironmentVariable("KILOBOT_NUM_ARENAS");
        if (!string.IsNullOrEmpty(env))
        {
            int parsed;
            if (int.TryParse(env, out parsed) && parsed > 0)
            {
                return parsed;
            }
        }
        return (int)Academy.Instance.EnvironmentParameters.GetWithDefault("num_arenas", defaultArenas);
    }

    void SpawnArenas(int n)
    {
        int columns = Mathf.CeilToInt(Mathf.Sqrt(n));
        for (int i = 0; i < n; i++)
        {
            int row = i / columns;
            int col = i % columns;
            Vector3 pos = new Vector3(col * arenaSpacing, 0f, row * arenaSpacing);
            GameObject go = Instantiate(swarmManagerPrefab, pos, Quaternion.identity);
            SwarmManager manager = go.GetComponent<SwarmManager>();
            manager.Configure(i, channel, imageLibrary);
            manager.SpawnInitial();
            channel.Register(i, manager);
            managers.Add(manager);
        }
    }

    void OnDestroy()
    {
        if (channel != null)
        {
            SideChannelManager.UnregisterSideChannel(channel);
        }
    }
}
