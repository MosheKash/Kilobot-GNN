using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

// One robot's requested pose, as sent over CriticChannel's KIND_SET_POSES.
// x/z are ARENA-LOCAL and unnormalized (the same units as arenaHalfExtent, i.e.
// what belief.py works in), NOT the [-1,1] normalized pair AppendNode reports.
// heading is in radians in the python convention: direction (cos h, sin h) in
// (x, z), which is why SetPosesCommand converts with yaw = 90deg - h rather
// than using it as a yaw directly (see that method).
public struct KilobotPose
{
    public int localIndex;
    public float x;
    public float z;
    public float heading;
}

public class SwarmManager : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject kilobotPrefab;
    public GameObject seedPrefab;
    public GameObject wallSeedPrefab;
    // Optional: a duplicate of wallSeedPrefab with a distinct body material,
    // for near-corner wall seeds (see WallSeedRobot.nearCorner). The ring gets
    // its distinct colour from code; a body material can only be set in the
    // Editor, which is what this second prefab is for. Left null, near-corner
    // seeds fall back to wallSeedPrefab -- ring colour is still correct, only
    // the body distinction is missing.
    public GameObject wallSeedNearCornerPrefab;

    [Header("Visualization")]
    // assign this arena's own floor/ground plane here to have it textured with the
    // current target formation image, so it is visually obvious whether kilobots are
    // sitting on the shape or not. Purely visual; leave unassigned to skip it entirely
    public Renderer floorRenderer;

    // resolved once in SpawnInitial() and reused for every spawned kilobot -- see that
    // method's own comment for why a "Kilobot" layer must exist in Project Settings >
    // Tags and Layers for this to do anything. -1 (LayerMask.NameToLayer's not-found
    // value) means collisions between kilobots are NOT being disabled.
    int kilobotLayer = -1;

    [Header("Population")]
    public int minKilobots = 20;
    public int maxKilobots = 50;
    // MUST match belief.py's KNOWN_START_HEADING and oracle_known_start_heading
    // exactly. The point of telling the filter a spawn heading is that it is a
    // physically-real prior; a mismatch replaces it with a confidently wrong
    // one. False gives a uniformly random spawn heading.
    public bool knownStartHeading = true;

    [Header("Arena")]
    public float arenaHalfExtent = 100f;
    public float spawnMargin = 10f;
    // minimum center-to-center distance enforced between every kilobot and
    // every other kilobot/seed/wall-seed at spawn time. Tune this to your
    // prefabs' actual collider size: too small and colliders can still
    // overlap and launch apart on spawn; too large and spawning many
    // kilobots gets slow or can fail to place all of them.
    public float minSpawnSeparation = 5f;

    const float IR_RANGE = 7f; // matches the real Kilobot platform's short-range IR (project units are cm)
    const int SEED_COUNT = 4;   // the origin point is gone; four corners remain
    const int WALL_SIZE = 4;
    // worst case a robot at the true boundary sees sqrt(WALL_SEED_INSET^2 + (WALL_SPACING/2)^2)
    // to the nearest wall seed; this must stay <= IR_RANGE. At INSET=5, IR_RANGE=7 that bounds
    // spacing to <= 9.8, so 8 leaves comfortable margin (worst case ~6.4). Must match
    // belief.WALL_SPACING on the python side.
    const float WALL_SPACING = 8f;
    // gap from the true boundary so wall seeds don't spawn embedded in the
    // physical wall barrier. Must match belief.WALL_SEED_INSET on the python side.
    const float WALL_SEED_INSET = 5f;
    const int MESSAGE_SIZE = 9;

    // Heartbeat: a robot that has gone this many decision ticks without any event
    // still gets a decision, so an isolated robot can re-steer instead of coasting
    // ballistically into a wall forever. 0 disables it (the historical semantics).
    // Read from KILOBOT_HEARTBEAT_TICKS, mirrored by config.py heartbeat_ticks on
    // the python side; launch.py sets both from the same variable.
    int heartbeatTicks;

    // KILOBOT_SEED_LAYOUT: "corners" (origin + four far corners) is the only
    // supported layout -- always use this. "cluster" (origin, (22,0), (11,19)
    // + two far corners) also exists below for backward compatibility only:
    // DEPRECATED, DO NOT USE, not run in a long time.
    // Must match the python side's belief.SEED_LAYOUTS and KILOBOT_SEED_LAYOUT.
    string seedLayout;

    bool showCommRadius;
    bool showTargetFloor;
    int floorRotationSteps;
    // KILOBOT_DEBUG_WALL_SEEDS -- gates both WALL_SEED_DUMP (static geometry,
    // once per arena at spawn) and WALL_SCAN_LIVE (the first 20 live sightings).
    bool debugWallSeeds;

    int arenaId;
    int imageId;
    int envStep;
    CriticChannel channel;
    ImageLibrary imageLibrary;

    List<KilobotAgent> kilobots = new List<KilobotAgent>();
    List<SeedRobot> seeds = new List<SeedRobot>();
    List<WallSeedRobot> wallSeeds = new List<WallSeedRobot>();

    bool pendingReset;
    bool pendingImage;
    int pendingImageId;
    int wallScanLogCount;

    List<KilobotPose> pendingPoses = new List<KilobotPose>();
    int poseWarnCount;

    // Control over the RNG that spawns the KILOBOT SWARM. Named for the swarm,
    // not "seed", because this project already uses that word for the landmark
    // robots -- KILOBOT_SEED_LAYOUT places the corner seeds, and neither those
    // nor the wall seeds are affected here at all: SpawnSeeds and SpawnWallSeeds
    // use fixed coordinates and draw no randomness, and SeedSwarmRng runs after
    // both of them. What this varies is exactly the four draws SpawnKilobots
    // makes: population count, each robot's position, each robot's cardinal
    // heading, and the heartbeat phase stagger. Those are also the only
    // UnityEngine.Random calls anywhere in Assets/Scripts.
    //
    // Unseeded by default -- Random keeps whatever state the engine gave it,
    // exactly what every run did before this existed.
    //
    // There are two knobs because there are two different jobs, and one value
    // cannot do both:
    //
    //   KILOBOT_SWARM_RNG  makes a whole RUN replayable. Episodes still differ
    //                      from one another (the value is mixed with a per-arena
    //                      respawn counter), because a fixed value should replay
    //                      a run, not collapse every episode into a copy of the
    //                      first. Read at spawn-time setup, and the only one that
    //                      can reach the very FIRST spawn: SpawnInitial runs from
    //                      SceneBootstrap.Start, before the parameters channel has
    //                      necessarily delivered anything -- the same problem that
    //                      makes KILOBOT_NUM_ARENAS exist.
    //
    //                      Deliberately NOT launch.py's KILOBOT_SEED, which seeds
    //                      torch and numpy: the player inherits this process's
    //                      environment, so sharing the name would silently seed
    //                      the swarm RNG for anyone who only wanted reproducible
    //                      network init -- and would hand every parallel player
    //                      in a multi-instance BC run the identical spawn stream,
    //                      since they all inherit the same value.
    //
    //   "swarm_rng"        on the EnvironmentParametersChannel, pins the NEXT
    //                      spawn exactly: same value, same arena, every time,
    //                      including two respawns of one long-lived player. This
    //                      is what a test needs -- the session-scoped fixture
    //                      reuses one player across tests, so anything mixed with
    //                      a respawn counter would hand two tests asking for the
    //                      same value two different arenas. Overrides the env var
    //                      whenever it has been delivered at all; -1 means
    //                      "explicitly unseeded", the sentinel below means
    //                      "never sent, fall back to KILOBOT_SWARM_RNG".
    const float SWARM_RNG_UNSET = -2f;
    int swarmRngEnv = -1;
    int spawnGeneration;

    List<float> nodeBuffer = new List<float>();
    List<float> sourceBuffer = new List<float>();
    List<float> targetBuffer = new List<float>();
    List<float> attrBuffer = new List<float>();
    List<float> edgeIndexBuffer = new List<float>();

    public void Configure(int id, CriticChannel ch, ImageLibrary lib)
    {
        arenaId = id;
        channel = ch;
        imageLibrary = lib;
    }

    public void SpawnInitial()
    {
        // resolves once per arena; harmless to repeat if multiple SwarmManager
        // instances each call this (Physics.IgnoreLayerCollision just sets one
        // flag in the project-wide, not per-arena, collision matrix -- setting
        // it more than once has no extra effect). Requires a "Kilobot" layer
        // to exist (Edit > Project Settings > Tags and Layers -> add "Kilobot"
        // in any empty User Layer slot) -- there is no way to create a layer
        // purely from code, Unity layers are a fixed set of 32 project-level
        // slots. This does NOT need the kilobotPrefab asset itself to be set
        // to that layer in the Inspector; SpawnKilobots below assigns it to
        // every spawned instance directly, so the prefab can stay on whatever
        // layer it already is.
        kilobotLayer = LayerMask.NameToLayer("Kilobot");
        if (kilobotLayer < 0)
        {
            Debug.LogError("SwarmManager: no 'Kilobot' layer found (Project Settings > " +
                            "Tags and Layers). Kilobot-kilobot collisions will NOT be " +
                            "disabled; everything else spawns and runs normally.");
        }
        else
        {
            // only ignores collisions WITHIN this layer -- kilobots still collide
            // normally with anything on a different layer (floor, arena walls,
            // seeds, wall seeds), since those were never added to this pair
            Physics.IgnoreLayerCollision(kilobotLayer, kilobotLayer, true);
        }
        heartbeatTicks = ParseIntEnv("KILOBOT_HEARTBEAT_TICKS", 0);
        // Swarm size, overridable from Python the same way heartbeatTicks is.
        // Previously Inspector-only, which meant --min-bots/--max-bots on the
        // training drivers were silently inert against a real build, and no
        // Python-side test could ask for a specific swarm size at all. Defaults
        // fall back to whatever the prefab was authored with, so an invocation
        // that sets neither behaves exactly as before.
        minKilobots = ParseIntEnv("KILOBOT_MIN_BOTS", minKilobots);
        maxKilobots = ParseIntEnv("KILOBOT_MAX_BOTS", maxKilobots);
        if (maxKilobots < minKilobots)
        {
            // Random.Range(min, max+1) would otherwise silently return min and
            // hide the misconfiguration; say so instead.
            Debug.LogWarning("KILOBOT_MAX_BOTS (" + maxKilobots + ") is below KILOBOT_MIN_BOTS ("
                             + minKilobots + ") -- clamping max up to min.");
            maxKilobots = minKilobots;
        }
        // NOT ParseIntEnv: that rejects negatives (returning the fallback), and
        // -1 is the meaningful "leave the RNG alone" value here.
        swarmRngEnv = ParseSignedIntEnv("KILOBOT_SWARM_RNG", -1);
        seedLayout = ParseStringEnv("KILOBOT_SEED_LAYOUT", "corners");
        showCommRadius = ParseBoolEnv("KILOBOT_SHOW_RADIUS", false);
        // defaults on, unlike showCommRadius: assigning floorRenderer in the Inspector is
        // already the deliberate opt-in (nobody does that by accident), and the expensive
        // part (loading/baking the image) happens regardless via imageLibrary.Sample() for
        // the reward, so gating this behind a second flag only adds a confusing way for it
        // to silently do nothing despite a correct Inspector setup. Set to false to disable
        // while keeping floorRenderer assigned
        showTargetFloor = ParseBoolEnv("KILOBOT_SHOW_TARGET_FLOOR", true);
        // Rotates the floor texture 180 degrees. Purely visual: no effect on
        // BakeImage, the reward, or the oracle. 180 is direction-agnostic, so
        // unlike the agents' 90-degree rotation this does not depend on getting
        // a rotational-direction convention right.
        floorRotationSteps = ((ParseIntEnv("KILOBOT_FLOOR_ROTATION_STEPS", 2) % 4) + 4) % 4;
        // KILOBOT_DEBUG_WALL_SEEDS: the two wall-seed dumps below. Off by
        // default -- WALL_SEED_DUMP alone is 104 lines PER ARENA, so a 16-arena
        // run opened with ~1700 lines of the same static geometry before a
        // single training number appeared. Turn it on when investigating wall
        // seeds specifically; the one-line summary is always printed instead.
        debugWallSeeds = ParseBoolEnv("KILOBOT_DEBUG_WALL_SEEDS", false);
        SpawnSeeds();
        SpawnWallSeeds();
        SeedSwarmRng();
        SpawnKilobots();
        Debug.Log("SwarmManager arena " + arenaId + ": " + kilobots.Count + " kilobots, "
                  + seeds.Count + " seeds, " + wallSeeds.Count + " wall seeds, layout="
                  + seedLayout + ", heartbeat=" + heartbeatTicks);
        LogWallSeedPositions();
    }

    // Re-seeds UnityEngine.Random immediately before a spawn, so the spawn is
    // reproducible regardless of how much randomness anything else consumed
    // beforehand. Seeding once at startup would not be enough: SpawnKilobots
    // runs again on every episode reset, and by then the shared global RNG has
    // been advanced by an unknown amount.
    void SeedSwarmRng()
    {
        spawnGeneration = spawnGeneration + 1;
        int pinned = Mathf.RoundToInt(SWARM_RNG_UNSET);
        if (Academy.IsInitialized)
        {
            pinned = Mathf.RoundToInt(
                Academy.Instance.EnvironmentParameters.GetWithDefault("swarm_rng", SWARM_RNG_UNSET));
        }
        // arenaId is mixed into both branches so parallel arenas don't all spawn
        // the identical swarm. Strides are large primes, so no (arena,
        // generation) pair can land on another's stream.
        if (pinned > Mathf.RoundToInt(SWARM_RNG_UNSET))
        {
            if (pinned >= 0)
            {
                Random.InitState(pinned + arenaId * 7919);
            }
            return;
        }
        if (swarmRngEnv >= 0)
        {
            Random.InitState(swarmRngEnv + arenaId * 7919 + spawnGeneration * 104729);
        }
    }

    // Prints every spawned wall seed's actual runtime world position and side,
    // verifying what could otherwise only be inferred from reading the spawn
    // code. Added for a specific investigation, and originally unconditional on
    // the reasoning that a one-time startup dump costs nothing -- which held for
    // one arena and stopped holding at sixteen: 104 lines each, all of it the
    // same static geometry, ahead of every actual training number. Now behind
    // KILOBOT_DEBUG_WALL_SEEDS, with the always-on summary in SpawnInitial
    // covering the "did the wall seeds spawn at all" question this answered
    // incidentally.
    void LogWallSeedPositions()
    {
        if (!debugWallSeeds)
        {
            return;
        }
        Debug.Log("WALL_SEED_DUMP arenaHalfExtent=" + arenaHalfExtent + " transform.position=" + transform.position);
        for (int s = 0; s < wallSeeds.Count; s++)
        {
            Vector3 p = wallSeeds[s].transform.position;
            Vector3 local = p - transform.position;
            Debug.Log("WALL_SEED_DUMP idx=" + s + " side=" + wallSeeds[s].side +
                      " world_pos=(" + p.x + ", " + p.y + ", " + p.z + ")" +
                      " local_pos=(" + local.x + ", " + local.y + ", " + local.z + ")");
        }
    }

    static int ParseIntEnv(string name, int fallback)
    {
        string env = System.Environment.GetEnvironmentVariable(name);
        int parsed;
        if (!string.IsNullOrEmpty(env) && int.TryParse(env, out parsed) && parsed >= 0)
        {
            return parsed;
        }
        return fallback;
    }

    static int ParseSignedIntEnv(string name, int fallback)
    {
        string env = System.Environment.GetEnvironmentVariable(name);
        int parsed;
        if (!string.IsNullOrEmpty(env) && int.TryParse(env, out parsed))
        {
            return parsed;
        }
        return fallback;
    }

    static string ParseStringEnv(string name, string fallback)
    {
        string env = System.Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(env))
        {
            return fallback;
        }
        return env.Trim().ToLowerInvariant();
    }

    static bool ParseBoolEnv(string name, bool fallback)
    {
        string env = System.Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(env))
        {
            return fallback;
        }
        string v = env.Trim().ToLowerInvariant();
        return v == "1" || v == "true" || v == "yes" || v == "on";
    }

    void SpawnSeeds()
    {
        float c = arenaHalfExtent - spawnMargin;
        // "cluster" is DEPRECATED, DO NOT USE -- kept only for backward
        // compatibility, not run in a long time. "corners" (the else branch
        // below) is the only supported layout.
        if (seedLayout == "cluster")
        {
            AddSeed(SeedType.UpperLeft, new Vector3(22f, 0f, 0f));
            AddSeed(SeedType.UpperRight, new Vector3(11f, 0f, 19f));
            AddSeed(SeedType.LowerLeft, new Vector3(-c, 0f, c));
            AddSeed(SeedType.LowerRight, new Vector3(c, 0f, -c));
        }
        else
        {
            AddSeed(SeedType.UpperLeft, new Vector3(-c, 0f, c));
            AddSeed(SeedType.UpperRight, new Vector3(c, 0f, c));
            AddSeed(SeedType.LowerLeft, new Vector3(-c, 0f, -c));
            AddSeed(SeedType.LowerRight, new Vector3(c, 0f, -c));
        }
    }

    void AddSeed(SeedType type, Vector3 localPos)
    {
        GameObject go = Instantiate(seedPrefab, transform.position + localPos, Quaternion.identity, transform);
        SeedRobot seed = go.GetComponent<SeedRobot>();
        seed.seedType = type;
        seeds.Add(seed);
        if (showCommRadius)
        {
            CommRadiusIndicator.Attach(go, IR_RANGE, CommRadiusIndicator.Kind.Seed);
        }
    }

    // How many seeds at each end of a wall broadcast their exact position, rather
    // than only a band reading. Python never mirrors this number -- it reads
    // whichever positions actually arrive over the message channel, so the two
    // sides cannot disagree about which seeds qualify (see
    // observation._resolve_wall_seed_xy).
    const int WALL_SEED_NEAR_CORNER_COUNT = 3;

    void SpawnWallSeeds()
    {
        if (wallSeedPrefab == null)
        {
            Debug.LogError("SwarmManager.wallSeedPrefab is not assigned in the Inspector; " +
                           "skipping wall seed spawn (kilobots will still spawn normally).");
            return;
        }
        float half = arenaHalfExtent;
        float inset = half - WALL_SEED_INSET;
        List<float> coords = new List<float>();
        for (float c = -half; c <= half + 0.01f; c += WALL_SPACING)
        {
            coords.Add(c);
        }
        for (int i = 0; i < coords.Count; i++)
        {
            // True for exactly the first/last
            // WALL_SEED_NEAR_CORNER_COUNT indices along this wall's own
            // sweep -- both of its own two ends, i.e. both of its own two
            // corners -- matching python's own
            // _wall_seed_near_corner_masks exactly, index for index.
            bool nearCorner = i < WALL_SEED_NEAR_CORNER_COUNT || i >= coords.Count - WALL_SEED_NEAR_CORNER_COUNT;
            float x = coords[i];
            AddWallSeed(WallSide.North, new Vector3(x, 0f, inset), nearCorner);
            AddWallSeed(WallSide.South, new Vector3(x, 0f, -inset), nearCorner);
            AddWallSeed(WallSide.East, new Vector3(inset, 0f, x), nearCorner);
            AddWallSeed(WallSide.West, new Vector3(-inset, 0f, x), nearCorner);
        }
    }

    void AddWallSeed(WallSide side, Vector3 localPos, bool nearCorner)
    {
        // Falls back to the ordinary prefab
        // whenever the near-corner one isn't assigned in the Inspector --
        // see wallSeedNearCornerPrefab's own comment above for why this
        // is a real, permanent fallback, not a placeholder to remove later.
        GameObject prefab = (nearCorner && wallSeedNearCornerPrefab != null) ? wallSeedNearCornerPrefab : wallSeedPrefab;
        GameObject go = Instantiate(prefab, transform.position + localPos, Quaternion.identity, transform);
        WallSeedRobot seed = go.GetComponent<WallSeedRobot>();
        if (seed == null)
        {
            Debug.LogError("The wall seed prefab used for this spawn (" + prefab.name +
                           ") has no WallSeedRobot component attached; destroying the spawned object.");
            Destroy(go);
            return;
        }
        seed.nearCorner = nearCorner;
        // wall seeds are stationary broadcast points, not physical bodies. two of
        // them spawn at the exact same world position at every corner (one per
        // adjacent wall -- geometrically both are in range there, though the
        // python side narrows to one per tick to match the one-IR-receiver
        // hardware), which a
        // non-kinematic rigidbody's collider resolves as a hard overlap and
        // launches; kill physics on the spawned instance so that never matters,
        // regardless of how the prefab itself is configured.
        Rigidbody rb = go.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }
        Collider col = go.GetComponent<Collider>();
        if (col != null)
        {
            col.isTrigger = true;
        }
        seed.side = side;
        wallSeeds.Add(seed);
        if (showCommRadius)
        {
            CommRadiusIndicator.Kind ringKind = nearCorner
                ? CommRadiusIndicator.Kind.WallSeedNearCorner
                : CommRadiusIndicator.Kind.WallSeed;
            CommRadiusIndicator.Attach(go, IR_RANGE, ringKind);
        }
    }

    void SpawnKilobots()
    {
        List<Vector3> occupied = new List<Vector3>();
        for (int s = 0; s < seeds.Count; s++)
        {
            occupied.Add(seeds[s].transform.position);
        }
        for (int s = 0; s < wallSeeds.Count; s++)
        {
            occupied.Add(wallSeeds[s].transform.position);
        }

        int count = Random.Range(minKilobots, maxKilobots + 1);
        for (int i = 0; i < count; i++)
        {
            Vector3 pos = FindClearSpawnPosition(occupied);
            occupied.Add(pos);
            // One of four cardinal rotations, so initial straight-line
            // exploration does not send every robot toward the same wall.
            //
            // cardinalIndex drives BOTH the physical rotation and pythonHeading,
            // which KilobotAgent.spawnHeading reports to Python every tick.
            // Deriving both from one draw is what guarantees the physical
            // rotation and the communicated belief can never disagree.
            //
            // A Unity rotation of 0 corresponds to a Python heading of +pi/2,
            // not 0 -- measured from real position data. belief.py's
            // KNOWN_START_HEADING names that; this rotation is unchanged.
            // Only cardinalIndex==0 is measured; 1-3 are derived by rotation
            // math. See belief.CARDINAL_HEADINGS and docs/code-history.md.
            //
            // NOTE: spawnHeading is a fixed-size observation change. Vector
            // Observation Space Size on this prefab's Behavior Parameters must
            // account for it in the Editor.
            int cardinalIndex = knownStartHeading ? Random.Range(0, 4) : 0;
            float pythonHeading = (Mathf.PI / 2f) - cardinalIndex * (Mathf.PI / 2f);
            Quaternion rot = knownStartHeading
                ? Quaternion.Euler(0f, cardinalIndex * 90f, 0f)
                : Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            GameObject go = Instantiate(kilobotPrefab, pos, rot, transform);
            if (kilobotLayer >= 0)
            {
                // recursive, not just go.layer = kilobotLayer: Physics collision
                // detection uses whichever GameObject the Collider component
                // itself sits on, which for a prefab built from child parts
                // (a separate visual mesh, a separate collider object, etc.)
                // may not be this root object -- setting the root's layer alone
                // would silently leave those children's collisions unaffected
                SetLayerRecursively(go, kilobotLayer);
            }
            KilobotAgent agent = go.GetComponent<KilobotAgent>();
            agent.arenaId = arenaId;
            agent.localIndex = i;
            agent.senderId = i + 1;
            agent.spawnHeading = knownStartHeading ? pythonHeading : 0f;
            // stagger heartbeat phases so a whole arena does not decide in lockstep
            agent.lastDecisionStep = envStep - (heartbeatTicks > 0 ? Random.Range(0, heartbeatTicks) : 0);
            kilobots.Add(agent);
            if (showCommRadius)
            {
                agent.ringRenderer = CommRadiusIndicator.Attach(go, IR_RANGE, CommRadiusIndicator.Kind.Kilobot);
                // The ring tracks the body colour (KilobotAgent.SetVisualState),
                // which Python only calls when KILOBOT_ORACLE_SEND_VISUAL_STATE
                // is on -- without this, a run with that flag off leaves a fresh
                // ring on whatever colour BuildMaterial gave it. State 0
                // (go_north) is correct regardless of the flag: it is what every
                // robot's oracle state starts as.
                agent.SetVisualState(0);
            }
        }
    }

    static void SetLayerRecursively(GameObject obj, int layer)
    {
        obj.layer = layer;
        foreach (Transform child in obj.transform)
        {
            SetLayerRecursively(child.gameObject, layer);
        }
    }

    // rejection-samples a position at least minSpawnSeparation from everything
    // already placed, so colliders never spawn overlapping and launch apart
    // under physics. Packing density here is low even at max population (100
    // kilobots plus ~25 seeds in an arena of this size), so this normally
    // succeeds on the first few attempts; the fallback below is a safety
    // valve, not the expected path.
    Vector3 FindClearSpawnPosition(List<Vector3> occupied)
    {
        const int MAX_ATTEMPTS = 50;
        Vector3 candidate = RandomArenaPosition();
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
            candidate = RandomArenaPosition();
            bool clear = true;
            for (int i = 0; i < occupied.Count; i++)
            {
                if (PlanarDistance(candidate, occupied[i]) < minSpawnSeparation)
                {
                    clear = false;
                    break;
                }
            }
            if (clear)
            {
                return candidate;
            }
        }
        Debug.LogWarning("SwarmManager: could not find a spawn position " + minSpawnSeparation +
                         " units clear of everything else after " + MAX_ATTEMPTS + " attempts; " +
                         "placing anyway, colliders may overlap. Consider a larger arena, fewer " +
                         "kilobots, or a smaller minSpawnSeparation.");
        return candidate;
    }

    Vector3 RandomArenaPosition()
    {
        float range = arenaHalfExtent - spawnMargin;
        float x = Random.Range(-range, range);
        float z = Random.Range(-range, range);
        return transform.position + new Vector3(x, 0f, z);
    }

    public void SetImageCommand(int newImageId)
    {
        pendingImage = true;
        pendingImageId = newImageId;
    }

    public void ResetCommand(int newImageId)
    {
        pendingReset = true;
        pendingImageId = newImageId;
    }

    public void SetRobotStates(List<int> states)
    {
        // Purely visual: sets each kilobot's body colour from the oracle's
        // per-robot state. states[i] corresponds to kilobots[i], both ordered by
        // localIndex, the same order AppendNode builds the snapshot in, so no
        // index mapping is needed. Sized defensively against a stale message
        // arriving just after a reset changed the robot count.
        int n = Mathf.Min(states.Count, kilobots.Count);
        for (int i = 0; i < n; i++)
        {
            kilobots[i].SetVisualState(states[i]);
        }
    }

    public void SetPosesCommand(List<KilobotPose> poses)
    {
        // Queued, not applied here: this runs while the side channel is being
        // drained, which is before Tick and therefore before a pending reset has
        // respawned anything. Applying immediately would write poses onto robots
        // that are about to be destroyed. ApplyPending runs the reset first and
        // these second, which is also the useful order -- "reset this arena, then
        // put the robots exactly here" is the whole point of the command.
        //
        // Appends rather than replaces so two messages arriving in one packet
        // both take effect.
        pendingPoses.AddRange(poses);
    }

    void ApplyPendingPoses()
    {
        for (int i = 0; i < pendingPoses.Count; i++)
        {
            KilobotPose p = pendingPoses[i];
            if (p.localIndex < 0 || p.localIndex >= kilobots.Count)
            {
                if (poseWarnCount < 20)
                {
                    poseWarnCount = poseWarnCount + 1;
                    Debug.LogWarning("SwarmManager: pose for localIndex " + p.localIndex +
                                     " ignored; arena " + arenaId + " has " + kilobots.Count +
                                     " kilobots. (A pose sent in the same packet as a reset is " +
                                     "applied AFTER the respawn, so index against the NEW count.)");
                }
                continue;
            }
            KilobotAgent a = kilobots[p.localIndex];
            // y is left alone: the caller works in the arena's 2D plane and has
            // no business knowing the prefab's resting height above the floor.
            Vector3 world = transform.position
                            + new Vector3(p.x, a.transform.position.y - transform.position.y, p.z);
            // heading is python's: direction (cos h, sin h) in (x, z). Unity yaw
            // is measured from +z toward +x, so its direction is (sin y, cos y),
            // giving y = pi/2 - h. That is the same relation SpawnKilobots
            // already encodes in the other direction (pythonHeading = pi/2 -
            // cardinalIndex * pi/2 alongside a yaw of cardinalIndex * 90).
            Quaternion rot = Quaternion.Euler(0f, 90f - p.heading * Mathf.Rad2Deg, 0f);
            // KilobotMovement drives the body with rb.MovePosition, which
            // interpolates from rb.position -- writing only the transform would
            // leave the rigidbody's own pose stale for a frame and drag the robot
            // back toward where it was. Momentum is cleared for the same reason:
            // a teleport should not arrive carrying the old location's velocity.
            Rigidbody rb = a.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.position = world;
                rb.rotation = rot;
            }
            a.transform.SetPositionAndRotation(world, rot);
        }
        pendingPoses.Clear();
    }

    public void Tick()
    {
        ApplyPending();
        ScanArena();
        SendSnapshot();
        RequestEligibleDecisions();
    }

    void ApplyPending()
    {
        if (pendingReset)
        {
            DoReset(pendingImageId);
            pendingReset = false;
            pendingImage = false;
        }
        else if (pendingImage)
        {
            imageId = pendingImageId;
            UpdateFloorTexture();
            pendingImage = false;
        }
        if (pendingPoses.Count > 0)
        {
            ApplyPendingPoses();
        }
    }

    void DoReset(int newImageId)
    {
        for (int i = 0; i < kilobots.Count; i++)
        {
            Destroy(kilobots[i].gameObject);
        }
        kilobots.Clear();
        imageId = newImageId;
        UpdateFloorTexture();
        SeedSwarmRng();
        SpawnKilobots();
    }

    void UpdateFloorTexture()
    {
        if (!showTargetFloor || floorRenderer == null || imageLibrary == null)
        {
            return;
        }
        Texture2D tex = imageLibrary.GetTexture(imageId);
        if (tex == null)
        {
            return;
        }
        for (int i = 0; i < floorRotationSteps; i++)
        {
            tex = Rotate90(tex);
        }
        // .material (not .sharedMaterial) makes a per-object instance the first time it
        // is touched, so this cannot bleed into other arenas that started from the same
        // material asset
        floorRenderer.material.mainTexture = tex;
    }

    // Rotates a copy of src by 90 degrees CCW; call N times for N*90 degrees.
    // Does not modify src (imageLibrary.GetTexture caches and returns the
    // same instance on every call, and BakeImage's own on-pixel geometry
    // reads this same cached texture -- mutating it in place would corrupt
    // the reward/oracle geometry as a side effect of a purely visual
    // rotation).
    //
    // This formula is a deliberate correction of the reverted phase-26/27
    // version, not a re-derivation from scratch trusted the same way twice:
    // that version assumed top-down pixel indexing (row 0 = top, matching a
    // human viewing an image file directly) when deriving the rotation
    // math, but GetPixels()/SetPixels() are actually bottom-up (row 0 =
    // bottom), Unity's own convention -- an internal mismatch that wasn't
    // caught before shipping the first time, even though the specific step
    // count chosen back then still happened to look correct when checked
    // against the real build. This version is instead derived by
    // conjugating the same, separately-verified 90-degree CCW rotation used
    // for BakeImage's onPoints (ImageLibrary.cs, (x,z)->(-z,x)) through the
    // pixel<->normalized-coordinate mapping, then checked exhaustively
    // against all 60 pixels of a small test canvas before being trusted
    // here -- not just spot-checked on one or
    // two hand-picked examples.
    Texture2D Rotate90(Texture2D src)
    {
        int w = src.width;
        int h = src.height;
        Color[] srcPixels = src.GetPixels();
        Color[] dstPixels = new Color[w * h];
        for (int newRow = 0; newRow < w; newRow++)
        {
            for (int newCol = 0; newCol < h; newCol++)
            {
                int oldCol = newRow;
                int oldRow = (h - 1) - newCol;
                dstPixels[newRow * h + newCol] = srcPixels[oldRow * w + oldCol];
            }
        }
        Texture2D dst = new Texture2D(h, w);
        dst.SetPixels(dstPixels);
        dst.Apply();
        return dst;
    }

    void ScanArena()
    {
        sourceBuffer.Clear();
        targetBuffer.Clear();
        attrBuffer.Clear();

        for (int i = 0; i < kilobots.Count; i++)
        {
            KilobotAgent a = kilobots[i];
            a.receivedMessages.Clear();
            for (int s = 0; s < SEED_COUNT; s++)
            {
                a.seedObs[s] = 0f;
            }
            for (int s = 0; s < WALL_SIZE; s++)
            {
                a.wallObs[s] = 0f;
            }

            float nearest = float.MaxValue;
            Vector3 pa = a.transform.position;

            for (int j = 0; j < kilobots.Count; j++)
            {
                if (j == i)
                {
                    continue;
                }
                KilobotAgent b = kilobots[j];
                float d = PlanarDistance(pa, b.transform.position);
                if (d < nearest)
                {
                    nearest = d;
                }
                if (d <= IR_RANGE)
                {
                    float strength = 1f / (1f + d);
                    a.receivedMessages.Add(BuildMessage(b, strength));
                    sourceBuffer.Add(b.localIndex);
                    targetBuffer.Add(a.localIndex);
                    attrBuffer.Add(strength);
                }
            }

            for (int s = 0; s < seeds.Count; s++)
            {
                float d = PlanarDistance(pa, seeds[s].transform.position);
                if (d <= IR_RANGE)
                {
                    int idx = (int)seeds[s].seedType;
                    a.seedObs[idx] = 1f / (1f + d);
                }
            }

            float[] wallBestDist = { float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue };
            int[] wallBestIdx = { -1, -1, -1, -1 };
            for (int s = 0; s < wallSeeds.Count; s++)
            {
                float d = PlanarDistance(pa, wallSeeds[s].transform.position);
                if (d <= IR_RANGE)
                {
                    int idx = (int)wallSeeds[s].side;
                    if (d < wallBestDist[idx])
                    {
                        wallBestDist[idx] = d;
                        wallBestIdx[idx] = s;
                    }
                }
            }
            for (int s = 0; s < WALL_SIZE; s++)
            {
                if (wallBestDist[s] < float.MaxValue)
                {
                    a.wallObs[s] = 1f / (1f + wallBestDist[s]);
                    int winnerSeedIdx = wallBestIdx[s];
                    // The specific winning wall seed's own known
                    // position, exposed as an additional row in the SAME
                    // (100,11) message channel robot-to-robot messages
                    // already use -- deliberately NOT a new or widened
                    // observation sensor, since that would need a matched,
                    // rebuilt Python-side shape and risks breaking the
                    // ml-agents connection if the two sides ever
                    // mismatch. Marked with a negative senderId (real
                    // robots are always >= 1, see agent.senderId = i + 1
                    // above) so the python side can distinguish this from
                    // a genuine robot message before the reception
                    // draw. wallObs[s] above is completely unchanged --
                    // this is purely additive, no existing signal is
                    // touched or replaced.
                    //
                    // Only actually sent when
                    // the winning seed is one of its own wall's near-corner
                    // ones (see WallSeedRobot.nearCorner and
                    // SpawnWallSeeds' own comment) -- a regular, mid-wall
                    // seed has no more claim to knowing its own precise
                    // position than it did before this phase, matching the
                    // python replica side exactly rather than sending
                    // everything and relying on the python side alone to
                    // discard the ones that shouldn't count.
                    if (wallSeeds[winnerSeedIdx].nearCorner)
                    {
                        float[] wsRow = new float[MESSAGE_SIZE + 2];
                        wsRow[0] = wallSeeds[winnerSeedIdx].transform.position.x;
                        wsRow[1] = wallSeeds[winnerSeedIdx].transform.position.z;
                        wsRow[MESSAGE_SIZE] = -(s + 1);
                        wsRow[MESSAGE_SIZE + 1] = a.wallObs[s];
                        a.receivedMessages.Add(wsRow);
                    }
                    if (debugWallSeeds && wallScanLogCount < 20)
                    {
                        wallScanLogCount = wallScanLogCount + 1;
                        Debug.Log("WALL_SCAN_LIVE robot=" + a.localIndex + " robot_pos=(" + pa.x + ", " + pa.z + ")" +
                                  " writing_to_wallObs_slot=" + s +
                                  " winning_seed_idx=" + winnerSeedIdx +
                                  " winning_seed_side=" + wallSeeds[winnerSeedIdx].side +
                                  " winning_seed_near_corner=" + wallSeeds[winnerSeedIdx].nearCorner +
                                  " winning_seed_pos=(" + wallSeeds[winnerSeedIdx].transform.position.x + ", " + wallSeeds[winnerSeedIdx].transform.position.z + ")" +
                                  " distance=" + wallBestDist[s] +
                                  " resulting_wallObs_value=" + a.wallObs[s]);
                    }
                }
            }

            if (nearest == float.MaxValue)
            {
                nearest = arenaHalfExtent;
            }
            a.nearestRobotDist = nearest;
        }
    }

    float[] BuildMessage(KilobotAgent b, float strength)
    {
        float[] row = new float[MESSAGE_SIZE + 2];
        for (int t = 0; t < MESSAGE_SIZE; t++)
        {
            row[t] = b.transmission[t];
        }
        row[MESSAGE_SIZE] = b.senderId;
        row[MESSAGE_SIZE + 1] = strength;
        return row;
    }

    void SendSnapshot()
    {
        nodeBuffer.Clear();
        int m = kilobots.Count;
        for (int i = 0; i < m; i++)
        {
            AppendNode(kilobots[i]);
        }

        int e = attrBuffer.Count;
        edgeIndexBuffer.Clear();
        for (int s = 0; s < e; s++)
        {
            edgeIndexBuffer.Add(sourceBuffer[s]);
        }
        for (int s = 0; s < e; s++)
        {
            edgeIndexBuffer.Add(targetBuffer[s]);
        }

        channel.SendSnapshot(arenaId, envStep, m, e, nodeBuffer, edgeIndexBuffer, attrBuffer);
        envStep = envStep + 1;
    }

    void AppendNode(KilobotAgent a)
    {
        Vector3 local = a.transform.position - transform.position;
        float px = local.x / arenaHalfExtent;
        float pz = local.z / arenaHalfExtent;

        Vector3 fwd = a.transform.forward;

        float dist;
        Vector2 dir;
        imageLibrary.Sample(imageId, px, pz, out dist, out dir);

        float c = a.nearestRobotDist / arenaHalfExtent;

        nodeBuffer.Add(px);
        nodeBuffer.Add(pz);
        nodeBuffer.Add(fwd.x);
        nodeBuffer.Add(fwd.z);
        nodeBuffer.Add(dist);
        nodeBuffer.Add(dir.x);
        nodeBuffer.Add(dir.y);
        nodeBuffer.Add(c);
        nodeBuffer.Add(a.movement.leftMotor);
        nodeBuffer.Add(a.movement.rightMotor);
        for (int t = 0; t < MESSAGE_SIZE; t++)
        {
            nodeBuffer.Add(a.transmission[t]);
        }
    }

    void RequestEligibleDecisions()
    {
        // A decision is an event: a neighbor transmission, a seed sighting, OR a
        // wall sighting. The old condition (messages only) silently discarded
        // every seed sighting a robot made while it had no neighbors in range,
        // which starved the split-observation actor of its only grounded
        // position signal and left isolated robots permanently frozen (they
        // could never issue a command).
        //
        // Wall seeds (2026-07-07): KILOBOT_SEED_LAYOUT's four corner landmarks
        // cover a small fraction of the arena, so most robots never range one
        // within an episode. Wall-lining seeds (see SpawnWallSeeds) guarantee
        // every point on every wall is within IR_RANGE of one, so a robot can
        // never coast indefinitely with zero information; a wall-only sighting
        // must count here the same way a landmark seed sighting does. Must
        // match the python side's own decision-eligibility rule.
        //
        // Heartbeat (2026-07-06): when KILOBOT_HEARTBEAT_TICKS > 0, a robot that
        // has gone that many ticks without deciding gets a decision anyway, with a
        // fully zero event. Motion stays ballistic between decisions (one constant
        // command composes into a single exact arc, the cheapest motion for the
        // pose filter to track), but coasting is no longer terminal: a robot that
        // overshot a beacon or is grinding on a wall gets a sparse chance to
        // re-steer. The python side must run with the same KILOBOT_HEARTBEAT_TICKS
        // so it commands these event-less deciders instead of zero-stopping them.
        for (int i = 0; i < kilobots.Count; i++)
        {
            bool seedVisible = false;
            for (int s = 0; s < SEED_COUNT; s++)
            {
                if (kilobots[i].seedObs[s] > 0f)
                {
                    seedVisible = true;
                    break;
                }
            }
            if (!seedVisible)
            {
                for (int s = 0; s < WALL_SIZE; s++)
                {
                    if (kilobots[i].wallObs[s] > 0f)
                    {
                        seedVisible = true;
                        break;
                    }
                }
            }
            bool heartbeatDue = heartbeatTicks > 0
                && envStep - kilobots[i].lastDecisionStep >= heartbeatTicks;
            if (kilobots[i].receivedMessages.Count > 0 || seedVisible || heartbeatDue)
            {
                kilobots[i].lastDecisionStep = envStep;
                kilobots[i].RequestDecision();
            }
        }
    }

    float PlanarDistance(Vector3 p, Vector3 q)
    {
        float dx = p.x - q.x;
        float dz = p.z - q.z;
        return Mathf.Sqrt(dx * dx + dz * dz);
    }
}
