using System;
using UnityEngine;

// True wheel-slip odometry noise -- the amended task's core dial (owner
// sign-off 2026-10-20). Real kilobots do not dead-reckon exactly; the sim's
// exactness was the unphysical part, and it is what made information
// non-perishable (anchor once, accurate forever), which in turn is a large
// part of why communication had no measurable value (docs/tuning.md phase
// 174: the channel's value ~= the noise band). Under slip, a robot's
// believed position diverges as it drives, so fresh measurements become a
// standing need rather than a one-time convenience.
//
// This is the mechanism KilobotMovement.cs's old comment prescribed when
// the previous domain-randomization attempt was removed: "model it so
// split_tick_motion can account for it, not as a Unity-side effect Python
// cannot see." The Python side now models it -- KILOBOT_MOTION_NOISE sets
// belief.MOTION_NOISE via launch.py -- so every particle filter honestly
// widens as error accumulates and conf decays. The old attempt's failure
// mode (invisible divergence: Python tracking exactly while Unity drifted)
// cannot recur, because the divergence is now the modelled, observable
// quantity the whole amended task hangs off.
//
// Components, matching real hardware:
//   KILOBOT_SLIP_SIGMA_BIAS  per-robot systematic wheel mismatch, drawn
//                            ONCE per robot at spawn. The linear-drift
//                            component -- the one only feedback loops
//                            (reception-based steering) can absorb.
//   KILOBOT_SLIP_SIGMA_TICK  per-FixedUpdate random slip. The random-walk
//                            component.
//   KILOBOT_SLIP_SEED        pins the noise stream. Default 0, NOT engine
//                            randomness: noise must never be the one
//                            nondeterminism in a paired comparison.
//
// Both sigmas default 0: the player is then bit-identical to the historical
// exact-motion build (DrawBias/DrawTick return 0 without consuming the
// stream, so even the RNG state is untouched).
//
// The RNG is a dedicated System.Random -- never UnityEngine.Random, whose
// stream is the swarm-spawn stream by design (see SwarmManager's swarm-rng
// comment: the four SpawnKilobots draws are the only UnityEngine.Random
// calls anywhere in Assets/Scripts, and that must stay true).
public static class MotionNoise
{
    public static float sigmaBias = 0f;
    public static float sigmaTick = 0f;

    static System.Random rng = new System.Random(0);
    static bool configured;

    // Idempotent: every SwarmManager instance (one per arena) calls this
    // from SpawnInitial, and KilobotMovement.Start calls it too, so a robot
    // can never draw its per-robot bias before the dials exist.
    public static void ConfigureFromEnv()
    {
        if (configured)
        {
            return;
        }
        configured = true;
        sigmaBias = SwarmManager.ParseFloatEnv("KILOBOT_SLIP_SIGMA_BIAS", 0f);
        sigmaTick = SwarmManager.ParseFloatEnv("KILOBOT_SLIP_SIGMA_TICK", 0f);
        int seed = SwarmManager.ParseSignedIntEnv("KILOBOT_SLIP_SEED", 0);
        rng = new System.Random(seed);
        if (sigmaBias > 0f || sigmaTick > 0f)
        {
            Debug.Log("MotionNoise ON: sigmaBias=" + sigmaBias.ToString("0.####")
                      + " sigmaTick=" + sigmaTick.ToString("0.####") + " seed=" + seed);
        }
    }

    // Box-Muller on two uniforms from the dedicated stream.
    static float NextGaussian()
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    public static float DrawBias()
    {
        return sigmaBias > 0f ? sigmaBias * NextGaussian() : 0f;
    }

    public static float DrawTick()
    {
        return sigmaTick > 0f ? sigmaTick * NextGaussian() : 0f;
    }
}
