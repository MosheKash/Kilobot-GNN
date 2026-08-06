using UnityEngine;

// A wall-lining seed broadcasts only which side of the arena it is on, not a
// precise position (unlike SeedRobot). Many of these line each wall so a
// kilobot can never get stuck without hearing from one; see WALL_SPACING in
// SwarmManager. Must match the python side's belief.WALL_AXIS / WALL_VAL
// ordering: North=0, East=1, South=2, West=3.
public enum WallSide
{
    North = 0,
    East = 1,
    South = 2,
    West = 3
}

public class WallSeedRobot : MonoBehaviour
{
    public WallSide side = WallSide.North;
    // True for exactly the seeds at each end of
    // this wall (see SwarmManager.SpawnWallSeeds' own comment for the
    // precise count and rationale) -- these still broadcast their own
    // exact position even with knownStartHeading's own oracle-side
    // wall-seed-position flag off, everywhere else on the same wall
    // staying position-silent as before.
    public bool nearCorner = false;
}
