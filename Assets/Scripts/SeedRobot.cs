using UnityEngine;

// The origin point is gone; the corner layout has four seeds. Must
// match the python side's belief.SEED_LAYOUTS ordering exactly -- both
// "corners" and "cluster" are [UpperLeft, UpperRight, LowerLeft, LowerRight]
// now.
public enum SeedType
{
    UpperLeft = 0,
    UpperRight = 1,
    LowerLeft = 2,
    LowerRight = 3
}

public class SeedRobot : MonoBehaviour
{
    public SeedType seedType = SeedType.UpperLeft;
}
