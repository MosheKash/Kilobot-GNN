using UnityEngine;

public class KilobotMovement : MonoBehaviour
{
    [Header("Motor Values (0 = off, 1 = full speed)")]
    [Range(0f, 1f)] public float leftMotor = 0f;
    [Range(0f, 1f)] public float rightMotor = 0f;

    [Header("Tuning")]
    [SerializeField] float moveSpeed = 1f;
    [SerializeField] float turnSpeed = 45f;

    Rigidbody rb;

    // True wheel-slip (MotionNoise.cs; the amended task, owner 2026-10-20).
    // slipBiasLeft/Right: this robot's persistent systematic wheel mismatch,
    // drawn once when this robot spawns -- the linear-drift component of its
    // odometry error. Applied to the EFFECTIVE wheel speeds and never
    // clamped: a slipping wheel at full command still slips, so a clamp at 1
    // would hide exactly the full-throttle case the renewal oracle's
    // feedback loops exist to absorb. The per-tick random component
    // (MotionNoise.DrawTick) is drawn fresh every FixedUpdate below. Both
    // are exactly 0 in a default launch -- motion is then bit-identical to
    // the historical exact-motion build, and no RNG draws are consumed.
    //
    // History, kept deliberately: a per-robot motor bias was tried before as
    // domain randomization and REMOVED, because Python's kinematic tracking
    // was exact by assumption while Unity drifted -- the divergence was
    // invisible by construction and caused a long heading-drift
    // investigation; the mechanism was deleted entirely and belief.
    // MOTION_NOISE set to 0 to match. The difference now: the drift is the
    // point of the amended task, and Python models it (KILOBOT_MOTION_NOISE
    // -> belief.MOTION_NOISE via launch.py), so every filter's conf honestly
    // decays as error accumulates instead of silently overstating accuracy.
    // The old comment's own prescription ("model it so split_tick_motion can
    // account for it") is finally followed.
    float slipBiasLeft;
    float slipBiasRight;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        MotionNoise.ConfigureFromEnv();
        slipBiasLeft = MotionNoise.DrawBias();
        slipBiasRight = MotionNoise.DrawBias();
    }

    void FixedUpdate()
    {
        float left = Mathf.Clamp01(leftMotor);
        float right = Mathf.Clamp01(rightMotor);
        // Effective wheel speeds under slip: commanded * (1 + persistent
        // bias + fresh random). The multipliers stay near 1 for any sane
        // sigma. A stopped robot (left = right = 0) stays exactly stopped --
        // error is frozen while parked, the property the arrival latch and
        // the settle metric both rely on.
        float effLeft = left * (1f + slipBiasLeft + MotionNoise.DrawTick());
        float effRight = right * (1f + slipBiasRight + MotionNoise.DrawTick());

        float forwardSpeed = (effLeft + effRight) * 0.5f * moveSpeed;
        float turnRate = (effLeft - effRight) * turnSpeed;

        transform.Rotate(0f, turnRate * Time.fixedDeltaTime, 0f);
        rb.MovePosition(rb.position + transform.forward * forwardSpeed * Time.fixedDeltaTime);
    }
}
