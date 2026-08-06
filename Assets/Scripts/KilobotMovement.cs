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

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    // NO domain randomization here, deliberately. A per-robot motor bias made a
    // robot commanded [1, 1] drift at a constant, per-robot rate -- invisible to
    // Python's kinematic tracking by construction, and the confirmed cause of a
    // long heading-drift investigation. Zeroing the [SerializeField] defaults
    // was not enough: a value already saved on the prefab does not pick up a
    // later change to the script's default, so the mechanism is gone entirely.
    // belief.MOTION_NOISE is 0 because this is. If domain randomization is
    // wanted for RL robustness, model it so split_tick_motion can account for
    // it, not as a Unity-side effect Python cannot see.
    void FixedUpdate()
    {
        float left = Mathf.Clamp01(leftMotor);
        float right = Mathf.Clamp01(rightMotor);

        float forwardSpeed = (left + right) * 0.5f * moveSpeed;
        float turnRate = (left - right) * turnSpeed;

        transform.Rotate(0f, turnRate * Time.fixedDeltaTime, 0f);
        rb.MovePosition(rb.position + transform.forward * forwardSpeed * Time.fixedDeltaTime);
    }
}
