using UnityEngine;
using UnityEngine.InputSystem;

public class FlyCam : MonoBehaviour
{
    [SerializeField] float moveSpeed = 50f;
    [SerializeField] float fastMultiplier = 3f;
    [SerializeField] float lookSensitivity = 2f;

    float pitch = 0f;
    float yaw = 0f;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        pitch = angles.x;
        yaw = angles.y;
    }

    void Update()
    {
        var mouse = Mouse.current;
        var keyboard = Keyboard.current;
        if (mouse == null || keyboard == null)
        {
            return;
        }

        if (mouse.rightButton.isPressed)
        {
            Vector2 delta = mouse.delta.ReadValue();
            pitch -= delta.y * lookSensitivity * Time.deltaTime * 10f;
            yaw += delta.x * lookSensitivity * Time.deltaTime * 10f;
            pitch = Mathf.Clamp(pitch, -89f, 89f);
            transform.eulerAngles = new Vector3(pitch, yaw, 0f);
        }

        float speed = moveSpeed;
        if (keyboard.leftShiftKey.isPressed)
        {
            speed = moveSpeed * fastMultiplier;
        }

        float x = 0f;
        if (keyboard.dKey.isPressed)
        {
            x += 1f;
        }
        if (keyboard.aKey.isPressed)
        {
            x -= 1f;
        }
        float y = 0f;
        if (keyboard.eKey.isPressed)
        {
            y += 1f;
        }
        if (keyboard.qKey.isPressed)
        {
            y -= 1f;
        }
        float z = 0f;
        if (keyboard.wKey.isPressed)
        {
            z += 1f;
        }
        if (keyboard.sKey.isPressed)
        {
            z -= 1f;
        }

        Vector3 move = new Vector3(x, y, z);
        transform.Translate(move.normalized * speed * Time.deltaTime, Space.Self);
    }
}
