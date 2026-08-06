using UnityEngine;

// Optional debug visualization: a flat, semi-transparent disc showing a
// robot's IR communication radius, parented under the robot so the robot
// itself still pokes up through the center and stays visible. Built entirely
// at runtime (no prefab or material asset needed) and gated behind
// KILOBOT_SHOW_RADIUS so it costs nothing when off.
public static class CommRadiusIndicator
{
    const int SEGMENTS = 32;
    // Low, to compensate for the ring tracking the body's exact hue: the two
    // are no longer a separately-tunable pairing.
    const float ALPHA = 0.18f;
    const float Y_OFFSET = 0.02f;

    public enum Kind
    {
        Kilobot,
        WallSeed,
        WallSeedNearCorner,
        Seed
    }

    static Mesh cachedMesh;
    static Material wallSeedMaterial;
    static Material wallSeedNearCornerMaterial;
    static Material seedMaterial;

    public static Renderer Attach(GameObject target, float radius, Kind kind)
    {
        Material mat = GetMaterial(kind);
        if (mat == null)
        {
            return null;
        }

        GameObject indicator = new GameObject("CommRadius");
        indicator.transform.SetParent(target.transform, false);
        // the mesh is built with radius in true world units; counter-scale
        // against the parent's own scale so the circle renders at the correct
        // absolute size in the world regardless of how the robot prefab itself
        // is scaled (a child's scale, and any local position offset, otherwise
        // multiplies with its parent's)
        Vector3 parentScale = target.transform.lossyScale;
        float invX = Mathf.Approximately(parentScale.x, 0f) ? 1f : 1f / parentScale.x;
        float invY = Mathf.Approximately(parentScale.y, 0f) ? 1f : 1f / parentScale.y;
        float invZ = Mathf.Approximately(parentScale.z, 0f) ? 1f : 1f / parentScale.z;
        indicator.transform.localPosition = new Vector3(0f, Y_OFFSET * invY, 0f);
        indicator.transform.localRotation = Quaternion.identity;
        indicator.transform.localScale = new Vector3(invX, invY, invZ);

        MeshFilter mf = indicator.AddComponent<MeshFilter>();
        mf.sharedMesh = GetMesh(radius);

        MeshRenderer mr = indicator.AddComponent<MeshRenderer>();
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        // The Kilobot ring tracks its body's oracle-state colour
        // (KilobotAgent.SetVisualState writes to the Renderer returned below),
        // so unlike WallSeed/Seed -- which never change and can share one
        // cached material -- it needs an independent material per robot.
        // .material, not .sharedMaterial, creates that per-instance copy;
        // sharedMaterial would recolour every ring built from the same asset.
        if (kind == Kind.Kilobot)
        {
            mr.material = mat;
        }
        else
        {
            mr.sharedMaterial = mat;
        }
        return mr;
    }

    static Mesh GetMesh(float radius)
    {
        if (cachedMesh == null)
        {
            cachedMesh = BuildCircleMesh(radius, SEGMENTS);
        }
        return cachedMesh;
    }

    static Material GetMaterial(Kind kind)
    {
        // The two seed colours are the deep blue / teal the seed prefabs'
        // own body materials use (an Editor-side asset this file cannot
        // reach), so a seed's ring and body read as one identity. See
        // KilobotAgent.SetVisualState for the palette.
        //
        // The Kilobot case has no fixed colour: the ring tracks the body's
        // current oracle-state colour. The value below is only the brief
        // default before SwarmManager triggers the first write at spawn.
        switch (kind)
        {
            case Kind.WallSeed:
                if (wallSeedMaterial == null)
                {
                    wallSeedMaterial = BuildMaterial(new Color(0.15f, 0.70f, 0.72f, ALPHA));
                }
                return wallSeedMaterial;
            case Kind.WallSeedNearCorner:
                // A brighter, more saturated cyan-blue: a genuine third
                // category rather than a blend of WallSeed's teal and Seed's
                // blue, while staying in the same cool family as both. These
                // are conceptually a wall seed upgraded partway toward a
                // corner seed (broadcasting an exact position), so a colour
                // between the two makes sense; this leans toward Seed's
                // own blue rather than an exact midpoint; a true midpoint
                // sat too close to WallSeed's own teal to read as
                // distinct at this indicator's small size and low alpha.
                if (wallSeedNearCornerMaterial == null)
                {
                    wallSeedNearCornerMaterial = BuildMaterial(new Color(0.20f, 0.60f, 0.90f, ALPHA));
                }
                return wallSeedNearCornerMaterial;
            case Kind.Seed:
                if (seedMaterial == null)
                {
                    seedMaterial = BuildMaterial(new Color(0.25f, 0.40f, 0.85f, ALPHA));
                }
                return seedMaterial;
            default:
                return BuildMaterial(new Color(0.96f, 0.94f, 0.88f, ALPHA));
        }
    }

    static Material BuildMaterial(Color color)
    {
        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null)
        {
            shader = Shader.Find("Unlit/Transparent");
        }
        if (shader == null)
        {
            Debug.LogError("CommRadiusIndicator: no compatible transparent shader found " +
                           "(tried Sprites/Default, Unlit/Transparent); radius circles disabled.");
            return null;
        }
        Material mat = new Material(shader);
        mat.color = color;
        return mat;
    }

    static Mesh BuildCircleMesh(float radius, int segments)
    {
        Mesh mesh = new Mesh();
        mesh.name = "CommRadiusDisc";
        Vector3[] vertices = new Vector3[segments + 1];
        vertices[0] = Vector3.zero;
        for (int i = 0; i < segments; i++)
        {
            float angle = i * 2f * Mathf.PI / segments;
            vertices[i + 1] = new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
        }
        // both winding orders per triangle, so the disc is visible from above
        // and below regardless of the shader's culling mode
        int[] triangles = new int[segments * 6];
        for (int i = 0; i < segments; i++)
        {
            int a = 0;
            int b = i + 1;
            int c = (i + 1) % segments + 1;
            int t = i * 6;
            triangles[t] = a;
            triangles[t + 1] = b;
            triangles[t + 2] = c;
            triangles[t + 3] = a;
            triangles[t + 4] = c;
            triangles[t + 5] = b;
        }
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }
}
