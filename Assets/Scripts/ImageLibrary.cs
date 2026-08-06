using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

public class ImageLibrary : MonoBehaviour
{
    public string formationsPath = "data/formations";
    public string searchPattern = "*.png";
    public int resolution = 64;
    public float onThreshold = 0.5f;

    float[][] distanceField;
    Vector2[][] directionField;
    Texture2D[] textures;
    string[] files;

    void Awake()
    {
        LoadFileList();
    }

    void LoadFileList()
    {
        string dir = ResolvePath();
        if (!Directory.Exists(dir))
        {
            Debug.LogError("ImageLibrary: formations folder not found at " + dir + ". Set formationsPath (absolute, or relative to the build/project root) to your Formations folder.");
            files = new string[0];
            distanceField = new float[0][];
            directionField = new Vector2[0][];
            textures = new Texture2D[0];
            return;
        }
        files = Directory.GetFiles(dir, searchPattern);
        Array.Sort(files, StringComparer.Ordinal);

        distanceField = new float[files.Length][];
        directionField = new Vector2[files.Length][];
        textures = new Texture2D[files.Length];
    }

    void EnsureBaked(int index)
    {
        if (distanceField[index] != null)
        {
            return;
        }
        Texture2D tex = LoadTexture(files[index]);
        textures[index] = tex;
        BakeImage(index, tex);
    }

    // returns the raw formation image for a target index (loading and caching it if this
    // is the first time), for anything visual that wants to display it rather than just
    // query the baked distance field, e.g. a floor texture. Null if the index is invalid
    // or the formations folder failed to load.
    public Texture2D GetTexture(int index)
    {
        if (distanceField == null || index < 0 || index >= distanceField.Length)
        {
            return null;
        }
        EnsureBaked(index);
        return textures[index];
    }

    string ResolvePath()
    {
        string path = Environment.GetEnvironmentVariable("KILOBOT_FORMATIONS");
        if (string.IsNullOrEmpty(path))
        {
            path = formationsPath;
        }
        if (Path.IsPathRooted(path))
        {
            return path;
        }
        return Path.Combine(Application.dataPath, "..", path);
    }

    Texture2D LoadTexture(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(bytes);
        return tex;
    }

    void BakeImage(int index, Texture2D tex)
    {
        int w = tex.width;
        int h = tex.height;
        Color[] pixels = tex.GetPixels();

        // The agents
        // (the actual reward geometry and the uncoordinated oracle, both
        // reading this same onPoints/baked field, independent of the
        // coordination-aware oracle's separate Python-side geometry and of
        // the floor's own visual display) rotated 90 degrees CCW.
        //
        // This step function is freshly re-derived, not reused from the
        // reverted phase-28 attempt: (x,z) -> (-z,x), the standard 2D CCW
        // rotation matrix applied directly to (nx,nz), verified against a
        // concrete example (east (1,0) rotates to north (0,1) under a
        // standard map view, X=right/east, Z=away-from-viewer/north) before
        // trusting it here.
        int rotationSteps = ((ParseIntEnv("KILOBOT_BAKE_ROTATION_STEPS", 1) % 4) + 4) % 4;

        List<Vector2> onPoints = new List<Vector2>();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Color px = pixels[y * w + x];
                float lum = (px.r + px.g + px.b) / 3f;
                if (lum > onThreshold)
                {
                    float nx = ((float)x / (w - 1)) * 2f - 1f;
                    float nz = ((float)y / (h - 1)) * 2f - 1f;
                    Vector2 p = new Vector2(nx, nz);
                    for (int s = 0; s < rotationSteps; s++)
                    {
                        p = new Vector2(-p.y, p.x);   // verified 90-degree CCW step: (x,z) -> (-z,x)
                    }
                    onPoints.Add(p);
                }
            }
        }

        int cells = resolution * resolution;
        float[] dist = new float[cells];
        Vector2[] dir = new Vector2[cells];

        for (int gz = 0; gz < resolution; gz++)
        {
            for (int gx = 0; gx < resolution; gx++)
            {
                float cx = ((float)gx / (resolution - 1)) * 2f - 1f;
                float cz = ((float)gz / (resolution - 1)) * 2f - 1f;
                Vector2 cellPos = new Vector2(cx, cz);

                float best = float.MaxValue;
                Vector2 bestDiff = Vector2.zero;
                for (int p = 0; p < onPoints.Count; p++)
                {
                    Vector2 diff = onPoints[p] - cellPos;
                    float dd = diff.sqrMagnitude;
                    if (dd < best)
                    {
                        best = dd;
                        bestDiff = diff;
                    }
                }

                int cell = gz * resolution + gx;
                if (onPoints.Count == 0)
                {
                    dist[cell] = 0f;
                    dir[cell] = Vector2.zero;
                }
                else
                {
                    dist[cell] = Mathf.Sqrt(best);
                    if (bestDiff.sqrMagnitude > 0f)
                    {
                        dir[cell] = bestDiff.normalized;
                    }
                    else
                    {
                        dir[cell] = Vector2.zero;
                    }
                }
            }
        }

        distanceField[index] = dist;
        directionField[index] = dir;
    }

    public void Sample(int index, float x, float z, out float distance, out Vector2 direction)
    {
        if (distanceField == null || index < 0 || index >= distanceField.Length)
        {
            distance = 1f;
            direction = Vector2.zero;
            return;
        }
        EnsureBaked(index);
        int gx = NormalizedToGrid(x);
        int gz = NormalizedToGrid(z);
        int cell = gz * resolution + gx;
        distance = distanceField[index][cell];
        direction = directionField[index][cell];
    }

    int NormalizedToGrid(float v)
    {
        float t = (v + 1f) * 0.5f;
        int g = Mathf.RoundToInt(t * (resolution - 1));
        if (g < 0)
        {
            g = 0;
        }
        if (g > resolution - 1)
        {
            g = resolution - 1;
        }
        return g;
    }

    // deliberately allows negative input, unlike SwarmManager.cs's own
    // ParseIntEnv -- the wrapping formula that consumes this
    // (((x%4)+4)%4) needs a negative input to wrap correctly into 0-3
    // rather than be rejected outright
    static int ParseIntEnv(string name, int fallback)
    {
        string env = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(env))
        {
            return fallback;
        }
        int result;
        if (int.TryParse(env, out result))
        {
            return result;
        }
        return fallback;
    }
}
