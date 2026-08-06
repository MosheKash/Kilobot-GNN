using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

// Headless Linux player build, callable from CI or a plain shell:
//
//   <editor> -batchmode -quit -nographics -projectPath <repo> \
//            -executeMethod BuildPlayer.BuildLinux -logFile -
//
// Exists because the repo had no build script at all -- the player in Builds/
// was produced by hand from the Editor, so any C# change meant a manual
// rebuild and there was no way to reproduce the artifact. Writes to the same
// Builds/Kilobot.x86_64 the Python side already defaults to
// (unity_env.DEFAULT_BUILD_PATH, launch.py's KILOBOT_BUILD_PATH).
public static class BuildPlayer
{
    const string OutputPath = "Builds/Kilobot.x86_64";

    static string[] EnabledScenes()
    {
        // whatever is ticked in Build Settings, in order -- not a hardcoded
        // list, so adding a scene in the Editor does not silently not-build
        var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
        if (scenes.Length == 0)
        {
            throw new Exception("No enabled scenes in Build Settings -- nothing to build.");
        }
        return scenes;
    }

    public static void BuildLinux()
    {
        var scenes = EnabledScenes();
        Debug.Log("BuildPlayer: building " + scenes.Length + " scene(s) -> " + OutputPath);
        foreach (var s in scenes)
        {
            Debug.Log("BuildPlayer:   scene " + s);
        }

        var options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = OutputPath,
            target = BuildTarget.StandaloneLinux64,
            options = BuildOptions.None,
        };

        BuildReport report = UnityEditor.BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;
        Debug.Log("BuildPlayer: result=" + summary.result
                  + " errors=" + summary.totalErrors
                  + " warnings=" + summary.totalWarnings
                  + " size=" + summary.totalSize + " bytes");

        if (summary.result != BuildResult.Succeeded)
        {
            // batchmode's own exit code does not otherwise reflect a failed
            // build, so a broken player would look like a clean run
            EditorApplication.Exit(1);
        }
        EditorApplication.Exit(0);
    }
}
