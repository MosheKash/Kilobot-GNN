using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using System.IO;
using UnityEngine;

public class PostBuildCopyGrpc : IPostprocessBuildWithReport
{
    public int callbackOrder => 0;

    public void OnPostprocessBuild(BuildReport report)
    {
        if (report.summary.platform != BuildTarget.StandaloneLinux64)
            return;

        string buildDir = Path.GetDirectoryName(report.summary.outputPath);
        string productName = Path.GetFileNameWithoutExtension(report.summary.outputPath);

        string src = Path.Combine(buildDir, productName + "_Data", "Plugins", "AnyCPU",
                                  "libgrpc_csharp_ext.x64.so");
        string dstDir = Path.Combine(buildDir, productName + "_Data", "Plugins", "x86_64");
        string dst = Path.Combine(dstDir, "libgrpc_csharp_ext.x64.so");

        if (!File.Exists(src))
        {
            Debug.LogWarning($"[PostBuild] libgrpc source not found at: {src}");
            return;
        }

        Directory.CreateDirectory(dstDir);
        File.Copy(src, dst, overwrite: true);
        Debug.Log($"[PostBuild] Copied libgrpc_csharp_ext.x64.so → {dstDir}");
    }
}