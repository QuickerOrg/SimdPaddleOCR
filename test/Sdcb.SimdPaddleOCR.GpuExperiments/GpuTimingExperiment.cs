using System.Reflection;
using System.Text.Json;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdPaddleOCR.Experimental;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using OcrCompiledModel = Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel;

internal static class GpuTimingExperiment
{
    internal static int Run(string[] args)
    {
        using var input = File.OpenRead(Path.Combine(args[1], "rec.onnx"));
        using var model = Model.Load(input);
        var backend = VkBackend.Default;
        var rows = new List<object>();
        foreach (var (batch, width) in new[] { (1, 320), (1, 800), (1, 1152), (8, 320), (16, 320) })
        {
            int[] shape = [batch, 3, 48, width];
            var head = ModelHeads.FindSoftmaxHead(model.Onnx, shape, OcrCompiledModel.Options)!.Projection!;
            string[] kernels = ["activation", "baseline", "subgroup", "online"];
            if(Environment.GetEnvironmentVariable("SIMD_OCR_GPU_PROFILE_FP32")=="1")kernels=["activation","baseline","fp16-control","fp32"];
            if (Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_GPU_PROFILE_REVERSE") == "1")
                Array.Reverse(kernels);
            foreach (string kernel in kernels)
            {
                bool compact = kernel != "activation";
                var options = OcrCompiledModel.Options with
                {
                    Vec4Eltwise = true,
                    OutputName = compact ? null : head.ActivationName,
                    Extensions = compact ? GpuCompactHead.ForKernel(kernel) : null,
                };
                using var session = backend.NewSession(model.Onnx, shape, options);
                for (int i = 0; i < session.Input.Length; i++) session.Input[i] = MathF.Sin(i * .037f);
                session.Run();
                var method = session.GetType().GetMethod("DebugProfile", BindingFlags.NonPublic | BindingFlags.Instance)
                    ?? throw new NotSupportedException("Pinned Vulkan session has no DebugProfile.");
                var timings = (Dictionary<string, double>)method.Invoke(session, null)!;
                var graph = session.GetType().GetField("_graph", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(session)!;
                var dispatches = (Array)graph.GetType().GetField("Dispatches", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(graph)!;
                var shaders = dispatches.Cast<object>().Select(d => (string)d.GetType().GetField("Name")!.GetValue(d)!).ToArray();
                var row = new { batch, width, compact, kernel, Device = backend.DeviceName,
                    ArenaBytes=(ulong)graph.GetType().GetProperty("ArenaBytes",BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(graph)!,
                    TotalGpuMs = timings.Values.Sum(), Timings = timings, Shaders = shaders,
                    DispatchTimings = session.PerDispatch!.Select(t => new { name = t.Name, milliseconds = t.Ms }).ToArray() };
                rows.Add(row);
                Console.WriteLine($"batch={batch} width={width} kernel={kernel} GPU={row.TotalGpuMs:F3}ms kernels={shaders.Length}");
            }
        }
        File.WriteAllText(args[2], JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
