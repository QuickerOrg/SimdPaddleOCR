using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// 使用实际 Small 模型权重，仅测融合投影，不包含 GPU、解码、HTTP 和模型加载。
internal static class CpuHeadTimingExperiment
{
    internal static int Run(string[] args)
    {
        using var input = File.OpenRead(Path.Combine(args[1], "rec.onnx"));
        using var model = Model.Load(input);
        var p = ModelHeads.FindSoftmaxHead(model.Onnx, [1, 3, 48, 320],
            Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel.Options)!.Projection!;
        var head = new CtcProjection(p.Weights, p.Bias, p.Inner, p.Columns);
        var results = new List<object>();
        using var process = Process.GetCurrentProcess();
        foreach (int rows in new[] { 8, 80, 320 })
        foreach (int threads in new[] { 1, 4 })
        {
            float[] features = Enumerable.Range(0, rows * p.Inner).Select(i => MathF.Sin(i * .037f)).ToArray();
            int[] ids = new int[rows]; float[] scores = new float[rows];
            var warmup = Stopwatch.StartNew();
            do
            {
                if (!head.ArgMax(features, 1, rows, ids, scores, threads))
                    throw new InvalidOperationException("Fused CTC path unavailable.");
            } while (warmup.Elapsed.TotalSeconds < 1);
            TimeSpan cpuBefore = process.TotalProcessorTime;
            var watch = Stopwatch.StartNew();
            int iterations = 0;
            do { head.ArgMax(features, 1, rows, ids, scores, threads); iterations++; }
            while (watch.Elapsed.TotalSeconds < 1);
            watch.Stop();
            double cpuMs = (process.TotalProcessorTime - cpuBefore).TotalMilliseconds;
            string hash = Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ids.AsSpan())));
            results.Add(new { rows, threads, p.Inner, p.Columns, iterations,
                wallMsPerCall = watch.Elapsed.TotalMilliseconds / iterations,
                cpuMsPerCall = cpuMs / iterations, indexHash = hash,
                scoreMin = scores.Min(), scoreMax = scores.Max(),
                Avx2 = Avx2.IsSupported, Avx512 = Avx512F.IsSupported });
        }
        File.WriteAllText(args[2], JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
