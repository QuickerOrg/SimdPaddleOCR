using System.Diagnostics;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Experimental;
using Sdcb.SimdOnnx.Backends.Vulkan;
using SkiaSharp;

internal static class PipelinePhaseExperiment
{
    internal static int Run(string[] args)
    {
        if (VkBackend.Default.DeviceName != Environment.GetEnvironmentVariable("SIMD_OCR_EXPECTED_GPU_NAME"))
            throw new Exception("Unexpected GPU.");
        string[] names = ["table-01-excel-light", "table-02-excel-dark", "table-05-wireless-mixed",
            "table-09-dense-schedule", "table-14-dashed-rules-only", "table-15-explorer-row-drift"];
        var images = names.Select(name =>
        {
            using var bitmap = SKBitmap.Decode(Path.Combine(args[2], name + ".png"));
            byte[] bytes = new byte[bitmap.Width * bitmap.Height * 3];
            for (int y = 0; y < bitmap.Height; y++) for (int x = 0; x < bitmap.Width; x++)
            { var c = bitmap.GetPixel(x, y); int i = (y * bitmap.Width + x) * 3; bytes[i] = c.Blue; bytes[i + 1] = c.Green; bytes[i + 2] = c.Red; }
            return (Bytes: bytes, Width: bitmap.Width, Height: bitmap.Height);
        }).ToArray();
        using var ocr = PaddleOcrAll.Load(Path.Combine(args[1], "det.onnx"), Path.Combine(args[1], "cls.onnx"),
            Path.Combine(args[1], "rec.onnx"), Path.Combine(args[1], "ppocrv6_dict.txt"), new PaddleOcrOptions
            { Detector = new() { Backend = OcrBackend.Vulkan }, Classifier = new() { Backend = OcrBackend.Vulkan },
                Recognizer = new() { Backend = OcrBackend.Vulkan }, LineWorkerCount = 2, DetIntraOpThreads = 4 });
        void Once(int n) { var image = images[n % 6]; ocr.Run(image.Bytes, image.Width, image.Height, true, image.Width * 3, ImagePixelFormat.Bgr24); }
        for (int n = 0; n < 12; n++) Once(n);
        int iterations = int.Parse(Environment.GetEnvironmentVariable("SIMD_OCR_PHASE_ITERATIONS") ?? "30");
        if (iterations is < 1 or > 3000) throw new ArgumentOutOfRangeException(nameof(iterations));
        Task.WaitAll(Enumerable.Range(0, 4).Select(w => Task.Factory.StartNew(() =>
        { for (int n = 0; n < 30; n++) Once(n + w); }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray());
        int totalImages = iterations * 4;
        PipelineProfiler.Enable(true);
        using var process = Process.GetCurrentProcess(); TimeSpan cpu = process.TotalProcessorTime;
        var watch = Stopwatch.StartNew();
        var tasks = Enumerable.Range(0, 4).Select(w => Task.Factory.StartNew(() =>
        { for (int n = 0; n < iterations; n++) Once(n + w); }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
        Task.WaitAll(tasks); watch.Stop(); var snapshot = PipelineProfiler.Snapshot(); PipelineProfiler.Enable(false);
        var rows = snapshot.Select((s, i) => new { name = PipelineProfiler.StageNames[i], msPerImage = s.Milliseconds / totalImages, calls = s.Calls }).ToArray();
        File.WriteAllText(args[3], JsonSerializer.Serialize(new { images = totalImages, seconds = watch.Elapsed.TotalSeconds,
            cpuMsPerImage = (process.TotalProcessorTime - cpu).TotalMilliseconds / totalImages,
            gpuDetResizes = GpuExactDetResize.Completed,
            rows }, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
