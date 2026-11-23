using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;

internal static class CorpusExperiment
{
    internal static int Run(string[] args)
    {
        string models = args[1], images = args[2], output = args[3];
        using var ocr = PaddleOcrAll.Load(Path.Combine(models,"det.onnx"),Path.Combine(models,"cls.onnx"),Path.Combine(models,"rec.onnx"),Path.Combine(models,"ppocrv6_dict.txt"),new PaddleOcrOptions {
            Detector=new(){Backend=OcrBackend.Vulkan},Recognizer=new(){Backend=OcrBackend.Vulkan},Classifier=new(){Backend=OcrBackend.Cpu},LineWorkerCount=4,DetIntraOpThreads=8
        });
        var rows = new List<object>();
        int failures = 0;
        foreach (string file in Directory.GetFiles(images,"*.png").Order())
        {
            try {
                byte[] data=File.ReadAllBytes(file);
                using var bitmap=SKBitmap.Decode(data);
                PaddleOcrResult Run() {
                    unsafe { return ocr.Run(new ReadOnlySpan<byte>((void*)bitmap.GetPixels(),bitmap.RowBytes*bitmap.Height),bitmap.Width,bitmap.Height,true,bitmap.RowBytes,ImagePixelFormat.Bgra32); }
                }
                var warm=Run();var result=Run();
                rows.Add(new {File=Path.GetFileName(file),Sha256=Convert.ToHexString(SHA256.HashData(data)),Success=true,Stable=JsonSerializer.Serialize(warm)==JsonSerializer.Serialize(result),Result=result});
            } catch(Exception ex) {failures++;rows.Add(new {File=Path.GetFileName(file),Success=false,Error=ex.ToString()});Console.WriteLine(Path.GetFileName(file)+": "+ex.Message);}
        }
        var states = new List<string>();
        foreach(string name in new[]{"_detector","_recognizer","_classifier"}){
            var component=typeof(PaddleOcrAll).GetField(name,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(ocr);
            var bag=component?.GetType().GetField("_sessions",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(component) as System.Collections.IEnumerable;
            if(bag is not null)foreach(var s in bag)states.Add(name+":"+s!.GetType().Name+":alive="+s.GetType().GetProperty("GpuAlive")?.GetValue(s));
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new {Compact=Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD"),SingleWave=Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE"),States=states.Distinct(),Rows=rows},new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Corpus: {rows.Count} images -> {output}");
        bool gpuAlive = states.Any(s => s.StartsWith("_detector:GpuOcrSession:alive=True"))
            && states.Any(s => s.StartsWith("_recognizer:GpuOcrSession:alive=True"))
            && !states.Any(s => s.Contains("GpuOcrSession:alive=False"));
        return failures == 0 && gpuAlive ? 0 : 1;
    }
}
