using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;

// 每种模式一个隔离进程，避免切换环境变量后复用已缓存的GPU图。
internal static class Fp32QualityExperiment
{
    private sealed record Case(string Id,string Path);
    internal static int Run(string[] args)
    {
        string models=args[1],output=args[3],mode=args[4];Directory.CreateDirectory(output);
        if(mode is not("cpu" or "baseline" or "fp16-control" or "fp32"))throw new ArgumentException("Unknown quality mode");
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD",mode=="cpu"?"0":"1");
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_KERNEL",mode);
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE","1");
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_CPU_SEEDED_T25","0");
        using var ocr=PaddleOcrAll.Load(Path.Combine(models,"det.onnx"),Path.Combine(models,"cls.onnx"),Path.Combine(models,"rec.onnx"),Path.Combine(models,"ppocrv6_dict.txt"),new PaddleOcrOptions{
            Detector=new(){Backend=OcrBackend.Vulkan},Classifier=new(){Backend=OcrBackend.Vulkan},Recognizer=new(){Backend=OcrBackend.Vulkan},LineWorkerCount=2,DetIntraOpThreads=4});
        var cases=JsonSerializer.Deserialize<Case[]>(File.ReadAllText(args[2]),new JsonSerializerOptions{PropertyNameCaseInsensitive=true})!;
        foreach(var item in cases)
        {
            using var bitmap=SKBitmap.Decode(item.Path);var first=RunImage(ocr,bitmap);var result=RunImage(ocr,bitmap);
            string[] states=States(ocr);
            if(states.Any(s=>s.Contains("alive=False"))||!states.Any(s=>s.StartsWith("_recognizer:GpuOcrSession:alive=True")))throw new InvalidOperationException("GPU fallback: "+string.Join(',',states));
            bool stable=JsonSerializer.Serialize(first)==JsonSerializer.Serialize(result);if(!stable)throw new InvalidDataException("Unstable output");
            File.WriteAllText(Path.Combine(output,item.Id+".json"),JsonSerializer.Serialize(new{item.Id,mode,stable,states,sha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(item.Path))),result,characters=result.Lines.Select(l=>l.EstimateCharacterBoxes()).ToArray()},new JsonSerializerOptions{WriteIndented=true}));
            Console.WriteLine($"{mode} {item.Id}: {result.Lines.Length} lines");
        }
        return 0;
    }
    private static unsafe PaddleOcrResult RunImage(PaddleOcrAll ocr,SKBitmap bitmap)=>ocr.Run(new ReadOnlySpan<byte>((void*)bitmap.GetPixels(),bitmap.RowBytes*bitmap.Height),bitmap.Width,bitmap.Height,true,bitmap.RowBytes,ImagePixelFormat.Bgra32);
    internal static string[] States(PaddleOcrAll ocr)
    {
        var states=new List<string>();
        foreach(string name in new[]{"_detector","_recognizer","_classifier"})
        {
            var component=typeof(PaddleOcrAll).GetField(name,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(ocr);
            var bag=component?.GetType().GetField("_sessions",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(component) as System.Collections.IEnumerable;
            if(bag is not null)foreach(var session in bag)states.Add(name+":"+session!.GetType().Name+":alive="+session.GetType().GetProperty("GpuAlive")?.GetValue(session));
        }
        return states.Distinct().ToArray();
    }
}
