using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;

if(args.Length==3 && args[0]=="--gpu-profile") return GpuTimingExperiment.Run(args);

if(args.Length==4 && args[0]=="--corpus") return CorpusExperiment.Run(args);

string models=args[0], images=args[1], output=args[2];
int seconds=int.Parse(args[3]), degree=int.Parse(args[4]), ctc=int.Parse(args[5]), prep=int.Parse(args[6]);
bool profile=args.Contains("--profile"), decode=args.Contains("--decode"), cpu=args.Contains("--cpu");
string[] files=["ui-01-business-notice.png","ui-02-chinese-news.png","ui-03-english-article.png","ui-04-receipt.png"];
var data=files.Select(n=>File.ReadAllBytes(Path.Combine(images,n))).ToArray();
var bitmaps=data.Select(d=>SKBitmap.Decode(d)).ToArray();
var options=new PaddleOcrOptions {
 Detector=new(){Backend=cpu?OcrBackend.Cpu:OcrBackend.Vulkan},
 Recognizer=new(){Backend=cpu?OcrBackend.Cpu:OcrBackend.Vulkan},
 Classifier=new(){Backend=args.Contains("--gpu-cls")?OcrBackend.Vulkan:OcrBackend.Cpu},
 LineWorkerCount=4, DetIntraOpThreads=8, GpuCtcIntraOpThreads=ctc, PreprocessWorkerCount=prep,
};
using var ocr=PaddleOcrAll.Load(Path.Combine(models,"det.onnx"),Path.Combine(models,"cls.onnx"),Path.Combine(models,"rec.onnx"),Path.Combine(models,"ppocrv6_dict.txt"),options);
PaddleOcrResult Run(int i) {
 if(!decode)return RunBitmap(bitmaps[i]);
 using var bitmap=SKBitmap.Decode(data[i]);return RunBitmap(bitmap);
}
unsafe PaddleOcrResult RunBitmap(SKBitmap bitmap)=>ocr.Run(new ReadOnlySpan<byte>((void*)bitmap.GetPixels(),bitmap.RowBytes*bitmap.Height),bitmap.Width,bitmap.Height,true,bitmap.RowBytes,ImagePixelFormat.Bgra32);
var mismatchExamples=new System.Collections.Concurrent.ConcurrentDictionary<string,string>();
var baselines=Enumerable.Range(0,4).Select(i=>JsonSerializer.Serialize(Run(i))).ToArray();
int warmupSeconds = int.Parse(Environment.GetEnvironmentVariable("SIMD_OCR_BENCH_WARMUP_SECONDS") ?? "5");
long warmStart = Stopwatch.GetTimestamp();
using (var warmReady = new CountdownEvent(degree))
using (var warmGo = new ManualResetEventSlim(false))
{
    var warmTasks = Enumerable.Range(0, degree).Select(worker => Task.Factory.StartNew(() =>
    {
        warmReady.Signal(); warmGo.Wait();
        int n = worker;
        do { Run(n++ % 4); } while (Stopwatch.GetElapsedTime(warmStart).TotalSeconds < warmupSeconds);
    }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
    warmReady.Wait(); warmStart = Stopwatch.GetTimestamp(); warmGo.Set();
    await Task.WhenAll(warmTasks);
}
// Establish the comparison after warming every caller/session shape, not on a cold route.
for (int i = 0; i < 4; i++) baselines[i] = JsonSerializer.Serialize(Run(i));
PipelineProfiler.Enable(profile);
var results=new List<(int Image,double Ms,double Done,bool Match)>[degree];
using var ready=new CountdownEvent(degree);using var go=new ManualResetEventSlim(false);
long started=0;
var jobs=Enumerable.Range(0,degree).Select(worker=>Task.Factory.StartNew(()=>{
 var list=new List<(int,double,double,bool)>();results[worker]=list;ready.Signal();go.Wait();int next=worker;
 while(Stopwatch.GetElapsedTime(started).TotalSeconds<seconds){int i=next++%4;long t=Stopwatch.GetTimestamp();var r=Run(i);double ms=Stopwatch.GetElapsedTime(t).TotalMilliseconds;double done=Stopwatch.GetElapsedTime(started).TotalSeconds;
 // Hash/result checks are outside the per-call latency; wall throughput includes this validation.
 string json=JsonSerializer.Serialize(r);if(json!=baselines[i])mismatchExamples.TryAdd(files[i],json);
 list.Add((i,ms,done,json==baselines[i]));}
},CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default)).ToArray();
ready.Wait();using var process=Process.GetCurrentProcess();TimeSpan cpuStart=process.TotalProcessorTime;
started=Stopwatch.GetTimestamp();go.Set();await Task.WhenAll(jobs);
double wall=Stopwatch.GetElapsedTime(started).TotalSeconds, cpuSeconds=(process.TotalProcessorTime-cpuStart).TotalSeconds;
var rows=results.SelectMany(r=>r).Where(r=>r.Done<=seconds).ToArray();
var stages=PipelineProfiler.Snapshot().Select((r,i)=>new{Name=PipelineProfiler.StageNames[i],r.Milliseconds,r.Calls}).ToArray();PipelineProfiler.Enable(false);
var states=new List<string>();
var batchesInfo=new List<object>();
foreach(string name in new[]{"_detector","_recognizer","_classifier"}){
 var component=typeof(PaddleOcrAll).GetField(name,BindingFlags.NonPublic|BindingFlags.Instance)!.GetValue(ocr);
 if(component is null)continue;
 var bag=component.GetType().GetField("_sessions",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(component) as System.Collections.IEnumerable;
 if(bag is not null)foreach(var s in bag) {
    states.Add(name+":"+s!.GetType().Name+":alive="+s.GetType().GetProperty("GpuAlive")?.GetValue(s));
    var runner=s.GetType().GetField("_runner",BindingFlags.NonPublic|BindingFlags.Instance)?.GetValue(s);
    if(runner is not null)batchesInfo.Add(new{Component=name,Count=runner.GetType().GetProperty("Count")?.GetValue(runner),Waves=runner.GetType().GetProperty("Waves")?.GetValue(runner)});
 }
}
if(!cpu&&(!states.Any(s=>s.Contains("GpuOcrSession:alive=True"))||states.Any(s=>s.Contains("GpuOcrSession:alive=False"))))throw new Exception("GPU session unavailable: "+string.Join(',',states));
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output,JsonSerializer.Serialize(new{CreatedUtc=DateTimeOffset.UtcNow,Device=cpu?"CPU":Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.DeviceName,CompactHead=Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD"),SingleWave=Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE"),WarmupSeconds=warmupSeconds,Batches=batchesInfo,GpuCls=args.Contains("--gpu-cls"),BinarySha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(PaddleOcrAll).Assembly.Location))),seconds,degree,ctc,prep,decode,cpu,profile,Completed=rows.Length,ImagesPerSecond=rows.Length/(double)seconds,CpuCores=cpuSeconds/wall,CpuMsPerImage=cpuSeconds*1000/rows.Length,MeanMs=rows.Average(r=>r.Ms),P95Ms=rows.Select(r=>r.Ms).Order().ElementAt(Math.Min(rows.Length-1,(int)(rows.Length*.95))),Mismatches=rows.Count(r=>!r.Match),WorkingSet=process.WorkingSet64,stages,states=states.Distinct(),Images=files.Select((n,i)=>new{File=n,Sha256=Convert.ToHexString(SHA256.HashData(data[i]))}),Baselines=baselines,MismatchExamples=mismatchExamples,Rows=rows.Select(r=>new{r.Image,r.Ms,r.Done,r.Match})},new JsonSerializerOptions{WriteIndented=true}));
foreach(var bitmap in bitmaps)bitmap.Dispose();
Console.WriteLine($"{output}: {rows.Length/(double)seconds:F2} img/s cpu={cpuSeconds/wall:F2} cores mismatches={rows.Count(r=>!r.Match)}");

return 0;
