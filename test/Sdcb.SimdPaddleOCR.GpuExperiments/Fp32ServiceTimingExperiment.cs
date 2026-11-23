using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using SkiaSharp;

// 仅离线整图API，4并发、同进程独立实例、平衡窗口顺序；不启动HTTP服务。
internal static class Fp32ServiceTimingExperiment
{
    internal static int Run(string[] args)
    {
        if(!OperatingSystem.IsWindows())throw new PlatformNotSupportedException();
        using var process=Process.GetCurrentProcess();process.ProcessorAffinity=(nint)0xc03c03;
        if((long)process.ProcessorAffinity!=0xc03c03)throw new InvalidOperationException("P-core affinity mismatch");
        string[] names=["ui-01-business-notice.png","ui-02-chinese-news.png","ui-03-english-article.png","ui-04-receipt.png"];
        var bitmaps=names.Select(n=>SKBitmap.Decode(Path.Combine(args[2],n))).ToArray();
        string[] modes=["cpu","baseline","fp16-control","fp32"];
        var instances=new Dictionary<string,PaddleOcrAll>();var expected=new Dictionary<string,string[]>();var rows=new List<object>();
        void Select(string mode){Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD",mode=="cpu"?"0":"1");Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_KERNEL",mode);}
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE","1");Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_CPU_SEEDED_T25","0");
        try
        {
            foreach(string mode in modes)
            {
                Select(mode);string models=args[1];
                var ocr=PaddleOcrAll.Load(Path.Combine(models,"det.onnx"),Path.Combine(models,"cls.onnx"),Path.Combine(models,"rec.onnx"),Path.Combine(models,"ppocrv6_dict.txt"),new PaddleOcrOptions{
                    Detector=new(){Backend=OcrBackend.Vulkan},Classifier=new(){Backend=OcrBackend.Vulkan},Recognizer=new(){Backend=OcrBackend.Vulkan},LineWorkerCount=2,DetIntraOpThreads=4,GpuCtcIntraOpThreads=4,PreprocessWorkerCount=2});
                instances[mode]=ocr;
                for(int i=0;i<4;i++)RunImage(ocr,bitmaps[i]);
                Window(mode,-1,3);expected[mode]=bitmaps.Select(b=>JsonSerializer.Serialize(RunImage(ocr,b))).ToArray();
            }
            // Williams平衡顺序，每个模式各占一次窗口位置，前后邻接也平衡。
            int[][] order=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]];
            for(int round=0;round<4;round++)foreach(int mode in order[round])Window(modes[mode],round,10);
            File.WriteAllText(args[3],JsonSerializer.Serialize(new{createdUtc=DateTimeOffset.UtcNow,affinity="0xc03c03",degree=4,secondsPerWindow=10,rounds=4,names,binarySha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(PaddleOcrAll).Assembly.Location))),rows},new JsonSerializerOptions{WriteIndented=true}));
            return 0;
        }
        finally{foreach(var instance in instances.Values)instance.Dispose();foreach(var bitmap in bitmaps)bitmap.Dispose();}

        void Window(string mode,int round,int seconds)
        {
            Select(mode);var ocr=instances[mode];using var ready=new CountdownEvent(4);using var go=new ManualResetEventSlim(false);
            long started=0;int[] counts=new int[4];var latencies=new List<double>[4];PaddleOcrResult?[] last=new PaddleOcrResult?[4];int[] lastIds=new int[4];
            var tasks=Enumerable.Range(0,4).Select(worker=>Task.Factory.StartNew(()=>{
                var times=new List<double>();latencies[worker]=times;int next=worker;ready.Signal();go.Wait();
                while(Stopwatch.GetElapsedTime(started).TotalSeconds<seconds)
                {int i=next++%4;long t=Stopwatch.GetTimestamp();last[worker]=RunImage(ocr,bitmaps[i]);lastIds[worker]=i;times.Add(Stopwatch.GetElapsedTime(t).TotalMilliseconds);counts[worker]++;}
            },CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default)).ToArray();
            ready.Wait();TimeSpan cpu=process.TotalProcessorTime;started=Stopwatch.GetTimestamp();go.Set();Task.WaitAll(tasks);
            double wall=Stopwatch.GetElapsedTime(started).TotalSeconds,cpuMs=(process.TotalProcessorTime-cpu).TotalMilliseconds;int completed=counts.Sum();
            double[] times=latencies.SelectMany(x=>x).Order().ToArray();
            string[] states=Fp32QualityExperiment.States(ocr);
            if(states.Any(s=>s.Contains("alive=False")))throw new InvalidOperationException("GPU fallback");
            if(round>=0)
            {
                for(int w=0;w<4;w++)if(JsonSerializer.Serialize(last[w])!=expected[mode][lastIds[w]])throw new InvalidDataException("Output changed within mode");
                rows.Add(new{round,mode,completed,wallSeconds=wall,imagesPerSecond=completed/wall,cpuMsPerImage=cpuMs/completed,cpuCores=cpuMs/1000/wall,p50Ms=times[times.Length/2],p95Ms=times[(int)(times.Length*.95)],states});
                Console.WriteLine($"round={round} mode={mode} ips={completed/wall:F2} cpu={cpuMs/completed:F2}ms p95={times[(int)(times.Length*.95)]:F2}ms");
            }
        }
    }
    private static unsafe PaddleOcrResult RunImage(PaddleOcrAll ocr,SKBitmap b)=>ocr.Run(new ReadOnlySpan<byte>((void*)b.GetPixels(),b.RowBytes*b.Height),b.Width,b.Height,true,b.RowBytes,ImagePixelFormat.Bgra32);
}
