using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdPaddleOCR.Experimental;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// 同进程、真实特征、交替10秒窗口；只测CPU投影，禁止据此推断服务吞吐。
internal static class SeededTimingExperiment
{
    private sealed record Feature(string file,int batch,int steps,int Inner,int Columns);
    private sealed record Work(float[] Data,int Batch,int Steps,int[] Ids,float[] Scores);
    internal static int Run(string[] args)
    {
        if(!System.OperatingSystem.IsWindows())throw new PlatformNotSupportedException();
        using var process=Process.GetCurrentProcess();
        long mask=Convert.ToInt64(args[4],16);int threads=int.Parse(args[5]);int rounds=int.Parse(args[6]);
        process.ProcessorAffinity=(nint)mask;
        if((long)process.ProcessorAffinity!=mask)throw new InvalidOperationException("Affinity mismatch");
        if(!Avx2.IsSupported||Avx512F.IsSupported)throw new InvalidOperationException("AVX2 prototype required");
        using var input=File.OpenRead(Path.Combine(args[1],"rec.onnx"));
        using var model=Model.Load(input);
        var p=ModelHeads.FindSoftmaxHead(model.Onnx,[1,3,48,320],Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel.Options)!.Projection!;
        var head=new CtcProjection(p.Weights,p.Bias,p.Inner,p.Columns);
        var items=JsonSerializer.Deserialize<Feature[]>(File.ReadAllText(args[2]),new JsonSerializerOptions{PropertyNameCaseInsensitive=true})!;
        var work=items.Select(f=>{
            using var file=File.OpenRead(Path.Combine(Path.GetDirectoryName(args[2])!,f.file));
            using var gzip=new GZipStream(file,CompressionMode.Decompress);using var memory=new MemoryStream();gzip.CopyTo(memory);
            byte[] data=memory.ToArray();var values=new float[data.Length/4];data.CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
            if(f.Inner!=p.Inner||f.Columns!=p.Columns||values.Length!=f.batch*f.steps*p.Inner)throw new InvalidDataException("Feature shape mismatch");
            return new Work(values,f.batch,f.steps,new int[f.batch*f.steps],new float[f.batch*f.steps]);
        }).ToArray();
        // 验证直接调用候选成功，不能把回退误记成候选速度。
        foreach(var w in work)if(!MatMul.TryArgMaxSeeded(w.Data,p.Weights,p.Bias??[],w.Ids,w.Scores,w.Batch,w.Steps,p.Inner,p.Columns,head.Packed,threads))
            throw new InvalidOperationException("Prototype fell back");
        string Hash()=>Convert.ToHexString(SHA256.HashData(work.SelectMany(w=>w.Ids).SelectMany(BitConverter.GetBytes).ToArray()));
        void Sweep(bool seeded){foreach(var w in work)if(!CpuSeededCtcHead.ArgMax(head,w.Data,w.Batch,w.Steps,w.Ids,w.Scores,threads,seeded))throw new InvalidOperationException("Head unavailable");}
        // 两版各预热至少3秒，模型加载、解压、JIT不计入窗口。
        foreach(bool seeded in new[]{false,true}){long t=Stopwatch.GetTimestamp();do{Sweep(seeded);}while(Stopwatch.GetElapsedTime(t).TotalSeconds<3);}
        Sweep(false);string expected=Hash();var results=new List<object>();
        for(int round=0;round<rounds;round++)foreach(bool seeded in round%2==0?new[]{false,true}:new[]{true,false})
        {
            GC.Collect();GC.WaitForPendingFinalizers();long allocations=GC.GetTotalAllocatedBytes();
            TimeSpan cpu=process.TotalProcessorTime;long t=Stopwatch.GetTimestamp();int count=0;
            do{Sweep(seeded);count++;}while(Stopwatch.GetElapsedTime(t).TotalSeconds<10);
            double wall=Stopwatch.GetElapsedTime(t).TotalMilliseconds,cpuMs=(process.TotalProcessorTime-cpu).TotalMilliseconds;
            long allocated=GC.GetTotalAllocatedBytes()-allocations;string hash=Hash();if(hash!=expected)throw new InvalidDataException("ArgMax hash mismatch");
            results.Add(new{round,seeded,sweeps=count,wallMs=wall,cpuMs,wallMsPerSweep=wall/count,cpuMsPerSweep=cpuMs/count,allocated,indexHash=hash});
            Console.WriteLine($"round={round} seeded={seeded} sweep={wall/count:F3}ms cpu={cpuMs/count:F3}ms");
        }
        File.WriteAllText(args[3],JsonSerializer.Serialize(new{createdUtc=DateTimeOffset.UtcNow,mask=$"0x{mask:x}",actualMask=$"0x{(long)process.ProcessorAffinity:x}",threads,rounds,positions=work.Sum(w=>w.Ids.Length),units=work.Length,p.Inner,p.Columns,
            manifestSha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(args[2]))),binarySha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(CpuSeededCtcHead).Assembly.Location))),results},new JsonSerializerOptions{WriteIndented=true}));
        return 0;
    }
}
