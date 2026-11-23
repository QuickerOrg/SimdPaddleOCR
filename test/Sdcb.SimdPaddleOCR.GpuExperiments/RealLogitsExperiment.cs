using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using SkiaSharp;
using OcrCompiled = Sdcb.SimdPaddleOCR.OnnxSharp.CompiledModel;

// 仅实验CLI使用：不修改识别库或生产选项。密集张量按unit释放，仅问题行落盘。
internal static class RealLogitsExperiment
{
    internal sealed record ImageCase(string Id, string Path, string? FocusText);
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };
    private static readonly int[] Thresholds = [20, 23, 25];
    private static readonly string[] Modes = ["oracle-final", "lane-running", "first-tile-broadcast"];
    internal sealed class Counter
    {
        public long Rows, VectorCalls, SkippedVectors, RelevantValues, ScalarTailValues, ScalarTailKept, NewMaxLanes;
        public long SkippedVectorsWithNewLaneMaximum;
        public double MaxAbsoluteError, MaxRelativeError, SumAbsoluteError;
    }
    private sealed class Density
    {
        internal readonly Dictionary<string, Counter> Counts = [];
        internal long Positions;
        internal int Columns;
        internal void Merge(Density source)
        {
            if(source.Positions==0)return;
            if(Columns!=0&&Columns!=source.Columns)throw new InvalidOperationException("Mixed vocabularies.");
            Columns=source.Columns;Positions+=source.Positions;
            foreach(var(key,value)in source.Counts)
            {
                if(!Counts.TryGetValue(key,out var c))Counts[key]=c=new();
                c.Rows+=value.Rows;c.VectorCalls+=value.VectorCalls;c.SkippedVectors+=value.SkippedVectors;c.RelevantValues+=value.RelevantValues;
                c.ScalarTailValues+=value.ScalarTailValues;c.ScalarTailKept+=value.ScalarTailKept;c.NewMaxLanes+=value.NewMaxLanes;
                c.SkippedVectorsWithNewLaneMaximum+=value.SkippedVectorsWithNewLaneMaximum;
                c.MaxAbsoluteError=Math.Max(c.MaxAbsoluteError,value.MaxAbsoluteError);c.MaxRelativeError=Math.Max(c.MaxRelativeError,value.MaxRelativeError);c.SumAbsoluteError+=value.SumAbsoluteError;
            }
        }
        internal double Add(ReadOnlySpan<float> values, int id, string kind)
        {
            Columns = values.Length; Positions++;
            int packed = Columns / 16 * 16;
            double maximum = values[id], denominator = 0;
            double[] kept = new double[9];
            Counter[] counters = new Counter[9];
            for (int m = 0; m < 3; m++) for (int t = 0; t < 3; t++)
            {
                string key = Modes[m] + "/T" + Thresholds[t] + "/" + kind;
                if (!Counts.TryGetValue(key, out var counter)) Counts[key] = counter = new();
                counters[m * 3 + t] = counter; counter.Rows++;
            }
            float[] maxima = Enumerable.Repeat(float.NegativeInfinity, 8).ToArray();
            float seed = float.NegativeInfinity;
            for (int c = 0; c < Math.Min(16, Columns); c++) seed = Math.Max(seed, values[c]);
            Span<int> relevant = stackalloc int[9];
            for (int col = 0; col < packed; col += 8)
            {
                relevant.Clear(); double mass = 0; int newMax = 0;
                for (int lane = 0; lane < 8; lane++)
                {
                    float value = values[col + lane];
                    if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite diagnostic logit.");
                    mass += Math.Exp(value - maximum);
                    if (value > maxima[lane]) newMax++;
                    for (int t = 0; t < 3; t++)
                    {
                        int threshold = Thresholds[t];
                        if (value >= maximum - threshold) relevant[t]++;
                        if (value >= maxima[lane] - threshold) relevant[3 + t]++;
                        // 首tile还未得到seed，前16列强制执行；后续同时使用seed和lane下界。
                        if (col < 16 || value >= Math.Max(seed, maxima[lane]) - threshold) relevant[6 + t]++;
                    }
                }
                denominator += mass;
                for (int i = 0; i < 9; i++)
                {
                    var counter = counters[i]; counter.VectorCalls++; counter.RelevantValues += relevant[i];
                    counter.NewMaxLanes += newMax;
                    if (relevant[i] == 0)
                    {
                        counter.SkippedVectors++;
                        if (newMax != 0) counter.SkippedVectorsWithNewLaneMaximum++;
                    }
                    else kept[i] += mass; // AVX2一次执行8个exp，不能把一个有效lane当成一次标量exp。
                }
                for (int lane = 0; lane < 8; lane++) maxima[lane] = Math.Max(maxima[lane], values[col + lane]);
            }
            float running = maxima.Max();
            for (int col = packed; col < Columns; col++)
            {
                float value = values[col];
                if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite diagnostic tail.");
                double mass = Math.Exp(value - maximum); denominator += mass;
                for (int m = 0; m < 3; m++) for (int t = 0; t < 3; t++)
                {
                    int i = m * 3 + t; counters[i].ScalarTailValues++;
                    float bound = m == 0 ? (float)maximum : m == 1 ? running : Math.Max(seed, running);
                    if (value >= bound - Thresholds[t]) { counters[i].ScalarTailKept++; kept[i] += mass; }
                }
                running = Math.Max(running, value);
            }
            for (int i = 0; i < 9; i++)
            {
                if (kept[i] < 1 - 1e-10) throw new InvalidDataException("Winner was discarded.");
                double absolute = Math.Max(0, 1 / kept[i] - 1 / denominator), relative = absolute * denominator;
                counters[i].MaxAbsoluteError = Math.Max(counters[i].MaxAbsoluteError, absolute);
                counters[i].MaxRelativeError = Math.Max(counters[i].MaxRelativeError, relative);
                counters[i].SumAbsoluteError += absolute;
                if (relative > Columns * Math.Exp(-Thresholds[i % 3]) + 1e-11)
                    throw new InvalidDataException("Truncation bound violated.");
            }
            return 1 / denominator;
        }
        internal object Report() => new { Positions, Columns, Rows = Counts.Select(p => new
        {
            key = p.Key, p.Value.Rows, p.Value.VectorCalls, p.Value.SkippedVectors,
            skipPercent = 100d * p.Value.SkippedVectors / p.Value.VectorCalls,
            p.Value.RelevantValues, p.Value.ScalarTailValues, p.Value.ScalarTailKept,
            p.Value.NewMaxLanes, p.Value.SkippedVectorsWithNewLaneMaximum,
            p.Value.MaxAbsoluteError, p.Value.MaxRelativeError, p.Value.SumAbsoluteError
        }).ToArray() };
    }

    internal static int Run(string[] args)
    {
        VerifyCounters();
        string models = args[1], output = args[3]; Directory.CreateDirectory(output);
        var cases = JsonSerializer.Deserialize<ImageCase[]>(File.ReadAllText(args[2]), Json)!;
        using var modelStream=File.OpenRead(Path.Combine(models,"rec.onnx"));
        using var model = Model.Load(modelStream);
        var backend = VkBackend.Default;
        if (backend.DeviceName != "NVIDIA GeForce RTX 4080") throw new InvalidOperationException("Unexpected GPU.");
        var options = OcrCompiled.Options with { Vec4Eltwise = true };
        var tail = ModelHeads.FindSoftmaxHead(model.Onnx, [1,3,48,320], options)!;
        var p = tail.Projection!; var projection = new CtcProjection(p.Weights, p.Bias, p.Inner, p.Columns);
        // 使用库实际解析后的字典，不自行解释BOM/空行。
        using var ocrCpu = Load(models, false);
        var recognizer = (PaddleOcrRecognizer)typeof(PaddleOcrAll).GetField("_recognizer", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(ocrCpu)!;
        string[] labels = (string[])typeof(PaddleOcrRecognizer).GetField("_labels", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(recognizer)!;
        int maxBatch = (int)typeof(PaddleOcrAll).GetField("_recBatchEffective", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(ocrCpu)!;
        var overall = new Density(); var reports = new List<object>();
        foreach (var image in cases)
        {
            Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD", "0");
            using var bitmap = SKBitmap.Decode(image.Path);
            PaddleOcrResult cpu = RunImage(ocrCpu, bitmap);
            PaddleOcrResult? gpu = null;
            if (image.FocusText is not null)
            {
                using var ocrGpu = Load(models, true); gpu = RunImage(ocrGpu, bitmap);
                Write(Path.Combine(output, "focus-full-ocr.json"), new { cpu, gpu,
                    cpuCharacters = cpu.Lines.Select(x => x.EstimateCharacterBoxes()), gpuCharacters = gpu.Lines.Select(x => x.EstimateCharacterBoxes()) });
            }
            Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD", "0");
            var density = new Density(); int mismatches = 0; double maxReplayScoreDelta = 0,maxDenseVsFusedScoreDelta=0; int units = 0;
            object? maxDenseScoreExample=null;Dictionary<string,int> shapeHistogram=[];
            Dictionary<string,double> denseScoreDeltaByKind=[];Dictionary<string,object> denseScoreExamplesByKind=[];
            foreach (int[] members in MakeUnits(cpu.Lines, maxBatch))
            {
                int width = cpu.Lines[members[0]].RecognitionTensorWidth;
                int[] shape = [members.Length,3,48,width];
                using var featuresSession = backend.NewSession(model.Onnx, shape, options with { OutputName = p.ActivationName });
                int[] contentWidths = FillInput(featuresSession.Input, bitmap, cpu.Lines, members, width, options.Nhwc && options.PhysicalIo);
                featuresSession.Run(); float[] features = featuresSession.Output.ToArray();
                int steps = featuresSession.OutputShape[^2], total = members.Length * steps;
                if(!Avx2.IsSupported || Avx512F.IsSupported || steps<4 || steps%4!=0)
                    throw new NotSupportedException("This diagnostic models the current AVX2 packed row gate only.");
                string shapeKey=$"batch={members.Length},width={width},steps={steps},AVX2";
                shapeHistogram[shapeKey]=shapeHistogram.GetValueOrDefault(shapeKey)+total;
                int[] ids = new int[total]; float[] scores = new float[total];
                if (!projection.ArgMax(features, members.Length, steps, ids, scores, 4)) throw new InvalidOperationException("Fused CPU head unavailable.");
                float[] logits = Dense(features, p, members.Length, steps);
                for (int b = 0; b < members.Length; b++)
                {
                    int line = members[b]; var original = cpu.Lines[line];
                    var replay = Decode(labels, ids.AsSpan(b * steps, steps), scores.AsSpan(b * steps, steps), contentWidths[b], width);
                    if (replay.Text != original.Text || !SameSpans(replay.CtcSpans!, original.CtcSpans!, 5e-6f)) mismatches++;
                    maxReplayScoreDelta = Math.Max(maxReplayScoreDelta, Math.Abs(replay.Score - original.RecognitionScore));
                    int previous = 0;
                    for (int t = 0; t < steps; t++)
                    {
                        int row = b * steps + t; var values = logits.AsSpan(row * p.Columns, p.Columns);
                        int winner = ArgMax(values); if (winner != ids[row]) throw new InvalidDataException("Dense logits differ from fused argmax.");
                        string kind = winner == 0 ? "blank" : winner == previous ? "repeat" : "emitted";
                        double reference=density.Add(values, winner, kind);
                        double delta=Math.Abs(reference-scores[row]);
                        if(delta>denseScoreDeltaByKind.GetValueOrDefault(kind))
                        {
                            denseScoreDeltaByKind[kind]=delta;
                            denseScoreExamplesByKind[kind]=new{line,step=t,winner,reference,fused=scores[row],delta};
                        }
                        if(delta>maxDenseVsFusedScoreDelta)
                        {
                            float denominatorReplay=ReplayDenominator(values);
                            if(Math.Abs(denominatorReplay-scores[row])>1e-7f)
                                throw new InvalidDataException("Dense logits cannot reproduce fused denominator arithmetic.");
                            maxDenseVsFusedScoreDelta=delta;
                            maxDenseScoreExample=new{line,step=t,winner,reference,fused=scores[row],denominatorReplay,delta};
                        }
                        previous = winner;
                    }
                    if (image.FocusText is not null && original.Text == image.FocusText)
                        Focus(output, backend, model, options, tail, p, features, logits, members.Length, steps, b,
                            shape, featuresSession.Input.ToArray(), labels, original, gpu!.Lines[line], replay, contentWidths[b], width);
                }
                units++;
            }
            var report = new { image.Id, image.Path, sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(image.Path))),
                model = "Small", p.Inner, p.Columns, maxBatch, units, lines = cpu.Lines.Length, replayMismatches = mismatches,
                maxReplayScoreDelta,maxDenseVsFusedScoreDelta,maxDenseScoreExample,denseScoreDeltaByKind,denseScoreExamplesByKind,shapeHistogram, density = density.Report() };
            overall.Merge(density);reports.Add(report); Write(Path.Combine(output, image.Id + ".json"), report);
            Console.WriteLine($"{image.Id}: {cpu.Lines.Length} lines, {density.Positions} positions, replay mismatches={mismatches}");
            if (mismatches != 0) throw new InvalidDataException("Reconstructed recognition unit did not reproduce pipeline.");
        }
        Write(Path.Combine(output, "summary.json"), new { backend.DeviceName, p.Inner, p.Columns, overall = overall.Report(), images = reports });
        return 0;
    }

    // 只用于定位既有数值差异，照搬当前AVX2分母的更新顺序；没有替换任何生产计算。
    private static float ReplayDenominator(ReadOnlySpan<float> values)
    {
        var maximum=Vector256.Create(float.NegativeInfinity);var sums=Vector256<float>.Zero;
        int packed=values.Length/16*16;
        for(int col=0;col<packed;col+=8)
        {
            var v=Vector256.Create(values[col],values[col+1],values[col+2],values[col+3],values[col+4],values[col+5],values[col+6],values[col+7]);
            var replace=Avx.Compare(v,maximum,FloatComparisonMode.OrderedGreaterThanNonSignaling);
            var e=Sdcb.SimdOnnx.Kernels.SimdMath.ExpCore(Avx.Or(Avx.Subtract(v,maximum),Vector256.Create(-0f)));
            sums=Avx.BlendVariable(Avx.Add(sums,e),Fma.MultiplyAdd(sums,e,Vector256<float>.One),replace);
            maximum=Avx.Max(maximum,v);
        }
        float m=maximum.GetElement(0);for(int lane=1;lane<8;lane++)m=Math.Max(m,maximum.GetElement(lane));
        double total=0;for(int lane=0;lane<8;lane++)if(sums.GetElement(lane)!=0)total+=sums.GetElement(lane)*Math.Exp((double)maximum.GetElement(lane)-m);
        float sum=(float)total;
        for(int col=packed;col<values.Length;col++)
        {
            float v=values[col];if(v>m){sum=sum*MathF.Exp(m-v)+1;m=v;}else sum+=MathF.Exp(v-m);
        }
        return Math.Min(1f,1f/sum);
    }

    private static void VerifyCounters()
    {
        float[] logits=new float[32];logits[0]=30;
        var d=new Density();d.Add(logits,0,"blank");
        foreach(int t in Thresholds)
            if(d.Counts[$"oracle-final/T{t}/blank"].SkippedVectors!=3 || d.Counts[$"lane-running/T{t}/blank"].SkippedVectors!=0 || d.Counts[$"first-tile-broadcast/T{t}/blank"].SkippedVectors!=2)
                throw new InvalidOperationException("Known 75%/0%/50% vector fixture failed.");
        logits=Enumerable.Repeat(-100f,32).ToArray();logits[0]=30;Array.Fill(logits,-50f,16,8);
        d=new Density();d.Add(logits,0,"blank");
        if(d.Counts["first-tile-broadcast/T25/blank"].SkippedVectorsWithNewLaneMaximum!=1)
            throw new InvalidOperationException("Seeded skip must report lane-state updates.");
        logits=new float[34];logits[33]=30;d=new Density();d.Add(logits,33,"emitted");
        if(d.Counts["oracle-final/T25/emitted"].ScalarTailKept!=1)
            throw new InvalidOperationException("Scalar-tail winner fixture failed.");
    }

    private static PaddleOcrAll Load(string models, bool compact)
    {
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD", compact ? "1" : "0");
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE", "1");
        Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_KERNEL", "baseline");
        return PaddleOcrAll.Load(Path.Combine(models,"det.onnx"),Path.Combine(models,"cls.onnx"),Path.Combine(models,"rec.onnx"),Path.Combine(models,"ppocrv6_dict.txt"),new PaddleOcrOptions
        { Detector=new(){Backend=OcrBackend.Vulkan}, Classifier=new(){Backend=OcrBackend.Vulkan}, Recognizer=new(){Backend=OcrBackend.Vulkan},LineWorkerCount=2,DetIntraOpThreads=4 });
    }
    private static unsafe PaddleOcrResult RunImage(PaddleOcrAll ocr, SKBitmap bitmap) => ocr.Run(
        new ReadOnlySpan<byte>((void*)bitmap.GetPixels(),bitmap.RowBytes*bitmap.Height),bitmap.Width,bitmap.Height,true,bitmap.RowBytes,ImagePixelFormat.Bgra32);
    private static IEnumerable<int[]> MakeUnits(PaddleOcrLine[] lines, int max)
    {
        Dictionary<int,List<int>> groups=[]; List<int[]> result=[];
        for(int i=0;i<lines.Length;i++)
        { int width=lines[i].RecognitionTensorWidth; if(!groups.TryGetValue(width,out var list))groups[width]=list=[];list.Add(i);if(list.Count==max){result.Add(list.ToArray());list.Clear();} }
        result.AddRange(groups.Values.Where(x=>x.Count>0).Select(x=>x.ToArray()));return result;
    }
    private static unsafe int[] FillInput(Span<float> input, SKBitmap bitmap,PaddleOcrLine[] lines,int[] members,int width,bool nhwc)
    {
        var source=new ReadOnlySpan<byte>((void*)bitmap.GetPixels(),bitmap.RowBytes*bitmap.Height);int[] resized=new int[members.Length];
        for(int b=0;b<members.Length;b++)
        {
            var line=lines[members[b]];byte[] crop=PPOCRCrop.Extract(source,bitmap.Width,bitmap.Height,bitmap.RowBytes,line.Box,out int w,out int h,ImagePixelFormat.Bgra32);
            if(line.AppliedRotationDegrees==180)PPOCRCrop.Rotate180(crop,w,h);
            resized[b]=PPOCRPreprocess.Rec(crop,w,h,w*3,width,input.Slice(b*3*48*width,3*48*width),nhwc);
        }
        return resized;
    }
    private static float[] Dense(float[] features,LinearProjection p,int batch,int steps,bool halfWeights=false,bool halfOutput=false)
    {
        float[] weights=halfWeights?p.Weights.Select(x=>(float)(Half)x).ToArray():p.Weights;
        float[]? bias=halfWeights?p.Bias?.Select(x=>(float)(Half)x).ToArray():p.Bias;
        var projection=new CtcProjection(weights,bias,p.Inner,p.Columns);
        float[] output=new float[batch*steps*p.Columns];
        Parallel.For(0,batch,b=>
        {
            MatMul.Try(features.AsSpan(b*steps*p.Inner,steps*p.Inner),weights,output.AsSpan(b*steps*p.Columns,steps*p.Columns),1,steps,p.Inner,p.Columns,projection.Packed);
            for(int row=0;row<steps;row++)
            {
                int start=(b*steps+row)*p.Columns;
                for(int c=0;c<p.Columns;c++)if(bias is not null)output[start+c]+=bias[c];
                // 融合AVX2列尾以bias起始并逐项乘加，与dense核的bias后加顺序不同。
                for(int c=p.Columns/16*16;c<p.Columns;c++)
                { float v=bias?[c]??0;for(int k=0;k<p.Inner;k++)v+=features[(b*steps+row)*p.Inner+k]*weights[k*p.Columns+c];output[start+c]=v; }
            }
        });
        if(halfOutput)for(int i=0;i<output.Length;i++)output[i]=(float)(Half)output[i];
        return output;
    }
    private static int ArgMax(ReadOnlySpan<float> values){int at=0;for(int i=1;i<values.Length;i++)if(values[i]>values[at])at=i;return at;}
    private static (int[] Ids,float[] Scores) Winners(float[] logits,int rows,int columns)
    {
        int[] ids=new int[rows];float[] scores=new float[rows];
        for(int row=0;row<rows;row++){var v=logits.AsSpan(row*columns,columns);int id=ArgMax(v);double sum=0;for(int c=0;c<columns;c++)sum+=Math.Exp((double)v[c]-v[id]);ids[row]=id;scores[row]=(float)(1/sum);}
        return(ids,scores);
    }
    private static PaddleOcrRecognitionResult Decode(string[] labels,ReadOnlySpan<int> ids,ReadOnlySpan<float> scores,int content,int width) =>
        PaddleOcrRecognizer.DecodeCtc(labels,labels.Max(x=>x.Length),true,[],ids,scores,ids.Length,content,width,false);
    private static bool SameSpans(PaddleOcrCtcSpan[] a,PaddleOcrCtcSpan[] b,float tolerance) => a.Length==b.Length && a.Zip(b).All(x=>x.First.Text==x.Second.Text&&x.First.StartColumn==x.Second.StartColumn&&x.First.EndColumn==x.Second.EndColumn&&Math.Abs(x.First.Score-x.Second.Score)<=tolerance);
    private static void Focus(string output,VkBackend backend,Model model,CompileOptions options,SoftmaxHead tail,LinearProjection p,
        float[] features,float[] cpuLogits,int batch,int steps,int member,int[] shape,float[] input,string[] labels,PaddleOcrLine cpuLine,PaddleOcrLine gpuLine,
        PaddleOcrRecognitionResult replay,int content,int width)
    {
        using var dense=backend.NewSession(model.Onnx,shape,options with { OutputName=tail.LogitsName });input.CopyTo(dense.Input);dense.Run();
        float[] gpuLogits=dense.Output.ToArray();var gpuWinner=Winners(gpuLogits,batch*steps,p.Columns);
        var gpuReplay=Decode(labels,gpuWinner.Ids.AsSpan(member*steps,steps),gpuWinner.Scores.AsSpan(member*steps,steps),content,width);
        // 独立projection-only图：同一份特征，避免把投影误差直接归因于主干。
        var builder=new OnnxModelBuilder(model.Onnx.Opset);var f=builder.Input("features",batch,steps,p.Inner);
        var w=builder.Constant("weights",p.Weights,p.Inner,p.Columns);var z=builder.Node("MatMul",[f,w],"logits");
        if(p.Bias is not null)z=builder.Node("Add",[z,builder.Constant("bias",p.Bias,p.Columns)],"biased");
        builder.Output(z);var headModel=builder.Build();
        using var headSession=backend.NewSession(headModel,[batch,steps,p.Inner],options);features.CopyTo(headSession.Input);headSession.Run();
        float[] fixedGpu=headSession.Output.ToArray();
        float[] halfWeights=Dense(features,p,batch,steps,true,false),halfOutput=Dense(features,p,batch,steps,false,true),both=Dense(features,p,batch,steps,true,true);
        var variants=new Dictionary<string,float[]>{{"cpu",cpuLogits},{"gpu-full",gpuLogits},{"gpu-fixed-features",fixedGpu},{"cpu-half-weights",halfWeights},{"cpu-half-output",halfOutput},{"cpu-half-both",both}};
        var details=new Dictionary<string,object>();
        foreach(var(key,values)in variants)
        {
            float[] line=values.AsSpan(member*steps*p.Columns,steps*p.Columns).ToArray();var win=Winners(line,steps,p.Columns);
            var decoded=Decode(labels,win.Ids,win.Scores,content,width);
            Dump(Path.Combine(output,"focus-"+key+".f32.gz"),line);
            details[key]=new{ decoded, rows=Enumerable.Range(0,steps).Select(t=>Top(line,t,p.Columns,labels)).ToArray() };
        }
        Dump(Path.Combine(output,"focus-features.f32.gz"),features.AsSpan(member*steps*p.Inner,steps*p.Inner));
        Dump(Path.Combine(output,"focus-normalized-input.f32.gz"),input.AsSpan(member*3*48*width,3*48*width));
        Write(Path.Combine(output,"focus-trace.json"),new{ shape,steps,p.Inner,p.Columns,member,content,width,cpuLine,gpuLine,replay,gpuReplay,
            gpuDenseStorageF16=dense.F16,fixedProjectionStorageF16=headSession.F16,
            gpuDenseMatchesPipeline=SameSpans(gpuReplay.CtcSpans!,gpuLine.CtcSpans!,5e-6f),
            fixedGpuMaxAbsDelta=MaxDelta(gpuLogits,fixedGpu),variants=details });
    }
    private static double MaxDelta(float[] a,float[] b){double d=0;for(int i=0;i<a.Length;i++)d=Math.Max(d,Math.Abs(a[i]-b[i]));return d;}
    private static object Top(float[] values,int row,int columns,string[] labels)
    {
        int start=row*columns,first=0,second=1;if(values[start+second]>values[start+first])(first,second)=(second,first);
        for(int c=2;c<columns;c++)if(values[start+c]>values[start+first]){second=first;first=c;}else if(values[start+c]>values[start+second])second=c;
        string Label(int id)=>id==0?"<blank>":id==labels.Length+1?" ":labels[id-1];double sum=0;for(int c=0;c<columns;c++)sum+=Math.Exp((double)values[start+c]-values[start+first]);
        return new{step=row,first,label=Label(first),logit=values[start+first],second,secondLabel=Label(second),secondLogit=values[start+second],margin=values[start+first]-values[start+second],probability=1/sum,blankLogit=values[start]};
    }
    private static void Dump(string path,ReadOnlySpan<float> values){using var f=File.Create(path);using var z=new GZipStream(f,CompressionLevel.Fastest);z.Write(MemoryMarshal.AsBytes(values));}
    private static void Write(string path,object value)=>File.WriteAllText(path,JsonSerializer.Serialize(value,Json));
}
