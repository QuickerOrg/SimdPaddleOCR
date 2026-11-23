#if NET10_0_OR_GREATER
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.Experimental;

// Opt-in experiment, deliberately absent from product options/defaults.
internal static class GpuCompactHead
{
    internal static bool Enabled => Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_HEAD") == "1";
    internal static bool SingleWave => Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE") == "1";
    private static readonly EngineExtensions Baseline = new EngineExtensions().Add(new CompactPass("ctc_compact"));
    private static readonly EngineExtensions Subgroup = new EngineExtensions().Add(new CompactPass("ctc_compact_subgroup"));
    private static readonly EngineExtensions Online = new EngineExtensions().Add(new CompactPass("ctc_compact_online"));
    internal static EngineExtensions Extensions => ForKernel(Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_COMPACT_KERNEL"));
    internal static EngineExtensions ForKernel(string? kernel) => kernel switch
    {
        "subgroup" => Subgroup,
        "online" => Online,
        "fp32" => GpuFp32Head.Fp32,
        "fp16-control" => GpuFp32Head.Fp16Control,
        _ => Baseline,
    };

    internal static void Unpack(ReadOnlySpan<float> packed, Span<int> indices, Span<float> scores)
    {
        if (packed.Length != indices.Length * 6 || scores.Length != indices.Length)
            throw new ArgumentException("Invalid compact CTC shape.");
        for (int i = 0; i < indices.Length; i++)
        {
            int offset = i * 6;
            indices[i] = (int)packed[offset] | ((int)packed[offset + 1] << 8);
            uint bits = 0;
            for (int k = 0; k < 4; k++) bits |= (uint)packed[offset + 2 + k] << (8 * k);
            scores[i] = BitConverter.Int32BitsToSingle((int)bits);
        }
    }

    private sealed class CompactPass(string shader) : GraphPass
    {
        public override PassStage Stage => PassStage.Physical;

        public override void Run(PassContext context)
        {
            if (context.Target != PlanTarget.Vulkan)
                throw new NotSupportedException("Compact CTC requires Vulkan.");
            int at = context.Producer(context.OutputValue);
            if (at < 0 || context.Nodes[at].Op != ExecOp.Softmax)
                throw new NotSupportedException("Compact CTC requires a terminal Softmax.");
            var node = context.Nodes[at];
            int[] shape = context.Values[node.In[0]].Shape;
            if (shape.Length != 3 || shape[^1] < 1024 || shape[^1] > 65536)
                throw new NotSupportedException("Unsupported CTC vocabulary shape.");
            int rows = checked(shape[0] * shape[1]);
            int columns = shape[^1];
            context.Values[context.OutputValue].Shape = [shape[0], shape[1], 6];
            context.Values[context.OutputValue].Layout = Layout.Identity(3);
            context.Nodes[at] = new ExecNode
            {
                Op = ExecOp.Ext, Ext = CompactOp.Instance, In = node.In, Out = node.Out,
                P = (rows, columns, shader), Name = "experimental-compact-ctc",
            };
        }
    }

    private sealed class CompactOp : ExtOp
    {
        internal static readonly CompactOp Instance = new();
        public override string Name => "ExperimentalCompactCtc";
        public override void RunCpu(InferenceSession session, ExecNode node) => throw new NotSupportedException();

        public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel graph, ExecNode node)
        {
            var (rows, columns, shader) = ((int, int, string))node.P!;
            yield return graph.Dispatch(node, shader, [graph.Bind(node.In[0]), graph.Bind(node.Out[0])],
                VkGraphModel.PcOfF([(uint)rows, (uint)columns]), (uint)rows, owner: typeof(GpuCompactHead).Assembly);
        }
    }
}
#endif
