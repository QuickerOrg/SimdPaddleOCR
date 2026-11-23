#if NET10_0_OR_GREATER
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdPaddleOCR.Experimental;

// 显式候选：保持整个主干和权重的存储精度，只将最后GEMM输出保存在FP32 scratch。
internal static class GpuFp32Head
{
    internal static readonly EngineExtensions Fp32 = new EngineExtensions().Add(new ProjectionPass(true));
    internal static readonly EngineExtensions Fp16Control = new EngineExtensions().Add(new ProjectionPass(false));
    private sealed record Parameters(int Rows,int Columns,int Inner,int Bias,bool FloatLogits);

    private sealed class ProjectionPass(bool floatLogits) : GraphPass
    {
        public override PassStage Stage=>PassStage.Physical;
        public override void Run(PassContext context)
        {
            if(context.Target!=PlanTarget.Vulkan)throw new NotSupportedException("FP32 CTC prototype requires Vulkan.");
            int at=context.Producer(context.OutputValue);
            if(at<0||context.Nodes[at].Op!=ExecOp.Softmax)throw new NotSupportedException("Terminal Softmax required.");
            var softmax=context.Nodes[at];int logits=softmax.In[0],producer=context.Producer(logits);
            // 实际Small尾部有reshape别名：按编译器的同一逻辑Buf追溯，不把arena重用当成生产者。
            int logitsBuffer=context.Values[logits].Buf;
            if(producer<0&&logitsBuffer>=0)producer=context.Nodes.FindIndex(n=>n.Out.Any(v=>context.Values[v].Buf==logitsBuffer));
            if(producer<0||context.Nodes[producer].Op!=ExecOp.Gemm)throw new NotSupportedException("Terminal Gemm required; tail="+string.Join(";",context.Nodes.TakeLast(5).Select(n=>$"{n.Op}:{n.Name} [{string.Join(',',n.In)}]->[{string.Join(',',n.Out)}]")));
            var gemm=context.Nodes[producer];var gp=(GemmP)gemm.P!;
            int[] shape=context.Values[logits].Shape;int inner=context.Values[gemm.In[0]].Shape[^1];
            // 本轮只验证Small：K=120、N=18710，原后端也走通用gemm而非tensor-core路径。
            if(shape.Length!=3||inner!=120||shape[^1]!=18710||gp.TransB||gp.Act!=ActivationKind.None||context.Values[gemm.In[1]].Floats is null)
                throw new NotSupportedException("FP32 prototype supports only the Small 120x18710 plain projection.");
            if(context.Nodes.Any(n=>!ReferenceEquals(n,softmax)&&n.In.Concat(n.ParamRefs()).Any(v=>context.Values[v].Buf==logitsBuffer)))
                throw new NotSupportedException("Projection logits must have a single reader.");
            context.Values[context.OutputValue].Shape=[shape[0],shape[1],6];
            context.Values[context.OutputValue].Layout=Layout.Identity(3);
            context.Nodes[at]=new ExecNode{Op=ExecOp.Ext,Ext=ProjectionOp.Instance,In=gemm.In,Out=softmax.Out,
                P=new Parameters(checked(shape[0]*shape[1]),shape[^1],inner,gp.BiasValue,floatLogits),Name="experimental-projection-ctc"};
            context.Nodes.RemoveAt(producer);
        }
    }

    private sealed class ProjectionOp : ExtOp
    {
        internal static readonly ProjectionOp Instance=new();
        public override string Name=>"ExperimentalFp32ProjectionCtc";
        public override IEnumerable<int> ParamRefs(ExecNode n){int bias=((Parameters)n.P!).Bias;if(bias>=0)yield return bias;}
        public override IEnumerable<(int value,ConstKind kind)> GpuConsts(ExecNode n)
        {
            yield return (n.In[1],ConstKind.Raw);
            int bias=((Parameters)n.P!).Bias;if(bias>=0)yield return (bias,ConstKind.F32);
        }
        public override void RunCpu(InferenceSession s,ExecNode n)=>throw new NotSupportedException();
        public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel graph,ExecNode n)
        {
            var p=(Parameters)n.P!;
            ulong bytes=checked((ulong)p.Rows*(ulong)p.Columns*(p.FloatLogits?4UL:(ulong)graph.Es));
            var scratch=graph.Scratch(bytes);
            yield return graph.Dispatch(n,p.FloatLogits?"ctc_projection32":"ctc_projection16",
                [graph.Bind(n.In[0]),graph.Bind(n.In[1]),graph.BindOrDummy(p.Bias,ConstKind.F32),scratch],
                VkGraphModel.PcOf((uint)p.Rows,(uint)p.Columns,(uint)p.Inner,p.Bias>=0?2u:0u,0,0,0),
                (uint)(p.Columns+63)/64,(uint)(p.Rows+63)/64,owner:typeof(GpuFp32Head).Assembly);
            yield return graph.Dispatch(n,p.FloatLogits?"ctc_reduce32":"ctc_compact",
                [scratch,graph.Bind(n.Out[0])],VkGraphModel.PcOf((uint)p.Rows,(uint)p.Columns),(uint)p.Rows,
                owner:typeof(GpuFp32Head).Assembly);
        }
    }
}
#endif
