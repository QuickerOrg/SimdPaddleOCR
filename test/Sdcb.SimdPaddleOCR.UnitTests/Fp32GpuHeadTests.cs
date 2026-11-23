#if !USE_NS20_LIBRARY
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdPaddleOCR.Experimental;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class Fp32GpuHeadTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Fp32OutputPreservesNearTieLostByHalfOutput(bool alias)
    {
        VkBackend backend;
        try{backend=VkBackend.Default;}catch(Exception ex){Assert.Skip("Vulkan unavailable: "+ex.Message);return;}
        const int k=120,c=18710,steps=4;
        int[] shape=alias?[steps,k]:[1,steps,k];
        var builder=new OnnxModelBuilder(11);var x=builder.Input("features",shape);
        float[] bias=Enumerable.Repeat(-30f,c).ToArray();bias[0]=21.327156f;bias[c-1]=21.331380f;
        var z=builder.Node("MatMul",[x,builder.Constant("weights",new float[k*c],k,c)],"logits");
        z=builder.Node("Add",[z,builder.Constant("bias",bias,c)],"biased");
        if(alias)z=builder.Node("Unsqueeze",[z],"alias",("axes",new[]{0}));
        builder.Output(builder.Node("Softmax",[z],"probabilities",("axis",2)));var model=builder.Build();
        foreach(string mode in new[]{"baseline","fp16-control","fp32"})
        {
            using var session=backend.NewSession(model,shape,OnnxSharp.CompiledModel.Options with{Extensions=GpuCompactHead.ForKernel(mode)});
            session.Input.Clear();session.Run();
            Assert.True(session.F16);int[] ids=new int[steps];float[] scores=new float[steps];GpuCompactHead.Unpack(session.Output,ids,scores);
            int expected=mode=="fp32"?c-1:0;Assert.All(ids,id=>Assert.Equal(expected,id));
            double probability=mode=="fp32"?1/(1+Math.Exp(bias[0]-bias[c-1])):.5;
            Assert.All(scores,score=>Assert.InRange(Math.Abs(score-probability),0,5e-6));
        }
    }
}
#endif
