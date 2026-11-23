#if !USE_NS20_LIBRARY
using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;
using Sdcb.SimdPaddleOCR.Experimental;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class CpuSeededCtcTests
{
    private static void RequireAvx2()
    { if(!Avx2.IsSupported || Avx512F.IsSupported)Assert.Skip("Prototype requires the AVX2 non-AVX512 dispatch."); }

    [Theory]
    [InlineData(0,0,4,1)]
    [InlineData(1,0,8,4)]
    [InlineData(2,0,8,4)]
    [InlineData(3,0,4,1)]
    [InlineData(2,1024,8,4)]
    [InlineData(3,-1024,8,4)]
    public void SeedDoesNotDuplicateMassOrChangeTiesAndTail(int scenario,int shift,int rows,int threads)
    {
        RequireAvx2();const int k=64,c=1030;
        float[] bias=Enumerable.Repeat((float)shift,c).ToArray();
        if(scenario==0)bias[0]+=30;
        if(scenario==1){bias[7]+=30;bias[15]+=30;bias[99]+=30;}
        if(scenario==2){bias[0]+=30;bias[500]+=31;bias[501]+=31;}
        if(scenario==3){bias[7]+=30;bias[15]+=30;bias[c-1]+=31;}
        var head=new CtcProjection(new float[k*c],bias,k,c);float[] features=new float[rows*k];
        int[] original=new int[rows],candidate=new int[rows];float[] a=new float[rows],b=new float[rows];
        Assert.True(head.ArgMax(features,1,rows,original,a,threads));
        Assert.True(CpuSeededCtcHead.ArgMax(head,features,1,rows,candidate,b,threads,true));
        Assert.Equal(original,candidate);
        int expected=Array.IndexOf(bias,bias.Max());double sum=bias.Sum(v=>Math.Exp((double)v-bias[expected]));
        foreach(float score in b){Assert.InRange(score,0,1);Assert.InRange(Math.Abs(score-1/sum),0,2e-6);}
        Assert.All(candidate,id=>Assert.Equal(expected,id));
    }

    [Theory]
    [InlineData(1,float.NaN)]
    [InlineData(1,float.PositiveInfinity)]
    [InlineData(1,float.NegativeInfinity)]
    [InlineData(33,float.NaN)]
    [InlineData(33,float.PositiveInfinity)]
    [InlineData(33,float.NegativeInfinity)]
    [InlineData(1029,float.NaN)]
    public void NonFiniteOutputKeepsOriginalResultOrException(int column,float bad)
    {
        RequireAvx2();const int k=64,c=1030,rows=8;
        float[] bias=new float[c];bias[0]=30;bias[column]=bad;
        var head=new CtcProjection(new float[k*c],bias,k,c);float[] features=new float[rows*k];
        int[] ia=new int[rows],ib=new int[rows];float[] sa=new float[rows],sb=new float[rows];
        Exception? a=Record.Exception(()=>head.ArgMax(features,1,rows,ia,sa,4));
        Exception? b=Record.Exception(()=>CpuSeededCtcHead.ArgMax(head,features,1,rows,ib,sb,4,true));
        if(a is not null){Assert.NotNull(b);Assert.Equal(a.GetBaseException().GetType(),b.GetBaseException().GetType());Assert.Equal(a.GetBaseException().Message,b.GetBaseException().Message);}
        else{Assert.Null(b);Assert.Equal(ia,ib);for(int i=0;i<rows;i++)Assert.InRange(Math.Abs(sa[i]-sb[i]),0,2e-6f);}
    }

    [Fact]
    public void ExplicitOffRetainsExactOriginalScores()
    {
        var head=new CtcProjection(new float[64*1030],new float[1030],64,1030);
        float[] features=new float[8*64];int[] ia=new int[8],ib=new int[8];float[] sa=new float[8],sb=new float[8];
        head.ArgMax(features,1,8,ia,sa,1);CpuSeededCtcHead.ArgMax(head,features,1,8,ib,sb,1,false);
        Assert.Equal(ia,ib);Assert.Equal(sa,sb);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void MergedUnitsWithRealProductsPreserveEveryRow(int threads)
    {
        RequireAvx2();const int k=64,c=1030,count=16;
        var random=new Random(2501);
        float[] weights=Enumerable.Range(0,k*c).Select(_=>(float)(random.NextDouble()-.5)).ToArray();
        float[] features=Enumerable.Range(0,count*k).Select(_=>(float)(random.NextDouble()-.5)).ToArray();
        float[] bias=Enumerable.Repeat(-30f,c).ToArray();bias[0]=0;bias[500]=.2f;bias[c-1]=-.1f;
        var head=new CtcProjection(weights,bias,k,c);
        int[] batches=[1,2],rows=[8,4],ia=new int[count],ib=new int[count];float[] sa=new float[count],sb=new float[count];
        Assert.True(head.ArgMaxUnits(features,batches,rows,ia,sa,threads));
        Assert.True(MatMul.TryArgMaxUnitsSeeded(features,weights,bias,ib,sb,batches,rows,k,c,head.Packed,threads));
        Assert.Equal(ia,ib);
        for(int row=0;row<count;row++)
        {
            double[] logits=new double[c];
            for(int col=0;col<c;col++){double value=bias[col];for(int inner=0;inner<k;inner++)value+=(double)features[row*k+inner]*weights[inner*c+col];logits[col]=value;}
            double maximum=logits.Max(),probability=1/logits.Sum(v=>Math.Exp(v-maximum));
            Assert.Equal(Array.IndexOf(logits,maximum),ib[row]);Assert.InRange(Math.Abs(sb[row]-probability),0,2e-6);
        }
    }
}
#endif
