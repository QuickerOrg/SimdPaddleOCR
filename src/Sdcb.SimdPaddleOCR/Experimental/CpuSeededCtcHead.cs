using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;

namespace Sdcb.SimdPaddleOCR.Experimental;

// 默认关闭，不扩展公共选项；不符合原型的shape/ISA自动回原实现。
internal static class CpuSeededCtcHead
{
    internal static bool Enabled => Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_CPU_SEEDED_T25") == "1";

    internal static bool ArgMax(CtcProjection head, ReadOnlySpan<float> features, int batch, int rows,
        Span<int> ids, Span<float> scores, int threads, bool? useSeeded = null)
    {
#if NET10_0_OR_GREATER
        if ((useSeeded ?? Enabled) && MatMul.TryArgMaxSeeded(features, head.Weights, head.Bias ?? [],
            ids, scores, batch, rows, head.Inner, head.Columns, head.Packed, threads)) return true;
#endif
        return head.ArgMax(features,batch,rows,ids,scores,threads);
    }

    internal static bool ArgMaxUnits(CtcProjection head, ReadOnlySpan<float> features,
        ReadOnlySpan<int> batches, ReadOnlySpan<int> rows, Span<int> ids, Span<float> scores, int threads)
    {
#if NET10_0_OR_GREATER
        if (Enabled && MatMul.TryArgMaxUnitsSeeded(features,head.Weights,head.Bias ?? [],ids,scores,
            batches,rows,head.Inner,head.Columns,head.Packed,threads)) return true;
#endif
        return head.ArgMaxUnits(features,batches,rows,ids,scores,threads);
    }
}
