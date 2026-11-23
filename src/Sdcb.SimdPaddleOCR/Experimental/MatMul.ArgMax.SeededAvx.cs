#if NET10_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static Sdcb.SimdOnnx.Decoding.Kernels.SimdOps;
using SimdMath = Sdcb.SimdOnnx.Kernels.SimdMath;

namespace Sdcb.SimdOnnx.Decoding.Kernels;

// 显式实验：矩阵乘体复制自固定子模块cb80be7，复用原归约与列尾，默认不调用。
internal static partial class MatMul
{
    internal static bool TryArgMaxSeeded(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, float[]? packedWeights, int threads)
    {
        if (!Avx2.IsSupported || Avx512F.IsSupported || packedWeights is null ||
            rows < 4 || rows % 4 != 0 || batch < 1 || !CanFuseArgMax(rows,inner,columns,packedWeights) ||
            input.Length != (long)batch*rows*inner || indices.Length != batch*rows || scores.Length != batch*rows ||
            (!bias.IsEmpty && bias.Length != columns)) return false;
        return MatMulArgMaxPackedSeededAvx(input,weights,packedWeights,bias,indices,scores,batch,rows,inner,columns,threads);
    }

    internal static bool TryArgMaxUnitsSeeded(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores, ReadOnlySpan<int> batches,
        ReadOnlySpan<int> rows, int inner,int columns,float[]? packedWeights,int threads)
    {
        if(batches.Length!=rows.Length || batches.IsEmpty)return false;
        int count=0;
        for(int u=0;u<rows.Length;u++)
        { if(rows[u]<4 || rows[u]%4!=0 || batches[u]<1)return false;count=checked(count+batches[u]*rows[u]); }
        // AVX2、无AVX512、全部rows%4==0：与原MergeKind=2完全相同，连续units合成一批。
        return TryArgMaxSeeded(input,weights,bias,indices,scores,1,count,inner,columns,packedWeights,threads);
    }
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe bool MatMulArgMaxPackedSeededAvx(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        int rowJobs = rows / 4;
        int jobs = checked(batch * rowJobs);
        int workers = Math.Min(ArgMaxWorkers(batch, rows, inner, columns, threads), jobs);
        bool hasBias = !bias.IsEmpty;
        int nonFiniteSeed = 0;
        fixed (float* inputPin = input, weightsPin = weights,
            packedPin = packedWeights, biasPin = bias)
        fixed (int* indicesPin = indices)
        fixed (float* scoresPin = scores)
        {
            float* inputPtr = inputPin, weightsPtr = weightsPin,
                packedPtr = packedPin, biasPtr = biasPin;
            int* indicesPtr = indicesPin;
            float* scoresPtr = scoresPin;
            // Column-block-major walk: a block of packed weight tiles
            // (~256 KB, L2-resident) is reused by every row quad of this
            // worker before moving on, instead of each quad streaming the
            // whole vocab matrix from L3. Each row still visits columns in
            // ascending order with the same per-element FMA chain, so the
            // maxima, indices and tie-breaks are unchanged.
            int tiles = columns >= 16 ? columns / 16 : 0;
            int blockTiles = Math.Max(1, 256 * 1024 / (inner * 16 * sizeof(float)));
            [MethodImpl(MethodImplCompat.AggressiveOptimization)]
            [SkipLocalsInit]
            void Worker(int w)
            {
                int myJobs = (jobs - w + workers - 1) / workers;
                int stateLength = checked(myJobs * 4);
                // Two <=4KB workspaces stay on the stack; larger states follow
                // the existing pool/discard policy. Initialize both before use.
                Vector256<float>[]? rentedMaxima = null, rentedIndices = null, rentedSums = null;
                try
                {
                    Span<Vector256<float>> allMaxima = stateLength <= 128
                        ? stackalloc Vector256<float>[stateLength]
                        : (rentedMaxima = PooledArrays.Rent<Vector256<float>>(stateLength)).AsSpan(0, stateLength);
                    Span<Vector256<float>> allIndices = stateLength <= 128
                        ? stackalloc Vector256<float>[stateLength]
                        : (rentedIndices = PooledArrays.Rent<Vector256<float>>(stateLength)).AsSpan(0, stateLength);
                    Span<Vector256<float>> allSums = stateLength <= 128
                        ? stackalloc Vector256<float>[stateLength]
                        : (rentedSums = PooledArrays.Rent<Vector256<float>>(stateLength)).AsSpan(0, stateLength);
                    allMaxima.Fill(Vector256.Create(float.NegativeInfinity));
                    allIndices.Clear();
                    allSums.Clear();
                    float* maxima = stackalloc float[4];
                    int* best = stackalloc int[4];
                    float* sums = stackalloc float[4];
                    Vector256<float>* tileOut = stackalloc Vector256<float>[8];
                    for (int t0 = 0; t0 < tiles; t0 += blockTiles)
                    {
                        int t1 = Math.Min(tiles, t0 + blockTiles);
                        for (int j = 0; j < myJobs; j++)
                        {
                            int job = w + j * workers;
                            int b = job / rowJobs, row = (job - b * rowJobs) * 4;
                            Span<Vector256<float>> vectorMaxima = allMaxima.Slice(j * 4, 4);
                            Span<Vector256<float>> vectorIndices = allIndices.Slice(j * 4, 4);
                            Span<Vector256<float>> vectorSums = allSums.Slice(j * 4, 4);
                            int inputBase = (b * rows + row) * inner;
                            for (int t = t0; t < t1; t++)
                            {
                                int col = t * 16;
                                Vector256<float> a0l = Vector256<float>.Zero, a0h = a0l;
                                Vector256<float> a1l = a0l, a1h = a0l;
                                Vector256<float> a2l = a0l, a2h = a0l;
                                Vector256<float> a3l = a0l, a3h = a0l;
                                float* tile = packedPtr + t * inner * 16;
                                for (int k = 0; k < inner; k++)
                                {
                                    Vector256<float> wl = Avx.LoadVector256(tile + k * 16);
                                    Vector256<float> wh = Avx.LoadVector256(tile + k * 16 + 8);
                                    Vector256<float> v0 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + k);
                                    Vector256<float> v1 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner + k);
                                    Vector256<float> v2 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner * 2 + k);
                                    Vector256<float> v3 = Avx.BroadcastScalarToVector256(inputPtr + inputBase + inner * 3 + k);
                                    a0l = AddMul(a0l, v0, wl); a0h = AddMul(a0h, v0, wh);
                                    a1l = AddMul(a1l, v1, wl); a1h = AddMul(a1h, v1, wh);
                                    a2l = AddMul(a2l, v2, wl); a2h = AddMul(a2h, v2, wh);
                                    a3l = AddMul(a3l, v3, wl); a3h = AddMul(a3h, v3, wh);
                                }
                                if (hasBias)
                                {
                                    Vector256<float> bl = Avx.LoadVector256(biasPtr + col);
                                    Vector256<float> bh = Avx.LoadVector256(biasPtr + col + 8);
                                    a0l = Avx.Add(a0l, bl); a0h = Avx.Add(a0h, bh);
                                    a1l = Avx.Add(a1l, bl); a1h = Avx.Add(a1h, bh);
                                    a2l = Avx.Add(a2l, bl); a2h = Avx.Add(a2h, bh);
                                    a3l = Avx.Add(a3l, bl); a3h = Avx.Add(a3h, bh);
                                }
                                tileOut[0] = a0l; tileOut[1] = a0h; tileOut[2] = a1l; tileOut[3] = a1h; tileOut[4] = a2l; tileOut[5] = a2h; tileOut[6] = a3l; tileOut[7] = a3h;
                                if (t == 0)
                                {
                                    if (!SeedFirstTile256(tileOut, vectorMaxima, vectorIndices, vectorSums))
                                    { Interlocked.Exchange(ref nonFiniteSeed, 1); return; }
                                }
                                else UpdateSeededTile256(tileOut, col, vectorMaxima, vectorIndices, vectorSums);
                            }
                        }
                    }
                    for (int j = 0; j < myJobs; j++)
                    {
                        int job = w + j * workers;
                        int b = job / rowJobs, row = (job - b * rowJobs) * 4;
                        Reduce256(allMaxima.Slice(j * 4, 4), allIndices.Slice(j * 4, 4), allSums.Slice(j * 4, 4),
                            maxima, best, sums);
                        FinishScalarTail(inputPtr, weightsPtr, biasPtr, hasBias,
                            indicesPtr, scoresPtr, b, row, 4, rows, inner, columns, tiles * 16,
                            maxima, best, sums);
                    }
                }
                finally
                {
                    PooledArrays.Return(rentedMaxima);
                    PooledArrays.Return(rentedIndices);
                    PooledArrays.Return(rentedSums);
                }
            }

            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
        return nonFiniteSeed == 0;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplCompat.AggressiveOptimization)]
    private static unsafe bool SeedFirstTile256(Vector256<float>* tile,
        Span<Vector256<float>> maxima,Span<Vector256<float>> indices,Span<Vector256<float>> sums)
    {
        for(int row=0;row<4;row++)
        {
            var low=tile[row*2];var high=tile[row*2+1];float maximum=float.NegativeInfinity;int index=0;
            for(int col=0;col<16;col++)
            {
                float value=col<8?low.GetElement(col):high.GetElement(col-8);
                // 原算法对混合非有限值的处理交给完整原路径，不改变其返回/异常行为。
                if(!float.IsFinite(value))return false;
                if(value>maximum){maximum=value;index=col;}
            }
            var seed=Vector256.Create(maximum);
            maxima[row]=seed;indices[row]=Vector256.Create((float)index);
            // 只计算真实首16项，每项一次；不能在8个lane重复加入种子winner。
            sums[row]=Avx.Add(SimdMath.ExpCore(Avx.Subtract(low,seed)),SimdMath.ExpCore(Avx.Subtract(high,seed)));
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplCompat.AggressiveOptimization)]
    private static unsafe void UpdateSeededTile256(Vector256<float>* tile,int column,
        Span<Vector256<float>> maxima,Span<Vector256<float>> indices,Span<Vector256<float>> sums)
    {
        for(int row=0;row<4;row++)
        {
            UpdateSeeded256(tile[2*row],column,row,maxima,indices,sums);
            UpdateSeeded256(tile[2*row+1],column+8,row,maxima,indices,sums);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UpdateSeeded256(Vector256<float> values,int column,int row,
        Span<Vector256<float>> maxima,Span<Vector256<float>> indices,Span<Vector256<float>> sums)
    {
        var threshold=Avx.Subtract(maxima[row],Vector256.Create(25f));
        // ordered比较不吞NaN；只有全部lane都严格较小，才能跳过整个ExpCore调用。
        if(Avx.MoveMask(Avx.Compare(values,threshold,FloatComparisonMode.OrderedLessThanNonSignaling))==255)return;
        Update256(values,column,row,maxima,indices,sums);
    }
}
#endif
