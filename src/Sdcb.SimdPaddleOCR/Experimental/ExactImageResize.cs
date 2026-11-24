using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Imaging;
#if !NETSTANDARD2_0
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Sdcb.SimdPaddleOCR;

// Opt-in exact integer resize; unsupported layouts/formats keep the engine implementation.
internal static unsafe class ExactImageResize
{
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_EXACT_RESIZE") == "1";
#if !NETSTANDARD2_0
    private static readonly bool GatherEnabled = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_RESIZE_GATHER") == "1";
    private sealed record Coefficients(int[] Offsets, short[] Weights, int[]? ChannelOffsets, int[]? ChannelWeights0, int[]? ChannelWeights1);
    private static long _retainedBytes;
    private static readonly ConcurrentDictionary<(int Source, int Destination), Coefficients> Cache = new();
    private static readonly object CacheGate = new();

    private static Coefficients? CachedCoefficients(int width, int resizedWidth)
    {
        var key = (width, resizedWidth);
        if (Cache.TryGetValue(key, out var value)) return value;
        if (resizedWidth > 4096) return null;
        lock (CacheGate)
        {
            if (Cache.TryGetValue(key, out value)) return value;
            // Immutable tables only; hard cap bounds retention on arbitrary request sizes.
            long bytes = (GatherEnabled ? 44L : 8L) * resizedWidth;
            if (Cache.Count >= 1024 || _retainedBytes + bytes > 32L * 1024 * 1024) return null;
            value = new(new int[resizedWidth], new short[resizedWidth * 2],
                GatherEnabled ? new int[resizedWidth * 3] : null,
                GatherEnabled ? new int[resizedWidth * 3] : null,
                GatherEnabled ? new int[resizedWidth * 3] : null);
            FillCoefficients(width, resizedWidth, value.Offsets, value.Weights);
            if (value.ChannelOffsets is not null)
                for (int x = 0; x < resizedWidth; x++) for (int c = 0; c < 3; c++)
                {
                    value.ChannelOffsets[x * 3 + c] = value.Offsets[x] * 3 + c;
                    value.ChannelWeights0![x * 3 + c] = value.Weights[x * 2];
                    value.ChannelWeights1![x * 3 + c] = value.Weights[x * 2 + 1];
                }
            _retainedBytes += bytes;
            Cache[key] = value;
            return value;
        }
    }

    private static void FillCoefficients(int width, int resizedWidth, int[] offsets, short[] coefficients)
    {
        for (int x = 0; x < resizedWidth; x++)
        {
            double coordinate = (x + 0.5) * width / resizedWidth - 0.5;
            int sx = checked((int)Math.Floor(coordinate));
            double fraction = coordinate - sx;
            if (sx < 0) { sx = 0; fraction = 0; }
            if (sx >= width - 1) { sx = width - 1; fraction = 0; }
            offsets[x] = sx;
            coefficients[x * 2] = checked((short)Math.Round((1 - fraction) * 2048, MidpointRounding.ToEven));
            coefficients[x * 2 + 1] = checked((short)Math.Round(fraction * 2048, MidpointRounding.ToEven));
        }
    }
#endif

    internal static void ResizeLinear(ReadOnlySpan<byte> source, int width, int height, int stride,
        PixelFormat format, int resizedWidth, int resizedHeight, Span<float> destination,
        TensorLayout layout, PixelNormalization normalization, int tensorWidth = 0, int threads = 1)
    {
        if (Enabled && TryResizeLinear(source, width, height, stride, format, resizedWidth, resizedHeight,
            destination, layout, normalization, tensorWidth, threads)) return;
        ImageTensor.ResizeLinear(source, width, height, stride, format, resizedWidth, resizedHeight,
            destination, layout, normalization, tensorWidth, threads);
    }

    internal static bool TryResizeLinear(ReadOnlySpan<byte> source, int width, int height, int stride,
        PixelFormat format, int resizedWidth, int resizedHeight, Span<float> destination,
        TensorLayout layout, PixelNormalization normalization, int tensorWidth = 0, int threads = 1)
    {
#if !NETSTANDARD2_0
        if (!Avx2.IsSupported || format != PixelFormat.Bgr24 || layout != TensorLayout.Nhwc ||
            normalization is null || normalization.Order != ChannelOrder.Bgr ||
            width <= 0 || height <= 0 || resizedWidth <= 0 || resizedHeight <= 0) return false;
        if (tensorWidth == 0) tensorWidth = resizedWidth;
        if (tensorWidth < resizedWidth || stride < 3L * width ||
            (height - 1L) * stride + 3L * width > source.Length ||
            3L * tensorWidth * resizedHeight > destination.Length || resizedWidth > int.MaxValue / 3) return false;
        Coefficients? cached = CachedCoefficients(width, resizedWidth);
        int[] offsets = cached?.Offsets ?? ArrayPool<int>.Shared.Rent(resizedWidth);
        short[] coefficients = cached?.Weights ?? ArrayPool<short>.Shared.Rent(checked(resizedWidth * 2));
        try
        {
            if (cached is null) FillCoefficients(width, resizedWidth, offsets, coefficients);
            int workers = Math.Max(1, Math.Min(threads, resizedHeight / 24));
            fixed (byte* src = source)
            fixed (float* dst = destination)
            fixed (float* lut = normalization.Table)
            {
                var job = new Job { Source = src, Destination = dst, Lut = lut, Width = width, Height = height,
                    Stride = stride, OutputWidth = resizedWidth, OutputHeight = resizedHeight,
                    TensorWidth = tensorWidth, Offsets = offsets, Coefficients = coefficients,
                    ChannelOffsets = cached?.ChannelOffsets, ChannelWeights0 = cached?.ChannelWeights0, ChannelWeights1 = cached?.ChannelWeights1 };
                if (workers == 1) job.Rows(0, 1);
                else Parallel.For(0, workers, w => job.Rows(w, workers));
            }
            return true;
        }
        finally
        {
            if (cached is null)
            {
                ArrayPool<int>.Shared.Return(offsets);
                ArrayPool<short>.Shared.Return(coefficients);
            }
        }
#else
        return false;
#endif
    }

#if !NETSTANDARD2_0
    private struct Job
    {
        internal byte* Source;
        internal float* Destination, Lut;
        internal int Width, Height, Stride, OutputWidth, OutputHeight, TensorWidth;
        internal int[] Offsets;
        internal int[]? ChannelOffsets, ChannelWeights0, ChannelWeights1;
        internal short[] Coefficients;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal readonly void Rows(int first, int step)
        {
            int count = checked(OutputWidth * 3);
            int[] a = ArrayPool<int>.Shared.Rent(count), b = ArrayPool<int>.Shared.Rent(count);
            try
            {
                fixed (int* p0 = a)
                fixed (int* p1 = b)
                {
                    for (int y = first; y < OutputHeight; y += step)
                    {
                        double coordinate = (y + 0.5) * Height / OutputHeight - 0.5;
                        int sy = checked((int)Math.Floor(coordinate));
                        double fraction = coordinate - sy;
                        int beta0 = checked((short)Math.Round((1 - fraction) * 2048, MidpointRounding.ToEven));
                        int beta1 = checked((short)Math.Round(fraction * 2048, MidpointRounding.ToEven));
                        Horizontal(Source + (long)Math.Clamp(sy, 0, Height - 1) * Stride, p0);
                        Horizontal(Source + (long)Math.Clamp(sy + 1, 0, Height - 1) * Stride, p1);
                        Store(p0, p1, beta0, beta1, Destination + (long)y * TensorWidth * 3, count, Lut);
                    }
                }
            }
            finally { ArrayPool<int>.Shared.Return(a); ArrayPool<int>.Shared.Return(b); }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private readonly void Horizontal(byte* row, int* destination)
        {
            if (ChannelOffsets is null)
            {
                for (int x = 0; x < OutputWidth; x++)
                {
                    int sx = Offsets[x], s0 = sx * 3, s1 = Math.Min(sx + 1, Width - 1) * 3, d = x * 3;
                    int c0 = Coefficients[x * 2], c1 = Coefficients[x * 2 + 1];
                    destination[d] = row[s0] * c0 + row[s1] * c1;
                    destination[d + 1] = row[s0 + 1] * c0 + row[s1 + 1] * c1;
                    destination[d + 2] = row[s0 + 2] * c0 + row[s1 + 2] * c1;
                }
                return;
            }
            int channel = 0;
            if (ChannelOffsets is not null)
            {
                fixed (int* offsets = ChannelOffsets)
                fixed (int* weights0 = ChannelWeights0)
                fixed (int* weights1 = ChannelWeights1)
                {
                    var three = Vector256.Create(3); var mask = Vector256.Create(255);
                    // Every gather reads four bytes. Leave the right edge to scalar code,
                    // so neither the last row nor a tightly packed source is overread.
                    int count = OutputWidth * 3, safeOffset = Width * 3 - 7;
                    for (; channel <= count - 8 && offsets[channel + 7] <= safeOffset; channel += 8)
                    {
                        var index = Avx.LoadVector256(offsets + channel);
                        var a = Avx2.And(Avx2.GatherVector256((int*)row, index, 1), mask);
                        var b = Avx2.And(Avx2.GatherVector256((int*)row, Avx2.Add(index, three), 1), mask);
                        var sum = Avx2.Add(Avx2.MultiplyLow(a, Avx.LoadVector256(weights0 + channel)),
                            Avx2.MultiplyLow(b, Avx.LoadVector256(weights1 + channel)));
                        Avx.Store(destination + channel, sum);
                    }
                }
            }
            for (; channel < OutputWidth * 3; channel++)
            {
                int x = channel / 3, c = channel % 3;
                int sx = Offsets[x], s0 = sx * 3, s1 = Math.Min(sx + 1, Width - 1) * 3, d = x * 3;
                int c0 = Coefficients[x * 2], c1 = Coefficients[x * 2 + 1];
                destination[channel] = row[s0 + c] * c0 + row[s1 + c] * c1;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Store(int* a, int* b, int beta0, int beta1, float* destination, int count, float* lut)
    {
        var c0 = Vector256.Create(beta0); var c1 = Vector256.Create(beta1);
        var two = Vector256.Create(2); var upper = Vector256.Create(255);
        var channel0 = Vector256.Create(0, 256, 512, 0, 256, 512, 0, 256);
        var channel1 = Vector256.Create(256, 512, 0, 256, 512, 0, 256, 512);
        var channel2 = Vector256.Create(512, 0, 256, 512, 0, 256, 512, 0);
        int i = 0;
        for (; i <= count - 8; i += 8)
        {
            var h0 = Avx2.ShiftRightArithmetic(Avx.LoadVector256(a + i), 4);
            var h1 = Avx2.ShiftRightArithmetic(Avx.LoadVector256(b + i), 4);
            var v0 = Avx2.ShiftRightArithmetic(Avx2.MultiplyLow(h0, c0), 16);
            var v1 = Avx2.ShiftRightArithmetic(Avx2.MultiplyLow(h1, c1), 16);
            var v = Avx2.ShiftRightArithmetic(Avx2.Add(Avx2.Add(v0, v1), two), 2);
            v = Avx2.Min(Avx2.Max(v, Vector256<int>.Zero), upper);
            var channels = i % 3 == 0 ? channel0 : i % 3 == 1 ? channel1 : channel2;
            Avx.Store(destination + i, Avx2.GatherVector256(lut, Avx2.Add(v, channels), 4));
        }
        for (; i < count; i++)
        {
            int v = ((((a[i] >> 4) * beta0 >> 16) + ((b[i] >> 4) * beta1 >> 16) + 2) >> 2);
            destination[i] = lut[i % 3 * 256 + Math.Clamp(v, 0, 255)];
        }
    }
#endif
}
