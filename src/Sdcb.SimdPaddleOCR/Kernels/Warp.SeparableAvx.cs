using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.SimdPaddleOCR.Kernels;

internal static partial class Warp
{
    internal static readonly bool SeparableEnabled = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_EXACT_CROP") == "1";

    // Caller guarantees the same interior 4x4 stencil as SampleCubicAvx.
    // Coordinates within 1e-10 of an integer have <1e-5 byte-value perturbation,
    // so the existing nearest-integer byte result is exactly the source pixel.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe bool TrySampleCubicSeparable(byte* source, int stride, double x, double y,
        int xBase, int yBase, byte* destination, int offset, PixelAccess access)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return false;
        double fx = x - xBase, fy = y - yBase;
        int nx = fx < .5 ? xBase : xBase + 1, ny = fy < .5 ? yBase : yBase + 1;
        int bpp = access.Bpp;
        if (Math.Abs(x - nx) <= 1e-10 && Math.Abs(y - ny) <= 1e-10)
        {
            byte* p = source + (long)ny * stride + nx * bpp;
            destination[offset] = p[access.SwapRedBlue ? 2 : 0];
            destination[offset + 1] = p[1];
            destination[offset + 2] = p[access.SwapRedBlue ? 0 : 2];
            return true;
        }
        var wx = CubicWeightVectorAvx(Avx.Subtract(Vector256.Create(x),
            Vector256.Create((double)(xBase - 1), xBase, xBase + 1, xBase + 2)));
        var wy = CubicWeightVectorAvx(Avx.Subtract(Vector256.Create(y),
            Vector256.Create((double)(yBase - 1), yBase, yBase + 1, yBase + 2)));
        var mask = Vector256.Create(long.MaxValue).AsDouble();
        var limit = Vector256.Create(2.0);
        if (Avx.MoveMask(Avx.Compare(Avx.And(wx, mask), limit, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling)) != 15 ||
            Avx.MoveMask(Avx.Compare(Avx.And(wy, mask), limit, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling)) != 15) return false;
        var x0 = Avx2.Permute4x64(wx, 0x00); var x1 = Avx2.Permute4x64(wx, 0x55);
        var x2 = Avx2.Permute4x64(wx, 0xaa); var x3 = Avx2.Permute4x64(wx, 0xff);
        byte* row = source + (long)(yBase - 1) * stride + (xBase - 1) * bpp;
        var acc = Avx.Multiply(HorizontalCubic(row, bpp, x0, x1, x2, x3), Avx2.Permute4x64(wy, 0x00));
        acc = Fma.MultiplyAdd(HorizontalCubic(row + stride, bpp, x0, x1, x2, x3), Avx2.Permute4x64(wy, 0x55), acc);
        acc = Fma.MultiplyAdd(HorizontalCubic(row + 2 * stride, bpp, x0, x1, x2, x3), Avx2.Permute4x64(wy, 0xaa), acc);
        acc = Fma.MultiplyAdd(HorizontalCubic(row + 3 * stride, bpp, x0, x1, x2, x3), Avx2.Permute4x64(wy, 0xff), acc);
        // Both accumulation orders are bounded by <1e-9 absolute error for
        // 8-bit samples and the checked |weight|<=2. Keep a wider 1e-6 guard
        // at byte-rounding discontinuities, where the original kernel wins.
        if (!TryStableByte(acc.GetElement(0), out byte c0) || !TryStableByte(acc.GetElement(1), out byte c1) ||
            !TryStableByte(acc.GetElement(2), out byte c2)) return false;
        destination[offset] = access.SwapRedBlue ? c2 : c0;
        destination[offset + 1] = c1;
        destination[offset + 2] = access.SwapRedBlue ? c0 : c2;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> HorizontalCubic(byte* p, int bpp,
        Vector256<double> x0, Vector256<double> x1, Vector256<double> x2, Vector256<double> x3)
    {
        var sum = Avx.Multiply(LoadPixelAvx(p), x0);
        sum = Fma.MultiplyAdd(LoadPixelAvx(p + bpp), x1, sum);
        sum = Fma.MultiplyAdd(LoadPixelAvx(p + 2 * bpp), x2, sum);
        return Fma.MultiplyAdd(LoadPixelAvx(p + 3 * bpp), x3, sum);
    }

    private static bool TryStableByte(double value, out byte result)
    {
        result = 0;
        if (!double.IsFinite(value)) return false;
        if (value <= 0) return true;
        if (value >= 255) { result = 255; return true; }
        double rounded = Math.Floor(value + .5);
        if (Math.Abs(Math.Abs(value - rounded) - .5) <= 1e-6) return false;
        result = checked((byte)rounded); return true;
    }
}
