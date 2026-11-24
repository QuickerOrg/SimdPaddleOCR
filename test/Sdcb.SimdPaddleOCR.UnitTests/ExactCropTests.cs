#if !USE_NS20_LIBRARY
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdPaddleOCR.Kernels;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class ExactCropTests
{
    [Fact]
    public unsafe void PreservesByteRoundingAndLeavesGuardedFallbackUntouched()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported || Avx512F.IsSupported || Warp.SeparableEnabled)
            Assert.Skip("AVX2 reference requires the crop experiment disabled");
        var random = new Random(20261125);
        int accepted = 0, guarded = 0;
        foreach (int bpp in new[] { 3, 4 }) foreach (bool swap in new[] { false, true })
        {
            int stride = 32 * bpp + 3;
            byte[] source = new byte[stride * 32], expected = new byte[3], actual = new byte[3];
            for (int i = 0; i < source.Length; i++) source[i] = (i % 2 == 0) ? (byte)0 : (byte)255;
            fixed (byte* src = source)
            fixed (byte* reference = expected)
            fixed (byte* candidate = actual)
            for (int n = 0; n < 10000; n++)
            {
                double fx = n % 5 switch { 0 => 0, 1 => 1e-12, 2 => -1e-12, 3 => .5, _ => random.NextDouble() };
                double fy = n % 3 == 0 ? .5 : random.NextDouble();
                double x = random.Next(3, 27) + fx, y = random.Next(3, 27) + fy;
                var access = new Warp.PixelAccess(bpp, swap);
                Warp.SampleCubic(src, 32, 32, stride, x, y, reference, 0, access);
                Array.Fill(actual, (byte)0xa5);
                if (Warp.TrySampleCubicSeparable(src, stride, x, y, (int)Math.Floor(x), (int)Math.Floor(y), candidate, 0, access))
                {
                    accepted++;
                    Assert.True(expected.AsSpan().SequenceEqual(actual), $"x={x:R} y={y:R} bpp={bpp} swap={swap}");
                }
                else { guarded++; Assert.All(actual, b => Assert.Equal((byte)0xa5, b)); }
            }
        }
        Assert.True(accepted > 1000);
        Assert.True(guarded > 0);
    }
}
#endif
