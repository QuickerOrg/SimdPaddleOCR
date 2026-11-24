#if !USE_NS20_LIBRARY
using System.Reflection;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Imaging;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class ExactImageResizeTests
{
    [Fact]
    public void ConcurrentResizesPreserveBitsAndBoundCoefficientRetention()
    {
        if (!Avx2.IsSupported) Assert.Skip("AVX2-specific candidate");
        var normalization = PixelNormalization.FromFunction((c, v) => (c * 256 + v - 300) / 73.0f);
        Parallel.For(0, 1400, n =>
        {
            int w = n + 1, h = 1 + n % 4, ow = 3 + n % 31, oh = 51, tw = ow + 5, stride = w * 3 + n % 5;
            byte[] source = new byte[stride * h]; new Random(n).NextBytes(source);
            foreach (bool changeSource in new[] { false, true })
            {
                if (changeSource) Array.Fill(source, (byte)(n % 256));
                float[] expected = Enumerable.Repeat(17.5f, tw * oh * 3).ToArray();
                float[] actual = (float[])expected.Clone();
                ImageTensor.ResizeLinear(source, w, h, stride, PixelFormat.Bgr24, ow, oh, expected,
                    TensorLayout.Nhwc, normalization, tw);
                Assert.True(ExactImageResize.TryResizeLinear(source, w, h, stride, PixelFormat.Bgr24,
                    ow, oh, actual, TensorLayout.Nhwc, normalization, tw));
                for (int i = 0; i < actual.Length; i++)
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
            }
        });
        object cache = typeof(ExactImageResize).GetField("Cache", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
        int retained = (int)cache.GetType().GetProperty("Count")!.GetValue(cache)!;
        Assert.InRange(retained, 1, 1024);
    }

    [Theory]
    [InlineData(PixelFormat.Rgb24, TensorLayout.Nhwc)]
    [InlineData(PixelFormat.Bgra32, TensorLayout.Nhwc)]
    [InlineData(PixelFormat.Rgba32, TensorLayout.Nhwc)]
    [InlineData(PixelFormat.Bgr24, TensorLayout.Nchw)]
    internal void UnsupportedFormatsAndLayoutsLeaveOutputUntouched(PixelFormat format, TensorLayout layout)
    {
        var normalization = PixelNormalization.Unit();
        float[] output = Enumerable.Repeat(19f, 12).ToArray();
        Assert.False(ExactImageResize.TryResizeLinear(new byte[16], 2, 2, 8, format, 2, 2, output, layout, normalization));
        Assert.All(output, x => Assert.Equal(19f, x));
    }
}
#endif
