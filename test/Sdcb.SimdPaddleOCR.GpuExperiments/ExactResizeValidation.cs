using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdPaddleOCR;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;

internal static class ExactResizeValidation
{
    internal static int Run(string output)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("AVX2 validation must run on AVX2.");
        var random = new Random(20261123);
        var normalization = PixelNormalization.FromFunction((c, v) => (c * 256 + v - 300) / 73.0f);
        int cases = 0; long elements = 0;
        foreach (int workers in new[] { 1, 4 })
        for (int n = 0; n < 300; n++)
        {
            int w = random.Next(1, 80), h = random.Next(1, 70);
            int ow = random.Next(1, 80), oh = random.Next(1, 90), tw = ow + random.Next(0, 9);
            if (n == 0) { w = 1024; h = 768; ow = 960; oh = 704; tw = 976; }
            int stride = w * 3 + random.Next(0, 8);
            byte[] source = new byte[stride * h]; random.NextBytes(source);
            float[] expected = Enumerable.Repeat(123.25f, tw * oh * 3).ToArray();
            float[] actual = (float[])expected.Clone();
            ImageTensor.ResizeLinear(source, w, h, stride, PixelFormat.Bgr24, ow, oh, expected,
                TensorLayout.Nhwc, normalization, tw, workers);
            if (!ExactImageResize.TryResizeLinear(source, w, h, stride, PixelFormat.Bgr24, ow, oh,
                actual, TensorLayout.Nhwc, normalization, tw, workers)) throw new Exception("Candidate not exercised.");
            for (int i = 0; i < expected.Length; i++)
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    throw new Exception($"Resize differs case={n} workers={workers} offset={i}: {expected[i]} vs {actual[i]}");
            if (ExactImageResize.TryResizeLinear(source, w, h, stride, PixelFormat.Bgr24, ow, oh,
                actual, TensorLayout.Nchw, normalization, tw, workers)) throw new Exception("Unsupported layout accepted.");
            cases++; elements += expected.Length;
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new { cases, elements, bitDifferences = 0,
            avx2 = Avx2.IsSupported, checks = "edges, strides, padding, vector tails, up/downscale, worker counts, NCHW rejection" },
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Exact resize: {cases} cases, {elements} floats, zero bit differences.");
        return 0;
    }
}
