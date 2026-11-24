using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Experimental;
using System.Text.Json;

internal static class GpuExactResizeValidation
{
    internal static int Run(string output)
    {
        string expectedGpu = Environment.GetEnvironmentVariable("SIMD_OCR_EXPECTED_GPU_NAME")
            ?? throw new InvalidOperationException("Expected GPU name is required.");
        if (VkBackend.Default.DeviceName != expectedGpu) throw new InvalidOperationException("Unexpected GPU.");
        var random = new Random(20261124);
        var normalization = PixelNormalization.FromFunction((c, v) => (c * 256 + v - 300) / 73.0f);
        long elements = 0;
        for (int n = 0; n < 300; n++)
        {
            int w = random.Next(1, 80), h = random.Next(1, 70), ow = random.Next(1, 90), oh = random.Next(1, 90);
            if (n == 0) { w = 1024; h = 768; ow = 960; oh = 704; }
            int stride = w * 3 + n % 8;
            byte[] source = new byte[(h - 1) * stride + w * 3]; random.NextBytes(source);
            float[] expected = new float[ow * oh * 3], actual = new float[expected.Length];
            ImageTensor.ResizeLinear(source, w, h, stride, PixelFormat.Bgr24, ow, oh, expected, TensorLayout.Nhwc, normalization);
            if (!GpuExactDetResize.TryResize(source, w, h, stride, ow, oh, actual, normalization, true, ImagePixelFormat.Bgr24))
                throw new Exception("GPU candidate not exercised.");
            for (int i = 0; i < actual.Length; i++)
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    throw new Exception($"GPU resize differs case={n} offset={i}: {expected[i]} vs {actual[i]}");
            elements += actual.Length;
        }
        if (GpuExactDetResize.Completed != 300) throw new Exception("GPU completion count differs.");
        File.WriteAllText(output, JsonSerializer.Serialize(new { cases = 300, elements, bitDifferences = 0,
            gpu = VkBackend.Default.DeviceName, gpuCompleted = GpuExactDetResize.Completed }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"GPU exact resize: 300 cases, {elements} floats, zero bit differences.");
        return 0;
    }
}
