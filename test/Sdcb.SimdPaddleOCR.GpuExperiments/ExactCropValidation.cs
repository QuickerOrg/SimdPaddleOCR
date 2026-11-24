using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.Kernels;

internal static class ExactCropValidation
{
    internal static unsafe int Run(string output)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported || Avx512F.IsSupported || Warp.SeparableEnabled)
            throw new InvalidOperationException("AVX2/FMA oracle requires AVX512 and the crop experiment disabled.");
        var random = new Random(20261125);
        int tested = 0, accepted = 0, fallback = 0, nearInteger = 0;
        foreach (int bpp in new[] { 3, 4 }) foreach (bool swap in new[] { false, true })
        {
            int stride = 64 * bpp + 7;
            byte[] source = new byte[64 * stride], expected = new byte[3], actual = new byte[3];
            random.NextBytes(source);
            fixed (byte* src = source)
            fixed (byte* a = expected)
            fixed (byte* b = actual)
            for (int n = 0; n < 100000; n++)
            {
                double fx = n % 4 == 0 ? 0 : n % 4 == 1 ? 1e-12 : n % 4 == 2 ? .5 : random.NextDouble();
                double fy = n % 5 == 0 ? 1 - 1e-12 : n % 5 == 1 ? 0 : random.NextDouble();
                double x = random.Next(2, 59) + fx, y = random.Next(2, 59) + fy;
                int xb = (int)Math.Floor(x), yb = (int)Math.Floor(y);
                var access = new Warp.PixelAccess(bpp, swap);
                Warp.SampleCubic(src, 64, 64, stride, x, y, a, 0, access);
                bool used = Warp.TrySampleCubicSeparable(src, stride, x, y, xb, yb, b, 0, access);
                if (used) accepted++;
                else { fallback++; Warp.SampleCubic(src, 64, 64, stride, x, y, b, 0, access); }
                if (Math.Abs(x - Math.Round(x)) <= 1e-10 && Math.Abs(y - Math.Round(y)) <= 1e-10) nearInteger++;
                if (!expected.AsSpan().SequenceEqual(actual)) throw new Exception($"Crop byte differs n={n} bpp={bpp} swap={swap} x={x:R} y={y:R}");
                tested++;
            }
        }
        File.WriteAllText(output, JsonSerializer.Serialize(new { tested, accepted, fallback, nearInteger, byteDifferences = 0 }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Exact crop: {tested} samples, {accepted} optimized, {fallback} guarded fallback, zero byte differences.");
        return 0;
    }
}
