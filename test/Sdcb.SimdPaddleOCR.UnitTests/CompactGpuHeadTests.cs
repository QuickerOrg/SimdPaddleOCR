#if !USE_NS20_LIBRARY
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdPaddleOCR.Experimental;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class CompactGpuHeadTests
{
    [Fact]
    public void CompactBytesPreserveLargeIndicesAndEveryScoreBit()
    {
        int[] ids = [0, 2049, 18709, 65535];
        float[] scores = [-3.1415927f, 0.9999999f, 23.123456f, float.NegativeInfinity];
        float[] packed = new float[ids.Length * 6];
        for (int i = 0; i < ids.Length; i++)
        {
            packed[i * 6] = ids[i] & 255;
            packed[i * 6 + 1] = ids[i] >> 8;
            uint bits = (uint)BitConverter.SingleToInt32Bits(scores[i]);
            for (int k = 0; k < 4; k++) packed[i * 6 + 2 + k] = (bits >> (8 * k)) & 255;
            // Simulate the fp16 activation storage; every byte must remain exact.
            for (int k = 0; k < 6; k++) packed[i * 6 + k] = (float)(Half)packed[i * 6 + k];
        }
        int[] gotIds = new int[ids.Length];
        float[] gotScores = new float[ids.Length];
        GpuCompactHead.Unpack(packed, gotIds, gotScores);
        Assert.Equal(ids, gotIds);
        Assert.Equal(scores.Select(BitConverter.SingleToInt32Bits), gotScores.Select(BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void CompactBytesRejectMismatchedLengths()
    {
        Assert.Throws<ArgumentException>(() => GpuCompactHead.Unpack(new float[6], new int[2], new float[2]));
        Assert.Throws<ArgumentException>(() => GpuCompactHead.Unpack(new float[6], new int[1], new float[2]));
    }

    [Fact]
    public void SingleWavePreservesEveryCompactUnit()
    {
        try { _ = VkBackend.Default; }
        catch (Exception ex) { Assert.Skip("Vulkan unavailable: " + ex.Message); return; }
        using var stream = ChineseV6TinyModel.Recognition.OpenRead();
        using var model = Model.Load(stream);
        using var compiled = new OnnxSharp.CompiledModel(model, new[] { 1, 3, 48, 128 }, 4);
        using var session = new GpuOcrSession(compiled, false);
        int[][] shapes = Enumerable.Range(0, 6).Select(i => new[] { 1, 3, 48, 128 + 32 * (i % 3) }).ToArray();
        string? previous = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE");
        float[][]? expected = null;
        try
        {
            foreach (bool single in new[] { false, true })
            {
                Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE", single ? "1" : "0");
                Span<float> input = session.ReshapeMany(shapes);
                for (int i = 0; i < input.Length; i++) input[i] = MathF.Sin(i * .037f);
                Assert.True(session.TryResolveManyHead(out int[] rows, out _));
                var got = new float[shapes.Length][];
                Assert.True(session.TryRunManyCompact((acts, offsets, first, count) =>
                {
                    for (int u = first; u < first + count; u++)
                        got[u] = acts.AsSpan(offsets[u], rows[u] * 6).ToArray();
                    return true;
                }));
                Assert.True(session.GpuAlive);
                var runner = (VkBatch)typeof(GpuOcrSession).GetField("_runner", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(session)!;
                Assert.Equal(single ? 1 : 3, runner.Waves);
                if (expected is null) expected = got;
                else for (int u = 0; u < got.Length; u++) Assert.Equal(expected[u], got[u]);
            }
        }
        finally { Environment.SetEnvironmentVariable("SIMD_OCR_EXPERIMENT_SINGLE_WAVE", previous); }
    }

    [Theory]
    [InlineData(1, 128)]
    [InlineData(2, 320)]
    public void CompactShaderMatchesDenseGpuArgMax(int batch, int width)
    {
        VkBackend backend;
        try { backend = VkBackend.Default; }
        catch (Exception ex) { Assert.Skip("Vulkan unavailable: " + ex.Message); return; }
        using var stream = ChineseV6TinyModel.Recognition.OpenRead();
        using var model = Model.Load(stream);
        int[] shape = [batch, 3, 48, width];
        var head = ModelHeads.FindSoftmaxHead(model.Onnx, shape, OnnxSharp.CompiledModel.Options)!;
        var options = OnnxSharp.CompiledModel.Options with { Vec4Eltwise = true };
        using var dense = backend.NewSession(model.Onnx, shape, options with { OutputName = head.LogitsName });
        using var compact = backend.NewSession(model.Onnx, shape, options with { Extensions = GpuCompactHead.Extensions });
        float[] input = Enumerable.Range(0, batch * 3 * 48 * width).Select(i => MathF.Sin(i * .037f)).ToArray();
        input.CopyTo(dense.Input); dense.Run();
        input.CopyTo(compact.Input); compact.Run();
        int columns = dense.OutputShape[^1];
        int rows = dense.Output.Length / columns;
        Assert.Equal(rows * 6, compact.Output.Length);
        int[] ids = new int[rows]; float[] scores = new float[rows];
        GpuCompactHead.Unpack(compact.Output, ids, scores);
        for (int row = 0; row < rows; row++)
        {
            var values = dense.Output.Slice(row * columns, columns);
            int expected = 0;
            for (int c = 1; c < columns; c++) if (values[c] > values[expected]) expected = c;
            Assert.Equal(expected, ids[row]);
            Assert.Equal(values[expected], scores[row]);
        }
    }
}
#endif
