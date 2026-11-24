#if NET10_0_OR_GREATER
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Imaging;

namespace Sdcb.SimdPaddleOCR.Experimental;

// A bounded, opt-in DET-only preprocessor. Buffers are process-lifetime cached like the backend.
internal static unsafe class GpuExactDetResize
{
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_GPU_DET_RESIZE") == "1";
    internal static readonly bool FusedEnabled = Environment.GetEnvironmentVariable("SIMD_OCR_EXPERIMENT_GPU_DET_FUSED") == "1";
    private static readonly SemaphoreSlim Slots = new(4, 4);
    private static readonly ConcurrentBag<Session> Pool = new();
    private static long _completed;
    internal static long Completed => Interlocked.Read(ref _completed);

    internal static bool TryRunIntegrated(VkInferenceSession model, ReadOnlySpan<byte> source,
        int width, int height, int stride, int resizedWidth, int resizedHeight, Span<float> output,
        PixelNormalization normalization, ImagePixelFormat format)
    {
        if (resizedWidth <= 0 || resizedHeight <= 0 || !model.F16 || !VkBackend.Default.Dev.PushDescriptors || model.InputCount != 1 ||
            model.InputBytes != 6UL * (uint)resizedWidth * (uint)resizedHeight || model.OutputBytes != (ulong)output.Length * 2 ||
            format != ImagePixelFormat.Bgr24 || normalization.Order != ChannelOrder.Bgr ||
            width <= 0 || height <= 0 || stride < width * 3L ||
            (height - 1L) * stride + width * 3L > source.Length) return false;
        Slots.Wait(); Session? session = null;
        try
        {
            if (Pool.TryTake(out session) && (!ReferenceEquals(session.Normalization, normalization) || !ReferenceEquals(session.Device, VkBackend.Default.Dev)))
            { session.Dispose(); session = null; }
            session ??= new Session(normalization);
            session.Run(source, width, height, stride, resizedWidth, resizedHeight, output, model);
            Interlocked.Increment(ref _completed); return true;
        }
        catch { session?.Dispose(); session = null; throw; }
        finally { if (session is not null) Pool.Add(session); Slots.Release(); }
    }

    internal static bool TryResize(ReadOnlySpan<byte> source, int width, int height, int stride,
        int resizedWidth, int resizedHeight, Span<float> output, PixelNormalization normalization,
        bool nhwc, ImagePixelFormat format)
    {
        if (!nhwc || format != ImagePixelFormat.Bgr24 || normalization is null ||
            normalization.Order != ChannelOrder.Bgr || width <= 0 || height <= 0 ||
            resizedWidth <= 0 || resizedHeight <= 0 || stride < 3L * width ||
            (height - 1L) * stride + width * 3L > source.Length ||
            output.Length != 3L * resizedWidth * resizedHeight) return false;
        Slots.Wait();
        Session? session = null;
        try
        {
            if (Pool.TryTake(out session) && (!ReferenceEquals(session.Normalization, normalization) || !ReferenceEquals(session.Device, VkBackend.Default.Dev)))
            { session.Dispose(); session = null; }
            session ??= new Session(normalization);
            session.Run(source, width, height, stride, resizedWidth, resizedHeight, output);
            Interlocked.Increment(ref _completed);
            return true;
        }
        catch
        {
            session?.Dispose(); session = null;
            throw; // Never count CPU fallback as a successful GPU preprocessing experiment.
        }
        finally { if (session is not null) Pool.Add(session); Slots.Release(); }
    }

    private sealed class Session : IDisposable
    {
        internal PixelNormalization Normalization { get; }
        internal VkDevice Device => _dev;
        private readonly VkDevice _dev;
        private readonly VkPipeline _pipeline = null!;
        private readonly IntPtr _commandPool, _command, _fence, _descriptorPool, _descriptorSet;
        private readonly VkBuffer _normalization = null!;
        private VkBuffer? _source, _x, _y, _output, _upload, _readback, _modelOutput;

        internal Session(PixelNormalization normalization)
        {
            Normalization = normalization;
            _dev = VkBackend.Default.Dev;
            try
            {
            // Use a conventional descriptor set so this experiment does not require push descriptors.
            using var stream = typeof(GpuExactDetResize).Assembly.GetManifestResourceStream(
                "Sdcb.SimdPaddleOCR.Experimental.Shaders.exact_det_resize.spv")
                ?? throw new InvalidOperationException("Missing exact DET resize shader.");
            using var bytes = new MemoryStream(); stream.CopyTo(bytes);
            IntPtr shader = _dev.NewShaderModule(bytes.ToArray());
            try { _pipeline = _dev.NewPipeline(shader, 5, 12, pushLayout: false); }
            finally { Vk.vkDestroyShaderModule(_dev.Device, shader, null); }
            _commandPool = _dev.NewCommandPool(); _command = _dev.AllocateCommandBuffer(_commandPool);
            _fence = _dev.NewFence(); _descriptorPool = _dev.NewDescriptorPool(1);
            _descriptorSet = _dev.AllocateDescriptorSet(_descriptorPool, _pipeline.SetLayout);
            _normalization = _dev.NewStorageBuffer(768 * sizeof(float), hostVisible: false);
            fixed (float* p = normalization.Table) _dev.Upload(_normalization, p, 768 * sizeof(float));
            }
            catch { Dispose(); throw; }
        }

        private void Ensure(ref VkBuffer? buffer, ulong bytes, bool host = false, bool readback = false)
        {
            if (buffer is not null && buffer.Size >= bytes) return;
            buffer?.Free(); buffer = null;
            buffer = _dev.NewStorageBuffer(bytes, hostVisible: host, preferHost: readback);
        }

        internal void Run(ReadOnlySpan<byte> source, int width, int height, int stride, int rw, int rh,
            Span<float> output, VkInferenceSession? model = null)
        {
            int sourceBytes = checked((height - 1) * stride + width * 3);
            ulong xOffset = ((ulong)sourceBytes + 3) & ~3UL;
            ulong yOffset = xOffset + (ulong)rw * 16;
            ulong uploadBytes = yOffset + (ulong)rh * 16;
            Ensure(ref _upload, uploadBytes, host: true);
            Ensure(ref _source, xOffset);
            Ensure(ref _x, checked((ulong)rw * 16)); Ensure(ref _y, checked((ulong)rh * 16));
            int resizedFloats = checked(rw * rh * 3);
            Ensure(ref _output, checked((ulong)resizedFloats * sizeof(float)));
            Ensure(ref _readback, checked((ulong)output.Length * sizeof(float)), host: true, readback: true);
            void* sourceMap = _upload!.Map();
            source[..sourceBytes].CopyTo(new Span<byte>(sourceMap, sourceBytes));
            int* x = (int*)((byte*)sourceMap + xOffset);
            for (int i = 0; i < rw; i++)
            {
                double coordinate = (i + 0.5) * width / rw - 0.5;
                int sx = checked((int)Math.Floor(coordinate)); double fraction = coordinate - sx;
                if (sx < 0) { sx = 0; fraction = 0; }
                if (sx >= width - 1) { sx = width - 1; fraction = 0; }
                x[i * 4] = sx; x[i * 4 + 1] = Math.Min(sx + 1, width - 1);
                x[i * 4 + 2] = checked((short)Math.Round((1 - fraction) * 2048, MidpointRounding.ToEven));
                x[i * 4 + 3] = checked((short)Math.Round(fraction * 2048, MidpointRounding.ToEven));
            }
            int* y = (int*)((byte*)sourceMap + yOffset);
            for (int i = 0; i < rh; i++)
            {
                double coordinate = (i + 0.5) * height / rh - 0.5;
                int sy = checked((int)Math.Floor(coordinate)); double fraction = coordinate - sy;
                y[i * 4] = Math.Clamp(sy, 0, height - 1); y[i * 4 + 1] = Math.Clamp(sy + 1, 0, height - 1);
                y[i * 4 + 2] = checked((short)Math.Round((1 - fraction) * 2048, MidpointRounding.ToEven));
                y[i * 4 + 3] = checked((short)Math.Round(fraction * 2048, MidpointRounding.ToEven));
            }
            _upload.Flush(0, uploadBytes); _upload.Unmap();
            VkBuffer[] buffers = [_source!, _x!, _y!, _normalization, _output!];
            for (int i = 0; i < buffers.Length; i++) _dev.BindBuffer(_descriptorSet, (uint)i, buffers[i], 0, buffers[i].Size);
            Vk.Check(Vk.vkResetCommandBuffer(_command, 0), "vkResetCommandBuffer");
            var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
            Vk.Check(Vk.vkBeginCommandBuffer(_command, &begin), "vkBeginCommandBuffer");
            var sourceCopy = new Vk.VkBufferCopy { Size = xOffset };
            Vk.vkCmdCopyBuffer(_command, _upload.Buffer, _source!.Buffer, 1, &sourceCopy);
            var xCopy = new Vk.VkBufferCopy { SrcOffset = xOffset, Size = (ulong)rw * 16 };
            Vk.vkCmdCopyBuffer(_command, _upload.Buffer, _x!.Buffer, 1, &xCopy);
            var yCopy = new Vk.VkBufferCopy { SrcOffset = yOffset, Size = (ulong)rh * 16 };
            Vk.vkCmdCopyBuffer(_command, _upload.Buffer, _y!.Buffer, 1, &yCopy);
            var uploaded = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead };
            Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0, 1, &uploaded, 0, null, 0, null);
            Vk.vkCmdBindPipeline(_command, VkConst.BindPointCompute, _pipeline.Pipeline);
            IntPtr set = _descriptorSet;
            Vk.vkCmdBindDescriptorSets(_command, VkConst.BindPointCompute, _pipeline.Layout, 0, 1, &set, 0, null);
            int* parameters = stackalloc int[3] { rw, rh, stride };
            Vk.vkCmdPushConstants(_command, _pipeline.Layout, VkConst.StageComputeShader, 0, 12, parameters);
            Vk.vkCmdDispatch(_command, (uint)((rw * rh + 255) / 256), 1, 1);
            VkBuffer resultBuffer = _output!;
            if (model is not null)
            {
                _dev.CmdComputeBarrier(_command);
                Convert("f32to16", _output!, 0, (ulong)resizedFloats * 4,
                    model.Arena, model.InputOffset, model.InputBytes, (uint)resizedFloats);
                _dev.CmdComputeBarrier(_command);
                model.RecordCompute(_command);
                _dev.CmdComputeBarrier(_command);
                Ensure(ref _modelOutput, (ulong)output.Length * 4);
                Convert("f16to32", model.Arena, model.OutputOffset, model.OutputBytes,
                    _modelOutput!, 0, (ulong)output.Length * 4, (uint)output.Length);
                resultBuffer = _modelOutput!;
            }
            var computed = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead };
            Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0, 1, &computed, 0, null, 0, null);
            var download = new Vk.VkBufferCopy { Size = (ulong)output.Length * sizeof(float) };
            Vk.vkCmdCopyBuffer(_command, resultBuffer.Buffer, _readback!.Buffer, 1, &download);
            var downloaded = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessHostRead };
            Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageTransfer, VkConst.PipelineStageHost, 0, 1, &downloaded, 0, null, 0, null);
            Vk.Check(Vk.vkEndCommandBuffer(_command), "vkEndCommandBuffer");
            _dev.Submit(_command, _fence); _dev.WaitFence(_fence);
            void* result = _readback!.Map(); _readback.Invalidate(0, (ulong)output.Length * sizeof(float));
            new ReadOnlySpan<float>(result, output.Length).CopyTo(output); _readback.Unmap();
        }

        private void Convert(string name, VkBuffer source, ulong sourceOffset, ulong sourceBytes,
            VkBuffer destination, ulong destinationOffset, ulong destinationBytes, uint elements)
        {
            VkPipeline pipeline = _dev.GetPipeline(name, 2, 4);
            uint groups = (elements + 255) / 256, gx = Math.Min(groups, 65535), gy = (groups + gx - 1) / gx;
            VkInferenceSession.RecordOne(_dev, _command, pipeline,
                [(source, sourceOffset, sourceBytes), (destination, destinationOffset, destinationBytes)],
                BitConverter.GetBytes(elements), gx, gy);
        }

        public void Dispose()
        {
            if (_commandPool != IntPtr.Zero) _dev.DestroyCommandPool(_commandPool);
            if (_fence != IntPtr.Zero) _dev.DestroyFence(_fence);
            if (_descriptorPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descriptorPool, null);
            _source?.Free(); _x?.Free(); _y?.Free(); _output?.Free(); _upload?.Free(); _readback?.Free(); _modelOutput?.Free(); _normalization?.Free();
            if (_pipeline is not null)
            {
                Vk.vkDestroyPipeline(_dev.Device, _pipeline.Pipeline, null);
                Vk.vkDestroyPipelineLayout(_dev.Device, _pipeline.Layout, null);
                Vk.vkDestroyDescriptorSetLayout(_dev.Device, _pipeline.SetLayout, null);
            }
        }
    }
}
#endif
