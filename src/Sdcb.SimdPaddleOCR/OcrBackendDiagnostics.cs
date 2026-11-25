namespace Sdcb.SimdPaddleOCR;

/// <summary>Read-only backend information for host startup validation.</summary>
public static class OcrBackendDiagnostics
{
    /// <summary>Initializes the selected Vulkan device and returns its name, or throws if unavailable.</summary>
    public static string GetVulkanDeviceName()
    {
#if NET10_0_OR_GREATER
        return Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.Device.Name;
#else
        throw new PlatformNotSupportedException("Vulkan requires the net10.0 build.");
#endif
    }
}
