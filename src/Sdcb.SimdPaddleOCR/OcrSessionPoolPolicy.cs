namespace Sdcb.SimdPaddleOCR;

/// <summary>Controls retention of idle inference sessions without limiting active request concurrency.</summary>
public sealed class OcrSessionPoolPolicy
{
    /// <summary>How long a returned session may remain idle before it is eligible for disposal.</summary>
    public TimeSpan IdleTimeout { get; init; } = TimeSpan.FromSeconds(60);
    /// <summary>Number of recently used sessions to keep warm, bounded by MaxPooledSessions.</summary>
    public int MinimumRetainedSessions { get; init; } = 1;
}
