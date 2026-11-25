using System.Diagnostics;

namespace Sdcb.SimdPaddleOCR.OnnxSharp;

/// <summary>Only returned resources are owned here; creation and all GPU work occur outside the lock.</summary>
internal sealed class IdleSessionPool<T> : IDisposable where T : class, IDisposable
{
    private readonly object _gate = new();
    private readonly List<(T Value, long Returned)> _idle = [];
    private readonly int _maximum, _minimum;
    private readonly double _timeoutTicks;
    private readonly Func<T, int> _capacity;
    private readonly Func<long> _clock;
    private Timer? _timer;
    private bool _disposed;

    internal IdleSessionPool(int maximum, OcrSessionPoolPolicy policy, Func<T, int> capacity,
        Func<long>? clock = null, bool automatic = true)
    {
        if (maximum < 0) throw new ArgumentOutOfRangeException(nameof(maximum));
        if (policy.IdleTimeout <= TimeSpan.Zero || policy.MinimumRetainedSessions < 0)
            throw new ArgumentOutOfRangeException(nameof(policy));
        _maximum = maximum;
        _minimum = Math.Min(maximum, policy.MinimumRetainedSessions);
        _timeoutTicks = policy.IdleTimeout.TotalSeconds * Stopwatch.Frequency;
        _capacity = capacity;
        _clock = clock ?? Stopwatch.GetTimestamp;
        // 定时器不强引用池：忘记 Dispose 的调用者不会因此永久保活整个模型。
        if (automatic)
        {
            var weak = new WeakReference<IdleSessionPool<T>>(this);
            int period = (int)Math.Max(100, Math.Min(30_000, policy.IdleTimeout.TotalMilliseconds / 2));
            Timer? timer = null;
            timer = new Timer(_ =>
            {
                if (weak.TryGetTarget(out var pool)) pool.TrimExpired();
                else timer?.Dispose();
            }, null, Timeout.Infinite, Timeout.Infinite);
            _timer = timer;
            timer.Change(period, period);
        }
    }

    internal int Count { get { lock (_gate) return _idle.Count; } }

    internal T? Take(int requiredVolume = 0)
    {
        lock (_gate)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(IdleSessionPool<T>));
            if (_idle.Count == 0) return null;
            int take = _idle.Count - 1;
            if (requiredVolume > 0)
            {
                int best = -1, largest = 0, bestWater = int.MaxValue, largestWater = -1;
                for (int i = 0; i < _idle.Count; i++)
                {
                    int water = _capacity(_idle[i].Value);
                    if (water > largestWater) { largestWater = water; largest = i; }
                    if (water >= requiredVolume && water < bestWater) { bestWater = water; best = i; }
                }
                take = best >= 0 ? best : largest;
            }
            T value = _idle[take].Value;
            _idle.RemoveAt(take);
            return value;
        }
    }

    internal void Return(T value)
    {
        lock (_gate)
        {
            if (!_disposed && _idle.Count < _maximum)
            {
                _idle.Add((value, _clock()));
                return;
            }
        }
        value.Dispose();
    }

    internal int TrimExpired()
    {
        List<T> removed = [];
        lock (_gate)
        {
            if (_disposed || _idle.Count <= _minimum) return 0;
            long now = _clock();
            // 保留最近归还的热会话。先摘除再释放，昂贵的驱动释放不占用租借锁。
            foreach (var candidate in _idle.OrderBy(e => e.Returned).ToArray())
            {
                if (_idle.Count <= _minimum || now - candidate.Returned < _timeoutTicks) break;
                _idle.Remove(candidate);
                removed.Add(candidate.Value);
            }
        }
        foreach (T value in removed)
        {
            try { value.Dispose(); }
            catch (Exception ex) { Trace.TraceError("Idle OCR session disposal failed: {0}", ex); }
        }
        return removed.Count;
    }

    public void Dispose()
    {
        T[] removed;
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            removed = _idle.Select(e => e.Value).ToArray();
            _idle.Clear();
        }
        _timer?.Dispose();
        foreach (T value in removed) value.Dispose();
    }
}

internal static class IdleSessionPolicy
{
    internal static IdleSessionPool<IOcrSession>? Create(int maximum, OcrSessionPoolPolicy? policy,
        Func<string, string?>? getEnvironmentVariable = null)
    {
        if (policy?.Enabled == false) return null;
        getEnvironmentVariable ??= Environment.GetEnvironmentVariable;
        // 兼容现有包装层的隔离试验入口；默认不启用，不改变既有缓存行为。
        if (policy is null && getEnvironmentVariable("SIMD_OCR_EXPERIMENT_SESSION_IDLE_SECONDS") is { } value)
        {
            if (!double.TryParse(value, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out double seconds) ||
                double.IsNaN(seconds) || double.IsInfinity(seconds) || seconds <= 0)
                throw new ArgumentException("Invalid SIMD_OCR_EXPERIMENT_SESSION_IDLE_SECONDS.");
            int keep = 1;
            if (getEnvironmentVariable("SIMD_OCR_EXPERIMENT_SESSION_IDLE_KEEP") is { } requested &&
                (!int.TryParse(requested, out keep) || keep < 0))
                throw new ArgumentException("Invalid SIMD_OCR_EXPERIMENT_SESSION_IDLE_KEEP.");
            policy = new OcrSessionPoolPolicy { IdleTimeout = TimeSpan.FromSeconds(seconds), MinimumRetainedSessions = keep };
        }
        return policy is null ? null : new IdleSessionPool<IOcrSession>(maximum, policy, s => s.HighWaterInputVolume);
    }
}
