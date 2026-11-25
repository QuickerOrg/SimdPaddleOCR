using System.Collections.Concurrent;
using System.Diagnostics;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Xunit;

namespace Sdcb.SimdPaddleOCR.UnitTests;

public class IdleSessionPoolTests
{
    private sealed class Resource(int size) : IDisposable
    {
        internal int Size = size, Disposals, InUse, InvalidDisposals;
        internal Action? OnDispose;
        public void Dispose()
        {
            if (Volatile.Read(ref InUse) != 0) Interlocked.Increment(ref InvalidDisposals);
            Interlocked.Increment(ref Disposals);
            OnDispose?.Invoke();
        }
    }

    [Fact]
    public void ExpiresOnlyReturnedSessionsAndKeepsMostRecent()
    {
        long now = 0;
        using var pool = new IdleSessionPool<Resource>(8, new() { IdleTimeout = TimeSpan.FromSeconds(1), MinimumRetainedSessions = 1 }, r => r.Size, () => now, false);
        var oldest = new Resource(100);var recent = new Resource(20);var borrowed = new Resource(50);
        pool.Return(oldest);now = Stopwatch.Frequency / 2;pool.Return(recent);pool.Return(borrowed);
        Assert.Same(borrowed, pool.Take(50));
        now = 2 * Stopwatch.Frequency;
        Assert.Equal(1,pool.TrimExpired());
        Assert.Equal(1,oldest.Disposals);Assert.Equal(0,borrowed.Disposals);Assert.Equal(0,recent.Disposals);
        Assert.Same(recent,pool.Take());
        pool.Return(borrowed);pool.Return(recent);
    }

    [Fact]
    public void PreservesBestFitThenLargestFallback()
    {
        using var pool = new IdleSessionPool<Resource>(8,new(),r => r.Size,automatic:false);
        var small=new Resource(10);var medium=new Resource(50);var big=new Resource(100);
        pool.Return(small);pool.Return(big);pool.Return(medium);
        Assert.Same(medium,pool.Take(40));Assert.Same(big,pool.Take(200));Assert.Same(small,pool.Take(1));
        pool.Return(small);pool.Return(medium);pool.Return(big);
    }

    [Fact]
    public void LateReturnAfterDisposeIsDisposedExactlyOnce()
    {
        var pool=new IdleSessionPool<Resource>(1,new(),r=>r.Size,automatic:false);
        var cached=new Resource(1);var active=new Resource(2);
        pool.Return(cached);pool.Dispose();pool.Dispose();pool.Return(active);
        Assert.Equal(1,cached.Disposals);Assert.Equal(1,active.Disposals);
        Assert.Throws<ObjectDisposedException>(()=>pool.Take());
    }

    [Fact]
    public void ZeroCapacityAndLargeMinimumPreserveExistingCapacityLimit()
    {
        using var pool=new IdleSessionPool<Resource>(0,new(){MinimumRetainedSessions=10},r=>r.Size,automatic:false);
        var resource=new Resource(1);pool.Return(resource);
        Assert.Equal(1,resource.Disposals);Assert.Null(pool.Take());
    }

    [Fact]
    public void TimerReclaimsWithoutAnotherRequest()
    {
        using var pool=new IdleSessionPool<Resource>(8,new(){IdleTimeout=TimeSpan.FromMilliseconds(100),MinimumRetainedSessions=0},r=>r.Size);
        var resource=new Resource(1);pool.Return(resource);
        Assert.True(SpinWait.SpinUntil(()=>Volatile.Read(ref resource.Disposals)==1,TimeSpan.FromSeconds(5)));
        Assert.Equal(0,pool.Count);
    }

    [Fact]
    public async Task SlowDisposalDoesNotBlockRentOrDisposeAnActiveLease()
    {
        long now=0;
        using var entered=new ManualResetEventSlim();using var release=new ManualResetEventSlim();
        using var pool=new IdleSessionPool<Resource>(8,new(){IdleTimeout=TimeSpan.FromSeconds(1),MinimumRetainedSessions=0},r=>r.Size,()=>now,false);
        var slow=new Resource(1){OnDispose=()=>{entered.Set();release.Wait(TimeSpan.FromSeconds(5));}};
        pool.Return(slow);now=2*Stopwatch.Frequency;
        var trimming=Task.Run(()=>pool.TrimExpired());
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(2)));
            var taking=Task.Run(()=>pool.Take());
            Assert.Same(taking,await Task.WhenAny(taking,Task.Delay(2000)));
            Assert.Null(await taking);
            var active=new Resource(2);pool.Return(active);Assert.Same(active,pool.Take());
            active.InUse=1;pool.Dispose();Assert.Equal(0,active.Disposals);
            active.InUse=0;pool.Return(active);Assert.Equal(1,active.Disposals);
        }
        finally{release.Set();await trimming;}
        Assert.Equal(1,slow.Disposals);
    }

    [Fact]
    public async Task ConcurrentRentReturnTrimAndShutdownNeverDisposeLeasedResources()
    {
        long now=0;var all=new ConcurrentBag<Resource>();
        var pool=new IdleSessionPool<Resource>(8,new(){IdleTimeout=TimeSpan.FromSeconds(1),MinimumRetainedSessions=1},r=>r.Size,()=>Interlocked.Read(ref now),false);
        await Task.WhenAll(Enumerable.Range(0,12).Select(_=>Task.Run(()=>
        {
            for(int i=0;i<200;i++)
            {
                Resource r=pool.Take(i%100) ?? Create();
                Assert.Equal(0,Interlocked.Exchange(ref r.InUse,1));
                Assert.Equal(0,Volatile.Read(ref r.Disposals));
                Thread.SpinWait(20);Interlocked.Exchange(ref r.InUse,0);pool.Return(r);
                Interlocked.Add(ref now,Stopwatch.Frequency/10);pool.TrimExpired();
            }
        })));
        pool.Dispose();
        Assert.All(all,r=>{Assert.Equal(1,r.Disposals);Assert.Equal(0,r.InvalidDisposals);});
        Resource Create(){var r=new Resource(100);all.Add(r);return r;}
    }
}
