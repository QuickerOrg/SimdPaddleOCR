using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;

internal static class HttpCorpusExperiment
{
    private sealed record Input(string Id, string Path, string Sha256, byte[] Payload);
    private sealed record Response(string Id, bool Success, int Status, double WireMs, double ValidatedMs,
        bool Mismatch, string? Signature, byte[] Body, string? Error);
    internal static async Task<int> Run(string[] args)
    {
        var url = new Uri(args[1]);
        if (!url.IsLoopback || url.Scheme != "http") throw new ArgumentException("Corpus harness requires a loopback HTTP endpoint.");
        string directory = Path.GetFullPath(args[2]), output = Path.GetFullPath(args[4]); Directory.CreateDirectory(output);
        int[] degrees = args[5].Split(',').Select(int.Parse).ToArray();
        if (degrees.Any(d => d is < 1 or > 8)) throw new ArgumentOutOfRangeException(nameof(degrees));
        using var manifest = JsonDocument.Parse(File.ReadAllBytes(args[3]));
        var inputs = manifest.RootElement.GetProperty("files").EnumerateArray().Select(item =>
        {
            string relative = item.GetProperty("path").GetString()!, file = Path.GetFullPath(Path.Combine(directory, relative));
            if (!Path.GetRelativePath(directory, file).StartsWith("..") && File.Exists(file))
            {
                byte[] bytes = File.ReadAllBytes(file); string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
                if (hash != item.GetProperty("sha256").GetString()) throw new InvalidDataException("Input SHA mismatch.");
                return new Input(item.GetProperty("id").GetString()!, relative, hash,
                    JsonSerializer.SerializeToUtf8Bytes(new { file = Convert.ToBase64String(bytes), includeTextElements = true, enableOrientationDetection = true }));
            }
            throw new InvalidDataException("Invalid corpus path.");
        }).ToArray();
        using var handler = new SocketsHttpHandler { MaxConnectionsPerServer = degrees.Max() };
        using var client = new HttpClient(handler) { BaseAddress = url, Timeout = TimeSpan.FromSeconds(120) };
        using var abort = new CancellationTokenSource(TimeSpan.FromMinutes(30));
        var expected = new ConcurrentDictionary<string, string>(); var summaries = new List<object>();
        async Task<Response> Call(Input image)
        {
            long started = Stopwatch.GetTimestamp();
            try
            {
                using var content = new ByteArrayContent(image.Payload); content.Headers.ContentType = new("application/json");
                using var response = await client.PostAsync("ocr/v2/text", content, abort.Token);
                byte[] body = await response.Content.ReadAsByteArrayAsync(abort.Token); double wire = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                bool success = response.IsSuccessStatusCode && body.AsSpan().IndexOf("\"isSuccess\":true"u8) >= 0;
                string? signature = response.IsSuccessStatusCode ? HttpLoadExperiment.StableBodySignature(body) : null;
                bool mismatch = signature is not null && expected.GetOrAdd(image.Id, signature) != signature;
                return new(image.Id, success, (int)response.StatusCode, wire, Stopwatch.GetElapsedTime(started).TotalMilliseconds,
                    mismatch, signature, body, null);
            }
            catch (Exception ex) when (!abort.IsCancellationRequested)
            { return new(image.Id, false, 0, Stopwatch.GetElapsedTime(started).TotalMilliseconds,
                Stopwatch.GetElapsedTime(started).TotalMilliseconds, false, null, [], ex.ToString()); }
        }
        foreach (int degree in degrees)
        foreach (bool warmup in new[] { true, false })
        {
            int cursor = -1, completed = 0; var results = new Response[inputs.Length];
            var watch = Stopwatch.StartNew();
            await Task.WhenAll(Enumerable.Range(0, degree).Select(async _ =>
            {
                int index;
                while ((index = Interlocked.Increment(ref cursor)) < inputs.Length)
                {
                    results[index] = await Call(inputs[index]); int done = Interlocked.Increment(ref completed);
                    if (done % 25 == 0 || done == inputs.Length) Console.WriteLine($"corpus degree={degree} warmup={warmup} completed={done}/{inputs.Length} elapsed={watch.Elapsed.TotalSeconds:F2}s");
                    if (results[index].Status == 0) throw new IOException("Corpus connection failed: " + results[index].Error);
                }
            }));
            watch.Stop();
            var wire = results.Select(r => r.WireMs).Order().ToArray();
            var summary = new { degree, warmup, requested = inputs.Length, completed, success = results.Count(r => r.Success),
                failures = results.Count(r => !r.Success), mismatches = results.Count(r => r.Mismatch),
                seconds = watch.Elapsed.TotalSeconds, imagesPerSecond = completed / watch.Elapsed.TotalSeconds,
                wireMeanMs = wire.Average(), wireP50Ms = wire[wire.Length / 2], wireP95Ms = wire[(int)Math.Ceiling(wire.Length * .95) - 1] };
            summaries.Add(summary);
            string folder = Path.Combine(output, (warmup ? "warmup-" : "measured-") + degree); Directory.CreateDirectory(folder);
            if (!warmup || results.Any(r => !r.Success))
            {
                string responses = Path.Combine(folder, "responses"); Directory.CreateDirectory(responses);
                foreach (var result in results.Where(r => !warmup || !r.Success)) File.WriteAllBytes(Path.Combine(responses, result.Id + ".json"), result.Body);
            }
            File.WriteAllText(Path.Combine(folder, "requests.json"), JsonSerializer.Serialize(results.Select(r => new {
                r.Id, r.Success, r.Status, r.WireMs, r.ValidatedMs, r.Mismatch, r.Signature, r.Error,
                responseBytes = r.Body.Length }), new JsonSerializerOptions { WriteIndented = true }));
            File.WriteAllText(Path.Combine(output, "summary.json"), JsonSerializer.Serialize(summaries, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine(JsonSerializer.Serialize(summary));
            if (summary.mismatches != 0) return 2;
        }
        return 0;
    }
}
