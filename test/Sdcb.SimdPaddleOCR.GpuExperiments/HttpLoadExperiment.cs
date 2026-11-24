using System.Diagnostics;
using System.Net.Http;
using System.Text;
using System.Security.Cryptography;
using System.Text.Json;

internal static class HttpLoadExperiment
{
    internal static async Task<int> Run(string[] args)
    {
        int port = int.Parse(args[1]); double seconds = double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture);
        if (port is < 1024 or > 65535 || seconds is < 1 or > 180) throw new ArgumentOutOfRangeException();
        string[] names = ["table-01-excel-light", "table-02-excel-dark", "table-05-wireless-mixed",
            "table-09-dense-schedule", "table-14-dashed-rules-only", "table-15-explorer-row-drift"];
        byte[][] payloads = names.Select(n => JsonSerializer.SerializeToUtf8Bytes(new {
            file = Convert.ToBase64String(File.ReadAllBytes(Path.Combine(args[2], n + ".png"))),
            includeTextElements = true, enableOrientationDetection = true })).ToArray();
        using var handler = new SocketsHttpHandler { MaxConnectionsPerServer = 4 };
        using var client = new HttpClient(handler) { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(45) };
        async Task<(string Signature, double Wire, double Parsed)> Call(int image)
        {
            using var content = new ByteArrayContent(payloads[image]); content.Headers.ContentType = new("application/json");
            long t = Stopwatch.GetTimestamp();
            using var response = await client.PostAsync("/ocr/v2/text", content);
            byte[] body = await response.Content.ReadAsByteArrayAsync();
            double wire = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
            if (args[0] == "--http-load-wire")
            {
                if (!response.IsSuccessStatusCode || body.AsSpan().IndexOf("\"isSuccess\":true"u8) < 0) throw new Exception("HTTP recognition failed.");
                string fullSignature = StableBodySignature(body);
                return (fullSignature, wire, Stopwatch.GetElapsedTime(t).TotalMilliseconds);
            }
            using var json = JsonDocument.Parse(body);
            var root = json.RootElement;
            if (!response.IsSuccessStatusCode || !root.GetProperty("isSuccess").GetBoolean()) throw new Exception("HTTP recognition failed.");
            var signature = new StringBuilder();
            foreach (var block in root.GetProperty("blocks").EnumerateArray())
                signature.Append(block.GetProperty("text").GetRawText()).Append(':').Append(block.GetProperty("boxPoints").GetRawText()).Append(';');
            return (signature.ToString(), wire, Stopwatch.GetElapsedTime(t).TotalMilliseconds);
        }
        string[] expected = new string[6];
        for (int i = 0; i < 6; i++) expected[i] = (await Call(i)).Signature;
        int counter = 0; object gate = new();
        var go = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var ready = new CountdownEvent(4);
        long start = 0;
        var jobs = Enumerable.Range(0, 4).Select(_ => Task.Run(async () =>
        {
            var times = new List<double>(); var wires = new List<double>(); var ids = new List<int>();
            ready.Signal(); await go.Task;
            while (true)
            {
                int n;
                lock (gate)
                {
                    if (Stopwatch.GetElapsedTime(start).TotalSeconds >= seconds && counter % 6 == 0) break;
                    n = counter++;
                }
                int image = n % 6; var result = await Call(image);
                if (result.Signature != expected[image]) throw new Exception("Concurrent text/box mismatch.");
                times.Add(result.Parsed); wires.Add(result.Wire); ids.Add(image);
            }
            return (times, wires, ids);
        })).ToArray();
        ready.Wait(); start = Stopwatch.GetTimestamp(); go.SetResult();
        var values = await Task.WhenAll(jobs); double wall = Stopwatch.GetElapsedTime(start).TotalSeconds;
        double[] parsed = values.SelectMany(v => v.times).Order().ToArray(), wireTimes = values.SelectMany(v => v.wires).Order().ToArray();
        int count = parsed.Length; var ids = values.SelectMany(v => v.ids).ToArray();
        var result = new { completed = count, wallSeconds = wall, imagesPerSecond = count / wall,
            p50Ms = parsed[count / 2], p95Ms = parsed[(int)Math.Ceiling(count * .95) - 1],
            wireP50Ms = wireTimes[count / 2], wireP95Ms = wireTimes[(int)Math.Ceiling(count * .95) - 1],
            caseCounts = names.Select((n, i) => (n, count: ids.Count(x => x == i))).ToDictionary(x => x.n, x => x.count), mismatches = 0 };
        string output = JsonSerializer.Serialize(result); File.WriteAllText(args[4], output); Console.WriteLine(output);
        return 0;
    }
    // Hash the entire response except its two volatile numeric timing values.
    // This validates scores and character geometry too, without building a
    // second client's full DOM on the measured server's slower CPU.
    internal static string StableBodySignature(byte[] body)
    {
        var ranges = new List<(int Start, int End)>();
        foreach (string marker in new[] { "\"decodeMs\":", "\"inferenceMs\":" })
        {
            byte[] token = Encoding.UTF8.GetBytes(marker);
            int index = body.AsSpan().IndexOf(token);
            if (index < 0) throw new Exception("Timing field missing.");
            int start = index + token.Length, end = start;
            while (end < body.Length && ((body[end] >= (byte)'0' && body[end] <= (byte)'9') || body[end] is (byte)'-' or (byte)'+' or (byte)'.' or (byte)'e' or (byte)'E')) end++;
            if (end == start) throw new Exception("Invalid timing field.");
            ranges.Add((start, end));
        }
        ranges.Sort((a,b) => a.Start.CompareTo(b.Start));
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        int offset = 0;
        foreach (var range in ranges) { hash.AppendData(body.AsSpan(offset, range.Start - offset)); offset = range.End; }
        hash.AppendData(body.AsSpan(offset)); return Convert.ToHexString(hash.GetHashAndReset());
    }

}
