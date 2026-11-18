using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

internal static class RuntimeFiles
{
    internal static readonly string Root = CacheRoot();

    private static string CacheRoot()
    {
        string configured = Environment.GetEnvironmentVariable("OVENDESK_CACHE_DIR");
        return String.IsNullOrWhiteSpace(configured)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "OvenDesk", "Optimized")
            : Path.GetFullPath(configured);
    }

    private static string Hash(Stream stream)
    {
        using (var sha = SHA256.Create())
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
    }

    internal static string Prepare(string resource, string prefix)
    {
        using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource))
        {
            string version = Hash(stream);
            stream.Position = 0;
            Directory.CreateDirectory(Root);
            string target = Path.Combine(Root, prefix + "-" + version.Substring(0, 16));
            using (var mutex = new Mutex(false, "Local\\OvenDesk-" + version))
            {
                try { mutex.WaitOne(); }
                catch (AbandonedMutexException) { }
                try
                {
                    using (var archive = new ZipArchive(stream, ZipArchiveMode.Read))
                    {
                        if (IsValid(archive, target))
                            return target;

                        string staging = Path.Combine(Root, "tmp-" + Guid.NewGuid().ToString("N").Substring(0, 12));
                        try
                        {
                            archive.ExtractToDirectory(staging);
                            if (!IsValid(archive, staging))
                                throw new IOException("运行文件校验失败。");
                            if (Directory.Exists(target))
                                Directory.Delete(target, true);
                            Directory.Move(staging, target);
                        }
                        finally
                        {
                            if (Directory.Exists(staging))
                                Directory.Delete(staging, true);
                        }
                    }
                    return target;
                }
                finally { mutex.ReleaseMutex(); }
            }
        }
    }

    private static bool IsValid(ZipArchive archive, string folder)
    {
        if (!Directory.Exists(folder))
            return false;
        foreach (var entry in archive.Entries)
        {
            if (entry.Name.Length == 0)
                continue;
            var file = new FileInfo(Path.Combine(folder, entry.FullName));
            if (!file.Exists || entry.Length != file.Length)
                return false;
            using (var expected = entry.Open())
            using (var actual = file.OpenRead())
                if (Hash(expected) != Hash(actual))
                    return false;
        }
        return true;
    }

    internal static ProcessStartInfo Python(string folder, string arguments)
    {
        var info = new ProcessStartInfo(Path.Combine(folder, "python.exe"), arguments) {
            WorkingDirectory = folder,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        info.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
        return info;
    }

    internal static string Quote(string path)
    {
        return "\"" + path + "\"";
    }
}

internal sealed class Candidate
{
    public string name { get; set; }
    public double score { get; set; }
}

internal sealed class Prediction
{
    public string error { get; set; }
    public Candidate[] candidates { get; set; }
    public string modelSha256 { get; set; }
    public string preprocessingSha256 { get; set; }
    public int[] inputSize { get; set; }
    public int[] tensorShape { get; set; }
    public string[] classes { get; set; }
    public string resizeMode { get; set; }
    public string modelVersion { get; set; }
    public string metadataSha256 { get; set; }
    public string tensorSha256 { get; set; }
    public string sourceSha256 { get; set; }
}

internal sealed class Classifier : IDisposable
{
    private readonly object gate = new object();
    private readonly JavaScriptSerializer json = new JavaScriptSerializer();
    private Process worker;
    private StringBuilder errors;
    private readonly string modelFolder;

    internal Classifier() : this(ModelLibrary.Active()) { }
    internal Classifier(string folder) { modelFolder = folder; }

    internal Prediction Run(string source, string diagnostic = null)
    {
        lock (gate)
        {
            if (worker == null || worker.HasExited)
            {
                Stop();
                Start();
            }

            Prediction prediction;
            try
            {
                byte[] request = Encoding.UTF8.GetBytes(
                    json.Serialize(new { source, diagnostic }) + "\n");
                worker.StandardInput.BaseStream.Write(request, 0, request.Length);
                worker.StandardInput.BaseStream.Flush();
                Task<string> response = worker.StandardOutput.ReadLineAsync();
                if (!response.Wait(TimeSpan.FromSeconds(30)))
                    throw new TimeoutException("识别超时，请重试。");
                if (response.Result == null)
                {
                    worker.WaitForExit();
                    lock (errors)
                        throw new IOException("识别进程已退出。" + Environment.NewLine + errors);
                }
                prediction = json.Deserialize<Prediction>(response.Result);
            }
            catch
            {
                Stop();
                throw;
            }

            if (prediction.error != null)
                throw new InvalidDataException(prediction.error);
            if (prediction.candidates == null || prediction.candidates.Length < 2)
            {
                Stop();
                throw new InvalidDataException("识别进程没有返回有效候选。");
            }
            return prediction;
        }
    }

    private void Start()
    {
        string folder = RuntimeFiles.Prepare("OvenDesk.DesktopAI.zip", "ai");
        var info = RuntimeFiles.Python(folder, "-X utf8 -u infer_desktop.py --worker");
        if (modelFolder != null) info.EnvironmentVariables["OVENDESK_MODEL_DIR"] = modelFolder;
        info.RedirectStandardInput = true;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        info.StandardOutputEncoding = Encoding.UTF8;
        info.StandardErrorEncoding = Encoding.UTF8;
        var log = new StringBuilder();
        errors = log;
        worker = new Process { StartInfo = info };
        worker.ErrorDataReceived += (sender, args) => {
            if (args.Data == null) return;
            lock (log)
                if (log.Length < 4096)
                    log.AppendLine(args.Data);
        };
        try
        {
            worker.Start();
            worker.BeginErrorReadLine();
        }
        catch
        {
            Stop();
            throw;
        }
    }

    private void Stop()
    {
        if (worker == null)
            return;
        try
        {
            if (!worker.HasExited)
                worker.Kill();
            worker.WaitForExit(2000);
        }
        catch (InvalidOperationException) { }
        finally
        {
            worker.Dispose();
            worker = null;
        }
    }

    public void Dispose()
    {
        lock (gate) Stop();
    }
}
