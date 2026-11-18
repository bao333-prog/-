using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Collections.Generic;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class DatasetImport
{
    internal static bool Show(Form owner, string folder = null)
    {
        using (var source = new FolderBrowserDialog { Description = "选择数据集总文件夹，里面按食材名分文件夹" })
        {
            if (folder == null && source.ShowDialog(owner) != DialogResult.OK) return false;
            if (folder != null) source.SelectedPath = folder;
            try {
                string workspace = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".."));
                string python = Path.Combine(workspace, "training", ".venv", "Scripts", "python.exe");
                string script = Path.Combine(workspace, "training", "auto_dataset.py");
                string output = Path.Combine(RuntimeFiles.Root, "datasets", Guid.NewGuid().ToString("N"));
                string validationFile = Path.Combine(RuntimeFiles.Root, "validation-folder.txt");
                string validation = File.Exists(validationFile) ? File.ReadAllText(validationFile).Trim() : "";
                string validationArgs = Directory.Exists(validation) ? " --validation " + RuntimeFiles.Quote(validation) : "";
                var info = new System.Diagnostics.ProcessStartInfo(python,
                    RuntimeFiles.Quote(script) + " --source " + RuntimeFiles.Quote(source.SelectedPath) +
                    " --output " + RuntimeFiles.Quote(output) + validationArgs) {
                    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };
                info.EnvironmentVariables["PYTHONUTF8"] = "1";
                using (var process = System.Diagnostics.Process.Start(info)) {
                    var stdout = process.StandardOutput.ReadToEndAsync();
                    var stderr = process.StandardError.ReadToEndAsync();
                    process.WaitForExit();
                    if (process.ExitCode != 0) throw new InvalidDataException(stderr.Result);
                    Directory.CreateDirectory(RuntimeFiles.Root);
                    File.WriteAllText(Path.Combine(RuntimeFiles.Root, "training-settings.json"),
                        new JavaScriptSerializer().Serialize(new TrainingSettings {
                            samples = Path.Combine(output, "samples.zip"),
                            splits = Path.Combine(output, "splits.json"),
                            checkpoint = Path.Combine(workspace, "training", "upstream", "model_C_v2.pt"),
                            output = Path.Combine(workspace, "training-runs")
                        }));
                    return true;
                }
            } catch (Exception ex) { MessageBox.Show(owner, ex.Message, "导入未完成"); return false; }
        }
    }

    internal static void Pack(string train, string validation, string output)
    {
        if (Path.GetFullPath(train).TrimEnd('\\').Equals(Path.GetFullPath(validation).TrimEnd('\\'), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("训练照片和验证照片需要使用不同文件夹。");
        var json = new JavaScriptSerializer();
        var rows = new List<object>();
        string stage = output + Guid.NewGuid().ToString("N");
        try {
            using (var zip = ZipFile.Open(stage, ZipArchiveMode.Create)) {
                foreach (var pair in new[] { Tuple.Create(train, "import-train"), Tuple.Create(validation, "import-val") }) {
                    int count = 0;
                    foreach (string category in Directory.GetDirectories(pair.Item1))
                        foreach (string photo in Directory.GetFiles(category, "*", SearchOption.AllDirectories)
                            .Where(p => new[] { ".jpg", ".jpeg", ".png", ".webp", ".bmp" }.Contains(Path.GetExtension(p).ToLowerInvariant()))) {
                            string name = "images/" + rows.Count + Path.GetExtension(photo);
                            zip.CreateEntryFromFile(photo, name);
                            rows.Add(new { image = name, label = Path.GetFileName(category), batch = pair.Item2,
                                layer = "未知", foil = "未知", device = "" });
                            count++;
                        }
                    if (count == 0) throw new InvalidDataException("没有找到照片，请确认按食材建立了子文件夹：" + pair.Item1);
                }
                using (var writer = new StreamWriter(zip.CreateEntry("labels.jsonl").Open()))
                    foreach (var row in rows) writer.WriteLine(json.Serialize(row));
            }
            if (File.Exists(output)) File.Replace(stage, output, null); else File.Move(stage, output);
            File.WriteAllText(output + ".splits.json", "{\"import-train\":\"train\",\"import-val\":\"val\"}");
        } finally { if (File.Exists(stage)) File.Delete(stage); }
    }
}
