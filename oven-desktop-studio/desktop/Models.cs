using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class ModelLibrary
{
    internal static readonly string Root = Path.Combine(RuntimeFiles.Root, "models");
    private static string Selection { get { return Path.Combine(Root, "active.txt"); } }

    internal static string[] FoodNames()
    {
        string folder = Active();
        using (var stream = folder == null ? typeof(ModelLibrary).Assembly.GetManifestResourceStream("OvenDesk.Food.json")
                                          : File.OpenRead(Path.Combine(folder, "food.json")))
        using (var reader = new StreamReader(stream)) {
            var config = new JavaScriptSerializer().Deserialize<System.Collections.Generic.Dictionary<string, object>>(reader.ReadToEnd());
            return ((System.Collections.IEnumerable)config["names"]).Cast<string>().ToArray();
        }
    }

    internal static string Active()
    {
        if (!File.Exists(Selection)) return null;
        string id = File.ReadAllText(Selection).Trim();
        if (id.Length == 0) return null;
        if (id != Path.GetFileName(id) || !Directory.Exists(Path.Combine(Root, id)))
            throw new IOException("所选模型不存在，请在模型管理中重新选择。");
        return Path.Combine(Root, id);
    }

    internal static void Activate(string folder)
    {
        Directory.CreateDirectory(Root);
        string staging = Selection + "." + Guid.NewGuid().ToString("N");
        try {
            File.WriteAllText(staging, folder == null ? "" : Path.GetFileName(folder));
            if (File.Exists(Selection)) File.Replace(staging, Selection, null);
            else File.Move(staging, Selection);
        } finally { if (File.Exists(staging)) File.Delete(staging); }
    }

    internal static string Import(string package)
    {
        Directory.CreateDirectory(Root);
        string staging = Path.Combine(Root, "import-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try {
            using (var zip = ZipFile.OpenRead(package))
                foreach (string name in new[] { "food.onnx", "food.json" }) {
                    var entries = zip.Entries.Where(e => e.FullName == name).ToArray();
                    if (entries.Length != 1) throw new InvalidDataException("ZIP 根目录需要各一份 food.onnx 和 food.json。");
                    entries[0].ExtractToFile(Path.Combine(staging, name));
                }
            using (var probe = new Classifier(staging)) probe.Run(null);
            string id;
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(package))
                id = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            string target = Path.Combine(Root, id);
            if (!Directory.Exists(target)) Directory.Move(staging, target);
            return target;
        } finally { if (Directory.Exists(staging)) Directory.Delete(staging, true); }
    }

    internal static async Task<string> Manage(Form owner, Action<Classifier> useReady)
    {
        using (var dialog = new Form { Text = "模型管理", Width = 760, Height = 420, StartPosition = FormStartPosition.CenterParent })
        {
            var list = new ListBox { Dock = DockStyle.Fill };
            var details = new TextBox { Dock = DockStyle.Bottom, Height = 120, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical };
            var buttons = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 45 };
            var import = new Button { Text = "导入模型 ZIP", AutoSize = true };
            var use = new Button { Text = "检查并使用所选版本", AutoSize = true };
            bool working = false;
            string chosen = null;
            list.Items.Add("内置模型");
            Directory.CreateDirectory(Root);
            foreach (string directory in Directory.GetDirectories(Root).Where(p => !Path.GetFileName(p).StartsWith("import-")))
                list.Items.Add(directory);
            list.SelectedIndex = 0;
            var active = Active();
            if (active != null && list.Items.Contains(active)) list.SelectedItem = active;
            details.Text = "当前版本：" + (active ?? "内置模型") + Environment.NewLine +
                "模型 ZIP 根目录放 food.onnx、food.json。配置需包含类别顺序、size、resize_mode、mean、std，可加 version。";
            import.Click += async (sender, args) => {
                if (working) return;
                using (var picker = new OpenFileDialog { Filter = "模型包|*.zip" }) {
                    if (picker.ShowDialog(dialog) != DialogResult.OK) return;
                    working = true;
                    details.Text = "正在检查模型和类别配置，请稍候…";
                    try {
                        string folder = await Task.Run(() => Import(picker.FileName));
                        if (!list.Items.Contains(folder)) list.Items.Add(folder);
                        list.SelectedItem = folder;
                        details.Text = "导入及运行检查成功。点击“检查并使用所选版本”启用。";
                    } catch (Exception ex) { details.Text = "导入失败：" + ex.Message; }
                    finally { working = false; }
                }
            };
            use.Click += async (sender, args) => {
                if (working) return;
                working = true;
                Classifier probe = null;
                try {
                    string folder = list.SelectedIndex == 0 ? null : (string)list.SelectedItem;
                    probe = new Classifier(folder);
                    details.Text = "正在加载所选模型…";
                    Prediction result = await Task.Run(() => probe.Run(null));
                    string summary = "版本：" + result.modelVersion + "\n类别：" + String.Join("、", result.classes);
                    if (MessageBox.Show(dialog, summary + "\n使用这个版本？", "模型检查通过", MessageBoxButtons.YesNo) == DialogResult.Yes) {
                        Activate(folder);
                        useReady(probe); probe = null;
                        chosen = folder ?? "";
                        working = false;
                        dialog.Close();
                    }
                } catch (Exception ex) { details.Text = "启用失败：" + ex.Message; }
                finally { if (probe != null) probe.Dispose(); working = false; }
            };
            dialog.FormClosing += (sender, args) => { if (working) args.Cancel = true; };
            buttons.Controls.Add(import);
            buttons.Controls.Add(use);
            var remove = new Button { Text = "删除所选模型", AutoSize = true };
            buttons.Controls.Add(remove);
            list.FormattingEnabled = true;
            list.Format += (s,e) => {
                string path = e.ListItem as string;
                if (path != null && Directory.Exists(path))
                    e.Value = Directory.GetCreationTime(path).ToString("yyyy-MM-dd HH:mm") + "  " + Path.GetFileName(path).Substring(0,8) + (path == active ? "  · 使用中" : "");
            };
            remove.Click += (s,e) => {
                if (working) return;
                string path = list.SelectedItem as string;
                if (list.SelectedIndex <= 0 || path == Active()) { details.Text = "请先切换到其他版本。内置模型始终保留。"; return; }
                try {
                    if (Path.GetDirectoryName(Path.GetFullPath(path)) != Path.GetFullPath(Root) ||
                        !System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileName(path), "^[a-f0-9]{64}$")) throw new IOException("无效的模型目录");
                    if (MessageBox.Show(dialog, "从模型库删除 " + list.GetItemText(path) + "？训练输出包仍可重新导入。", "删除模型", MessageBoxButtons.YesNo) != DialogResult.Yes) return;
                    var trash = Path.Combine(RuntimeFiles.Root, "removed-models");
                    Directory.CreateDirectory(trash);
                    Directory.Move(path, Path.Combine(trash, Guid.NewGuid().ToString("N")));
                    list.Items.Remove(path); list.SelectedIndex = 0; details.Text = "模型已从列表删除。";
                } catch (Exception ex) { details.Text = ex.Message; }
            };
            dialog.Controls.Add(list);
            dialog.Controls.Add(details);
            dialog.Controls.Add(buttons);
            Appearance.Attach(dialog);
            dialog.ShowDialog(owner);
            return await Task.FromResult(chosen);
        }
    }
}

internal static class Comparison
{
    internal static async Task Show(Form owner, string source, Classifier current)
    {
        if (source == null) throw new InvalidOperationException("请先选择图片或拍照。");
        using (var dialog = new Form { Text = "同图对照", Width = 900, Height = 650, StartPosition = FormStartPosition.CenterParent })
        {
            var report = new TextBox { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both };
            var buttons = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 65 };
            var url = new TextBox { Width = 300, Text = "http://127.0.0.1:8000/api/predict" };
            var web = new Button { Text = "对照网页版", AutoSize = true };
            var save = new Button { Text = "保存对照报告", AutoSize = true };
            var reference = new Button { Text = "导入原始模型结果 JSON", AutoSize = true };
            string temp = Path.Combine(Path.GetTempPath(), "OvenCompare-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temp);
            string snapshot = Path.Combine(temp, "source.image");
            File.Copy(source, snapshot);
            bool working = true;
            var json = new JavaScriptSerializer();
            dialog.Shown += async (sender, args) => {
                try {
                    var result = await Task.Run(() => current.Run(snapshot, Path.Combine(temp, "current-input.png")));
                    File.WriteAllText(Path.Combine(temp, "current.json"), json.Serialize(result));
                    report.Text = Summary("当前 EXE", result);
                    using (var original = new Classifier(null)) {
                        var baseline = await Task.Run(() => original.Run(snapshot, Path.Combine(temp, "builtin-input.png")));
                        File.WriteAllText(Path.Combine(temp, "builtin.json"), json.Serialize(baseline));
                        report.AppendText("\r\n" + Summary("内置模型基线", baseline));
                        report.AppendText("\r\n输入张量相同：" + (result.tensorSha256 == baseline.tensorSha256));
                        report.AppendText("\r\n第一候选相同：" + (result.candidates[0].name == baseline.candidates[0].name));
                    }
                } catch (Exception ex) { report.AppendText("\r\n本地对照失败：" + ex.Message); }
                finally { working = false; }
            };
            web.Click += async (sender, args) => {
                if (working) return;
                Uri endpoint;
                if (!Uri.TryCreate(url.Text, UriKind.Absolute, out endpoint) || !endpoint.IsLoopback) {
                    MessageBox.Show(dialog, "请输入本机网页版接口地址，例如 http://127.0.0.1:8000/api/predict"); return;
                }
                working = true;
                try {
                    using (var client = new HttpClient { Timeout = TimeSpan.FromSeconds(40) })
                    using (var form = new MultipartFormDataContent()) {
                        var bytes = new ByteArrayContent(File.ReadAllBytes(snapshot));
                        byte[] signature = File.ReadAllBytes(snapshot);
                        bool png = signature.Length >= 8 && signature[0] == 137 && signature[1] == 80;
                        bool webp = signature.Length >= 12 && Encoding.ASCII.GetString(signature, 8, 4) == "WEBP";
                        string mime = png ? "image/png" : webp ? "image/webp" : "image/jpeg";
                        bytes.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(mime);
                        form.Add(bytes, "image", png ? "comparison.png" : webp ? "comparison.webp" : "comparison.jpg");
                        using (var response = await client.PostAsync(endpoint, form)) {
                            string body = await response.Content.ReadAsStringAsync();
                            File.WriteAllText(Path.Combine(temp, "web-response.txt"), body);
                            report.AppendText("\r\n网页版 HTTP " + (int)response.StatusCode + "\r\n" + body +
                                "\r\n发送的是同一份原图。网页版未提供的模型哈希和输入张量无法核实。");
                        }
                    }
                } catch (Exception ex) { report.AppendText("\r\n网页版调用失败：" + ex.Message); }
                finally { working = false; }
            };
            reference.Click += (sender, args) => {
                if (working) return;
                using (var picker = new OpenFileDialog { Filter = "原始模型结果|*.json" }) {
                    if (picker.ShowDialog(dialog) != DialogResult.OK) return;
                    try {
                        string text = File.ReadAllText(picker.FileName);
                        var result = json.Deserialize<Prediction>(text);
                        var local = json.Deserialize<Prediction>(File.ReadAllText(Path.Combine(temp, "current.json")));
                        if (result.sourceSha256 != local.sourceSha256 || String.IsNullOrEmpty(result.sourceSha256))
                            throw new InvalidDataException("原始模型结果需要提供与当前图片一致的 sourceSha256。");
                        File.WriteAllText(Path.Combine(temp, "reference.json"), text);
                        report.AppendText("\r\n" + Summary("原始模型结果", result));
                    } catch (Exception ex) { MessageBox.Show(dialog, ex.Message); }
                }
            };
            save.Click += (sender, args) => {
                if (working) return;
                using (var picker = new SaveFileDialog { Filter = "对照报告|*.zip", FileName = "OvenDesk-comparison.zip" }) {
                    if (picker.ShowDialog(dialog) != DialogResult.OK) return;
                    string stage = picker.FileName + "." + Guid.NewGuid().ToString("N");
                    try {
                        File.WriteAllText(Path.Combine(temp, "summary.txt"), report.Text);
                        ZipFile.CreateFromDirectory(temp, stage);
                        if (File.Exists(picker.FileName)) File.Replace(stage, picker.FileName, null);
                        else File.Move(stage, picker.FileName);
                        MessageBox.Show(dialog, "对照报告已保存。");
                    } catch (Exception ex) { MessageBox.Show(dialog, ex.Message); }
                    finally { if (File.Exists(stage)) File.Delete(stage); }
                }
            };
            dialog.FormClosing += (sender, args) => { if (working) args.Cancel = true; };
            buttons.Controls.Add(url); buttons.Controls.Add(web); buttons.Controls.Add(reference); buttons.Controls.Add(save);
            dialog.Controls.Add(report); dialog.Controls.Add(buttons);
            try { Appearance.Attach(dialog);
            dialog.ShowDialog(owner); }
            finally { Directory.Delete(temp, true); }
        }
        await Task.FromResult(0);
    }

    private static string Summary(string title, Prediction prediction)
    {
        return title + "\r\n版本：" + prediction.modelVersion + "\r\n模型 SHA256：" + prediction.modelSha256 +
            "\r\n类别：" + String.Join("、", prediction.classes ?? new string[0]) +
            "\r\n输入处理：" + prediction.resizeMode + "\r\n输入张量 SHA256：" + prediction.tensorSha256 +
            "\r\n" + String.Join("\r\n", prediction.candidates.Select(c => c.name + "  " + c.score.ToString("P4"))) + "\r\n";
    }
}

