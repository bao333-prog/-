using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal sealed class TrainingSettings
{
    public string samples { get; set; }
    public string splits { get; set; }
    public string checkpoint { get; set; }
    public string output { get; set; }
}

internal static class Training
{
    internal static bool ModelAdded;
    private static readonly string SettingsFile = Path.Combine(RuntimeFiles.Root, "training-settings.json");
    private static readonly System.Web.Script.Serialization.JavaScriptSerializer Json = new System.Web.Script.Serialization.JavaScriptSerializer();

    internal static void Show(Form owner)
    {
        ModelAdded = false;
        using (var dialog = new Form { Text = "训练与升级 · 使用步骤", Width = 720, Height = 370, StartPosition = FormStartPosition.CenterParent })
        {
            var instructions = new Label { Dock = DockStyle.Fill, Padding = new Padding(20), Text =
                "点击“选数据集并训练”，选中包含各食材子文件夹的总文件夹。程序会整理照片、训练并添加新版本。训练进度会显示在新窗口。\n\n已有独立验证集可先在主界面选择。连拍照片应分批拍摄。"+
                "\n\n再次使用上次的数据，可点“使用上次数据训练新版本”。" };
            var actions = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 90 };
            var import = new Button { Text = "选数据集并训练", AutoSize = true };
            var photos = new Button { Text = "打开实拍照片文件夹", AutoSize = true };
            var start = new Button { Text = "重新选择训练材料", AutoSize = true };
            var update = new Button { Text = "使用上次数据训练新版本", AutoSize = true };
            update.Click += (sender, args) => Start(dialog, true);

            import.Click += (sender, args) => { FromFolder(dialog); if (ModelAdded) dialog.Close(); };
            photos.Click += (sender, args) => Records.OpenFolder();
            start.Click += (sender, args) => Start(dialog, false);
            actions.Controls.Add(import); actions.Controls.Add(photos); actions.Controls.Add(start); actions.Controls.Add(update);
            dialog.Controls.Add(instructions); dialog.Controls.Add(actions);
            Appearance.Attach(dialog);
            dialog.ShowDialog(owner);
        }
    }

    internal static void FromFolder(Form owner, string folder = null)
    {
        ModelAdded = false;
        if (DatasetImport.Show(owner, folder)) Start(owner, true);
    }

    private static void Start(Form owner, bool reuse)
    {
        string workspace = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".."));
        string script = Path.Combine(workspace, "training", "run.py");
        string python = Path.Combine(workspace, "training", ".venv", "Scripts", "python.exe");
        if (!File.Exists(script) || !File.Exists(python))
        {
            MessageBox.Show(owner, "训练需要源码目录和单独的训练环境。请先运行 training/Setup.ps1。\n" +
                "训练说明：" + Path.Combine(workspace, "training", "README.md"), "训练环境尚未准备");
            return;
        }
        using (var samples = new OpenFileDialog { Title = "选择已确认训练素材 ZIP", Filter = "训练素材|*.zip" })
        using (var splits = new OpenFileDialog { Title = "选择拍摄批次划分 JSON", Filter = "批次划分|*.json" })
        using (var checkpoint = new OpenFileDialog { Title = "选择训练起点权重", Filter = "PyTorch 权重|*.pt",
            InitialDirectory = Path.Combine(workspace, "training", "upstream"), FileName = "model_C_v2.pt" })
        using (var output = new FolderBrowserDialog { Description = "选择训练输出的父文件夹，将自动建立独立目录" })
        {
            TrainingSettings settings = null;
            try {
                if (reuse && File.Exists(SettingsFile))
                    settings = Json.Deserialize<TrainingSettings>(File.ReadAllText(SettingsFile));
            } catch (Exception ex) {
                MessageBox.Show(owner, "读取上次训练设置失败：" + ex.Message);
                return;
            }
            if (settings != null && (!File.Exists(settings.samples) || !File.Exists(settings.splits) || !File.Exists(settings.checkpoint))) {
                MessageBox.Show(owner, "上次选择的训练文件已移动，请点击“重新选择训练材料”。");
                return;
            }
            if (settings == null) {
                if (samples.ShowDialog(owner) != DialogResult.OK || splits.ShowDialog(owner) != DialogResult.OK ||
                    checkpoint.ShowDialog(owner) != DialogResult.OK || output.ShowDialog(owner) != DialogResult.OK) return;
                settings = new TrainingSettings { samples = samples.FileName, splits = splits.FileName,
                    checkpoint = checkpoint.FileName, output = output.SelectedPath };
                Directory.CreateDirectory(RuntimeFiles.Root);
                File.WriteAllText(SettingsFile, Json.Serialize(settings));
            }
            samples.FileName = settings.samples;
            splits.FileName = settings.splits;
            checkpoint.FileName = settings.checkpoint;
            output.SelectedPath = settings.output;
            string run = Path.Combine(output.SelectedPath, "training-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N").Substring(0, 6));
            var info = new ProcessStartInfo(python,
                "-u " + RuntimeFiles.Quote(script) + " --samples " + RuntimeFiles.Quote(samples.FileName) +
                " --splits " + RuntimeFiles.Quote(splits.FileName) + " --checkpoint " + RuntimeFiles.Quote(checkpoint.FileName) +
                " --output " + RuntimeFiles.Quote(run)) {
                UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true,
                WorkingDirectory = workspace
            };
            info.EnvironmentVariables["PYTHONUTF8"] = "1";
            using (var dialog = new Form { Text = "训练进度", Width = 850, Height = 600, StartPosition = FormStartPosition.CenterParent })
            using (var process = new Process { StartInfo = info, EnableRaisingEvents = true })
            {
                var log = new TextBox { Dock = DockStyle.Fill, Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Both };
                var folder = new Button { Text = "打开输出文件夹", Dock = DockStyle.Bottom, Enabled = false };
                folder.Click += (sender, args) => Process.Start("explorer.exe", RuntimeFiles.Quote(run));
                dialog.Controls.Add(log);
                dialog.Controls.Add(folder);
                bool running = false;
                DataReceivedEventHandler received = (sender, args) => {
                    if (args.Data != null && !dialog.IsDisposed)
                        dialog.BeginInvoke(new Action(() => log.AppendText(args.Data + Environment.NewLine)));
                };
                process.OutputDataReceived += received;
                process.ErrorDataReceived += received;
                dialog.Shown += (sender, args) => {
                    try {
                        process.Start(); running = true;
                        process.BeginOutputReadLine(); process.BeginErrorReadLine();
                        System.Threading.Tasks.Task.Run(() => {
                            process.WaitForExit();
                            dialog.BeginInvoke(new Action(() => {
                                folder.Enabled = Directory.Exists(run);
                                if (process.ExitCode != 0) {
                                    running = false;
                                    log.AppendText("\r\n训练失败，当前模型继续使用。请查看上面的错误信息。");
                                    return;
                                }
                                log.AppendText("\r\n训练完成，正在检查并导入新版…");
                                System.Threading.Tasks.Task.Run(() => {
                                    try {
                                        ModelLibrary.Import(Path.Combine(run, "model-package.zip"));
                                        dialog.BeginInvoke(new Action(() => {
                                            ModelAdded = true;
                                            running = false;
                                            log.AppendText("\r\n新模型已加入可选列表。要使用它，请打开“模型管理 / 切换版本”。");
                                        }));
                                    } catch (Exception ex) {
                                        dialog.BeginInvoke(new Action(() => {
                                            running = false;
                                            log.AppendText("\r\n更新失败：" + ex.Message);
                                        }));
                                    }
                                });
                            }));
                        });
                    } catch (Exception ex) { log.Text = ex.Message; }
                };
                dialog.FormClosing += (sender, args) => {
                    if (running) { args.Cancel = true; MessageBox.Show(dialog, "训练仍在进行，请等待完成。"); }
                };
                Appearance.Attach(dialog);
            dialog.ShowDialog(owner);
            }
        }
    }
}
