using System;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 2 && args[0] == "--verify-image")
            {
                using (var image = Desk.ReadImage(args[1])) return image.Width > 0 ? 0 : 1;
            }
            if (args.Length == 3 && args[0] == "--verify-classifier")
            {
                using (var classifier = new Classifier())
                    File.WriteAllLines(args[2], classifier.Run(Path.GetFullPath(args[1])).candidates
                        .Select(c => c.name + "\t" + c.score.ToString("F6", CultureInfo.InvariantCulture)));
                return 0;
            }
            if (args.Length == 2 && args[0] == "--self-test")
            {
                using (var classifier = new Classifier())
                    File.WriteAllText(args[1], new JavaScriptSerializer().Serialize(classifier.Run(null)));
                return 0;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Desk());
            return 0;
        }
        catch (Exception ex)
        {
            try
            {
                File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                    "startup-error.txt"), ex.ToString());
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            if (args.Length == 0)
                MessageBox.Show(ex.Message, "OvenDesk");
            else if (args.Length >= 2 && (args[0] == "--self-test" || args[0] == "--verify-classifier"))
                File.WriteAllText(args[args.Length - 1] + ".error.txt", ex.ToString());
            return 1;
        }
    }
}

internal sealed class Desk : Form
{
    private ToolTip usageTips;
    private Classifier classifier = new Classifier();
    private readonly PictureBox preview = new PictureBox { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom };
    private readonly Label status = new Label { AutoSize = true, MaximumSize = new Size(870, 0) };
    private readonly Label cameraStatus = new Label { AutoSize = true, Text = "摄像头已关闭" };
    private readonly Label quality = new Label { AutoSize = true };
    private readonly Label checkResult = new Label { AutoSize = true, MaximumSize = new Size(850, 0), Text = "尚未检查程序" };
    private string currentRecord;
    private readonly Label confirmed = new Label { AutoSize = true };
    private readonly ComboBox cameras = new ComboBox { Width = 200, DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly ComboBox ingredient = new ComboBox { Width = 145, MaxDropDownItems = 14 };
    private readonly ComboBox state = new ComboBox { Width = 85, DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly NumericUpDown weight = new NumericUpDown { Maximum = 50000, Width = 85 };
    private readonly FlowLayoutPanel candidates = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Fill };
    private readonly Timer timer = new Timer { Interval = 65 };
    private readonly Stopwatch deviceRefresh = Stopwatch.StartNew();
    private readonly Button capture;
    private readonly Button export;
    private CameraStream camera;
    private byte[] displayed;
    private int brightness, contrast = 100, liveGeneration;
    private decimal exposureEv;
    private int saturation = 100;
    private bool OriginalColors { get { return brightness == 0 && contrast == 100 && exposureEv == 0 && saturation == 100; } }
    private string openedPhoto;
    private byte[] frozenFrame;
    private bool liveRunning;
    private DateTime nextLive;
    private readonly CheckBox live = new CheckBox { Text = "实时识别", AutoSize = true, Anchor = AnchorStyles.Left };
    private bool reviewing;
    private bool wantsCamera;
    private bool busy;
    private bool updatingDevices;
    private string cameraRuntime;
    private string diagnosticImage;
    private Prediction lastPrediction;
    private string inputPath;
    private string inputDescription;
    private readonly SampleFields sampleFields = new SampleFields();
    private DateTime nextCameraStart;
    private int cameraRequest;
    private bool openingCamera;

    internal Desk()
    {
        Text = "OvenDesk 食材识别";
        ClientSize = new Size(1120, 860);
        AutoScaleMode = AutoScaleMode.Dpi;
        MinimumSize = new Size(1040, 720);
        Font = new Font("Microsoft YaHei UI", 10f);
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(22), ColumnCount = 1, RowCount = 10 };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        for (int i = 0; i < 8; i++) layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(layout);

        var commands = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Fill };
        commands.Controls.Add(Button("打开图片", async () => {
            using (var dialog = Images())
                if (dialog.ShowDialog(this) == DialogResult.OK) await OpenImage(dialog.FileName);
        }));
        commands.Controls.Add(cameras);
        commands.Controls.Add(live);
        commands.Controls.Add(Button("照片参数", EditPhoto));
        live.CheckedChanged += async (s,e) => {
            liveGeneration++;
            if (!live.Checked) { cameraStatus.Text = "实时识别已关闭"; return; }
            if (busy) { live.Checked = false; return; }
            reviewing = false; SetPrediction(null);
            wantsCamera = true;
            try { if (camera == null && !openingCamera) await StartCamera(); }
            catch (Exception ex) { live.Checked = false; status.Text = ex.Message; }
        };
        commands.Controls.Add(Button("打开 / 关闭摄像头", async () => {
            wantsCamera = !wantsCamera;
            if (!wantsCamera) { StopCamera(); status.Text = "摄像头已关闭。"; return; }
            SetPrediction(null);
            await StartCamera();
        }));
        capture = Button("识别并拍照", CapturePhoto);
        capture.Enabled = false;
        commands.Controls.Add(capture);
        commands.Controls.Add(Button("检查程序", SelfTest));
        commands.Controls.Add(Button("选择训练底库", () => {
            using (var picker = new FolderBrowserDialog { Description = "选择含 train、val、test 或 manifest.csv 的14类原始数据集" }) {
                if (picker.ShowDialog(this) == DialogResult.OK) {
                    Directory.CreateDirectory(RuntimeFiles.Root);
                    File.WriteAllText(Path.Combine(RuntimeFiles.Root,"replay-base.txt"),picker.SelectedPath);
                    status.Text = "训练底库：" + picker.SelectedPath;
                }
            }
            return Task.FromResult(0);
        }));
        commands.Controls.Add(Button("验证集", () => {
            using (var options = new Form { Text = "验证集", Width = 400, Height = 155, StartPosition = FormStartPosition.CenterParent }) {
                var row = new FlowLayoutPanel { Dock = DockStyle.Fill };
                row.Controls.Add(new Button { Text = "拍摄验证照片", AutoSize = true, DialogResult = DialogResult.Yes });
                row.Controls.Add(new Button { Text = "选择已有验证集", AutoSize = true, DialogResult = DialogResult.No });
                options.Controls.Add(row);
                var choice = options.ShowDialog(this);
                if (choice == DialogResult.Yes) {
                    wantsCamera = false; StopCamera(); Collection.Show(this, true, false, cameras.SelectedItem as CameraDevice);
                    return Task.FromResult(0);
                }
                if (choice != DialogResult.No) return Task.FromResult(0);
            }
            using (var picker = new FolderBrowserDialog { Description = "选择未参加训练的验证照片总文件夹，内部按食材名称分类" }) {
                if (picker.ShowDialog(this) == DialogResult.OK) {
                    Directory.CreateDirectory(RuntimeFiles.Root);
                    File.WriteAllText(Path.Combine(RuntimeFiles.Root, "validation-folder.txt"), picker.SelectedPath);
                    MessageBox.Show(this, "已记住验证集。下次导入训练数据集时会自动使用。");
                }
            }
            return Task.FromResult(0);
        }));
        commands.Controls.Add(Button("拍照并训练", () => {
            wantsCamera = false;
            StopCamera();
            Training.ModelAdded = false;
            Collection.Show(this, false, true, cameras.SelectedItem as CameraDevice);
            if (Training.ModelAdded) status.Text = "新模型已加入模型管理，可手动切换。";
            return Task.FromResult(0);
        }));
        commands.Controls.Add(Button("普通拍照", () => {
            wantsCamera = false; StopCamera(); Collection.Show(this, false, false, cameras.SelectedItem as CameraDevice);
            return Task.FromResult(0);
        }));
        commands.Controls.Add(Button("选择数据集并训练", () => {
            busy = true;
            try {
                Training.FromFolder(this);
                if (Training.ModelAdded) status.Text = "新模型已加入模型管理，可手动切换。";
            } finally { busy = false; }
            return Task.FromResult(0);
        }));
        commands.Controls.Add(Button("模型管理 / 切换版本", async () => {
            busy = true;
            try {
                string chosen = await ModelLibrary.Manage(this, ready => { var previous = classifier; classifier = ready; previous.Dispose(); });
                if (chosen != null) {
                    ingredient.Items.Clear(); ingredient.Items.AddRange(ModelLibrary.FoodNames());
                    SetPrediction(null);
                    status.Text = "模型已切换，请重新选择图片或拍摄。当前：" + (chosen.Length == 0 ? "内置模型" : Path.GetFileName(chosen));
                }
            } finally { busy = false; }
        }));
        commands.Controls.Add(Button("同图对照", async () => {
            busy = true;
            try { await Comparison.Show(this, inputPath, classifier); }
            finally { busy = false; }
        }));
        commands.Controls.Add(Button("识别记录 / 训练素材", () => { Records.Show(this); return Task.FromResult(0); }));
        export = Button("保存问题报告", Export);
        export.Enabled = true;
        commands.Controls.Add(export);
        layout.Controls.Add(commands, 0, 0);
        preview.BackColor = Color.FromArgb(232, 232, 237);
        preview.AllowDrop = true;
        preview.DragEnter += (sender, args) => {
            if (args.Data.GetDataPresent(DataFormats.FileDrop)) args.Effect = DragDropEffects.Copy;
        };
        preview.DragDrop += async (sender, args) => {
            try {
            string[] files = args.Data.GetData(DataFormats.FileDrop) as string[];
            if (files != null && files.Length == 1) await OpenImage(files[0]);
            } catch (Exception ex) { status.Text = ex.Message; }
        };
        layout.Controls.Add(preview, 0, 1);
        layout.Controls.Add(status, 0, 2);
        layout.Controls.Add(candidates, 0, 3);

        var fields = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Fill };
        fields.Controls.Add(new Label { Text = "食材", AutoSize = true });
        ingredient.Items.AddRange(ModelLibrary.FoodNames());
        fields.Controls.Add(ingredient);
        state.Items.AddRange(new object[] { "冷藏", "冷冻", "常温" });
        state.SelectedIndex = 2;
        fields.Controls.Add(state);
        fields.Controls.Add(new Label { Text = "重量（克，0 表示未填）", AutoSize = true });
        fields.Controls.Add(weight);
        fields.Controls.Add(Button("确认并保存训练素材", () => {
            if (currentRecord == null) throw new InvalidOperationException("请先识别一张照片。");
            if (String.IsNullOrWhiteSpace(ingredient.Text))
                throw new InvalidOperationException("请选择或填写食材名称。");
            Records.Confirm(currentRecord, ingredient.Text.Trim(), sampleFields);
            confirmed.Text = "已保存训练素材：" + ingredient.Text.Trim() + " · " + state.Text +
                (weight.Value == 0 ? "" : " · " + weight.Value + " 克");
            return Task.FromResult(0);
        }));
        fields.Controls.Add(Button("清除", () => {
            SetPrediction(null);
            if (!wantsCamera) ReplacePreview(null);
            status.Text = "已清除识别结果。";
            return Task.FromResult(0);
        }));
        layout.Controls.Add(fields, 0, 4);
        layout.Controls.Add(confirmed, 0, 5);
        layout.Controls.Add(cameraStatus, 0, 6);
        layout.Controls.Add(quality, 0, 7);
        layout.Controls.Add(checkResult, 0, 8);
        layout.Controls.Add(sampleFields, 0, 9);
        status.Font = new Font(Font.FontFamily, 15f, FontStyle.Bold);

        WorkspaceLayout.Arrange(this, layout, commands, preview, status, candidates, fields,
            confirmed, cameraStatus, quality, checkResult, sampleFields, Button);
        usageTips = Appearance.Attach(this);
        status.Text = "打开照片，开始识别";
        cameras.SelectedIndexChanged += async (sender, args) => {
            if (wantsCamera && !updatingDevices)
            {
                try { await StartCamera(); }
                catch (Exception ex) { status.Text = ex.Message; }
            }
        };
        timer.Tick += async (sender, args) => {
            try { await Tick(); }
            catch (Exception ex) { status.Text = ex.Message; }
        };
        Shown += async (sender, args) => {
            try { await RefreshDevices(); }
            catch (Exception ex) { status.Text = "无法枚举摄像头。 " + ex.Message; }
            timer.Start();
            try { cameraRuntime = await Task.Run(() => RuntimeFiles.Prepare("OvenDesk.DesktopAI.zip", "ai")); }
            catch (Exception ex) { if (!IsDisposed) cameraStatus.Text = ex.Message; }
        };
        FormClosing += (sender, args) => {
            if (busy) { args.Cancel = true; status.Text = "请等当前识别完成。"; }
        };
        FormClosed += (sender, args) => {
            timer.Dispose();
            StopCamera();
            classifier.Dispose();
            ReplacePreview(null);
            DeleteDiagnostic();
        };
    }

    private Button Button(string title, Func<Task> action)
    {
        var button = new Button { AutoSize = true, Text = title };
        button.Click += async (sender, args) => {
            if (busy) return;
            try { await action(); }
            catch (Exception ex) { status.Text = ex.Message; }
        };
        if (usageTips != null) Appearance.Apply(candidates, usageTips);
        return button;
    }

    private static OpenFileDialog Images()
    {
        return new OpenFileDialog { Filter = "食材图片|*.jpg;*.jpeg;*.png;*.webp" };
    }

    private async Task OpenImage(string path)
    {
        if (busy) return;
        wantsCamera = false;
        StopCamera();
        sampleFields.SetDevice("");
        SetPrediction(null);
        busy = true;
        try
        {
            status.Text = "正在读取图片…";
            openedPhoto = path;
            using (var original = await Task.Run(() => ReadImage(path))) ReplacePreview(Adjust(original));
            inputPath = OriginalColors ? Records.Snapshot(path) : SnapshotPreview();
            await Recognize(inputPath, Path.GetFileName(path));
        }
        finally { busy = false; }
    }

    internal static Image ReadImage(string path)
    {
        byte[] signature = new byte[12];
        using (var stream = File.OpenRead(path)) stream.Read(signature, 0, signature.Length);
        bool webp = System.Text.Encoding.ASCII.GetString(signature, 8, 4) == "WEBP";
        if (!webp)
        {
            using (var image = Image.FromFile(path)) return new Bitmap(image);
        }
        string folder = RuntimeFiles.Prepare("OvenDesk.WebP.zip", "webp");
        string output = Path.GetTempFileName();
        try
        {
            var info = RuntimeFiles.Python(folder, "decode.py " + RuntimeFiles.Quote(Path.GetFullPath(path)) +
                " " + RuntimeFiles.Quote(output));
            using (var process = Process.Start(info))
            {
                if (!process.WaitForExit(15000))
                {
                    process.Kill();
                    throw new TimeoutException("WebP 解码超时。");
                }
                if (process.ExitCode != 0) throw new IOException("无法打开 WebP 图片。");
            }
            using (var image = Image.FromFile(output)) return new Bitmap(image);
        }
        finally { File.Delete(output); }
    }

    private void ReplacePreview(Image image)
    {
        var previous = preview.Image;
        preview.Image = image;
        if (previous != null) previous.Dispose();
        if (image != null && !wantsCamera) quality.Text = Records.Quality(image);
    }

    private bool enumerating;
    private async Task RefreshDevices()
    {
        if (enumerating) return;
        enumerating = true;
        System.Collections.Generic.List<CameraDevice> devices;
        try { devices = await Task.Run(() => Cameras.List()); } finally { enumerating = false; }
        if (IsDisposed) return;
        var selected = cameras.SelectedItem as CameraDevice;
        if (cameras.Items.Cast<CameraDevice>().Select(d => d.Id).SequenceEqual(devices.Select(d => d.Id)))
            return;
        updatingDevices = true;
        try
        {
            cameras.Items.Clear();
            cameras.Items.AddRange(devices.ToArray());
            int index = selected == null ? -1 : devices.FindIndex(d => d.Id == selected.Id);
            cameras.SelectedIndex = index >= 0 ? index : devices.FindIndex(d =>
                d.Name.IndexOf("IR Camera", StringComparison.OrdinalIgnoreCase) < 0);
        }
        finally { updatingDevices = false; }
        if (wantsCamera) StopCamera();
    }

    private async Task StartCamera()
    {
        StopCamera();
        int request = cameraRequest;
        var device = cameras.SelectedItem as CameraDevice;
        nextCameraStart = DateTime.UtcNow.AddSeconds(2);
        if (device == null) { cameraStatus.Text = "摄像头未连接，接入后会重试。"; return; }
        cameraStatus.Text = "摄像头连接中…";
        openingCamera = true;
        try
        {
            if (cameraRuntime == null)
                cameraRuntime = await Task.Run(() => RuntimeFiles.Prepare("OvenDesk.DesktopAI.zip", "ai"));
            if (request == cameraRequest && wantsCamera && !IsDisposed)
                camera = new CameraStream(cameraRuntime, device.Index);
        }
        finally
        {
            if (request == cameraRequest) openingCamera = false;
        }
    }

    private Bitmap Adjust(Image source)
    {
        if (OriginalColors) return new Bitmap(source);
        float scale = contrast / 100f, shift = brightness / 100f + (1 - scale) / 2;
        var matrix = new System.Drawing.Imaging.ColorMatrix();
        // Approximate linear-light exposure for the gamma-encoded camera image.
        scale *= (float)Math.Pow(2, (double)exposureEv / 2.2);
        float chroma = saturation / 100f;
        float gray = scale * (1 - chroma);
        matrix.Matrix00 = matrix.Matrix01 = matrix.Matrix02 = gray * 0.2126f;
        matrix.Matrix10 = matrix.Matrix11 = matrix.Matrix12 = gray * 0.7152f;
        matrix.Matrix20 = matrix.Matrix21 = matrix.Matrix22 = gray * 0.0722f;
        matrix.Matrix00 += scale * chroma; matrix.Matrix11 += scale * chroma; matrix.Matrix22 += scale * chroma;
        matrix.Matrix40 = matrix.Matrix41 = matrix.Matrix42 = shift;
        var result = new Bitmap(source.Width, source.Height);
        using (var graphics = Graphics.FromImage(result))
        using (var attr = new System.Drawing.Imaging.ImageAttributes()) {
            attr.SetColorMatrix(matrix);
            graphics.DrawImage(source, new Rectangle(Point.Empty, result.Size), 0, 0, source.Width, source.Height, GraphicsUnit.Pixel, attr);
        }
        return result;
    }

    private string SnapshotPreview()
    {
        string path = Path.GetTempFileName();
        try { preview.Image.Save(path, System.Drawing.Imaging.ImageFormat.Png); return Records.Snapshot(path); }
        finally { File.Delete(path); }
    }

    private async Task EditPhoto()
    {
        using (var dialog = new Form { Text = "照片参数", Width = 450, Height = 405, StartPosition = FormStartPosition.CenterParent })
        using (var panel = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 135, AutoScroll = true })
        using (var sample = new PictureBox { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom })
        using (var stream = new MemoryStream(frozenFrame ?? displayed ?? new byte[0]))
        using (var original = openedPhoto != null ? ReadImage(openedPhoto) : stream.Length > 0 ? Image.FromStream(stream) : null) {
            var light = new NumericUpDown { Minimum = -75, Maximum = 75, Value = brightness, Width = 65 };
            var gain = new NumericUpDown { Minimum = 25, Maximum = 200, Value = contrast, Width = 68 };
            var exposure = new NumericUpDown { Minimum = -3, Maximum = 3, DecimalPlaces = 1, Increment = 0.1m, Value = exposureEv, Width = 68 };
            var color = new NumericUpDown { Minimum = 0, Maximum = 200, Value = saturation, Width = 68 };
            decimal oldEv = exposureEv;
            int oldColor = saturation;
            int oldLight = brightness, oldGain = contrast;
            Action refresh = () => {
                brightness = (int)light.Value; contrast = (int)gain.Value; exposureEv = exposure.Value; saturation = (int)color.Value;
                if (original != null) { var old = sample.Image; sample.Image = Adjust(original); if (old != null) old.Dispose(); }
            };
            var reset = new Button { Text = "恢复默认", AutoSize = true };
            reset.Click += (s,e) => { light.Value = 0; gain.Value = 100; exposure.Value = 0; color.Value = 100; };
            exposure.ValueChanged += (s,e) => refresh(); color.ValueChanged += (s,e) => refresh();
            light.ValueChanged += (s,e) => refresh(); gain.ValueChanged += (s,e) => refresh();
            panel.Controls.AddRange(new Control[] { new Label { Text = "亮度", AutoSize = true }, light, new Label { Text = "对比度 %", AutoSize = true }, gain, new Label { Text = "曝光补偿 EV", AutoSize = true }, exposure, new Label { Text = "饱和度 %", AutoSize = true }, color, reset, new Button { Text = "应用", DialogResult = DialogResult.OK } });
            dialog.Controls.Add(sample); dialog.Controls.Add(panel);
            busy = true;
            try {
                refresh();
                if (dialog.ShowDialog(this) != DialogResult.OK) { brightness = oldLight; contrast = oldGain; exposureEv = oldEv; saturation = oldColor; return; }
                liveGeneration++;
                if (original != null && (reviewing || openedPhoto != null)) {
                    ReplacePreview(Adjust(original)); inputPath = SnapshotPreview(); await Recognize(inputPath, "调整后的照片");
                }
            } finally { busy = false; if (sample.Image != null) sample.Image.Dispose(); }
        }
    }

    private void StopCamera()
    {
        reviewing = false;
        openedPhoto = null; frozenFrame = null; liveGeneration++;
        cameraRequest++;
        openingCamera = false;
        if (camera != null) { camera.Dispose(); camera = null; }
        displayed = null;
        capture.Enabled = false;
        cameraStatus.Text = "摄像头已关闭 / 画面已定格";
    }

    private async Task Tick()
    {
        if (busy || !wantsCamera) return;
        if (deviceRefresh.ElapsedMilliseconds >= 1000)
        {
            deviceRefresh.Restart();
            await RefreshDevices();
            if (IsDisposed || !wantsCamera || busy) return;
        }
        if (camera == null)
        {
            if (!openingCamera && DateTime.UtcNow >= nextCameraStart) await StartCamera();
            return;
        }
        byte[] frame = camera.FreshFrame();
        capture.Enabled = frame != null;
        if (frame == null)
        {
            cameraStatus.Text = "尚未收到画面，正在尝试恢复。请检查设备权限或占用。";
            if (displayed != null) { displayed = null; ReplacePreview(null); }
            if (camera.Age.ElapsedMilliseconds >= 6000 && DateTime.UtcNow >= nextCameraStart)
                await StartCamera();
            return;
        }
        cameraStatus.Text = "摄像头已连接，画面正常";
        if (reviewing) { cameraStatus.Text = "摄像头已连接 · 正在查看已拍照片"; return; }
        if (ReferenceEquals(frame, displayed)) return;
        using (var stream = new MemoryStream(frame))
        using (var image = Image.FromStream(stream))
            ReplacePreview(Adjust(image));
        displayed = frame;
        if (live.Checked && !liveRunning && DateTime.UtcNow >= nextLive) {
            liveRunning = true;
            int generation = liveGeneration;
            string temp = Path.GetTempFileName();
            try {
                var worker = classifier;
                using (var adjusted = OriginalColors ? null : new Bitmap(preview.Image)) {
                    var prediction = await Task.Run(() => {
                        if (adjusted == null) File.WriteAllBytes(temp, frame);
                        else adjusted.Save(temp, System.Drawing.Imaging.ImageFormat.Png);
                        return worker.Run(temp);
                    });
                if (!IsDisposed && !busy && !reviewing && live.Checked && generation == liveGeneration && worker == classifier) {
                    var top = prediction.candidates[0];
                    status.Text = "实时 · " + top.name + "  " + top.score.ToString("P0") +
                        (top.score < 0.65 || top.score - prediction.candidates[1].score < 0.15 ? "（待确认）" : "");
                }
                }
            } catch (Exception ex) { if (!IsDisposed && generation == liveGeneration) { live.Checked = false; status.Text = ex.Message; } }
            finally { File.Delete(temp); nextLive = DateTime.UtcNow.AddSeconds(1); liveRunning = false; }
        }
    }

    private async Task CapturePhoto()
    {
        byte[] frame = camera == null ? null : camera.FreshFrame();
        if (frame == null) throw new InvalidOperationException("请等摄像头恢复画面后再拍摄。");
        busy = true;
        SetPrediction(null);
        string temp = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(temp, frame);
            using (var stream = new MemoryStream(frame))
            using (var image = Image.FromStream(stream))
                ReplacePreview(Adjust(image));
            frozenFrame = frame; openedPhoto = null; liveGeneration++;
            inputPath = OriginalColors ? Records.Snapshot(temp) : SnapshotPreview();
            var selectedDevice = cameras.SelectedItem as CameraDevice;
            sampleFields.SetDevice(selectedDevice == null ? "" : selectedDevice.Name + " | " + selectedDevice.Id);
            reviewing = true;
            cameraStatus.Text = "摄像头已连接 · 正在查看已拍照片";
            quality.Text = Records.Quality(preview.Image);
            await Recognize(inputPath, "摄像头拍照");
        }
        finally
        {
            busy = false;
            File.Delete(temp);
        }
    }

    private async Task Recognize(string path, string description)
    {
        status.Text = "正在识别…";
        currentRecord = null;
        DeleteDiagnostic();
        string png = Path.GetTempFileName();
        try
        {
            Prediction prediction = await Task.Run(() => classifier.Run(path, png));
            diagnosticImage = png;
            inputDescription = description;
            SetPrediction(prediction);
            currentRecord = Records.Save(path, prediction);
        }
        catch (Exception ex)
        {
            inputDescription = description;
            lastPrediction = new Prediction { error = ex.Message };
            status.Text = "识别失败：" + ex.Message;
            if (new FileInfo(png).Length > 0) diagnosticImage = png;
            export.Enabled = true;
            throw;
        }
        finally
        {
            if (diagnosticImage != png) File.Delete(png);
        }
    }

    private void SetPrediction(Prediction prediction)
    {
        lastPrediction = prediction;
        export.Enabled = true;
        while (candidates.Controls.Count > 0) candidates.Controls[0].Dispose();
        ingredient.Text = "";
        confirmed.Text = "";
        if (prediction == null) { inputPath = null; currentRecord = null; quality.Text = ""; DeleteDiagnostic(); return; }
        var top = prediction.candidates[0];
        bool uncertain = top.score < 0.65 || top.score - prediction.candidates[1].score < 0.15;
        status.Text = uncertain ? "无法确认。请选择候选食材、手动填写，或重拍。" : "识别建议：" + top.name;
        ingredient.Items.Clear();
        ingredient.Items.AddRange(prediction.classes);
        if (!uncertain) ingredient.Text = top.name;
        foreach (var result in prediction.candidates)
            candidates.Controls.Add(Button(result.name + "  " + result.score.ToString("P0"), () => {
                ingredient.Text = result.name;
                return Task.FromResult(0);
            }));
        Appearance.Apply(candidates, usageTips);
    }

    private async Task SelfTest()
    {
        busy = true;
        checkResult.Text = "正在检查模型…";
        var report = new System.Text.StringBuilder();
        try
        {
            try {
                var result = await Task.Run(() => classifier.Run(null));
                report.AppendLine("✓ 模型加载及测试推理成功，支持 " + result.classes.Length + " 类食材。");
            } catch (Exception ex) { report.AppendLine("× 模型检查失败：" + ex.Message); }
            checkResult.Text = report.ToString() + "正在检查摄像头取流…";
            try {
                var devices = Cameras.List();
                report.AppendLine("检测到摄像头：" + devices.Count + " 个");
                if (devices.Count > 0) {
                    CameraStream probe = camera;
                    bool owns = probe == null;
                    try {
                        if (owns) {
                            string folder = await Task.Run(() => RuntimeFiles.Prepare("OvenDesk.DesktopAI.zip", "ai"));
                            var device = cameras.SelectedItem as CameraDevice ?? devices[0];
                            probe = new CameraStream(folder, device.Index);
                        }
                        var wait = Stopwatch.StartNew();
                        while (probe.FreshFrame() == null && wait.ElapsedMilliseconds < 8000) await Task.Delay(100);
                        report.AppendLine(probe.FreshFrame() == null
                            ? "× 摄像头未返回画面，请检查权限、连接或其他程序占用。"
                            : "✓ 摄像头已返回画面。");
                    } finally { if (owns && probe != null) probe.Dispose(); }
                } else report.AppendLine("× 请连接摄像头后再检查。");
            } catch (Exception ex) { report.AppendLine("× 摄像头检查失败：" + ex.Message); }
            checkResult.Text = report.ToString();
            MessageBox.Show(this, report.ToString(), "检查结果");
        }
        finally { busy = false; }
    }

    private Task Export()
    {
        using (var dialog = new SaveFileDialog { Filter = "识别诊断压缩包|*.zip", FileName = "OvenDesk-diagnostics.zip" })
        {
            if (dialog.ShowDialog(this) != DialogResult.OK) return Task.FromResult(0);
            string staged = dialog.FileName + "." + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                using (var archive = System.IO.Compression.ZipFile.Open(staged, System.IO.Compression.ZipArchiveMode.Create))
                {
                    if (diagnosticImage != null)
                        System.IO.Compression.ZipFileExtensions.CreateEntryFromFile(archive, diagnosticImage, "input.png");
                    using (var writer = new StreamWriter(archive.CreateEntry("result.json").Open()))
                        writer.Write(new JavaScriptSerializer().Serialize(new { source = inputDescription, result = lastPrediction, check = checkResult.Text, camera = cameraStatus.Text, status = status.Text }));
                }
                if (File.Exists(dialog.FileName)) File.Replace(staged, dialog.FileName, null);
                else File.Move(staged, dialog.FileName);
                status.Text = "问题报告已保存：" + dialog.FileName;
                if (MessageBox.Show(this, status.Text + "\n打开所在文件夹？", "保存完成", MessageBoxButtons.YesNo) == DialogResult.Yes)
                    Process.Start("explorer.exe", "/select," + RuntimeFiles.Quote(dialog.FileName));
            }
            finally { if (File.Exists(staged)) File.Delete(staged); }
        }
        return Task.FromResult(0);
    }

    private void DeleteDiagnostic()
    {
        if (diagnosticImage == null) return;
        File.Delete(diagnosticImage);
        diagnosticImage = null;
    }
}






