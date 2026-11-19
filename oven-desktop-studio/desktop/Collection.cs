using System;
using System.IO;
using System.Drawing;
using System.Windows.Forms;
using System.Web.Script.Serialization;

internal static class Collection
{
    internal static void Show(Form owner, bool validation = false, bool training = false, CameraDevice preferred = null)
    {
        using (var form = new Form { Text = validation ? "采集验证照片" : training ? "拍照并训练" : "普通拍照", Width = 850, Height = 650, StartPosition = FormStartPosition.CenterParent })
        {
            var photo = new PictureBox { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom };
            var controls = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 240, AutoScroll = true };
            var label = new ComboBox { Width = 120, MaxDropDownItems = 14 };
            label.Items.AddRange(ModelLibrary.FoodNames());
            var count = new NumericUpDown { Minimum = 1, Maximum = 10000, Value = 30 };
            var status = new Label { AutoSize = true, Text = "填写正确食材名称，拍摄不同摆放和批次。" };
            var fields = new SampleFields { Dock = DockStyle.None, Width = 780 };
            var save = new Button { Text = "单拍一张", AutoSize = true };
            var train = new Button { Text = "用已确认照片训练新版本", AutoSize = true, Enabled = false };
            var folder = new Button { Text = "选择保存文件夹", AutoSize = true };
            var batch = new Button { Text = "开始新拍摄批次", AutoSize = true };
            string session = DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N").Substring(0, 6);
            int saved = 0, deviceIndex = 0;
            CameraStream camera = null;
            var timer = new Timer { Interval = 150 };
            controls.Controls.Add(new Label { Text = "食材名称", AutoSize = true });
            controls.Controls.Add(label);
            controls.Controls.Add(new Label { Text = "计划张数", AutoSize = true });
            controls.Controls.Add(count);
            var location = new TextBox { Width = 740, ReadOnly = true };
            Action showLocation = () => location.Text = Path.Combine(Records.PhotoFolder, validation ? "验证数据集" : training ? "已确认训练集" : "采集数据集");
            showLocation();
            var burst = new Button { Text = "开始连拍", AutoSize = true };
            var interval = new NumericUpDown { Minimum = 1, Maximum = 60, Value = 2, Width = 55 };
            var shots = new Timer();
            byte[] last = null;
            Action stop = () => { shots.Stop(); burst.Text = "开始连拍"; folder.Enabled = batch.Enabled = label.Enabled = count.Enabled = interval.Enabled = true; };
            Action reset = () => { saved = 0; train.Enabled = training && Directory.Exists(location.Text); session = Guid.NewGuid().ToString("N"); };
            label.TextChanged += (s,e) => reset();
            controls.Controls.Add(folder); controls.Controls.Add(batch);
            controls.Controls.Add(location); controls.Controls.Add(fields); controls.Controls.Add(burst); controls.Controls.Add(new Label {Text="间隔（秒）", AutoSize=true}); controls.Controls.Add(interval); controls.Controls.Add(save); controls.Controls.Add(train); controls.Controls.Add(status);
            form.Controls.Add(photo); form.Controls.Add(controls);
            folder.Click += (sender, args) => { string old = Records.PhotoFolder; Records.ChooseFolder(form); showLocation(); if (old != Records.PhotoFolder) reset(); };
            batch.Click += (sender, args) => { reset(); status.Text = "新批次已开始，请改变实物或拍摄场次。"; };
            save.Click += (sender, args) => {
                try {
                    string name = label.Text.Trim();
                    if (String.IsNullOrEmpty(name) || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || name == "." || name == "..")
                        throw new InvalidDataException("请填写有效食材名称。");
                    var frame = camera == null ? null : camera.FreshFrame();
                    if (frame == null) throw new IOException("摄像头尚未返回画面。");
                    if (shots.Enabled && Object.ReferenceEquals(frame, last)) return;
                    string collectionRoot = Path.Combine(Records.PhotoFolder, validation ? "验证数据集" : training ? "已确认训练集" : "采集数据集");
                    string target = Path.Combine(collectionRoot, name);
                    var record = new SavedRecord { label = name, date = DateTimeOffset.Now.ToString("o") };
                    fields.Apply(record);
                    if (training) {
                        using (var review = new Form { Text = "确认这张照片的食材", Width = 580, Height = 460, StartPosition = FormStartPosition.CenterParent })
                        using (var stream = new MemoryStream(frame))
                        using (var frozen = Image.FromStream(stream)) {
                            review.Controls.Add(new PictureBox { Dock = DockStyle.Fill, Image = frozen, SizeMode = PictureBoxSizeMode.Zoom });
                            review.Controls.Add(new Button { Text = "确认是“" + name + "”，加入训练集", Dock = DockStyle.Bottom, Height = 42, DialogResult = DialogResult.OK });
                            review.Controls.Add(new Button { Text = "放弃这张", Dock = DockStyle.Bottom, DialogResult = DialogResult.Cancel });
                            bool resume = shots.Enabled;
                            shots.Stop();
                            if (review.ShowDialog(form) != DialogResult.OK) { stop(); return; }
                            if (resume) shots.Start();
                        }
                    }
                    Directory.CreateDirectory(target);
                    CapturePair.Write(target, record, frame, String.IsNullOrWhiteSpace(record.batch) ? session : record.batch, deviceIndex, training);
                    last = frame; saved++;
                    if (shots.Enabled && saved >= count.Value) stop();
                    if (validation) {
                        Directory.CreateDirectory(RuntimeFiles.Root);
                        File.WriteAllText(Path.Combine(RuntimeFiles.Root, "validation-folder.txt"), collectionRoot);
                    }
                    train.Enabled = training;
                    status.Text = "已保存 " + saved + " / " + count.Value + " 张，位置：" + target;
                } catch (Exception ex) { stop(); MessageBox.Show(form, ex.Message); }
            };
            burst.Click += (s,e) => {
                if (shots.Enabled) { stop(); return; }
                if (saved >= count.Value) reset();
                shots.Interval = (int)interval.Value * 1000;
                folder.Enabled = batch.Enabled = label.Enabled = count.Enabled = interval.Enabled = false;
                burst.Text = "暂停连拍"; shots.Start(); save.PerformClick();
            };
            shots.Tick += (s,e) => save.PerformClick();
            train.Visible = training; train.Enabled = training && Directory.Exists(location.Text);
            count.ValueChanged += (sender, args) => train.Enabled = training && Directory.Exists(location.Text);
            train.Click += (sender, args) => {
                stop();
                Training.FromFolder(form, location.Text);
                if (Training.ModelAdded) form.Close();
            };
            timer.Tick += (sender, args) => {
                byte[] frame = camera == null ? null : camera.FreshFrame();
                if (frame == null) return;
                try {
                    using (var stream = new MemoryStream(frame))
                    using (var image = Image.FromStream(stream)) {
                        var previous = photo.Image; photo.Image = new Bitmap(image);
                        if (previous != null) previous.Dispose();
                    }
                } catch (ArgumentException) { }
            };
            form.Shown += async (sender, args) => {
                try {
                    var devices = await System.Threading.Tasks.Task.Run(() => Cameras.List());
                    if (devices.Count == 0) throw new IOException("未找到摄像头。");
                    string runtime = await System.Threading.Tasks.Task.Run(() => RuntimeFiles.Prepare("OvenDesk.DesktopAI.zip", "ai"));
                    if (form.IsDisposed) return;
                    var selected = preferred == null ? devices[0] : devices.Find(d => d.Id == preferred.Id);
                    if (selected == null) throw new IOException("所选摄像头已断开，请重新连接。");
                    deviceIndex = selected.Index; camera = new CameraStream(runtime, deviceIndex);
                    fields.SetDevice(selected.Name);
                    timer.Start();
                } catch (Exception ex) { status.Text = ex.Message; }
            };
            Appearance.Attach(form);
            try { form.ShowDialog(owner); }
            finally { shots.Dispose(); timer.Dispose(); if (camera != null) camera.Dispose(); if (photo.Image != null) photo.Image.Dispose(); }
        }
    }
}




