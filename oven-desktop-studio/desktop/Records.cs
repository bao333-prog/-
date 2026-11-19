using System;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal sealed class SavedRecord
{
    public string date { get; set; }
    public string label { get; set; }
    public string layer { get; set; }
    public string foil { get; set; }
    public string batch { get; set; }
    public string device { get; set; }
    public Prediction prediction { get; set; }
}

internal static class Records
{
    internal static string PhotoFolder { get { return Root; } }
    internal static void OpenFolder()
    {
        Directory.CreateDirectory(Root);
        System.Diagnostics.Process.Start("explorer.exe", RuntimeFiles.Quote(Root));
    }
    private static string Root {
        get {
            string settings = Path.Combine(RuntimeFiles.Root, "photo-folder.txt");
            return File.Exists(settings) ? File.ReadAllText(settings).Trim() : Path.Combine(RuntimeFiles.Root, "records");
        }
    }

    internal static void ChooseFolder(Form owner)
    {
        using (var picker = new FolderBrowserDialog { Description = "选择以后保存照片和标签的位置", SelectedPath = Root }) {
            if (picker.ShowDialog(owner) != DialogResult.OK) return;
            Directory.CreateDirectory(RuntimeFiles.Root);
            File.WriteAllText(Path.Combine(RuntimeFiles.Root, "photo-folder.txt"), picker.SelectedPath);
            MessageBox.Show(owner, "以后照片保存到：" + picker.SelectedPath + "\n已有照片保留在原位置。");
        }
    }
    private static readonly JavaScriptSerializer Json = new JavaScriptSerializer();

    internal static string Snapshot(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        string hash;
        using (var sha = SHA256.Create())
            hash = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "");
        string folder = Path.Combine(Root, hash);
        Directory.CreateDirectory(folder);
        string target = Path.Combine(folder, "original.image");
        if (!File.Exists(target))
        {
            string staging = target + Guid.NewGuid().ToString("N");
            try {
                File.WriteAllBytes(staging, bytes);
                try { File.Move(staging, target); }
                catch (IOException) { if (!File.Exists(target)) throw; }
            } finally { if (File.Exists(staging)) File.Delete(staging); }
        }
        return target;
    }

    internal static string Save(string source, Prediction prediction)
    {
        string image = Snapshot(source);
        string file = Path.Combine(Path.GetDirectoryName(image), "record.json");
        SavedRecord record = File.Exists(file) ? Json.Deserialize<SavedRecord>(File.ReadAllText(file)) : new SavedRecord();
        record.date = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        record.prediction = prediction;
        Write(file, record);
        return file;
    }

    private static void Write(string file, SavedRecord record)
    {
        string staging = file + Guid.NewGuid().ToString("N");
        try {
            File.WriteAllText(staging, Json.Serialize(record));
            if (File.Exists(file)) File.Replace(staging, file, null);
            else File.Move(staging, file);
        } finally { if (File.Exists(staging)) File.Delete(staging); }
    }

    internal static void Confirm(string file, string label, SampleFields fields)
    {
        SavedRecord record = Json.Deserialize<SavedRecord>(File.ReadAllText(file));
        record.label = label;
        fields.Apply(record);
        Write(file, record);
    }

    internal static string Quality(Image image)
    {
        using (var small = new Bitmap(image, 96, 96))
        {
            double sum = 0, edges = 0;
            for (int y = 1; y < 96; y++)
                for (int x = 1; x < 96; x++)
                {
                    Color c = small.GetPixel(x, y);
                    Color previous = small.GetPixel(x - 1, y);
                    sum += (c.R + c.G + c.B) / 3.0;
                    edges += Math.Abs(c.R - previous.R) + Math.Abs(c.G - previous.G) + Math.Abs(c.B - previous.B);
                }
            double brightness = sum / (95 * 95);
            if (brightness < 45) return "画面偏暗，建议增加光线后重拍。";
            if (brightness > 235) return "画面偏亮，建议减少反光后重拍。";
            if (edges / (95 * 95 * 3) < 3) return "画面细节较少，可能失焦或食材较远，请靠近并保持稳定。";
            return "画面亮度和细节检查通过，请确保食材清楚可见。";
        }
    }

    internal static void Show(Form owner)
    {
        Directory.CreateDirectory(Root);
        using (var dialog = new Form { Text = "识别记录与训练素材", Width = 920, Height = 580, StartPosition = FormStartPosition.CenterParent })
        {
            var list = new ListView { Dock = DockStyle.Fill, View = View.Details, FullRowSelect = true, MultiSelect = false };
            list.Columns.Add("时间", 160);
            list.Columns.Add("识别候选", 160);
            list.Columns.Add("确认标签", 160);
            var photo = new PictureBox { Dock = DockStyle.Right, Width = 300, SizeMode = PictureBoxSizeMode.Zoom };
            foreach (string file in Directory.GetFiles(Root, "record.json", SearchOption.AllDirectories).OrderByDescending(File.GetLastWriteTimeUtc))
            {
                try {
                    var record = Json.Deserialize<SavedRecord>(File.ReadAllText(file));
                    var item = new ListViewItem(new[] { record.date, record.prediction.candidates[0].name, record.label ?? "未确认" });
                    item.Tag = file;
                    list.Items.Add(item);
                } catch (Exception ex) { if (!(ex is IOException || ex is ArgumentException)) throw; }
            }
            list.SelectedIndexChanged += (sender, args) => {
                if (list.SelectedItems.Count == 0) return;
                var old = photo.Image;
                try { photo.Image = Desk.ReadImage(Path.Combine(Path.GetDirectoryName((string)list.SelectedItems[0].Tag), "original.image")); }
                catch { photo.Image = null; }
                if (old != null) old.Dispose();
            };
            var buttons = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 160, AutoScroll = true };
            var metadata = new SampleFields { Dock = DockStyle.None, Width = 850 };
            buttons.Controls.Add(metadata);
            var label = new TextBox { Width = 140 };
            list.SelectedIndexChanged += (sender, args) => {
                if (list.SelectedItems.Count == 0) return;
                try {
                    var record = Json.Deserialize<SavedRecord>(File.ReadAllText((string)list.SelectedItems[0].Tag));
                    label.Text = record.label ?? "";
                    metadata.LoadRecord(record);
                } catch (Exception ex) { MessageBox.Show(dialog, ex.Message); }
            };
            var edit = new Button { Text = "修改确认标签", AutoSize = true };
            edit.Click += (sender, args) => {
                if (list.SelectedItems.Count == 0 || String.IsNullOrWhiteSpace(label.Text)) return;
                try {
                    Confirm((string)list.SelectedItems[0].Tag, label.Text.Trim(), metadata);
                    list.SelectedItems[0].SubItems[2].Text = label.Text.Trim();
                } catch (Exception ex) { MessageBox.Show(dialog, ex.Message); }
            };
            var export = new Button { Text = "导出已确认训练素材", AutoSize = true };
            var openFolder = new Button { Text = "打开照片文件夹", AutoSize = true };
            openFolder.Click += (sender, args) => OpenFolder();
            buttons.Controls.Add(openFolder);
            export.Click += (sender, args) => {
                using (var save = new SaveFileDialog { Filter = "训练素材|*.zip", FileName = "OvenDesk-training.zip" })
                {
                    if (save.ShowDialog(dialog) != DialogResult.OK) return;
                    string staging = save.FileName + Guid.NewGuid().ToString("N");
                    try {
                        int count = 0;
                        using (var zip = ZipFile.Open(staging, ZipArchiveMode.Create))
                        using (var manifest = new StreamWriter(zip.CreateEntry("labels.jsonl").Open()))
                        {
                            foreach (string file in Directory.GetFiles(Root, "record.json", SearchOption.AllDirectories))
                            {
                                var record = Json.Deserialize<SavedRecord>(File.ReadAllText(file));
                                if (String.IsNullOrWhiteSpace(record.label)) continue;
                                string id = Path.GetFileName(Path.GetDirectoryName(file));
                                string name = "images/" + id + ".image";
                                // Entries must be written one at a time, after the manifest is buffered.
                                count++;
                                manifest.WriteLine(Json.Serialize(new { image = name, label = record.label, layer = record.layer ?? "未知", foil = record.foil ?? "未知", batch = record.batch, device = record.device, group = String.IsNullOrWhiteSpace(record.batch) ? null : record.batch, imageSha256 = id, model = record.prediction.modelSha256 }));
                            }
                        }
                        using (var zip = ZipFile.Open(staging, ZipArchiveMode.Update))
                            foreach (string file in Directory.GetFiles(Root, "record.json", SearchOption.AllDirectories))
                            {
                                var record = Json.Deserialize<SavedRecord>(File.ReadAllText(file));
                                if (String.IsNullOrWhiteSpace(record.label)) continue;
                                string folder = Path.GetDirectoryName(file);
                                zip.CreateEntryFromFile(Path.Combine(folder, "original.image"), "images/" + Path.GetFileName(folder) + ".image");
                            }
                        if (File.Exists(save.FileName)) File.Replace(staging, save.FileName, null);
                        else File.Move(staging, save.FileName);
                        MessageBox.Show(dialog, "已导出 " + count + " 张确认照片。");
                    } catch (Exception ex) { MessageBox.Show(dialog, ex.Message); }
                    finally { if (File.Exists(staging)) File.Delete(staging); }
                }
            };
            buttons.Controls.Add(label);
            buttons.Controls.Add(edit);
            buttons.Controls.Add(export);
            buttons.Controls.Add(new Label { AutoSize = true, Text = "确认照片会保存为训练素材。继续训练需要原训练工程和权重。" });
            dialog.Controls.Add(list);
            dialog.Controls.Add(photo);
            dialog.Controls.Add(buttons);
            Appearance.Attach(dialog);
            dialog.ShowDialog(owner);
            if (photo.Image != null) photo.Image.Dispose();
        }
    }
}
