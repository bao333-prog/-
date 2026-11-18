using System;
using System.IO;
using System.Drawing;
using System.Security.Cryptography;
using System.Web.Script.Serialization;

internal static class CapturePair
{
    internal static void Write(string target, SavedRecord record, byte[] frame, string session, int deviceIndex, bool confirmed)
    {
        string[] names = "南瓜 吐司 牛排 玉米 西兰花 鸡翅 芦笋 培根 三文鱼 整鸡 土豆 香肠 薯条 可颂".Split(' ');
        string[] keys = "pumpkin toast steak corn broccoli chicken-wings asparagus bacon salmon whole-chicken potato sausage fries croissant".Split(' ');
        int index = Array.IndexOf(names, record.label);
        if (confirmed && index < 0) throw new InvalidDataException("当前模型没有这个食材类别：" + record.label);
        string foil = record.foil == "有锡纸" ? "have" : record.foil == "无锡纸" ? "no" : "unknown";
        string layer = record.layer == "上层" ? "high" : record.layer == "中层" ? "mid" : record.layer == "下层" ? "low" : "unknown";
        string prefix = (index < 0 ? record.label : keys[index]) + "-" + foil + "-" + layer;
        int seq = 1;
        foreach (string file in Directory.EnumerateFiles(target, prefix + "_*.jpg")) {
            int n;
            if (Int32.TryParse(Path.GetFileNameWithoutExtension(file).Substring(prefix.Length + 1), out n)) seq = Math.Max(seq, n + 1);
        }
        string stem = Path.Combine(target, prefix + "_" + seq.ToString("D4"));
        int width, height;
        using (var stream = new MemoryStream(frame))
        using (var img = Image.FromStream(stream)) { width = img.Width; height = img.Height; }
        string hash;
        using (var sha = SHA256.Create()) hash = BitConverter.ToString(sha.ComputeHash(frame)).Replace("-", "").ToLowerInvariant();
        string stamp = DateTimeOffset.Now.ToString("o");
        var data = new { schema = "het-oven-capture/2", image_file = Path.GetFileName(stem) + ".jpg", sample_prefix = prefix,
            photo_sequence = seq, created_host_local = stamp, width, height, format = "JPG", source_representation = "full_decoded_camera_frame",
            sensor_raw = false, resized = false, cropped = false, sha256 = hash, label_status = confirmed ? "confirmed" : "pending_manual_review",
            camera_context = new { camera_alias = record.device, session_id = session, physical_sample_id = prefix,
                candidate_class_key_unreviewed = (string)null, placement = (string)null, lighting = (string)null,
                device_index = deviceIndex, device_name = record.device, backend = (string)null, fourcc_reported = (string)null,
                driver_properties_last_observed = (object)null, human_label_confirmed = confirmed, driver_properties_at_capture = (object)null,
                captured_at_host_approx = record.date } };
        using (var jpg = new FileStream(stem + ".jpg", FileMode.CreateNew, FileAccess.Write)) {
            try {
                jpg.Write(frame, 0, frame.Length);
                using (var json = new FileStream(stem + ".json", FileMode.CreateNew, FileAccess.Write))
                using (var writer = new StreamWriter(json)) writer.Write(new JavaScriptSerializer().Serialize(data));
            } catch { jpg.Close(); File.Delete(stem + ".jpg"); throw; }
        }
    }
}
