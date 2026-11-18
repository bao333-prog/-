using System.Windows.Forms;

internal sealed class SampleFields : FlowLayoutPanel
{
    private readonly ComboBox layer = Choice("未知", "上层", "中层", "下层");
    private readonly ComboBox foil = Choice("未知", "有锡纸", "无锡纸");
    private readonly TextBox batch = new TextBox { Width = 160 };
    private readonly TextBox device = new TextBox { Width = 190 };

    internal SampleFields()
    {
        AutoSize = true;
        Dock = DockStyle.Fill;
        Add("层位", layer);
        Add("锡纸", foil);
        Add("拍摄批次", batch);
        Add("拍摄设备", device);
    }

    private static ComboBox Choice(params string[] values)
    {
        var box = new ComboBox { Width = 85, DropDownStyle = ComboBoxStyle.DropDownList };
        box.Items.AddRange(values);
        box.SelectedIndex = 0;
        return box;
    }

    private void Add(string caption, Control control)
    {
        Controls.Add(new Label { Text = caption, AutoSize = true });
        Controls.Add(control);
    }

    internal void LoadRecord(SavedRecord record)
    {
        layer.SelectedItem = record.layer ?? "未知";
        foil.SelectedItem = record.foil ?? "未知";
        if (layer.SelectedIndex < 0) layer.SelectedIndex = 0;
        if (foil.SelectedIndex < 0) foil.SelectedIndex = 0;
        batch.Text = record.batch ?? "";
        device.Text = record.device ?? "";
    }

    internal void SetDevice(string name) { device.Text = name; }

    internal void Apply(SavedRecord record)
    {
        record.layer = layer.Text;
        record.foil = foil.Text;
        record.batch = batch.Text.Trim();
        record.device = device.Text.Trim();
    }
}
