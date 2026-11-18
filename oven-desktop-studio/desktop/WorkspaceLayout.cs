using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

internal static class WorkspaceLayout
{
    private static TableLayoutPanel Grid(int columns)
    {
        var grid = new TableLayoutPanel { ColumnCount = columns, AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink,
            Dock = DockStyle.Top, Margin = Padding.Empty, Padding = new Padding(4) };
        for (int i = 0; i < columns; i++) grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / columns));
        return grid;
    }

    private static void Field(TableLayoutPanel grid, string caption, Control input, int column, int row)
    {
        grid.Controls.Add(new Label { Text = caption, AutoSize = true, Anchor = AnchorStyles.Left }, column, row);
        input.Dock = DockStyle.Fill;
        input.AccessibleName = caption;
        grid.Controls.Add(input, column, row + 1);
    }

    internal static void Arrange(Form form, TableLayoutPanel old, FlowLayoutPanel commands,
        PictureBox preview, Control status, Control candidates, FlowLayoutPanel fields,
        Control confirmed, Control camera, Control quality, Control checks, Control samples,
        Func<string, Func<System.Threading.Tasks.Task>, Button> makeButton)
    {
        var root = Grid(2);
        root.BackColor = Color.FromArgb(248, 238, 225); root.Dock = DockStyle.Fill; root.AutoSize = false; root.Padding = new Padding(18);
        root.ColumnStyles[0].Width = 57; root.ColumnStyles[1].Width = 43;
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        var heading = new Label { Text = "OvenDesk  /  食材工作台", AutoSize = true, Font = new Font(form.Font.FontFamily, 16, FontStyle.Bold) };
        root.Controls.Add(heading, 0, 0); root.SetColumnSpan(heading, 2);
        var left = Grid(1); left.AutoSize = false; left.Dock = DockStyle.Fill; left.Margin = new Padding(0, 0, 12, 0);
        left.BackColor = Color.FromArgb(255, 250, 243);
        left.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        for (int i = 0; i < 3; i++) left.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        preview.Paint += (sender, args) => {
            if (preview.Image != null || preview.Width < 100 || preview.Height < 100) return;
            var g = args.Graphics;
            var bounds = preview.ClientRectangle;
            using (var wash = new System.Drawing.Drawing2D.LinearGradientBrush(bounds,
                Color.FromArgb(253, 235, 213), Color.FromArgb(231, 218, 239), 38f)) g.FillRectangle(wash, bounds);
            int x = bounds.Width / 2, y = bounds.Height / 2 - 20;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            using (var disc = new SolidBrush(Color.FromArgb(175, Color.White)))
            using (var ink = new Pen(Color.FromArgb(119, 74, 65), 2.4f)) {
                g.FillEllipse(disc, x-43, y-43, 86, 86);
                g.DrawRectangle(ink, x-22, y-19, 44, 38);
                g.DrawRectangle(ink, x-16, y-4, 32, 17);
                g.DrawLine(ink, x-16, y-11, x-5, y-11);
                g.DrawEllipse(ink, x+10, y-13, 4, 4);
            }
            TextRenderer.DrawText(g, "将食材照片拖到这里", preview.Font,
                new Rectangle(0,y+50,bounds.Width,32), Color.FromArgb(83,62,66),
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
        };
        left.Controls.Add(preview, 0, 0);
        left.Controls.Add(camera, 0, 1);
        var device = Grid(2);
        var actions = Grid(4);
        Func<string, Control> take = name => commands.Controls.Cast<Control>().First(c => c.Text == name);
        var cameras = commands.Controls.OfType<ComboBox>().Single();
        cameras.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        device.Controls.Add(cameras, 0, 0);
        device.Controls.Add(take("打开 / 关闭摄像头"), 1, 0);
        int col = 0;
        foreach (var name in new[] { "打开图片", "拍照并定格", "识别这张", "重新拍摄" })
            actions.Controls.Add(take(name), col++, 0);
        left.Controls.Add(device, 0, 2); left.Controls.Add(actions, 0, 3); root.Controls.Add(left, 0, 1);

        var tabs = new TabControl { Dock = DockStyle.Fill, Padding = new Point(13, 7) };
        var info = new TabPage("照片信息") { AutoScroll = true, BackColor = Color.FromArgb(255, 251, 246) };
        var manage = new TabPage("采集与模型") { AutoScroll = true, BackColor = Color.FromArgb(255, 251, 246) };
        tabs.TabPages.Add(info); tabs.TabPages.Add(manage); root.Controls.Add(tabs, 1, 1);
        status.Padding = new Padding(9, 5, 9, 5); status.BackColor = Color.FromArgb(255, 240, 219);
        var body = Grid(1); body.Padding = new Padding(10);
        info.Controls.Add(body);
        body.Controls.Add(status, 0, 0); body.Controls.Add(candidates, 0, 1); body.Controls.Add(quality, 0, 2);
        var entries = Grid(2); entries.BackColor = Color.FromArgb(248, 246, 252); entries.Padding = new Padding(10);
        var input = fields.Controls.Cast<Control>().Where(c => c is ComboBox || c is NumericUpDown).ToArray();
        Field(entries, "食材", input[0], 0, 0); Field(entries, "食材状态", input[1], 1, 0);
        Field(entries, "重量（克，选填）", input[2], 0, 2);
        var meta = samples.Controls.Cast<Control>().Where(c => c is ComboBox || c is TextBox).ToArray();
        Field(entries, "烤箱层位", meta[0], 1, 2); Field(entries, "锡纸", meta[1], 0, 4);
        Field(entries, "拍摄批次", meta[2], 1, 4); Field(entries, "拍摄设备", meta[3], 0, 6);
        entries.SetColumnSpan(meta[3], 2);
        body.Controls.Add(entries, 0, 3);
        var submit = Grid(2); submit.ColumnStyles[0].Width = 68; submit.ColumnStyles[1].Width = 32;
        col = 0; foreach (var button in fields.Controls.OfType<Button>().ToArray()) submit.Controls.Add(button, col++, 0);
        body.Controls.Add(submit, 0, 4); body.Controls.Add(confirmed, 0, 5);
        var tools = Grid(2); tools.Padding = new Padding(10);
        manage.Controls.Add(tools);
        int index = 0;
        foreach (var name in new[] { "普通拍照", "拍照并训练", "选择数据集并训练", "验证集" })
        { tools.Controls.Add(take(name), index % 2, index / 2); index++; }
        foreach (var button in commands.Controls.Cast<Control>().ToArray())
        { tools.Controls.Add(button, index % 2, index / 2); index++; }
        tools.Controls.Add(checks, 0, (index + 1) / 2); tools.SetColumnSpan(checks, 2);
        var footer = Grid(4); footer.BackColor = Color.FromArgb(239, 225, 208); footer.ColumnStyles.Clear();
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); footer.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize)); footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        footer.Controls.Add(new Label { Text = "照片位置", AutoSize = true, Anchor = AnchorStyles.Left }, 0, 0);
        var path = new TextBox { Text = Records.PhotoFolder, ReadOnly = true, Anchor = AnchorStyles.Left | AnchorStyles.Right };
        footer.Controls.Add(path, 1, 0);
        form.Activated += (s,e) => path.Text = Records.PhotoFolder;
        footer.Controls.Add(makeButton("更改保存位置", () => { Records.ChooseFolder(form); path.Text = Records.PhotoFolder; return System.Threading.Tasks.Task.FromResult(0); }), 2, 0);
        footer.Controls.Add(makeButton("打开照片文件夹", () => { Records.OpenFolder(); return System.Threading.Tasks.Task.FromResult(0); }), 3, 0);
        root.Controls.Add(footer, 0, 2); root.SetColumnSpan(footer, 2);
        // SampleFields remains the owner of its values even after the inputs are reparented.
        form.Controls.Remove(old); old.Dispose(); form.Controls.Add(root);
    }
}


