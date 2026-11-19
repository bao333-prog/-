using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

internal static class Appearance
{
    private static readonly Dictionary<string, string> Help = new Dictionary<string, string> {
        { "使用上次数据训练新版本", "首次选择训练材料。以后复用这些文件，训练完成后自动检查并添加到模型列表。" },
        { "重新选择训练材料", "更换素材包、批次划分、起点权重或输出目录，并开始训练更新。" },
        { "开始连拍", "按设定间隔拍摄，到计划张数自动停止。再次点击可暂停。" },
        { "更改保存位置", "选择保存实拍照片和标签的总文件夹。" },
        { "选择保存文件夹", "更改保存位置，下面会显示当前路径。" },
        { "删除所选模型", "从模型库移除选中版本。要删除当前版本，请先切换模型。" },
        { "选数据集并训练", "选择按食材分类的照片总文件夹，自动整理并训练新模型。" },
        { "导入我的数据集", "选择按食材分文件夹的训练照片和验证照片，自动生成素材包及批次文件。" },
        { "打开实拍照片文件夹", "打开本程序保存照片的位置。每张照片按文件指纹单独存放。" },
        { "打开照片文件夹", "打开本程序保存的原图和标签文件夹。" },
        { "开始训练", "依次选择素材包、批次文件、起点权重和输出位置。" },
        { "打开图片", "选择食材照片，程序会显示照片并开始识别。也可以把照片拖入画面。" },
        { "打开 / 关闭摄像头", "打开摄像头预览；再次点击即可关闭。" },
        { "识别并拍照", "拍下当前画面并识别。再次点击可拍下一张。" },
        { "识别这张", "让当前模型识别这张照片。看错时可以修改食材名称。" },
        { "检查程序", "检查模型能否运行、摄像头能否返回画面，并显示每一步的结果。" },
        { "训练与升级", "用已确认的照片继续训练，生成新模型和对比报告。需要训练环境、素材包及批次划分文件。" },
        { "模型管理 / 切换版本", "导入新版 ONNX 模型包并切换。只有内置模型时，需先导入新版本才能比较或切换。" },
        { "同图对照", "先选择照片，再比较当前模型与内置模型。网页版启动后，还能比较它的返回结果。" },
        { "识别记录 / 训练素材", "查看保存的照片、修改正确标签，导出已确认的训练素材。" },
        { "保存问题报告", "把识别结果、错误和处理后的图片保存为 ZIP，方便排查。保存后可打开所在文件夹。" },
        { "确认并保存训练素材", "请先核对食材、层位、锡纸和批次。保存照片与正确答案，供后续训练使用。" },
        { "清除", "清除当前照片的识别结果，方便开始下一次识别。" },
        { "导入模型 ZIP", "选择包含 food.onnx 和 food.json 的模型包，程序会先检查能否运行。" },
        { "检查并使用所选版本", "检查选中模型，确认后用于后续识别。选择内置模型可以切回原版。" },
        { "对照网页版", "先启动这台电脑上的网页版服务，再发送同一张照片查看结果。" },
        { "导入原始模型结果 JSON", "导入训练工程生成的预测结果。结果中的图片指纹必须与当前照片一致。" },
        { "保存对照报告", "保存本次对照的原图、结果和输入处理信息，便于分析差异。" },
        { "修改确认标签", "先选择历史照片，再修改食材、层位、锡纸、批次或设备，点击保存。" },
        { "导出已确认训练素材", "把已确认照片及标签导出为 ZIP，供训练使用。" },
        { "打开输出文件夹", "打开这次训练的报告、权重和新模型包所在文件夹。" }
    };

    internal static void Apply(Control root, ToolTip tips)
    {

        foreach (Control control in root.Controls)
        {
            var button = control as Button;
            if (button != null)
            {
                button.FlatStyle = FlatStyle.Flat;
                button.FlatAppearance.BorderSize = 0;
                string name = button.Text;
                Color fill = Color.FromArgb(240, 242, 245), ink = Color.FromArgb(48, 55, 67);
                if (name == "识别并拍照") { fill = Color.FromArgb(166, 66, 33); ink = Color.White; }
                else if (name == "确认并保存训练素材") { fill = Color.FromArgb(30, 107, 93); ink = Color.White; }
                else if (name.Contains("训练") || name.Contains("模型") || name == "验证集") {
                    fill = Color.FromArgb(236, 231, 250); ink = Color.FromArgb(77, 59, 129);
                } else if (name.Contains("拍") || name.Contains("摄像头")) {
                    fill = Color.FromArgb(224, 241, 236); ink = Color.FromArgb(27, 89, 78);
                }
                button.UseVisualStyleBackColor = false;
                button.BackColor = SystemInformation.HighContrast ? SystemColors.Control : fill;
                button.ForeColor = SystemInformation.HighContrast ? SystemColors.ControlText : ink;
                button.Padding = new Padding(8, 4, 8, 4);
                button.Margin = new Padding(4, 5, 4, 5);
                button.Cursor = Cursors.Hand;
                button.FlatAppearance.MouseOverBackColor = ControlPaint.Dark(button.BackColor, 0.08f);
                button.SizeChanged += (sender, args) => Round(button);
                Round(button);
                string text;
                tips.SetToolTip(button, Help.TryGetValue(button.Text, out text) ? text : "点击选择这项结果。");
            }
            else if (control is FlowLayoutPanel)
            {
                control.Padding = new Padding(4);
                control.BackColor = Color.FromArgb(248, 238, 225);
            }
            else if (control is TextBox || control is ComboBox || control is NumericUpDown)
            {
                control.Margin = new Padding(4, 2, 6, 6);
                control.BackColor = Color.White;
            }
            else if (control is Label)
                control.Margin = new Padding(7, 8, 7, 6);
            Apply(control, tips);

        }
    }

    private static void Round(Control control)
    {
        if (control.Width < 20 || control.Height < 20) return;
        using (var path = new GraphicsPath())
        {
            const int d = 16;
            path.AddArc(0, 0, d, d, 180, 90);
            path.AddArc(control.Width - d, 0, d, d, 270, 90);
            path.AddArc(control.Width - d, control.Height - d, d, d, 0, 90);
            path.AddArc(0, control.Height - d, d, d, 90, 90);
            path.CloseFigure();
            var old = control.Region;
            control.Region = new Region(path);
            if (old != null) old.Dispose();
        }
    }

    internal static ToolTip Attach(Form form)
    {
        form.BackColor = Color.FromArgb(248, 238, 225);
        form.Font = new Font("Microsoft YaHei UI", 10f);
        var tips = new ToolTip { InitialDelay = 450, ReshowDelay = 100, AutoPopDelay = 15000, ShowAlways = true };
        Apply(form, tips);
        form.FormClosed += (sender, args) => tips.Dispose();
        return tips;
    }
}




