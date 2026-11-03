# GitHub Pages 发布

仓库根目录 docs 是专用公开站点目录，仅包含网页、ONNX模型与类别配置；内部工作文档在 oven-app/docs，不在发布目录。

## 首次启用（需仓库所有者）

打开 https://github.com/bao333-prog/-/settings/pages 。
Build and deployment → Source 选择 Deploy from a branch，分支选择 main，目录选择 /docs，保存。
成功部署后默认网址为 https://bao333-prog.github.io/-/ 。这是预期地址，启用并构建成功之前不能作为已上线网址。

当前协作账号有 push 权限，但 admin=false、maintain=false。创建Pages接口返回404，仓库 has_pages=false，必须由所有者完成首次启用。

## 技术说明

GitHub Pages 不运行 Flask/Python，公开版使用 ONNX Runtime Web 1.22.0（CDN）在浏览器内推理。
模型约6.1MB，首次使用需下载运行时及模型；图片不上传。发布用模型可由访客下载，这是客户端推理的必要条件。
保留白底、等比例补边、RGB归一化与类别顺序。浏览器 Canvas 与 Pillow 的缩放插值可能存在差异，不能直接沿用 Python 验证集成绩作为网页版成绩。
同一输入张量导出校验：PyTorch 与 ONNX logits 最大误差 5.4836e-6，通过 rtol/atol=1e-4 检查。此检查不是实际食材准确率评测。

## 更新

在 oven-app 目录执行：

```
python reproduce/export_web.py --model models/model_C_v2.pt --output ../docs
python reproduce/build_pages.py
```

将源码和生成的 docs 一起提交 main。后续 main 更新会在 Pages 启用后触发发布。
导出依赖：onnx==1.23.2、onnxruntime==1.31.1、onnxscript==0.7.2；基础运行依赖见 requirements-validated.txt。
