# 采样优化 A：网站发布记录

网页及本地后端统一切换到 `model_optimized_A.pt`，显示“十四类采样优化A”，版本 `optimized_A_001bf328`。与本地对比版 ONNX/PT 字节完全相同，仅移除名称中的对比标记并采用正式资源路径。

## 模型与验证

- 来源：用户提供《采样优化A_第二批验证_代码与权重.zip》中的 models/model_optimized_A.pt；未选择 records/official_initialization.pt。训练成果归提供模型的组员，本次提交为网站接入与发布。
- PT SHA256：`001bf3284aa37bd00172eba198e1f081140adce7eed741c95ffab2dfb5213837`。
- ONNX SHA256：`66885e6775cdf2b1977583c4c8bf174d725c46a85094179fd85620964bde6201`。
- MobileNetV3-Small，224×224、RGB/NCHW，白底 letterbox 和权重指定的 ImageNet 归一化；网页配置的 mean/std 按 Python 实际使用的 float32 保存。输出14类 logits 后进行 softmax。
- 类别按权重顺序：南瓜、吐司、牛排、玉米、西兰花、鸡翅、芦笋、培根、三文鱼、整鸡、土豆、香肠、薯条、可颂。
- 原包 prepare_image/image_tensor 与网站项目相应 Python 函数 AST 相同；导出随机张量最大 logits 差 3.34e-6。15张现有图片 PyTorch/ONNX 最高候选一致15/15，最大 logits 差约1.77e-5，见[逐图转换校验](采样优化A转换校验.json)。包含重复图片，不是准确率评测。
- 新默认模型下4项 Python 真实接口测试和25项 Node 模型、预处理与摄像头测试通过。自动浏览器上传界面未新增实测记录；本次沿用本地对比版相同权重及前端。
- 模型包说明 A/B 评测均属开发对照，整鸡缺实拍评测，不能据此声称独立实拍准确率。网页仍要求人工确认。

## 复现与回滚

在仓库根目录的项目依赖环境，设置 PYTHONPATH 为 oven-app;oven-app/src 后，可运行：

```powershell
python oven-app/reproduce/export_web.py --model oven-app/models/model_optimized_A.pt --output docs --id optimized_A --name "十四类采样优化A" --version optimized_A_001bf328
python oven-app/reproduce/verify_web_export.py --model oven-app/models/model_optimized_A.pt --web-dir docs --images "你的原图目录"
```

如重新导出，mean/std应与Python实际使用的float32常量对齐。本地后端默认使用新PT；若已有OVEN_MODEL环境变量，它仍会覆盖默认路径。

上一版南瓜可颂 PT 继续保留；网页 ONNX、model.json 和后端选择配置在提交 `ddb0da14cf3da3cba6c76655b7b5ac4b2276d4ba` 中可恢复。旧ONNX哈希为 `366c38043e1c4d520d734f949dae64832cce085778204726737796a7d1bd2156`。

“箱见-本地模型对比-采样优化A”目录中的原版本备份和恢复原模型入口仍保留，完整性再次核验通过。该入口只恢复本地对比网页。线上回滚需回退本次发布，或同时恢复旧提交的 docs/model.onnx、docs/model.json、oven-app/backend/app.py，并同步启动说明后重新发布。

保留现有图片上传、USB摄像头、预处理、人工确认与模型哈希缓存参数。未改团队模型归档，也未发布训练原图及本机路径清单。由Codex辅助实现，网站提交作者lichenyi-home。
