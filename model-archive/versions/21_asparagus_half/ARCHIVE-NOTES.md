# 半截芦笋补充

归档日期：2026-11-28。本次只保存此前训练结果，没有重新训练或重测食品图片。

- 主权重：[model.pt](model.pt)，SHA256 `e2e6c3199fba10dc3ab52df6159f7043b9f05b529aa1009dc5ba1856d7102460`。
- 14类MobileNetV3-Small；输入224×224，预处理与类别顺序以权重及对应代码为准。
- 共训练23轮，选定第17轮（raw）。
- 初始化：official_imagenet_backbone_random_14_class_head。数据版本的继承不表示继承上一版模型参数。
- 共训练23轮，最佳第17轮；验证190/193（98.45%）；开发测试163/167（97.60%），芦笋28/28。
- 本目录食品测试是已公开复用的开发对照，历史部分还有连续照片及南瓜同源问题；具体限制见原报告。没有证明所有食品个体独立。
- 整鸡缺实拍评测；部分验证协议也缺牛排。本次没有用归档加载检查产生新的准确率。
- 模型为电脑端PyTorch权重；实际板卡为Luckfox Pico Ultra BW/RV1106G2（128MB），本版未转换或通过板端内存验收。
- 源码、配置、报告和来源清单按原字节保留。食品照片/预览HTML不重复上传，排除项见archive-excluded-preview-files.json；原交付指纹清单仍是完整本地交付包的历史证据，归档完整性以总SHA256SUMS.json为准。
- 原始本机路径是历史记录；复训需要另备对应数据并调整配置副本，不能把缺失数据当作随包提供。

CPU使用方法：先安装本目录pyproject.toml所列依赖或requirements.txt，进入本目录后运行：

```powershell
python predict.py "D:/实拍/食物.jpg" --checkpoint model.pt --device cpu
```

[返回总目录](../../README.md)。
