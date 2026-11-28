# 南瓜可颂补充

归档日期：2026-11-28。本次只保存此前训练结果，没有重新训练或重测食品图片。

- 主权重：[models/model_C_pumpkin_croissant.pt](models/model_C_pumpkin_croissant.pt)，SHA256 `9db30ce0eedee1fe8d53843eaff34d3b49e435c3f2222417ce79db21584c674a`。
- 14类MobileNetV3-Small；输入224×224，预处理与类别顺序以权重及对应代码为准。
- 共训练10轮，选定第10轮（raw）。
- 初始化：14_C_geometry。数据版本的继承不表示继承上一版模型参数。
- 选模验证77张；完整验证80/83（另6张南瓜仅诊断）；开发测试82/84（97.62%）。南瓜评测与训练存在同源关联。
- 本目录食品测试是已公开复用的开发对照，历史部分还有连续照片及南瓜同源问题；具体限制见原报告。没有证明所有食品个体独立。
- 整鸡缺实拍评测；部分验证协议也缺牛排。本次没有用归档加载检查产生新的准确率。
- 模型为电脑端PyTorch权重；实际板卡为Luckfox Pico Ultra BW/RV1106G2（128MB），本版未转换或通过板端内存验收。
- 源码、配置、报告和来源清单按原字节保留。食品照片/预览HTML不重复上传，排除项见archive-excluded-preview-files.json；原交付指纹清单仍是完整本地交付包的历史证据，归档完整性以总SHA256SUMS.json为准。
- 原始本机路径是历史记录；复训需要另备对应数据并调整配置副本，不能把缺失数据当作随包提供。

CPU使用方法：先安装本目录pyproject.toml所列依赖或requirements.txt，进入本目录后运行：

```powershell
python -m pip install -e .
python predict.py --model models/model_C_pumpkin_croissant.pt --image "D:/实拍/食物.jpg"
```

[返回总目录](../../README.md)。
