# C_real_20261105 分类头增强候选

日期：2026-11-18。任务负责人：@MascoTxTs。候选是供小组比较的训练成果，尚未替换网站模型。

## 结果

固定143张比较候选图，完整重新加载父模型和候选模型进行预测：

| 指标 | 原C_real_20261105 | 本候选 |
|---|---:|---:|
| 正确张数 | 122/143 | 127/143 |
| 总体准确率 | 85.31% | 88.81% |
| 有样本的11类宏平均准确率 | 81.71% | 88.74% |

总体增加5张，提升3.50个百分点；11类宏平均提升7.03个百分点。

| 食材 | 比较样本 | 原模型正确 | 候选正确 | 准确率变化（百分点） |
|---|---:|---:|---:|---:|
| 南瓜 | 0 | 未评测 | 未评测 | — |
| 吐司 | 12 | 12/12 | 12/12 | +0.00 |
| 牛排 | 6 | 3/6 | 6/6 | +50.00 |
| 玉米 | 12 | 10/12 | 10/12 | +0.00 |
| 西兰花 | 21 | 20/21 | 21/21 | +4.76 |
| 鸡翅 | 6 | 3/6 | 5/6 | +33.33 |
| 芦笋 | 28 | 26/28 | 24/28 | -7.14 |
| 培根 | 6 | 6/6 | 6/6 | +0.00 |
| 三文鱼 | 6 | 6/6 | 5/6 | -16.67 |
| 整鸡 | 0 | 未评测 | 未评测 | — |
| 土豆 | 13 | 13/13 | 13/13 | +0.00 |
| 香肠 | 12 | 5/12 | 6/12 | +8.33 |
| 薯条 | 21 | 18/21 | 19/21 | +4.76 |
| 可颂 | 0 | 未评测 | 未评测 | — |

芦笋从26/28降至24/28，三文鱼从6/6降至5/6，必须与牛排、鸡翅等改善一起评估。香肠仅6/12，玉米仍10/12，后续需更多独立样本检查。

比较集合缺南瓜、整鸡和可颂，无法计算完整十四类宏平均或判断三类保留率。没有新增类别，本轮不使用“新增类别准确率”混淆新增数据域效果。比较图的实际食品个体独立性未确认，结果不是独立验收成绩；没有使用它们训练或选轮次。

## 训练设置

- 父权重：model-archive/versions/08_C_real_sampled/models/model_C_real.pt，SHA-256见base_provenance.json。
- 保持14类顺序与224输入、白底等比例补边、ImageNet归一化。
- 使用修正版训练2471张和新增牛排205张，总2676张，覆盖12类。
- 修正版暂缓366张继续排除；整鸡、可颂没有新训练图，保留类别。
- 冻结features、BN运行状态和分类头前段，仅训练最后线性层；无新训练图的两个类别对应输出行逐步恢复为原权重。
- 类别加权交叉熵 + 0.5×父模型KL蒸馏，温度2，Adam学习率0.0003，固定24轮，seed42，CPU4线程，不增加随机增强。
- 缓存冻结特征加快CPU训练；未训练参数逐张量核对不变。即使缺样本类别输出行不变，其他类别竞争概率仍可能变化，不能保证其准确率不变。
- 最后固定轮次为候选，比较集不参与调参和选模。保存了逐轮损失、准确率与逐图比较预测。

数据清单中有249张被标为旧实拍训练回放、10张被标为此前开发测试图，记录随public_manifest保留。因此不将2676张统称为全新独立样本。图片本体、原JSON、设备信息及本机绝对路径不上传。

## 复现

使用项目Python环境，准备本地数据目录：其中图片相对路径与public_manifest.csv一致。将public_manifest.csv复制为本地数据目录的manifest.csv即可供脚本使用；data_summary保留了数量与使用边界。源压缩包照片通过小组约定方式取得。

在仓库根目录执行（把YOUR_DATA替换为本地数据目录）：

```powershell
python oven-app/experiments/C_real_head_candidate_20261118/train_candidate.py --source-code oven-app --data YOUR_DATA --base model-archive/versions/08_C_real_sampled/models/model_C_real.pt --output YOUR_RUN
python oven-app/experiments/C_real_head_candidate_20261118/evaluate_candidate.py --source-code oven-app --data YOUR_DATA --base model-archive/versions/08_C_real_sampled/models/model_C_real.pt --run YOUR_RUN
```

输出目录必须为尚不存在的新目录，防止覆盖已有实验。当前脚本针对冻结的2676/143数据清单做数量和哈希检查，不适用于随意替换数据后直接沿用。

## 使用候选

```powershell
python oven-app/predict.py --model oven-app/experiments/C_real_head_candidate_20261118/model_C_real_head_candidate.pt --image YOUR_IMAGE
```

web-export含匹配的ONNX及配置，数值导出检查通过（最大logit误差约4.35e-6）；未发布到网站，未执行该候选的浏览器实图一致性测试，也未测开发板延迟。使用前先完成PR审查和小组评估。

依赖：沿用oven-app/requirements-validated.txt中的实际验证组合；本轮PyTorch 2.15.0+cpu、Python 3.12。导出依赖沿用onnx==1.23.2、onnxruntime==1.31.1、onnxscript==0.7.2。
