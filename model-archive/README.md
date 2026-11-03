# 食材识别：全部历史模型归档

归档日期：2026-11-03。本目录包含八个实际完成训练的版本、一个明确未训练的环境检查权重，以及每版原始交付代码、配置、报告与来源校验记录。全部为 **MobileNetV3-Small 单标签分类**；YOLO仅讨论过迁移，尚无已训练YOLO权重。

## 版本索引

验证集用于选模；历史网图测试、复用测试和新增鸡翅测试子集的口径不同，不能直接横向排名，更不能当真实实物成绩。

| 版本 | 类别数 | 最佳轮/总轮数 | 已有结果及边界 |
|---|---:|---:|---|
| [六类临时试跑](versions/00_smoke_six_class/ARCHIVE-NOTES.md) | 6 | 8/8 | 验证14/17（82.35%）；仅流程试跑，未作为正式A起点。 |
| [正式六类A](versions/01_formal_A/ARCHIVE-NOTES.md) | 6 | 1/20 | 历史网图测试158/159（99.37%）。 |
| [六类A优化候选v2](versions/02_A_candidate_v2/ARCHIVE-NOTES.md) | 6 | 9/20 | 验证157/157（100%）；没有独立测试成绩，未作为B起点。 |
| [第一次增量十类B](versions/03_increment_B/ARCHIVE-NOTES.md) | 10 | 7/20 | 历史测试270/277（97.47%）；旧六类下降3.14个百分点，未达≤2个百分点目标。 |
| [十类B增强候选](versions/04_B_augmentation/ARCHIVE-NOTES.md) | 10 | 9/20 | 验证269/275（97.82%）；与B同起点重做增量，不是从B继续训练；未重测历史测试。 |
| [十类B鸡翅补强](versions/05_B_wings_supplement/ARCHIVE-NOTES.md) | 10 | 19/20 | 验证292/293（99.66%）；新增鸡翅测试16/18（88.89%），该成绩只针对新增鸡翅子集。 |
| [第二次增量十四类C](versions/06_increment_C/ARCHIVE-NOTES.md) | 14 | 9/20 | 历史复用测试388/409（94.87%）；旧十类下降4.07个百分点，未达保持目标。 |
| [十四类C知识保持与增强候选v2](versions/07_C_retention_v2/ARCHIVE-NOTES.md) | 14 | 12/20 | 验证395/407（97.05%）；旧类287/293、新类108/114；旧类验证下降1.71个百分点，未重测测试集，不能宣称测试通过。 |

## 模型关系

```text
官方ImageNet预训练 ── 临时六类试跑（独立结束）
                 ├─ 正式六类A ── 十类B
                 │           └─ 十类B增强候选 ── 十类B鸡翅补强 ── 十四类C
                 │                                           └─ 十四类C v2候选
                 └─ 六类A v2候选（不是B的父模型）
```

十四类C和C v2均从鸡翅补强B开始；C v2没有接着C训练。B增强候选从A重做第一次增量；鸡翅补强没有新增类别。环境检查是随机未训练权重，单独保存在[auxiliary](auxiliary/environment_check/README.md)。不同目录中相同的父模型权重为原包保留副本，不重复计数。

## 下载与预测

可克隆整个仓库，也可下载仓库ZIP后打开本目录。主模型文件都是约6 MB的实际PyTorch权重，直接纳入Git；不需要为本目录单独下载LFS对象。仓库其他历史ZIP仍沿用原LFS规则。

需要哪版就进入对应版本目录，在独立虚拟环境中安装它，不要把八版源码同时加入Python路径。例如在仓库根目录执行：

```powershell
cd model-archive/versions/07_C_retention_v2
py -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe predict.py --model models/model_C_v2.pt --image "D:/实拍/食物.jpg"
```

正式A、A v2、B、增强B、补强B和C也各自提供原始 `predict.py`。临时试跑保留原始命令行代码及 `reference/` 配置，不提供独立predict脚本；仅用于追溯流程。预测类别顺序必须读取当前权重的 `class_names`，不要手工套用其他版本。分数是已知类别间的相对概率，不能保证未知食物识别。

## 完整性检查

无需安装训练依赖即可检查文件指纹：

```powershell
python model-archive/verify_archive.py
```

已安装PyTorch、torchvision、NumPy和Pillow时，可增加逐版新进程CPU加载和有限输出检查：

```powershell
python model-archive/verify_archive.py --load-models
```

[catalog.json](catalog.json)记录主权重、类别顺序、父版本和成绩口径；[provenance.json](provenance.json)逐项记录原交付包成员指纹；[SHA256SUMS.json](SHA256SUMS.json)覆盖本目录文件（清单自身及本地检查输出除外）。[归档检查记录](verification.json)说明本次实际做过的检查。加载检查使用合成输入，只验证流程，不产生新的准确率。

## 数据与复现范围

原包内 `src/`、`tests/`、`reproduce/`、配置和报告按字节保留，补充训练记录位于每版 `archive-records/`。本目录不重复上传原始食品图片。历史配置、脚本和报告中的本机绝对路径是原始证据，无法在其他电脑直接使用；复训前需要恢复对应数据版本、分组和划分，并修改配置副本中的数据、输出及父权重路径。

旧训练记录的损失统计方式存在历史差异，详见各版原始报告；本次只归档，不修订历史数值。部分同源风险、旧鸡翅原图缺失、可颂来源缺失等问题仍保留。文件完整不等于数据完整可复训。

用户已反馈吐司、香肠、可颂、牛排、培根、整鸡的真实实物识别较弱；本次未增加实拍数据、未训练适配模型、未宣称这些问题已解决。目标烤箱、计数能力和RK3506部署指标均未通过本次归档验证。
