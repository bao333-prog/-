# 十二类目标域训练：A′ 与 B′ 对照

训练日期：2026-11-05
数据：`按食材分类/`，1428 张烤箱实拍图，12 类
环境：Apple M5 / 24 GB / Python 3.12 / PyTorch 2.15.0 / MPS

本目录记录把 A v2 配方与 B 配方迁移到 12 类烤箱实拍数据的第一轮实验。原始图片不在仓库中，只提交代码、划分清单、配置、权重、评测明细和报告。

## 结果

| 模型 | 配方 | 最佳轮 | 验证集 | 独立 test | test macro F1 |
|---|---|---:|---:|---:|---:|
| **A′** | gentle，lr=3e-4 | 16 | 96.75% | **96.05%** | **95.91%** |
| B′ | gentle_plus，lr=1e-4 | 11 | 93.50% | 87.72% | 86.49% |
| B+（控制实验） | gentle_plus，lr=3e-4 | 17 | 97.56% | 91.23% | 88.59% |

A′ 的 test 错误共 9 张：三文鱼→培根 3、可颂→玉米 4、土豆→可颂 2。

B′ 的主要错误：吐司→玉米 10、香肠→鸡翅 10。控制学习率和初始化后，B+ 的吐司→玉米错误消失，但香肠→鸡翅仍然 10 张，说明 gentle_plus 的色温和压缩增强会削弱颜色相近食材的区分。

原模型 zero-shot 在 target test 上明显下降：

| 模型 | 可评估类别 | 样本 | 准确率 |
|---|---|---:|---:|
| A v2（六类） | 5 类（缺南瓜） | 111 | 62.16% |
| B（十类） | 8 类（缺南瓜、整鸡） | 157 | 49.04% |

## 数据与划分

- 总图片 1428，12 类；南瓜、整鸡在新数据集中没有样本，按确认方案退出 12 类体系。
- train 954 / val 246 / test 228。
- 以 `sample_prefix` 物理样本组为单位划分，同一组只进一个 split，按类别分层，seed=42。
- test 是“留出物理样本组”，与 train 同批 4 个采集 session，不是外部独立验收集。
- 标签来自文件夹归类；原始 JSON 仍为 `pending_manual_review`，尚未逐张人工确认。

各类图片数：

| 类别 | 总数 | train | val | test |
|---|---:|---:|---:|---:|
| 三文鱼 | 93 | 63 | 15 | 15 |
| 可颂 | 110 | 74 | 18 | 18 |
| 吐司 | 104 | 65 | 20 | 19 |
| 土豆 | 114 | 78 | 18 | 18 |
| 培根 | 97 | 69 | 14 | 14 |
| 牛排 | 79 | 33 | 31 | 15 |
| 玉米 | 252 | 172 | 40 | 40 |
| 芦笋 | 112 | 77 | 18 | 17 |
| 薯条 | 122 | 82 | 20 | 20 |
| 西兰花 | 119 | 85 | 17 | 17 |
| 香肠 | 97 | 67 | 15 | 15 |
| 鸡翅 | 129 | 89 | 20 | 20 |

牛排只有 3 个物理样本组，train 只有 33 张，相关指标波动较大。

## 代码改动

两组实验代码分别来自仓库已有的归档版本：

- A 配方：`model-archive/versions/02_A_candidate_v2`
- B 配方：`model-archive/versions/04_B_augmentation`

本目录 `code/` 下是复制后的源码，在原代码上只增加了一个配置项 `init_backbone_from`：

- 加载父模型的 `features.*`；
- 丢弃 `classifier.*`；
- 按当前 12 类重新初始化分类头。

原归档版本和 `model-archive` 的校验文件未修改。

## 目录结构

```text
experiments/target12/
├── README.md
├── REPORT.md                 # 完整训练与评测报告
├── models/                   # 三个 PyTorch 权重
├── configs/                  # 三组训练配置
├── dataset/
│   ├── manifest.csv          # 1428 行清单与 split
│   └── split_summary.json
├── runs/                     # 每组的历史曲线、摘要、配置、日志
├── eval/                     # test 汇总、逐张预测、混淆矩阵
├── scripts/                  # 数据划分、统一评测、冒烟测试
└── code/
    ├── A_recipe_src/         # A 配方改造源码
    ├── B_recipe_src/         # B 配方改造源码
    └── predict.py
```

## 预测

```bash
cd experiments/target12
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision pillow numpy

PYTHONPATH=code/A_recipe_src .venv/bin/python code/predict.py \
  --model models/A_target12_gentle.pt \
  --image "你的图片.jpg"
```

预测类别顺序从权重的 `class_names` 读取，不要手工套用其他版本。

## 复现训练

`configs/` 中的路径是原作者机器的绝对路径。换机器时需要修改：

- `manifest`
- `data_root`
- `output_dir`
- `init_backbone_from`（指向 A v2 或 B 的权重）

原始图片需要恢复到 `data_root/按食材分类/...`，或者在 `manifest.csv` 中改 `image_path`。

然后：

```bash
PYTHONPATH=code/A_recipe_src .venv/bin/python -m oven_food train configs/A_target12_gentle.json
PYTHONPATH=code/B_recipe_src .venv/bin/python -m oven_food train configs/B_target12_gentle_plus.json
```

## 边界

- 单 seed、单次划分；test 只有 228 张，每类 14–40 张，A′ 与 B+ 的 4.8 个百分点约等于 11 张图，存在抽样波动。
- test 与 train 同批采集，不能代替跨批次、跨设备、跨光照的独立验收。
- 标签尚未人工确认，当前成绩按文件夹弱标签计算。
- 1,524,006 参数按 INT8 约 1.45 MB，超过课题 Flash 500 KB 上限，端侧部署尚未验证。
