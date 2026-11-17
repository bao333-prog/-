"""从十四类C v2继续实拍适配，复用原训练入口。"""
from pathlib import Path
import json
import sys
import time
from collections import Counter
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oven_food.config import load_experiment_config
from oven_food.train import train_experiment
from oven_food.dataset import FoodImageDataset

# 数据加载进程数为0，记录每张图在训练中的实际抽样次数。
# 原始读取函数只调用一次，记录不消耗随机数。
sample_counts = Counter()
original_getitem = FoodImageDataset.__getitem__

def tracked_getitem(dataset, index):
    sample = original_getitem(dataset, index)
    if dataset.augmentation_profile != "baseline":
        sample_counts[dataset.records[index].image_path] += 1
    return sample

FoodImageDataset.__getitem__ = tracked_getitem

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
config = load_experiment_config(RUN / "train_config.json")
started = time.perf_counter()
try:
    checkpoint = train_experiment(config)
except Exception as error:
    (RUN / "execution.json").write_text(
        json.dumps(dict(status="failed", error=str(error), seconds=time.perf_counter()-started)),
        encoding="utf-8")
    raise
(RUN / "execution.json").write_text(
    json.dumps(dict(status="completed", checkpoint=str(checkpoint), seconds=time.perf_counter()-started)),
    encoding="utf-8")
(RUN / "sample_counts.json").write_text(
    json.dumps(dict(sample_counts), ensure_ascii=False, indent=2), encoding="utf-8")
