"""使用纯几何增强训练；需显式启动本脚本，原数据和历史模型不改写。"""
from pathlib import Path
import json
import sys
import time
from collections import Counter
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / "source_snapshot"))
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

assert torch.cuda.is_available(), "GPU不可用，停止而不自动回退CPU"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
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
