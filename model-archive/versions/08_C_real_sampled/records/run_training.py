"""从十四类C v2继续实拍适配，复用原训练入口。"""
from pathlib import Path
import json
import sys
import time
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oven_food.config import load_experiment_config
from oven_food.train import train_experiment

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
