"""本轮通用路径和读写工具；不导入共享训练源码，防止串用历史模块。"""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'runs/asparagus_half_20261126_172248'
INTEGRATED = ROOT / 'runs/integrated_food_20261126_115154'

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
