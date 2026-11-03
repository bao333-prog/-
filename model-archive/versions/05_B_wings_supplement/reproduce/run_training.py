"""保持十类标签不变的补强训练；复用项目入口，不走新增类别的20张回放分支。"""
from pathlib import Path
import hashlib
import json
import time
import traceback
import torch
from oven_food.config import load_experiment_config
from oven_food.train import train_experiment

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
audit = json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(audit['parent_checkpoint']).read_bytes()).hexdigest()==audit['parent_sha256']
config = load_experiment_config(OUT/'train_config.json')
assert not config.new_classes and config.epochs==20
started = time.perf_counter()
result = dict(status='running',stage='chicken_supplement_same_ten_classes',threads=4)
try:
    checkpoint = train_experiment(config)
    result.update(status='completed',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status='failed',error=repr(exc))
    traceback.print_exc()
    raise
finally:
    result['training_seconds']=time.perf_counter()-started
    result['protected_models_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in audit['protected_models'].items())
    (OUT/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
