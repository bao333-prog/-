from pathlib import Path
import json,time,traceback,hashlib
import torch
from oven_food.config import load_experiment_config
from oven_food.train import train_experiment
out=Path(__file__).resolve().parent;torch.set_num_threads(4);started=time.perf_counter()
result={"status":"running","threads":4,"stage":"A_candidate_v2"}
try:
    path=train_experiment(load_experiment_config(out/"train_config.json"))
    result.update(status="completed",checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status="failed",error=repr(exc));traceback.print_exc();raise
finally:
    result["training_seconds"]=time.perf_counter()-started
    (out/"execution.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
