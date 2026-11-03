"""从正式六类A继续学习四个新类；旧A只读，保存独立十类B。"""
from pathlib import Path
import hashlib,json,time,traceback
import torch
from oven_food.config import load_experiment_config
import oven_food.train as training

OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
baseline=json.loads((OUT/"baseline.json").read_text(encoding="utf-8"));BASE=Path(baseline["parent_dir"])
assert hashlib.sha256((BASE/"training/best.pt").read_bytes()).hexdigest()==baseline["parent_sha256"]
original_expand=training.expand_classifier
def audited_expand(network,*,num_classes):
    weight=network.classifier[-1].weight.detach().clone();bias=network.classifier[-1].bias.detach().clone()
    backbone=network.features[0][0].weight.detach().clone()
    result=original_expand(network,num_classes=num_classes)
    assert torch.equal(result.classifier[-1].weight[:len(weight)],weight)
    assert torch.equal(result.classifier[-1].bias[:len(bias)],bias)
    assert torch.equal(result.features[0][0].weight,backbone)
    (OUT/"initialization_audit.json").write_text(json.dumps({"old_outputs":len(weight),"new_outputs":num_classes,
        "old_head_rows_identical_before_training":True,"backbone_unchanged_during_expansion":True},indent=2),encoding="utf-8")
    return result
training.expand_classifier=audited_expand
started=time.perf_counter();result={"status":"running","stage":"B_augmentation_trial","threads":4}
try:
    path=training.train_experiment(load_experiment_config(OUT/"train_config.json"))
    result.update(status="completed",checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status="failed",error=repr(exc));traceback.print_exc();raise
finally:
    result["training_seconds"]=time.perf_counter()-started
    result["parent_A_unchanged"]=hashlib.sha256((BASE/"training/best.pt").read_bytes()).hexdigest()==baseline["parent_sha256"]
    (OUT/"execution.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
