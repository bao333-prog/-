"""改进框架重做第二次增量；核对教师参数和BN全程冻结，旧模型只读。"""
from pathlib import Path
import os
import copy,hashlib,json,time,traceback
import torch
from oven_food.config import load_experiment_config
import oven_food.train as training
from oven_food.manifest import read_manifest
OUT=Path(os.environ.get("OVEN_RUN_DIR", Path(__file__).resolve().parents[1]/"records")).resolve();torch.set_num_threads(4)
a=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(a['parent_checkpoint']).read_bytes()).hexdigest()==a['parent_sha256']
original_copy=copy.deepcopy;teachers=[]
def capture_teacher(value):
    result=original_copy(value)
    if isinstance(result,torch.nn.Module):teachers.append((result,{k:v.detach().clone() for k,v in result.state_dict().items()}))
    return result
# 只覆盖训练模块局部绑定，不改动torch等库所使用的标准copy模块。
from types import SimpleNamespace
training.copy=SimpleNamespace(deepcopy=capture_teacher)
started=time.perf_counter();execution=dict(status='running',stage='C_retention_v2',threads=4)
try:
    path=training.train_experiment(load_experiment_config(OUT/'train_config.json'))
    assert read_manifest(OUT/'training/training_manifest.csv')==read_manifest(OUT/'expected_training_manifest.csv')
    assert len(teachers)==1
    teacher,before=teachers[0]
    assert not teacher.training and all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(torch.equal(value,teacher.state_dict()[key]) for key,value in before.items())
    execution.update(status='completed',checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     teacher_parameters_and_bn_unchanged=True,teacher_has_no_gradients=True)
except BaseException as exc:
    execution.update(status='failed',error=repr(exc));traceback.print_exc();raise
finally:
    execution['training_seconds']=time.perf_counter()-started
    execution['protected_models_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in a['protected_models'].items())
    (OUT/'execution.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(execution,ensure_ascii=False),flush=True)
