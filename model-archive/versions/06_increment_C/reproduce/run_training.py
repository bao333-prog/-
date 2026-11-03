"""第二次增量，运行时核对扩层前后原十类所有参数与缓冲区均保留。"""
from pathlib import Path
import hashlib,json,time,traceback
import torch
from oven_food.config import load_experiment_config
import oven_food.train as training
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(audit['parent_checkpoint']).read_bytes()).hexdigest()==audit['parent_sha256']
original=training.expand_classifier
def audited_expand(model,*,num_classes):
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    result=original(model,num_classes=num_classes)
    after=result.state_dict()
    for key,value in before.items():
        candidate=after[key][:len(value)] if key in {'classifier.3.weight','classifier.3.bias'} else after[key]
        assert torch.equal(value,candidate),key
    assert before['classifier.3.weight'].shape[0]==10 and after['classifier.3.weight'].shape[0]==14
    (OUT/'initialization_audit.json').write_text(json.dumps(dict(old_outputs=10,new_outputs=14,
        all_existing_parameters_and_buffers_unchanged_during_expansion=True),indent=2),encoding='utf-8')
    return result
training.expand_classifier=audited_expand
start=time.perf_counter();result=dict(status='running',stage='second_increment_C',threads=4)
try:
    path=training.train_experiment(load_experiment_config(OUT/'train_config.json'))
    result.update(status='completed',checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status='failed',error=repr(exc));traceback.print_exc();raise
finally:
    result['training_seconds']=time.perf_counter()-start
    result['protected_models_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==d for p,d in audit['protected_models'].items())
    (OUT/'execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
