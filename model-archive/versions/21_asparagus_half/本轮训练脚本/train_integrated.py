"""半截芦笋补充训练：GPU、独立初始化、验证选模；训练不读取测试图片。"""
import sys
import time
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from prepare_data import RUN, ROOT, load, dump, sha

sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.config import load_experiment_config
import oven_food.train as training
from family_sampler import FamilyCoverageSampler
from validation_monitor import validate_with_class_metrics
from continuation import should_stop


def main():
    assert torch.cuda.is_available(),'GPU不可用，停止'
    review=load(RUN/'integration_preflight_review.json')
    assert review['approved'] and not review['errors']
    for name,digest in review['file_sha256'].items():assert sha(RUN/name)==digest,name
    assert Path(training.__file__).resolve().is_relative_to(RUN/'source_snapshot')
    assert sha(RUN/'official_initialization.pt')=='5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=load_experiment_config(RUN/'train_config.json')
    rows=load(RUN/'training_integrated.json');names=load(RUN/'official_class_names.json')
    weights=load(RUN/'sampling_weights.json');counts=Counter();validation_history=[]
    original=training.FoodImageDataset.__getitem__
    started=time.perf_counter()

    def tracked(dataset,index):
        result=original(dataset,index)
        if dataset.augmentation_profile=='oven_geometry':counts[dataset.records[index].image_path]+=1
        return result

    def sampler(records,target_fraction,torch_module,sampler_type):
        assert [r.image_path for r in records]==[r['actual_input'] for r in rows]
        return FamilyCoverageSampler(rows,weights,seed=config.seed,coverage_interval=4)

    def validate(model,loader,criterion,device,torch_module):
        loss,accuracy,stats=validate_with_class_metrics(model,loader,criterion,device,names)
        epoch=len(validation_history)+1
        validation_history.append({'epoch':epoch,'seconds':time.perf_counter()-started,**stats})
        dump(RUN/'validation_class_history.json',validation_history)
        dump(RUN/'training_progress.json',dict(epoch=epoch,validation_loss=loss,
             validation_accuracy=accuracy,elapsed_seconds=time.perf_counter()-started))
        print('VALIDATION',epoch,stats['correct'],'/',stats['total'],flush=True)
        return loss,accuracy

    def epoch_complete(state):
        history=state['history'];epoch=state['epoch']
        dump(RUN/'training/history.json',history)
        # 末轮权重与最佳权重分开保存；保留优化器和随机状态，避免误把best当作末轮。
        torch.save(dict(model_state=state['model'].state_dict(),epoch=epoch,class_names=names,
          model_name=config.model_name,image_size=config.image_size,config=asdict(config),
          optimizer_state=state['optimizer'].state_dict(),
          scheduler_state=state['scheduler'].state_dict() if state['scheduler'] else None,
          history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
          sampler_rng=state['sampler'].generator.get_state(),sampler_epoch=state['sampler'].epoch,
          augmentation_rng=state['train_dataset'].augmenter._rng.getstate(),
          loader_rng=state['loader_generator'].get_state(),sample_counts=dict(counts)),RUN/'training/last.pt')
        stop=should_stop(history)
        dump(RUN/'continuation_decision.json',dict(epoch=epoch,stop=stop,nominal_epochs=20,
            maximum_epochs=60,patience_after_extension=3,
            last_three_validation_accuracy=[r['validation_accuracy'] for r in history[-3:]],
            reason=('继续：不足20轮或末段仍满足验证提升规则' if not stop else
                    '停止：20轮末段无继续上升、扩展后连续3轮无准确率新高，或达到60轮上限'),
            test_used_for_decision=False))
        return stop

    training.FoodImageDataset.__getitem__=tracked
    training._weighted_sampler=sampler;training._validate_epoch=validate
    training._maximum_epochs=60;training._epoch_complete=epoch_complete
    dump(RUN/'execution.json',dict(status='running',epochs_requested=20,training_started=True,
         gpu=torch.cuda.get_device_name(0),test_used_for_selection=False))
    try:
        checkpoint=training.train_experiment(config)
        epochs=len(validation_history);assert 20<=epochs<=60
        dump(RUN/'sample_counts.json',dict(counts))
        expected=Counter();preview=FamilyCoverageSampler(rows,weights,seed=42,coverage_interval=4)
        for _ in range(epochs):expected.update(rows[i]['actual_input'] for i in preview)
        assert counts==expected and set(counts)=={r['actual_input'] for r in rows}
        dump(RUN/'expected_sampling_counts.json',dict(expected))
        dump(RUN/'execution.json',dict(status='completed',epochs=epochs,seconds=time.perf_counter()-started,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),training_started=True,
            test_used_for_selection=False,initialization_sha256=sha(RUN/'official_initialization.pt'),
            gpu=torch.cuda.get_device_name(0),sample_reads=sum(counts.values())))
    except Exception as exc:
        dump(RUN/'execution.json',dict(status='failed',error=repr(exc),seconds=time.perf_counter()-started))
        dump(RUN/'partial_sample_counts.json',dict(counts));raise


if __name__=='__main__':main()
