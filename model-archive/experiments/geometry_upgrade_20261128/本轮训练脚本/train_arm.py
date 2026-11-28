"""复用冻结训练入口；普通模型与EMA并行比较，测试在此进程不可见。"""
import argparse
import copy
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn
from support import RUN, ROOT, load, dump, sha

sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.config import load_experiment_config
from oven_food.dataset import FoodImageDataset, IMAGENET_MEAN, IMAGENET_STD
from oven_food.augment import prepare_image
from oven_food.manifest import read_manifest
from oven_food.model import build_model
import oven_food.train as training
from family_sampler import FamilyCoverageSampler
from scene_sampler import SceneBalancedSampler
from selection_ema import selection_key, create_ema, update_ema, ema_model_state
from validation_monitor import validate_with_class_metrics
from continuation import should_stop
from rng_utils import isolated_validation_rng


def checkpoint_payload(state,config,names,epoch,stats,loss,variant):
    return dict(model_state=state,class_names=names,model_name=config.model_name,image_size=config.image_size,
       epoch=epoch,validation_accuracy=stats['correct']/stats['total'],validation_loss=loss,
       selection_key=list(selection_key(stats,loss)),selection_stats=stats,variant=variant,
       preprocessing=dict(resize_mode='letterbox',mean=IMAGENET_MEAN.tolist(),std=IMAGENET_STD.tolist()),
       config=asdict(config),initialization_sha256=sha(RUN/'official_initialization.pt'))


def run(directory,*,smoke=False):
    options=load(directory/'options.json');config=load_experiment_config(directory/'config.json')
    if not smoke:
        review=load(directory/'preflight.json');assert review['approved'] and not review['errors']
        for path,digest in review['file_sha256'].items():assert sha(Path(path))==digest,path
        assert torch.cuda.is_available(),'GPU不可用，停止'
        assert config.device=='cuda:0'
    assert Path(training.__file__).resolve().is_relative_to(RUN/'source_snapshot')
    assert sha(config.checkpoint)=='5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    torch.set_num_threads(4)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=load(directory/'training.json');weights=load(directory/'weights.json')
    names=load(RUN/'official_class_names.json');by_path={r['actual_input']:r for r in rows}
    counts=Counter();history=[];best={};ema=None;active_sampler=None;started=time.perf_counter()
    base_getitem=FoodImageDataset.__getitem__

    class TrackedDataset(FoodImageDataset):
        def __getitem__(self,index):
            record=self.records[index];row=by_path.get(record.image_path)
            if self.augmentation_profile=='oven_preserve' and row is not None:
                if options['fixed_target_background'] and row['environment']=='target':
                    # C分支实际几何来自离线食材旋转，完整烤箱背景不再整体镜像/旋转。
                    from PIL import Image
                    with Image.open(record.image_path) as image:
                        image=prepare_image(image,image_size=config.image_size,resize_mode='letterbox')
                    pixels=(np.asarray(image,dtype=np.float32)/255.-IMAGENET_MEAN)/IMAGENET_STD
                    result=torch.from_numpy(pixels.transpose(2,0,1).copy()),self.class_to_index[record.food_label]
                else:result=base_getitem(self,index)
                counts[record.image_path]+=1
                return result
            return base_getitem(self,index)

    def sampler(records,target_fraction,torch_module,sampler_type):
        nonlocal active_sampler
        assert [r.image_path for r in records]==[r['actual_input'] for r in rows]
        if options['scene_balanced']:
            scene_ids=load(RUN/'scene_index.json')['family_to_scene']
            active_sampler=SceneBalancedSampler(rows,weights,scene_ids,seed=config.seed)
        else:active_sampler=FamilyCoverageSampler(rows,weights,seed=config.seed,coverage_interval=4)
        return active_sampler

    def optimizer_complete(state):
        nonlocal ema
        if ema is None:ema=create_ema(state['model'],decay=options['ema_decay'])
        update_ema(ema,state['model'])

    def record_best(model_state,epoch,stats,loss,variant):
        key=selection_key(stats,loss)
        if variant not in best or key>tuple(best[variant]['key']):
            path=directory/'training'/('best_'+variant+'.pt');path.parent.mkdir(exist_ok=True)
            torch.save(checkpoint_payload(model_state,config,names,epoch,stats,loss,variant),path)
            best[variant]=dict(path=str(path),key=list(key),epoch=epoch,stats=stats,loss=loss)

    def validate(model,loader,criterion,device,torch_module):
        epoch=len(history)+1
        loss,accuracy,stats=validate_with_class_metrics(model,loader,criterion,device,names)
        record_best(model.state_dict(),epoch,stats,loss,'raw')
        assert ema is not None
        # 额外迭代验证DataLoader会消耗CPU RNG，隔离后不扰动live训练轨迹。
        with isolated_validation_rng():
            ema_loss,ema_accuracy,ema_stats=validate_with_class_metrics(ema.module,loader,criterion,device,names)
        record_best(ema_model_state(ema),epoch,ema_stats,ema_loss,'ema')
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-started,
          raw=dict(loss=loss,accuracy=accuracy,**stats),ema=dict(loss=ema_loss,accuracy=ema_accuracy,**ema_stats)))
        dump(directory/'validation_history.json',history);dump(directory/'best_candidates.json',best)
        dump(directory/'progress.json',dict(epoch=epoch,raw_correct=stats['correct'],ema_correct=ema_stats['correct'],total=stats['total'],seconds=time.perf_counter()-started))
        print('VALIDATION',directory.name,epoch,'raw',stats['correct'],'ema',ema_stats['correct'],'/',stats['total'],flush=True)
        return loss,accuracy

    def epoch_complete(state):
        epoch=state['epoch'];train_history=state['history'];dump(directory/'training/history.json',train_history)
        sampler_state=(active_sampler.state_dict() if options['scene_balanced'] else
             dict(epoch=active_sampler.epoch,generator=active_sampler.generator.get_state()))
        # 在epoch边界保存：num_workers=0，没有尚未消费的预取批次。
        torch.save(dict(model_state=state['model'].state_dict(),optimizer_state=state['optimizer'].state_dict(),
          scheduler_state=state['scheduler'].state_dict() if state['scheduler'] else None,
          ema_state=ema.state_dict(),sampler_state=sampler_state,epoch=epoch,class_names=names,
          config=asdict(config),options=options,best=best,history=train_history,validation_history=history,
          python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
          augmentation_rng=state['train_dataset'].augmenter._rng.getstate(),loader_rng=state['loader_generator'].get_state(),
          sample_counts=dict(counts)),directory/'training/last.pt')
        stop=(epoch>=config.epochs if smoke else should_stop(train_history))
        dump(directory/'continuation.json',dict(epoch=epoch,stop=stop,test_used=False,
          recent_raw_validation_accuracy=[r['validation_accuracy'] for r in train_history[-3:]],
          policy='20轮末段净提升则继续，扩展至少3轮，连续3轮无整体准确率新高停止；最多60轮'))
        return stop

    training.FoodImageDataset=TrackedDataset;training._weighted_sampler=sampler
    training._validate_epoch=validate;training._optimizer_complete=optimizer_complete
    training._epoch_complete=epoch_complete;training._maximum_epochs=config.epochs if smoke else 60
    dump(directory/'execution.json',dict(status='running',smoke=smoke,test_used_for_selection=False))
    try:
        training.train_experiment(config)
        dump(directory/'sample_counts.json',dict(counts))
        assert {r['family_id'] for r in rows if counts.get(r['actual_input'],0)}>={r['family_id'] for r in rows}
        # EMA的BN重估仅使用本分支训练原图，固定baseline，不读任何验证/测试图。
        raw_rows=load(directory/'raw_training.json');raw_paths={r['actual_input'] for r in raw_rows}
        records=read_manifest(directory/'train_validation.csv')
        raw_records=list({r.image_path:r for r in records if r.split=='train' and r.image_path in raw_paths}.values())
        assert len(raw_records)==len({r['family_id'] for r in raw_rows})
        raw_data=FoodImageDataset(raw_records,data_root=ROOT,class_names=names,augmentation_profile='baseline',
                  image_size=config.image_size,resize_mode='letterbox')
        bn_loader=DataLoader(raw_data,batch_size=config.batch_size,shuffle=False,num_workers=0)
        cp=torch.load(best['ema']['path'],map_location='cpu',weights_only=False)
        bn_model=build_model(config.model_name,num_classes=len(names),pretrained=False)
        bn_model.load_state_dict(cp['model_state'],strict=True);bn_model.to(config.device)
        update_bn(bn_loader,bn_model,device=torch.device(config.device))
        val_records=[r for r in records if r.split=='val']
        val_data=FoodImageDataset(val_records,data_root=ROOT,class_names=names,augmentation_profile='baseline',
                    image_size=config.image_size,resize_mode='letterbox')
        val_loader=DataLoader(val_data,batch_size=config.batch_size,shuffle=False,num_workers=0)
        loss,accuracy,stats=validate_with_class_metrics(bn_model,val_loader,torch.nn.CrossEntropyLoss(),torch.device(config.device),names)
        record_best(bn_model.state_dict(),cp['epoch'],stats,loss,'ema_bn')
        dump(directory/'bn_audit.json',dict(split='train',images=len(raw_records),training_families=len(raw_records),validation_or_test_used_for_bn=False))
        dump(directory/'best_candidates.json',best)
        chosen=max(best.values(),key=lambda b:tuple(b['key']))
        dump(directory/'selection.json',dict(**chosen,checkpoint_sha256=sha(chosen['path']),test_used=False))
        dump(directory/'execution.json',dict(status='completed',epochs=len(history),seconds=time.perf_counter()-started,
           sample_reads=sum(counts.values()),selected=chosen['path'],checkpoint_sha256=sha(chosen['path']),
           test_used_for_selection=False,initialization_sha256=sha(config.checkpoint)))
    except Exception as exc:
        dump(directory/'execution.json',dict(status='failed',error=repr(exc),seconds=time.perf_counter()-started))
        dump(directory/'partial_sample_counts.json',dict(counts));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['A224','B320','C320']);args=parser.parse_args()
    run(RUN/args.arm)
