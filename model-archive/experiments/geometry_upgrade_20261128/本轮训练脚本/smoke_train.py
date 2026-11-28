"""用独立合成小数据验证EMA、保存重载和train-only BN流程；不接触实拍评测。"""
from pathlib import Path
import hashlib
import sys
import numpy as np
from PIL import Image
from support import RUN, ROOT, PREVIOUS, load, dump
sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.manifest import ManifestRecord,write_manifest
from train_arm import run


def main():
    directory=RUN/'_smoke_cpu';directory.mkdir(exist_ok=True)
    names=load(RUN/'official_class_names.json');train=[];val=[];rng=np.random.default_rng(42)
    for i,name in enumerate(names):
        for j in range(3):
            p=directory/f'{i}_{j}.png';Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(p)
            split='train' if j<2 else 'val';env='proxy' if j==0 else 'target'
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            row=dict(actual_input=str(p),family_id=digest,label=name,split=split,environment=env,kind='original',
                era='synthetic',sampling_group=f'{split}_{i}_{j}',source_group=f'{split}_{i}_{j}',source_session=f'{split}_{i}_{j}')
            (train if split=='train' else val).append(row)
    def record(r):return ManifestRecord(image_path=r['actual_input'],food_label=r['label'],environment=r['environment'],
        session_id=r['source_group'],split=r['split'],camera_position='fixed',lighting_condition='fixed',steam_level='none',occlusion_level='none')
    write_manifest(directory/'train_validation.csv',[record(r) for r in train+val])
    for file,rows in [('training',train),('raw_training',train),('validation',val)]:dump(directory/(file+'.json'),rows)
    dump(directory/'weights.json',[1/len(train)]*len(train))
    config=load(PREVIOUS/'train_config.json');config.update(experiment_name='synthetic_smoke_only',
       manifest=str(directory/'train_validation.csv'),output_dir=str(directory/'training'),
       checkpoint=str(RUN/'official_initialization.pt'),image_size=64,epochs=2,batch_size=8,device='cpu',augmentation='oven_preserve')
    dump(directory/'config.json',config)
    dump(directory/'options.json',dict(scene_balanced=False,fixed_target_background=False,ema_decay=.99))
    run(directory,smoke=True)
    result=load(directory/'execution.json');assert result['status']=='completed' and result['epochs']==2
    assert set(load(directory/'best_candidates.json'))=={'raw','ema','ema_bn'}
    import torch
    from oven_food.model import build_model
    chosen=load(directory/'selection.json');cp=torch.load(chosen['path'],map_location='cpu',weights_only=False)
    model=build_model('mobilenet_v3_small',num_classes=14,pretrained=False);model.load_state_dict(cp['model_state'],strict=True)
    assert cp['class_names']==names
    assert load(directory/'bn_audit.json')['images']==28
    print('PASS: 2 CPU epochs, raw/EMA/EMA-BN saved, 14-class strict reload, BN read training 28 only',flush=True)


if __name__=='__main__':main()
