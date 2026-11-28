"""Independent A/B audit. Hash cache bytes and decoded RGB; never run an evaluation."""
import csv
import hashlib
import io
import itertools
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import torch
from PIL import Image
from support import RUN, PREVIOUS, load, dump, sha
from family_sampler import FamilyCoverageSampler


def audit(name):
    size=int(name[1:])
    folder=RUN/name
    other=RUN/('B320' if name=='A224' else 'A224')
    read=lambda n: load(folder/(n+'.json'))
    cfg=read('config'); opt=read('options')
    for n,omit in [('config',{'image_size','manifest','output_dir','experiment_name'}),('options',{'arm','input_size'})]:
        a,b=read(n),load(other/(n+'.json'))
        assert {k:v for k,v in a.items() if k not in omit}=={k:v for k,v in b.items() if k not in omit}
    assert cfg['augmentation']=='oven_preserve' and cfg['image_size']==opt['input_size']==size
    assert opt['ema_decay']==.99 and not any(opt[k] for k in ('scene_balanced','fixed_target_background','foreground_rotation'))
    classes=load(RUN/'official_class_names.json')
    cp=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['class_names']==classes and len(classes)==14 and cp['epoch']==0
    assert sha(cfg['checkpoint'])=='5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    train,val,test,raw=[read(k) for k in ('training','validation','test','raw_training')]
    assert [len(t) for t in (train,val,test,raw)]==[7696,193,167,5591]
    raws={r['family_id'] for r in raw}
    assert len(raws)==5590 and set(r['family_id'] for r in train)==raws
    assert set(r['label'] for r in raw)==set(classes)
    overlap={}
    for (a,aa),(b,bb) in itertools.combinations(zip(('train','val','test'),(train,val,test)),2):
        for field in ('source','cache_source','family_id','raw_sha256','pixel_sha256','prepared_pixel_sha256','actual_input','source_session'):
            def keys(xx):
                return {r[field] for r in xx if field!='source_session' or r['environment']=='target'}
            overlap[a+'_'+b+'_'+field]=len(keys(aa)&keys(bb))
            assert not keys(aa)&keys(bb),(a,b,field)
    for rows,previous,split in [(train,'training_integrated','train'),(raw,'training_base','train'),(val,'validation_base','val'),(test,'test_base','test')]:
        old=load(PREVIOUS/(previous+'.json'))
        assert len(old)==len(rows)
        for curr,base in zip(rows,old):
            for k in ('source','raw_sha256','family_id','label','environment','split','source_session','kind'):
                assert curr[k]==base[k],(k,curr['source'])
            assert curr['baseline_input']==base['actual_input'] and curr['split']==split
            if size==224:assert curr['prepared_pixel_sha256']==base['prepared_pixel_sha256']
    for filename,rows in [('train_validation.csv',train+val),('test_only.csv',test)]:
        with (folder/filename).open(encoding='utf-8-sig',newline='') as f: lines=list(csv.DictReader(f))
        assert len(lines)==len(rows)
        for l,r in zip(lines,rows):
            assert [l[k] for k in ('image_path','food_label','split','environment','session_id')]==[r[k] for k in ('actual_input','label','split','environment','source_group')]
    weights=read('weights')
    assert weights==load(PREVIOUS/'sampling_weights.json') and math.isclose(sum(weights),1.)
    sampler=FamilyCoverageSampler(train,weights,seed=42,coverage_interval=4)
    sample=[list(sampler) for _ in range(5)]
    assert all(len(s)==5590 for s in sample)
    assert all({train[i]['family_id'] for i in s}==raws for s in [sample[0],sample[4]])
    assert all(train[i]['kind']=='original' for i in sample[0])
    paths=list(set(RUN.glob('*.py'))-{RUN/'foreground.py',RUN/'test_foreground.py'})
    paths+=list((RUN/'source_snapshot/oven_food').glob('*.py'))
    paths += [RUN/f for f in ('official_class_names.json','official_initialization.pt','scene_index.json','scene_audit.json','frozen_external_files.json')]
    paths += [folder/f for f in ('config.json','options.json','data_audit.json','training.json','raw_training.json','validation.json','test.json','weights.json','train_validation.csv','test_only.csv')]
    paths.append(RUN/'data'/str(size)/'audit.json')
    hashes={str(p.resolve()):sha(p) for p in paths}
    unique={}
    for r in train+val+test:
        if r['actual_input'] in unique:assert unique[r['actual_input']]['prepared_sha256']==r['prepared_sha256']
        unique[r['actual_input']]=r
        assert r['prepared_resolution']==size
        if r['kind']=='reviewed_food_geometry':
            assert r['cache_source']==r['synthetic_source_path'] and r['cache_source_sha256']==r['synthetic_source_sha256']
        else:
            assert r['cache_source_sha256']==r['family_id']
            if r['cache_source']!=r['source']:
                assert sha(r['cache_source'])==sha(r['source'])==r['family_id']
        if r['kind']=='annotation_roi':
            l,t,x,y=r['crop_xyxy'];w,h=r['cache_source_size']
            assert r['cache_crop_xyxy']==r['crop_xyxy'] and 0<=l<x<=w and 0<=t<y<=h
    def check(r):
        p=Path(r['actual_input'])
        assert p.is_relative_to(RUN/'data'/str(size)/'inputs'/r['split'])
        buf=p.read_bytes(); digest=hashlib.sha256(buf).hexdigest()
        assert digest==r['prepared_sha256'],p
        with Image.open(io.BytesIO(buf)) as im:
            assert im.size==(size,size) and im.mode=='RGB'
            assert hashlib.sha256(im.tobytes()).hexdigest()==r['prepared_pixel_sha256'],p
        return str(p),digest
    with ThreadPoolExecutor(6) as pool:hashes.update(pool.map(check,unique.values()))
    assert set((RUN/'data'/str(size)/'inputs').rglob('*.png'))=={Path(s) for s in unique}
    audit_record=dict(approved=True,errors=[],file_sha256=hashes,cache_unique_verified=len(unique),cross_split_overlap=overlap,
        train_views=dict(Counter(r['kind'] for r in train)),sampled_unique_families_first_five=[len({train[i]['family_id'] for i in s}) for s in sample],
        validation_counts=dict(Counter(r['label'] for r in val)),validation_missing_classes=sorted(set(classes)-{r['label'] for r in val}),
        raw_rows=len(raw),families=len(raws),validation_rows=len(val),test_rows=len(test),
        low_resolution_proxy_sources=sum(r['environment']=='proxy' and r['source_already_low_resolution'] for r in raw),
        isolation_scope='原图文件及其SHA、家族、旧像素、新缓存像素、已知实拍会话逐对零交叉。现有元数据无法证明不同会话中绝无同一件实物。',
        sampler='首轮每个家族选一个原图；每四轮轮换，其他轮按冻结家族权重有放回选择。两臂不使用场景均衡。',
        optimizer_ema='真实optimizer.step后更新独立AveragedModel；raw/EMA验证均inference_mode。epoch末在验证及调度器后保存末状态。',
        bn_and_selection='仅原始train家族按路径去重用于EMA副本BN。raw/EMA/EMA-BN仅按验证的非空类宏召回、最差召回、整体准确率、负交叉熵选取。普通best.pt仅归档。',
        previous_manifest_match=True,official_epoch=0,official_classes=classes,model_test_evaluation=False,
        external_freeze_entries=len(load(RUN/'frozen_external_files.json')),external_full_rehash_deferred_until_final=True)
    return audit_record


if __name__=='__main__':
    for name in sys.argv[1:] or ['A224','B320']:
        result=audit(name)
        result['source_rerender_sample']=load(RUN/(name+'_source_review.json'))
        result['ema_validation_rng']='EMA-only validation restores global CPU RNG via fork_rng(devices=[]); CUDA eval is deterministic.'
        dump(RUN/(name+'_review_evidence.json'),result)
        dump(RUN/name/'preflight.json',result)
        print(name,'checked',result['cache_unique_verified'],flush=True)


