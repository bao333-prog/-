"""仅添加漏掉的121张训练原图；重用已冻结的旧缓存，原始照片一律只读。"""
import hashlib
import shutil
import sys
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from inspect_new import RUN, ROOT, PREVIOUS, load, dump, sha


def main():
    # 复用旧版对应的源码和官方初始化，不继承旧版任务训练后的权重。
    shutil.copytree(PREVIOUS/'source_snapshot', RUN/'source_snapshot',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['prepare_data.py','family_sampler.py','test_family_sampler.py','joint_support.py',
                 'validation_monitor.py','predict.py','evaluate_final.py','official_initialization.pt',
                 'official_class_names.json']:
        shutil.copy2(PREVIOUS/name,RUN/name)
    sys.path.insert(0,str(RUN/'source_snapshot'))
    from oven_food.augment import prepare_image
    from oven_food.manifest import ManifestRecord, validate_manifest, write_manifest
    from joint_support import hierarchical_weights
    from family_sampler import FamilyCoverageSampler

    source_code=RUN/'source_snapshot/oven_food/train.py'
    code=source_code.read_text(encoding='utf-8')
    old='for epoch in range(1, config.epochs + 1):'
    assert code.count(old)==1
    code=code.replace(old,"for epoch in range(1, globals().get('_maximum_epochs', config.epochs) + 1):")
    old='math.cos(math.pi * epoch / config.epochs)'
    assert code.count(old)==1
    code=code.replace(old,'math.cos(math.pi * min(epoch, config.epochs) / config.epochs)')
    old='        if scheduler is not None:\n            scheduler.step()\n'
    assert code.count(old)==1
    code=code.replace(old, old+"        # 本次私有实验钩子：保存最后状态并按验证曲线决定是否延长，不访问测试集。\n        hook = globals().get('_epoch_complete')\n        if hook is not None and hook(locals()):\n            break\n")
    source_code.write_text(code,encoding='utf-8')

    base={s:load(PREVIOUS/f'{s}_base.json') for s in ['training','validation','test']}
    combined=load(PREVIOUS/'training_integrated.json')
    new=load(RUN/'new_inventory_pending.json')
    assert len(new)==121 and not any(r['existing_file_match'] for r in new)
    approved=[]
    for r in new:
        assert sha(r['source'])==r['sha256'] and sha(r['metadata'])==r['metadata_sha256']
        assert not r['held_out_rgb_overlap'] and not r['held_out_session_overlap']
        r.update(visual_status='approved',visual_review='root viewed all 121 on contact_01..07; no obvious hand/tool occlusion; small/dark but recognizable asparagus')
        sid=r['sha256']; session=r['source_session']; prefix=Path(r['source']).stem.rsplit('_',1)[0]
        target=RUN/'inputs/train/芦笋'/f'{sid}_full.png'; target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(r['source']) as image:
            prepared=prepare_image(image,image_size=224,resize_mode='letterbox'); prepared.save(target)
        approved.append(dict(label='芦笋',environment='target',split='train',era='second_real',
            source=r['source'],actual_input=str(target),raw_sha256=sid,input_sha256=sha(target),
            source_session=session,recorded_session=session,source_group='real|芦笋|'+session,
            sampling_group=session+'|'+prefix,physical_sample_id=r['physical_sample_id'],
            metadata=r['metadata'],metadata_sha256=r['metadata_sha256'],restored=False,
            historical_source=r['source'],pixel_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),
            family_id=sid,kind='original',previous_actual_input=None,prepared_sha256=sha(target),
            prepared_pixel_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),
            added_this_run=True,capture_level=prefix.rsplit('-',1)[-1],
            visual_review=r['visual_review']))
    dump(RUN/'new_inventory_approved.json',new)
    base['training']+=approved; combined+=approved
    for split,rs in base.items(): dump(RUN/f'{split}_base.json',rs)
    dump(RUN/'training_integrated.json',combined)

    # 用文件指纹和固定224输入像素再次阻止新来源或派生视图跨集合。
    for a,b in [('training','validation'),('training','test'),('validation','test')]:
        for field in ['family_id','prepared_pixel_sha256']:
            assert not ({r[field] for r in base[a]} & {r[field] for r in base[b]}),(a,b,field)
        sessions=lambda rs:{r['source_session'] for r in rs if r['environment']=='target'}
        assert not sessions(base[a]) & sessions(base[b]),(a,b,'session')
    held_pixels={r['prepared_pixel_sha256'] for r in base['validation']+base['test']}
    assert not {r['prepared_pixel_sha256'] for r in combined} & held_pixels
    assert len(combined)==7696 and len(base['training'])==5591
    assert len({r['family_id'] for r in base['training']})==5590
    # 旧训练视图和评测清单逐项不变，只在尾部增加这批原图。
    assert combined[:-121]==load(PREVIOUS/'training_integrated.json')
    for split in ['validation','test']: assert base[split]==load(PREVIOUS/f'{split}_base.json')

    def record(r):
        return ManifestRecord(image_path=r['actual_input'],food_label=r['label'],environment=r['environment'],
          session_id=r['source_group'],camera_position='fixed_oven_camera',lighting_condition='fixed',
          steam_level='none',occlusion_level='none',split=r['split'])
    for filename,rs in [('train_validation.csv',combined+base['validation']),('test_only.csv',base['test'])]:
        manifest=[record(r) for r in rs];validate_manifest(manifest,check_files=True);write_manifest(RUN/filename,manifest)
    raw_weights=hierarchical_weights(base['training'],target_share=.75,max_expected_draws=4.)
    family_mass=defaultdict(float)
    for r,w in zip(base['training'],raw_weights): family_mass[r['family_id']]+=w
    family_kinds=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(combined):family_kinds[r['family_id']][r['kind']].append(i)
    weights=[0.]*len(combined)
    for family,groups in family_kinds.items():
        shares={'annotation_roi':.25 if 'annotation_roi' in groups else 0.,'reviewed_food_geometry':.10 if 'reviewed_food_geometry' in groups else 0.}
        shares['original']=1-sum(shares.values())
        for kind,indices in groups.items():
            for i in indices:weights[i]=family_mass[family]*shares[kind]/len(indices)
        assert math.isclose(sum(weights[i] for indices in groups.values() for i in indices),family_mass[family],abs_tol=1e-14)
    assert math.isclose(sum(weights),1.,abs_tol=1e-10) and min(weights)>0
    dump(RUN/'sampling_weights.json',weights)
    preview=Counter();sampler=FamilyCoverageSampler(combined,weights,seed=42)
    for epoch in range(1,61):
        preview.update(combined[i]['actual_input'] for i in sampler)
        if epoch in [20,60]:dump(RUN/f'expected_sampling_counts_{epoch}.json',dict(preview))
    config=load(PREVIOUS/'train_config.json')
    config.update(experiment_name=RUN.name,manifest=str(RUN/'train_validation.csv'),
                  output_dir=str(RUN/'training'),checkpoint=str(RUN/'official_initialization.pt'))
    dump(RUN/'train_config.json',config)

    # 冻结所有实际读取的输入以及原图/侧车文件，训练后再次逐文件验证。
    checks={}
    for r in combined+base['validation']+base['test']:
        checks[r['actual_input']]=r['prepared_sha256']
    for rs in base.values():
        for r in rs:
            checks[r['source']]=r['family_id']
            if r.get('metadata'): checks[r['metadata']]=r['metadata_sha256']
    checks[str(PREVIOUS/'training/best.pt')]=sha(PREVIOUS/'training/best.pt')
    def check(item):
        p,digest=item
        if sha(p)!=digest:raise ValueError('文件指纹不一致：'+p)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,checks.items()))
    dump(RUN/'frozen_external_files.json',checks)
    report={'added_raw_images':121,'old_raw_rows':5470,'new_raw_rows':5591,'unique_source_families':5590,
       'integrated_train_views':len(combined),'view_counts':dict(Counter(r['kind'] for r in combined)),
       'raw_per_class':dict(Counter(r['label'] for r in base['training'])),
       'roi_per_class':dict(Counter(r['label'] for r in combined if r['kind']=='annotation_roi')),
       'validation_originals':193,'test_originals':167,'cross_split_file_pixel_known_session_overlap':0,
       'old_views_and_eval_unchanged':True,'official_initialization_sha256':sha(RUN/'official_initialization.pt'),
       'inherited_task_trained_weights':False,'light_or_color_augmentation':False,
       'source_files_frozen':len(checks),'new_rows_share_single_training_session':True,
       'physical_individual_independence_proven':False,'errors':[]}
    dump(RUN/'prepared_data_audit.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
