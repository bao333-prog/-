"""按独立开关生成实验配置；同一来源的派生数量不增加其总权重。"""
import argparse
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from support import RUN, ROOT, PREVIOUS, load, dump, sha
from data_ops import cache_row

sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.manifest import ManifestRecord, write_manifest, validate_manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A224','B320','C320'],required=True);args=parser.parse_args()
    name=args.arm;size=224 if name=='A224' else 320;directory=RUN/name;directory.mkdir(exist_ok=True)
    rows=load(RUN/'data'/str(size)/'integrated.json');weights=load(PREVIOUS/'sampling_weights.json')
    assert len(rows)==len(weights)==7696
    options=dict(arm=name,input_size=size,ema_decay=.99,scene_balanced=name=='C320',
       foreground_rotation=name=='C320',fixed_target_background=name=='C320',
       selection='macro_min_overall_negative_loss',nominal_epochs=20,maximum_epochs=60)
    if name=='C320':
        approved=load(RUN/'foreground_reviewed.json')
        assert approved['root_visual_review_complete']
        by_family={r['family_id']:r for r in load(RUN/'data'/str(size)/'training.json')}
        additions=[]
        for s in approved['sources']:
            assert s['review_status']=='approved' and s['split']=='train'
            parent=by_family[s['source_sha256']];assert parent['label']==s['label']
            row={**parent,'kind':'foreground_rotation','foreground_source_path':s['output'],
                 'foreground_source_sha256':s['output_sha256'],'foreground_evidence':s,
                 'actual_input':s['output']}
            additions.append(cache_row(row,size))
        additions_by_family=Counter(r['family_id'] for r in additions)
        old_mass=defaultdict(float)
        for r,w in zip(rows,weights):old_mass[r['family_id']]+=w
        weights=[w*(.9 if r['family_id'] in additions_by_family else 1.) for r,w in zip(rows,weights)]
        weights += [old_mass[r['family_id']]*.1/additions_by_family[r['family_id']] for r in additions]
        rows+=additions
        options['foreground_review_sha256']=sha(RUN/'foreground_reviewed.json')
        options['foreground_views']=len(additions)
    assert math.isclose(sum(weights),1.,abs_tol=1e-10) and min(weights)>0
    val=load(RUN/'data'/str(size)/'validation.json');test=load(RUN/'data'/str(size)/'test.json')
    for split,rs in [('training',rows),('validation',val),('test',test),('raw_training',load(RUN/'data'/str(size)/'training.json'))]:
        dump(directory/(split+'.json'),rs)
    def record(r):return ManifestRecord(image_path=r['actual_input'],food_label=r['label'],environment=r['environment'],
          session_id=r['source_group'],camera_position='fixed_oven_camera',lighting_condition='fixed',
          steam_level='none',occlusion_level='none',split=r['split'])
    for filename,rs in [('train_validation.csv',rows+val),('test_only.csv',test)]:
        records=[record(r) for r in rs];validate_manifest(records,check_files=True);write_manifest(directory/filename,records)
    config=load(PREVIOUS/'train_config.json')
    config.update(experiment_name='geometry_upgrade_'+name,manifest=str(directory/'train_validation.csv'),
          output_dir=str(directory/'training'),checkpoint=str(RUN/'official_initialization.pt'),
          image_size=size,augmentation='oven_preserve')
    dump(directory/'config.json',config);dump(directory/'options.json',options);dump(directory/'weights.json',weights)
    dump(directory/'data_audit.json',dict(raw_rows=5591,train_views=len(rows),source_families=5590,
       validation=193,test=167,view_counts=dict(Counter(r['kind'] for r in rows)),options=options))
    print(name,len(rows),'train views prepared',flush=True)


if __name__=='__main__':main()
