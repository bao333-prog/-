"""Read-only dataset/initialization review. Only write the review result."""
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps
import torch

run = Path(__file__).resolve().parents[1]
old = run.parent / 'integrated_food_20261126_115154'
sys.path.insert(0, str(run))
from continuation import should_stop


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def pixel(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def original_pixel(path):
    with Image.open(path) as image:
        return pixel(image.convert('RGB'))


def audit(report):
    base, views, val, test = [read(run / name) for name in (
        'training_base.json', 'training_integrated.json', 'validation_base.json', 'test_base.json')]
    added = base[-121:]
    inventory = read(run / 'new_inventory_approved.json')
    assert len(added) == len(inventory) == 121 and views[-121:] == added
    assert base[:-121] == read(old / 'training_base.json')
    assert views[:-121] == read(old / 'training_integrated.json')
    assert val == read(old / 'validation_base.json') and test == read(old / 'test_base.json')
    counts = Counter(r['kind'] for r in views)
    assert counts == {'original':5591,'annotation_roi':2011,'reviewed_food_geometry':94}
    assert len(val)==193 and len(test)==167 and len({r['family_id'] for r in base})==5590
    trainval = csv_rows(run / 'train_validation.csv')
    original_csv = csv_rows(old / 'train_validation.csv')
    assert trainval[:len(views)-121] + trainval[len(views):] == original_csv
    assert sha(run/'test_only.csv') == sha(old/'test_only.csv')
    for rec, row in zip(trainval, views + val, strict=True):
        assert (rec['image_path'],rec['split'],rec['food_label']) == (row['actual_input'],row['split'],row['label'])
    assert {r['split'] for r in views}=={'train'} and {r['split'] for r in val}=={'val'}
    for x,y in combinations([views,val,test],2):
        for key in ['family_id','source','actual_input','prepared_pixel_sha256']:
            assert {r[key] for r in x}.isdisjoint(r[key] for r in y), key
        sessions = lambda rows: {r['source_session'] for r in rows if r['environment']=='target'}
        assert sessions(x).isdisjoint(sessions(y))
    with ThreadPoolExecutor(4) as pool:
        held_pixels=set(pool.map(original_pixel,[r['source'] for r in val+test]))
    held_paths={r['source'] for r in val+test}
    preview=read(run/'expected_sampling_counts_20.json')
    assert set(preview)=={r['actual_input'] for r in views} and sum(preview.values()) == 20*5590
    weights=read(run/'sampling_weights.json')
    assert len(weights)==len(views) and min(weights)>0 and math.isclose(sum(weights),1)
    for row,item in zip(added,inventory,strict=True):
        assert row['label']=='芦笋' and row['kind']=='original'
        assert row['source']==item['source'] and row['family_id']==sha(row['source'])==item['sha256']
        assert sha(row['metadata'])==item['metadata_sha256']
        meta=read(Path(row['metadata']))
        assert meta['sha256']==item['sha256'] and meta['camera_context']['session_id']==row['source_session']
        assert row['prepared_sha256']==sha(row['actual_input']) and preview[row['actual_input']]>=5
        assert item['visual_status']=='approved' and len(item['nearest_eval'])==3
        assert all(e['source'] in held_paths for e in item['nearest_eval'])
        with Image.open(row['source']) as image, Image.open(row['actual_input']) as cached:
            rgb=ImageOps.exif_transpose(image).convert('RGB')
            assert pixel(rgb)==item['full_rgb_sha256'] and pixel(rgb) not in held_pixels
            assert rgb.size==(1920,1080) and cached.size==(224,224)
            expected=Image.new('RGB',(224,224),'white')
            expected.paste(rgb.resize((224,126),Image.Resampling.BILINEAR),(0,49))
            assert expected.tobytes()==cached.convert('RGB').tobytes()
    assert len({r['full_rgb_sha256'] for r in inventory})==121
    assert not {r['family_id'] for r in added} & {r['family_id'] for r in base[:-121]}
    config=read(run/'train_config.json')
    assert Path(config['checkpoint'])==run/'official_initialization.pt' and Path(config['manifest'])==run/'train_validation.csv'
    assert config['device']=='cuda:0' and config['epochs']==20 and config['seed']==42 and config['num_workers']==0
    init=sha(run/'official_initialization.pt')
    assert init==sha(old/'official_initialization.pt')=='5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    frozen=read(run/'frozen_external_files.json')
    previous=old/'training/best.pt'
    assert len(frozen)==17852 and frozen[str(previous)]==sha(previous)!=init
    for r in views+val+test:
        assert frozen[r['actual_input']]==r['prepared_sha256']
    for r in base+val+test:
        assert frozen[r['source']]==r['family_id']
        if r.get('metadata'): assert frozen[r['metadata']]==r['metadata_sha256']
    ck=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    names=read(run/'official_class_names.json')
    assert ck['epoch']==0 and ck['class_names']==names and len(names)==14 and set(names)=={r['label'] for r in base}
    suite=[.95]*17+[.90,.91,.92]
    for extra,stop in [([],False),([.92],False),([.92]*2,False),([.92]*3,True)]:
        assert should_stop([{'validation_accuracy':v} for v in suite+extra])==stop,'extension patience'
    report.update(train_rows=len(views),train_families=5590,added=121,validation=len(val),test=len(test),
                  view_counts=dict(counts),old_views_and_splits_identical=True,
                  train_plus_validation_csv_checked=True,heldout_full_rgb_checked=360,
                  new_geometry='1920x1080 -> 224x126, centered in 224x224 at (0,49), all 121 matched bytewise',
                  new_sessions=dict(Counter(r['source_session'] for r in added)),
                  session_also_in_old_training=bool({r['source_session'] for r in added}&{r['source_session'] for r in base[:-121]}),
                  split_source_session_cached_pixel_overlap=0,new_vs_heldout_raw_rgb_overlap=0,
                  nearest_reviewed_by_lead=12,physical_independence_proven=False,
                  initialization_sha256=init,initialization_epoch=0,frozen_inventory_entries=len(frozen),
                  min_new_preflight_draws_in_20_epochs=min(preview[r['actual_input']] for r in added))


if __name__=='__main__':
    names='prepare_data.py prepare_update.py train_integrated.py evaluate_final.py continuation.py family_sampler.py joint_support.py validation_monitor.py train_config.json official_initialization.pt official_class_names.json training_base.json validation_base.json test_base.json training_integrated.json train_validation.csv test_only.csv sampling_weights.json frozen_external_files.json new_inventory_approved.json inspect_new.py'.split()
    names += sorted(p.relative_to(run).as_posix() for p in (run/'source_snapshot/oven_food').rglob('*.py'))
    report=dict(approved=False,errors=[],file_sha256={n:sha(run/n) for n in names})
    try:
        audit(report)
        assert all(sha(run/n)==value for n,value in report['file_sha256'].items())
        report['approved']=True
    except Exception as exc:
        import traceback
        report['errors'].append(traceback.format_exc())
    dest=run/'integration_preflight_review.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report['approved'],report['errors'])
    sys.exit(0 if report['approved'] else 1)
