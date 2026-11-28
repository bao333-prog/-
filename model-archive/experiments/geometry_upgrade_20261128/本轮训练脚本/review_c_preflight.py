"""Independent C-only preflight; run on CPU. Revalidate the B fingerprints inherited by C."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
from support import RUN,load,dump,sha
from scene_sampler import SceneBalancedSampler,BRANCH_FIELDS
sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.augment import prepare_image

C,B=RUN/'C320',RUN/'B320'
lock={}
def check(p,expected=None):
    p=Path(p).resolve();digest=sha(p)
    assert expected is None or digest==expected,p
    return str(p),digest
def freeze(p,expected=None):
    k,v=check(p,expected);lock[k]=v

b_audit=load(B/'preflight.json')
assert b_audit['approved'] and not b_audit['errors']
# The B-specific manifest/config hashes are separately replaced with C hashes below.
common=[(p,v) for p,v in b_audit['file_sha256'].items() if not Path(p).is_relative_to(B)]
with ThreadPoolExecutor(4) as pool: lock.update(pool.map(lambda pv:check(*pv),common))
for p in [*RUN.glob('*.py'),*C.glob('*.csv'),*(C/k for k in ('config.json','options.json','weights.json','training.json','validation.json','test.json','raw_training.json','data_audit.json'))]:
    freeze(p)
for p in ('foreground_reviewed.json','foreground_manifest.json','foreground_visual_review.json','foreground_frozen_external.json','foreground_rejected_manifest.json','foreground_verification.json'):
    freeze(RUN/p)
for p,d in load(RUN/'foreground_frozen_external.json').items():freeze(p,d)
cfg=load(C/'config.json');opts=load(C/'options.json');bcfg=load(B/'config.json')
assert all(cfg[k]==bcfg[k] for k in bcfg if k not in ('manifest','experiment_name','output_dir'))
assert Path(cfg['manifest'])==C/'train_validation.csv' and Path(cfg['output_dir'])==C/'training'
assert opts['fixed_target_background'] and opts['scene_balanced'] and opts['foreground_rotation'] and opts['input_size']==320
for n in ('raw_training.json','validation.json','test.json','test_only.csv'):assert sha(C/n)==sha(B/n)
base=load(B/'training.json');train=load(C/'training.json');assert train[:len(base)]==base and len(train)==7755
fg=train[len(base):]; parent={r['family_id']:r for r in load(C/'raw_training.json')}
w=load(C/'weights.json');wb=load(B/'weights.json');assert len(w)==len(train) and len(wb)==len(base) and min(w)>0
review=load(RUN/'foreground_reviewed.json')
assert opts['foreground_review_sha256']==sha(RUN/'foreground_reviewed.json') and review['root_visual_review_complete'] and not review['user_review_explicit']
heldout=load(C/'validation.json')+load(C/'test.json')
held_ids={r['family_id'] for r in heldout};held_pixels={r['prepared_pixel_sha256'] for r in heldout}
held_sessions={r['source_session'] for r in heldout if r['environment']=='target'}
assert len(fg)==len(review['sources'])==len({r['family_id'] for r in fg})==59
ids={r['family_id'] for r in fg}
assert not held_ids&ids and ids<=set(parent)
assert set(r['actual_input'] for r in fg).isdisjoint(r['actual_input'] for r in base)
for r,s in zip(fg,review['sources']):
    family=r['family_id'];e=s['evidence']
    assert s['training_eligible'] and s['review_status']=='approved'
    assert r['kind']=='foreground_rotation' and r['environment']=='target' and r['split']==s['split']=='train'
    assert s==r['foreground_evidence'] and s['label']==parent[family]['label']==r['label']
    assert s['source_sha256']==family==sha(parent[family]['source'])
    assert e['mask_usage']=='full_food_only' and e['mask_status']=='verified' and e['scale']==1 and e['food_photometric_transform']=='none'
    assert r['cache_source']==r['foreground_source_path']==s['output'] and r['cache_source_sha256']==s['output_sha256']
    assert r['source_session']==parent[family]['source_session'] and r['source_session'] not in held_sessions
    assert r['prepared_pixel_sha256'] not in held_pixels
    for k,h in [(s['source_path'],family),(s['mask_path'],s['mask_sha256']),(s['output'],s['output_sha256']),
                (e['erase_mask_path'],e['erase_mask_sha256']),(e['final_contact_sheet'],e['final_contact_sheet_sha256']),(r['actual_input'],r['prepared_sha256'])]:freeze(k,h)
    src=np.array(Image.open(s['source_path']).convert('RGB'));pic=Image.open(s['output']).convert('RGB')
    mask=np.array(Image.open(s['mask_path']).convert('L'))>0
    erased=np.array(Image.open(e['erase_mask_path']).convert('L'))>0
    assert list(pic.size)==e['output_image_size'] and list(pic.size)==r['cache_source_size'] and mask.shape==src.shape[:2]==erased.shape
    assert np.all(erased[mask]) and not np.any(mask[[0,-1],:]) and not np.any(mask[:,[0,-1]])
    yy,xx=np.where(mask);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
    assert [cx,cy]==e['center']
    rad=np.radians(s['angle']);dx=xx+.5-cx;dy=yy+.5-cy
    px=cx+np.cos(rad)*dx+np.sin(rad)*dy;py=cy-np.sin(rad)*dx+np.cos(rad)*dy
    assert min(px.min(),py.min())>=2 and px.max()<=pic.width-2 and py.max()<=pic.height-2
    alpha=Image.fromarray(mask.astype(np.uint8)*255).rotate(s['angle'],Image.Resampling.BICUBIC,center=tuple(e['center']))
    diff=np.any(np.asarray(pic)!=src,axis=2)
    assert not np.any(diff&~(erased|(np.asarray(alpha)>0))) and diff.sum()==e['changed_pixels']
    assert np.array_equal(np.asarray(prepare_image(pic,image_size=320,resize_mode='letterbox')),np.asarray(Image.open(r['actual_input'])))
    assert hashlib.sha256(Image.open(r['actual_input']).convert('RGB').tobytes()).hexdigest()==r['prepared_pixel_sha256']
with (C/'train_validation.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
assert [(r['image_path'],r['split'],r['food_label'],r['environment'],r['session_id']) for r in manifest]==[(r['actual_input'],r['split'],r['label'],r['environment'],r['source_group']) for r in train+load(C/'validation.json')]
bm=defaultdict(float);cm=defaultdict(float);branches=defaultdict(float)
for r,a,b in zip(base,wb,w):
    fid=r['family_id'];bm[fid]+=a;branches[tuple(r[k] for k in BRANCH_FIELDS)]+=a
    assert math.isclose(b,a*(.9 if fid in ids else 1.),rel_tol=1e-12)
for r,b in zip(train,w):cm[r['family_id']]+=b
for r,b in zip(fg,w[len(base):]):assert math.isclose(b,bm[r['family_id']]*.1,rel_tol=1e-12)
assert all(math.isclose(bm[k],cm[k],rel_tol=1e-12) for k in bm)
scene=load(RUN/'scene_index.json')['family_to_scene'];sampler=SceneBalancedSampler(train,w,scene,seed=42)
assert set(scene)==set(parent) and set(sampler.branch_probabilities)==set(branches)
assert all(math.isclose(sampler.branch_probabilities[k],branches[k],rel_tol=1e-12) for k in branches)
for k,sg in sampler.branch_scenes.items():assert all(math.isclose(sampler.scene_probabilities[s],branches[k]/len(sg),rel_tol=1e-12) for s in sg)
cycles={};read_counts=Counter()
for e in range(1,21):
    draws=list(sampler)
    assert len(draws)==len(parent)
    if e==1:
        assert all(train[i]['kind']=='original' for i in draws)
        assert {train[i]['family_id'] for i in draws}==set(parent)
    else:
        for i in draws:
            fid=train[i]['family_id'];sid=scene[fid];seen=cycles.setdefault(sid,set());assert fid not in seen
            seen.add(fid)
            if len(seen)==len(sampler.scenes[sid]['families']):seen.clear()
    read_counts.update(train[i]['actual_input'] for i in draws if train[i]['kind']=='foreground_rotation')
out=dict(approved=True,errors=[],file_sha256=lock,train_views=len(train),training_families=len(parent),scene_count=len(sampler.scenes),
    hierarchy_branch_count=len(branches),branch_probability_preserved=True,scene_equal_probability_within_branch=True,
    first_epoch_all_original_families=True,actual_sampler_twenty_epoch_simulation=dict(draws_per_epoch=len(parent),foreground_draws=sum(read_counts.values()),distinct_foreground_drawn=len(read_counts),scope='CPU sampler simulation only; actual training counts must come from sample_counts.json'),
    foreground_rows=len(fg),source_mask_erase_output_and_cached_pixels_checked=len(fg),background_changes_outside_erased_or_rotated_mask=0,
    number_of_final_visual_pages_reviewed_by_root=len(review['reviewed_contact_pages']),explicit_user_per_image_review=False,heldout_family_pixel_and_known_session_overlaps=0,
    bn='Original train only, identical to B320',training_behavior='All target rows use baseline letterbox and ImageNet normalization. Proxy rows use oven_preserve. Both routes increment counts at actual __getitem__ reads.',
    analysis_limit='Combined foreground views, fixed target camera geometry and scene balancing. Individual effect cannot be isolated. No physical-individual independence claim.',
    heldout_results_read=False,model_test_evaluation=False)
dump(C/'preflight.json',out)
print('C320 approved',len(lock),'fingerprints',len(read_counts),'distinct foreground simulated',sum(read_counts.values()),'draws')
