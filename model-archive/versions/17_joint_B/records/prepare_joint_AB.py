"""冻结联合训练池及两套评测定义，不改写原图或已有模型。"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import shutil
import sys

from PIL import Image
import torch
from torchvision.models import MobileNet_V3_Small_Weights

ROOT = Path.cwd()
RUN = Path((ROOT / '.superpowers/latest_joint_AB.txt').read_text().strip())
BASE = ROOT / 'runs/C_pumpkin_croissant_20261118_160802'
sys.path.insert(0, str(BASE / 'source_snapshot'))
sys.path.insert(0, str(RUN))
from oven_food.augment import prepare_image
from oven_food.manifest import ManifestRecord, read_manifest, write_manifest, validate_manifest
from oven_food.model import build_model
from oven_food.config import load_experiment_config
from joint_support import hierarchical_weights


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(p):
    return str(Path(p).resolve()).casefold()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_row(r, split):
    return ManifestRecord(r['actual_input'], r['label'], r['environment'], r['source_group'], 'fixed', 'oven', 'none', 'none', split)


names = read(BASE / 'data_audit.json')['class_names']
old = read(ROOT / 'runs/C_pumpkin_supplement_20261107/inventory.json')
old_by_hash = {r['raw_sha256']: r for r in old}
old_beef = [r for r in old if r['label'] == '牛排']
assert len(old_beef) == 79
forbidden_beef_hashes = {r['raw_sha256'] for r in old_beef}
quality = read(RUN / 'quality_exclusions.json')
removed = {key(r['source']) for r in quality}
current = read(BASE / 'training_inventory.json')
extra = read(RUN / 'restored_candidates.json')
matched_removed = [r for r in current + extra if key(r['image']) in removed]
assert len(matched_removed) == 10
real_rows = []
current_hashes = {r['raw_sha256'] for r in current}


def real_row(r, restored):
    image = Path(r['image'])
    assert sha(image) == r['raw_sha256']
    assert r['raw_sha256'] not in forbidden_beef_hashes
    origin = old_by_hash.get(r['raw_sha256'], r)
    era = 'first_real' if r['raw_sha256'] in old_by_hash else 'second_real'
    assert not (r['label'] == '牛排' and era == 'first_real')
    if restored:
        cached = RUN / 'prepared_restored' / r['label'] / (r['raw_sha256'] + '.png')
        cached.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as im:
            prepare_image(im, image_size=224, resize_mode='letterbox').save(cached)
    else:
        cached = Path(r['prepared'])
        assert sha(cached) == r['prepared_sha256']
    physical = origin.get('physical_sample_id') or origin.get('sequence') or image.stem.rsplit('_',1)[0]
    session = origin['session']
    metadata = image.with_suffix('.json')
    return dict(label=r['label'], environment='target', split='train', era=era, source=str(image), actual_input=str(cached),
                raw_sha256=r['raw_sha256'], input_sha256=sha(cached), source_session=session,
                recorded_session=r['session'], source_group='real|' + r['label'] + '|' + session,
                sampling_group=session + '|' + physical, physical_sample_id=physical,
                metadata=str(metadata) if metadata.is_file() else '', metadata_sha256=sha(metadata) if metadata.is_file() else '',
                restored=restored, historical_source=origin['image'])


requests = [(r, False) for r in current if key(r['image']) not in removed]
requests += [(r, True) for r in extra if key(r['image']) not in removed]
with ThreadPoolExecutor(max_workers=4) as pool:
    real_rows = list(pool.map(lambda p: real_row(*p), requests))
assert len(real_rows) == 3990
full_web_manifest = ROOT / 'runs/C_retention_v2_20261029_231408/manifest.csv'
full_web = read_manifest(full_web_manifest)
web_train = [r for r in full_web if r.split == 'train' and r.environment == 'proxy']
excluded_web_beef = [r for r in web_train if r.food_label == '牛排']
assert len(web_train) == 1884 and len(excluded_web_beef) == 140
web_rows = []
for r in web_train:
    if r.food_label == '牛排':
        continue
    cached = Path(r.image_path)
    if not cached.is_absolute():
        cached = ROOT / 'runs/C_retention_v2_20261029_231408' / cached
    with Image.open(cached) as im:
        im.load()
        assert im.size == (224, 224)
    web_rows.append(dict(label=r.food_label, environment='proxy', split='train', era='web', source=str(cached),
                         actual_input=str(cached), raw_sha256='', input_sha256=sha(cached), source_session=r.session_id,
                         recorded_session=r.session_id, source_group='web|' + r.food_label + '|' + r.session_id,
                         sampling_group=r.session_id, physical_sample_id=r.session_id, metadata='', metadata_sha256='',
                         restored=False, historical_source=str(full_web_manifest)))
assert len(web_rows) == 1744
training_rows = real_rows + web_rows
assert len(training_rows) == 5734 and len({r['actual_input'] for r in training_rows}) == 5734
beef = [r for r in training_rows if r['label'] == '牛排']
assert len(beef) == 205 and all(r['environment'] == 'target' and r['era'] == 'second_real' for r in beef)
training = [manifest_row(r, 'train') for r in training_rows]

eval_inventory = read(BASE / 'eval_inventory.json')
assert len(eval_inventory) == 167
second_rows = []
for r in eval_inventory:
    assert sha(r['source']) == r['sha256'] and sha(r['prepared']) == r['prepared_sha256']
    second_rows.append(dict(label=r['label'], environment='target', split='test', era='second_eval', source=r['source'],
                            actual_input=r['prepared'], raw_sha256=r['sha256'], input_sha256=r['prepared_sha256'],
                            source_session=r['source_session'], source_group='real|' + r['label'] + '|' + r['source_session'],
                            original_half=r['split']))
val_A = [manifest_row(r, 'val') for r in second_rows if r['original_half'] == 'val']
test_A = [manifest_row(r, 'test') for r in second_rows if r['original_half'] == 'test']
test_B = [manifest_row(r, 'test') for r in second_rows]
first_snapshot = read(ROOT / 'runs/food_verify_20261107/input_snapshot.json')
first_rows = []
for r in first_snapshot:
    if r['split'] != 'validation' or r['label'] == '牛排':
        continue
    assert sha(r['image']) == r['sha256']
    first_rows.append(dict(label=r['label'], environment='target', split='val', era='first_validation', source=r['image'],
                           actual_input=r['prepared'], raw_sha256=r['sha256'], input_sha256=sha(r['prepared']),
                           source_session=r['session'], source_group='real|' + r['label'] + '|' + r['session'], original_half='val'))
val_B = [manifest_row(r, 'val') for r in first_rows]
assert tuple(map(len, [val_A, test_A, val_B, test_B])) == (83, 84, 193, 167)
assert not {r['raw_sha256'] for r in real_rows} & {r['raw_sha256'] for r in second_rows + first_rows}
assert not {r['input_sha256'] for r in training_rows} & {r['input_sha256'] for r in second_rows + first_rows}
assert not {r.source_group if hasattr(r,'source_group') else r.session_id for r in val_B} & {r.session_id for r in training}
validate_manifest(training + val_B, data_root=ROOT, check_files=True)
validate_manifest(val_A, data_root=ROOT, check_files=True)
shared_A = sorted({r.session_id for r in training} & {r.session_id for r in val_A})
assert shared_A == ['real|南瓜|session-20261117-2fc02ecc9c09']
for filename, rows in [('manifest.csv', training + val_B), ('training_manifest.csv', training), ('validation_A.csv', val_A),
                       ('validation_B.csv', val_B), ('test_A.csv', test_A), ('test_B.csv', test_B)]:
    write_manifest(RUN / filename, rows)
dump(RUN / 'training_inventory.json', training_rows)
dump(RUN / 'second_eval_inventory.json', second_rows)
dump(RUN / 'first_validation_inventory.json', first_rows)
csv_write(RUN / '训练逐图来源.csv', training_rows)
csv_write(RUN / '质量剔除清单.csv', quality)
dump(RUN / 'excluded_first_beef.json', old_beef)
dump(RUN / 'excluded_web_beef.json', [dict(image=r.image_path, label=r.food_label) for r in excluded_web_beef])

weights = hierarchical_weights(training_rows, target_share=.75)
dump(RUN / 'sampling_weights.json', weights)
mass = {n: sum(w for w,r in zip(weights,training_rows) if r['label']==n) for n in names}
assert all(abs(m-1/14) < 1e-10 for m in mass.values())
shutil.copytree(BASE / 'source_snapshot', RUN / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

# 官方骨干与固定14类随机新头，不读取任何历史训练权重作为初始化。
torch.set_num_threads(4)
torch.manual_seed(42)
model = build_model('mobilenet_v3_small', num_classes=14, pretrained=True)
official_path = Path(torch.hub.get_dir()) / 'checkpoints/mobilenet_v3_small-047dcff4.pth'
official_sha = sha(official_path)
assert official_sha == '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
official = torch.load(official_path, map_location='cpu', weights_only=True)
state = model.state_dict()
for k, v in official.items():
    if k not in {'classifier.3.weight','classifier.3.bias'}:
        assert torch.equal(v, state[k])
assert state['classifier.3.weight'].shape == (14, 1024)
initial = RUN / 'official_initialization.pt'
torch.save(dict(model_state=state, class_names=names, model_name='mobilenet_v3_small', image_size=224, epoch=0,
                preprocessing=dict(resize_mode='letterbox', mean=[.485,.456,.406], std=[.229,.224,.225]),
                initialization='official ImageNet pretrained backbone plus new random 14-class head', official_sha256=official_sha), initial)
dump(RUN / 'initialization_audit.json', dict(official_url=MobileNet_V3_Small_Weights.DEFAULT.url, official_sha256=official_sha,
                                            initial_sha256=sha(initial), canonical_class_names=names, initialized_from_previous_model=False))
cfg = read(BASE / 'train_config.json')
cfg.update(experiment_name='joint_AB_shared_training', manifest=str(RUN/'manifest.csv'), output_dir=str(RUN/'selection_B'),
           checkpoint=str(initial), pretrained=False, epochs=20, learning_rate=.0003, target_fraction=.75, class_balanced_loss=False)
dump(RUN / 'train_config.json', cfg)
load_experiment_config(RUN / 'train_config.json')
counts = [dict(食材=n, 第一批实拍=sum(r['label']==n and r['era']=='first_real' for r in training_rows),
               第二批实拍=sum(r['label']==n and r['era']=='second_real' for r in training_rows),
               网图=sum(r['label']==n and r['environment']=='proxy' for r in training_rows),
               训练合计=sum(r['label']==n for r in training_rows), A验证=sum(r.food_label==n for r in val_A),
               A测试=sum(r.food_label==n for r in test_A), B验证=sum(r.food_label==n for r in val_B), B测试=sum(r.food_label==n for r in test_B)) for n in names]
csv_write(RUN / '每类数量.csv', counts)
audit = dict(training_images=5734, real_images=3990, historical_web_images=1744, removed_quality=10,
             excluded_first_beef_train=79, excluded_first_beef_validation=21, excluded_web_beef=140, beef_only_second_real=205,
             restored_real_images=sum(r['restored'] for r in real_rows), class_names=names, per_class=counts,
             validation_A=83, test_A=84, validation_B=193, test_B=167, shared_training_A_validation_groups=shared_A,
             shared_training_B_validation_groups=[], independent_test=False, canonical_mapping=True,
             previous_model_not_initialization=True, initial_sha256=sha(initial), official_sha256=official_sha,
             first_epoch_full_coverage=True, remaining_epochs_weighted=19, target_share_with_both_domains=.75,
             raw_train_eval_exact_overlap=0, cached_train_eval_exact_overlap=0,
             quality_review_scope='新增1189张历史图逐张缩略图浏览，29张疑点放大；当前已选数据沿用前轮质量清理并复核已知薯条遮挡。未宣称全部原尺寸逐图人工验收。')
dump(RUN / 'data_audit.json', audit)
print(json.dumps(dict(run=str(RUN), training=5734, real=3990, web=1744, beef_only_second_real=205, restored=audit['restored_real_images'], A=[83,84], B=[193,167], init_sha256=sha(initial)), ensure_ascii=False), flush=True)
