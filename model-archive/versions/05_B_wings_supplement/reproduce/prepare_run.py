"""准备十类鸡翅补强；原图只读，同源关联贯通旧session与新增来源组。"""
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import shutil

from PIL import Image
from oven_food.augment import prepare_image
from oven_food.manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT/'runs/increment_B_20261025_165256'
PARENT = ROOT/'runs/B_augmentation_trial_20261026_123837'
REVIEW = ROOT/'runs/chicken_wings_review_20261029_171649'
old = read_manifest(PREVIOUS/'manifest.csv')
old_reviews = json.loads((ROOT/'runs/formal_A_20261025_113459/review_and_split.json').read_text(encoding='utf-8'))
old_wings = {r['id']:r for r in old_reviews if r['class']=='鸡翅' and r['included']}
new = json.loads((REVIEW/'reviewed_inventory.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 并查集合并历史session及新来源关系，不能仅检查路径是否重复。
parents = {f'o{key}':f'o{key}' for key in old_wings}
parents.update({f'n{r["index"]}':f'n{r["index"]}' for r in new})


def find(key):
    while parents[key]!=key:
        parents[key]=parents[parents[key]]
        key=parents[key]
    return key


def union(a,b):
    parents[find(b)]=find(a)


old_groups = defaultdict(list)
for key,row in old_wings.items():
    old_groups[row['group']].append('o'+key)
new_groups = defaultdict(list)
for row in new:
    new_groups[row['group_id']].append('n'+str(row['index']))
for group in list(old_groups.values())+list(new_groups.values()):
    for key in group[1:]:
        union(group[0],key)
for row in new:
    for match in row['old_confirmed_matches']:
        union('n'+str(row['index']),'o'+match['old_id'])
union('n76','o鸡翅_072')  # 已复核并记录的保守同源关联。
known_holdout_groups = {find('o'+key) for key,row in old_wings.items() if row['split'] in {'val','test'}}
removed = [row for key,row in old_wings.items() if row['split']=='train' and find('o'+key) in known_holdout_groups]
removed_paths = {r['prepared_path'] for r in removed}
assert len(removed)==7
by_prepared = {r['prepared_path']:key for key,r in old_wings.items()}
records = []
for record in old:
    if record.split=='test' or record.image_path in removed_paths:
        continue
    if record.food_label=='鸡翅':
        record=replace(record,session_id='supp_wings_'+find('o'+by_prepared[record.image_path]))
    records.append(record)
frozen_test = []
new_used = []
for row in new:
    split = row['proposed_split']
    if not split:
        continue
    assert sha(Path(row['path']))==row['sha256']
    group = 'supp_wings_'+find('n'+str(row['index']))
    if split=='train':
        assert find('n'+str(row['index'])) not in known_holdout_groups, row['filename']
    if split=='test':
        frozen_test.append(ManifestRecord(row['path'],'鸡翅','proxy',group,'unknown_web','unknown_web','none','none','test'))
        continue
    destination = OUT/'prepared_new_wings'/f"{row['index']:03d}.png"
    destination.parent.mkdir(exist_ok=True)
    with Image.open(row['path']) as image:
        prepare_image(image,image_size=224,resize_mode='letterbox').save(destination)
    records.append(ManifestRecord(str(destination),'鸡翅','proxy',group,'unknown_web','unknown_web','none','none',split))
    new_used.append(dict(index=row['index'],original_path=row['path'],prepared_path=str(destination),split=split,group=group,sha256=row['sha256']))
validate_manifest(records,check_files=True)
validate_manifest(records+frozen_test,check_files=True)
assert Counter(r.split for r in records)=={'train':1355,'val':293}
assert Counter(r['split'] for r in new_used)=={'train':83,'val':18}
assert len(frozen_test)==18
assert not {r.image_path for r in records} & {r.image_path for r in old if r.split=='test'}
assert not {r.session_id for r in records} & {r.session_id for r in frozen_test}
counts = Counter(r.food_label for r in records if r.split=='train')
assert len(counts)==10 and counts['鸡翅']==134
write_manifest(OUT/'manifest.csv',records)
write_manifest(OUT/'frozen_new_chicken_test.csv',frozen_test)
(OUT/'training').mkdir()
write_manifest(OUT/'training/training_manifest.csv',records)

config = json.loads((PARENT/'train_config.json').read_text(encoding='utf-8'))
config.update(experiment_name='ten_class_B_chicken_supplement',manifest=str(OUT/'manifest.csv'),data_root=str(OUT),
              output_dir=str(OUT/'training'),checkpoint=str(PARENT/'training/best.pt'),new_classes=[],
              epochs=20,learning_rate=.00005,augmentation='gentle_plus',seed=42)
(OUT/'train_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
protected = [ROOT/'runs/formal_A_20261025_113459/training/best.pt',PREVIOUS/'training/best.pt',PARENT/'training/best.pt']
snapshot = dict(stage='ten_class_same_labels_chicken_supplement',parent_checkpoint=str(protected[-1]),
                parent_sha256=sha(protected[-1]),protected_models={str(p):sha(p) for p in protected},
                training_count=1355,validation_count=293,fresh_chicken_test_count=18,
                training_counts=dict(counts),new_train_count=83,new_validation_count=18,
                removed_old_train=[dict(id=r['id'],path=r['prepared_path'],reason='新增来源关系和旧session贯通后关联旧验证/测试；本轮隔离') for r in removed],
                old_prepared_sha256={r.image_path:sha(Path(r.image_path)) for r in old},
                new_original_sha256={r['path']:r['sha256'] for r in new},new_used=new_used,
                source_sha256={p.name:sha(p) for p in (OUT/'source_snapshot/oven_food').glob('*.py')},
                selection='highest combined 293-image validation accuracy, tie lower unweighted validation loss',
                fresh_test_policy='only evaluate 18 fresh chicken images after checkpoint selection; never auto tune based on score',
                old_test_evaluation=False,
                limits=['这是十类同标签的数据补强微调，使用全部可用旧训练图，不是新类别增量，也不再限制每旧类20张回放。',
                        '虽然本轮隔离7张旧训练图，父模型仍可能学过它们；旧验证275张存在历史同源遗漏风险，只作开发对照。',
                        '新测试只有18张鸡翅，不能代表其他九类或检验六类相对A的独立精度下降。',
                        '来源域和视觉分组仍可能遗漏转载；所有结果是网图效果，不等于烤箱或RK3506验收。'])
(OUT/'data_audit.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(REVIEW/'usage_summary.json',OUT/'incoming_data_review_summary.json')
plan='''# 本轮执行记录

用户已授权开始鸡翅补强训练。复用现有入口与配置，不修改公共训练代码。

1. 从2026年10月26日增强B候选继续十类全参数微调，不从A重做、不增加类别。
2. 使用全部可用原训练图，剔除新增关联的7张旧鸡翅训练图，加83张新增鸡翅，共1355张。
3. 旧验证275加新增验证18，共293；新测试18单独冻结。旧277张测试不评测。
4. 20轮、lr=0.00005、AdamW、wd=0.0001、batch16、CPU四线程、seed42、gentle_plus、类别加权。
5. 仅按验证准确率/同分损失选最佳；新进程加载后再比较父模型和补强模型，最后评测18张新测试。
6. 校验原图、旧准备副本、受保护A/B哈希。独立保存报告、清单、模型及代码包，不自动追加训练。

Ruling: 使用全部旧训练数据，而不继续每旧类20张回放限制——本轮目标是同一十类的数据补强，避免新鸡翅挤掉其他类别。
Ruling: 仅隔离本轮新增发现的旧训练关联，历史父模型不重训——不能借此声称已消除历史泄漏，旧验证指标保留开发口径。
Ruling: 采用最新增强B作父模型，20轮小学习率微调——沿用已交付增强方案，减少重新构建A/B链的工作；不能把变化仅归因于新增83张鸡翅。
'''
(OUT/'执行计划与记录.md').write_text(plan,encoding='utf-8')
print(json.dumps({k:snapshot[k] for k in ['training_count','validation_count','fresh_chicken_test_count','training_counts','removed_old_train']},ensure_ascii=False,indent=2))
