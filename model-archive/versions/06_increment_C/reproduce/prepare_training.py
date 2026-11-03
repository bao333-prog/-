"""第二次增类10→14：筛选、来源隔离、固定划分、输入封存和实际回放预核。"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,random,shutil
from PIL import Image
from oven_food.augment import prepare_image
from oven_food.manifest import ManifestRecord,read_manifest,write_manifest,validate_manifest
from oven_food.train import _select_replay_records
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
PARENT=ROOT/'runs/B_wings_supplement_20261029_174246'
rows=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
pairs=json.loads((OUT/'internal_similarity_candidates.json').read_text(encoding='utf-8'))
assert len(rows)==800 and all(r['decoded'] for r in rows)
assert not json.loads((OUT/'old_visual_overlap_candidates.json').read_text(encoding='utf-8'))
assert not any(r['old_hash_matches'] or r['metadata_sha256_matches'] is False for r in rows)
classes=['土豆','香肠','薯条','可颂']
excluded={'土豆_094':'来源为图像编辑平台素材，缺实拍身份佐证，暂缓；不声称已证明AI生成。',
          '香肠_062':'来源商品标题及页面名指向熟制，但描述又称raw，自相矛盾，暂缓。',
          '薯条_105':'明显深色轮廓和抠图锯齿，暂缓。',
          '薯条_109':'外观偏楔形块，条状薯条标签边界不清，暂缓。',
          '薯条_128':'抠图后大面积深色背景残留，暂缓。'}
for i in list(range(38,53))+list(range(81,89))+list(range(120,135)):
    excluded[f'可颂_{i:03d}']='明显低清像素块和抠图边缘，近似处理版本成组暂缓。'
assert len(excluded)==43
parent={r['id']:r['id'] for r in rows};edges=[]
def find(k):
    while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
    return k
def union(a,b,reason):
    parent[find(b)]=find(a);edges.append(dict(left=a,right=b,reason=reason))
seen={}
for r in rows:
    for kind,value in [('file',r['sha256']),('content',r.get('content_sha256')),('original',r['original_sha256']),('image_url',r['image_url'])]:
        if not value:continue
        key=(r['food_class'],kind,value)
        if key in seen:union(seen[key],r['id'],kind)
        seen[key]=r['id']
    domain=r['source_domain']
    if domain:
        # 可识别的同一公司多语言站按同一来源域处理。
        if domain in {'europlant.biz','europlant.cz'}:domain='europlant'
        key=(r['food_class'],'domain',domain)
        if key in seen:union(seen[key],r['id'],'conservative_source_domain '+domain)
        seen[key]=r['id']
for p in pairs:
    if p['cross_label']:raise ValueError('cross-label duplicate requires review')
    union(p['left'],p['right'],f"visual_similarity d={p['distance']} mae={p['pixel_mae']:.4f}")
# 无来源记录的可颂中，大量连续图片是同一构图的角度/裁切/缩放版本。
# 人工保守视觉组只用于避免已发现的近似图跨集合，不证明真实母图数。
families=[[2,4,5],[12,13],list(range(14,20)),list(range(20,24)),[24,25,26],
          list(range(27,33)),list(range(33,38)),list(range(38,53)),list(range(53,60)),
          list(range(60,68)),list(range(68,77)),list(range(77,81)),list(range(81,89)),
          list(range(89,97)),list(range(97,111)),list(range(111,120)),list(range(120,135)),
          list(range(135,139)),list(range(139,145)),list(range(145,151)),list(range(151,164)),
          list(range(164,169)),list(range(169,176)),list(range(176,181)),[181,182],
          list(range(183,188)),list(range(188,199)),[199,200]]
for group in families:
    for i in group[1:]:union(f'可颂_{group[0]:03d}',f'可颂_{i:03d}','manual_conservative_visual_family')
potato_manual=[[188,192,193,196]]
for group in potato_manual:
    for i in group[1:]:union(f'土豆_{group[0]:03d}',f'土豆_{i:03d}','manual_similar_catalogue_composition')
grouped=defaultdict(list)
for r in rows:
    if r['id'] not in excluded:grouped[find(r['id'])].append(r)
rng=random.Random(42);assignments={};group_counts={}
def choose(pool,target,label):
    possible={0:[]}
    for i,g in enumerate(pool):
        for n,ids in list(possible.items()):possible.setdefault(n+len(g),ids+[i])
    choices=[(n,ids) for n,ids in possible.items() if n>0 and len(ids)<len(pool)]
    if not choices:raise ValueError(label+'无法按来源组划分训练和评估')
    return min(choices,key=lambda item:abs(item[0]-target))[1]
for label in classes:
    groups=[g for g in grouped.values() if g[0]['food_class']==label]
    group_counts[label]=len(groups);rng.shuffle(groups)
    count=sum(map(len,groups))
    for split in ['test','val']:
        ids=choose(groups,round(count*.15),label)
        for i in ids:assignments[find(groups[i][0]['id'])]=split
        groups=[g for i,g in enumerate(groups) if i not in ids]
    for group in groups:assignments[find(group[0]['id'])]='train'
new_records=[];review=[]
for row in rows:
    if row['id'] in excluded:
        review.append({**row,'included':False,'reason':excluded[row['id']]});continue
    group='Cnew_'+find(row['id']);split=assignments[find(row['id'])]
    path=OUT/'prepared_images'/row['food_class']/(row['id']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(row['path']) as image:prepare_image(image,image_size=224,resize_mode='letterbox').save(path)
    new_records.append(ManifestRecord(str(path),row['food_class'],'proxy',group,'unknown_web','unknown_web','none','none',split))
    review.append({**row,'included':True,'split':split,'group':group,'prepared_path':str(path)})
old=read_manifest(PARENT/'manifest.csv')
old_test=[r for r in read_manifest(ROOT/'runs/increment_B_20261025_165256/manifest.csv') if r.split=='test']
old_test+=read_manifest(PARENT/'frozen_new_chicken_test.csv')
assert len(old_test)==295
old+=old_test
combined=old+new_records;validate_manifest(combined,check_files=True)
write_manifest(OUT/'manifest.csv',combined)
old_names=['南瓜','吐司','牛排','玉米','西兰花','鸡翅','芦笋','培根','三文鱼','整鸡']
replay=_select_replay_records([r for r in old if r.split=='train'],old_names,per_class=20,seed=42)
assert Counter(r.food_label for r in replay)==Counter({n:20 for n in old_names})
actual=replay+[r for r in new_records if r.split=='train']+[r for r in combined if r.split=='val']
write_manifest(OUT/'expected_training_manifest.csv',actual)
summary={label:dict(Counter(r.split for r in new_records if r.food_label==label)) for label in classes}
config=json.loads((PARENT/'train_config.json').read_text(encoding='utf-8'))
config.update(experiment_name='second_increment_C_four_new_classes',manifest=str(OUT/'manifest.csv'),data_root=str(OUT),
              output_dir=str(OUT/'training'),checkpoint=str(PARENT/'training/best.pt'),new_classes=classes,
              replay_per_class=20,learning_rate=.0001,epochs=20,augmentation='gentle_plus')
(OUT/'train_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected=[ROOT/'runs'/folder/'training/best.pt' for folder in ['formal_A_20261025_113459','increment_B_20261025_165256','B_augmentation_trial_20261026_123837','B_wings_supplement_20261029_174246']]
audit=dict(stage='second_class_increment_10_to_14',parent_dir=str(PARENT),parent_checkpoint=str(PARENT/'training/best.pt'),
           parent_sha256=sha(PARENT/'training/best.pt'),protected_models={str(p):sha(p) for p in protected},
           old_class_names=old_names,new_class_names=classes,class_names=old_names+classes,
           new_input_count=800,new_included=len(new_records),new_excluded=len(excluded),
           new_split=summary,new_source_or_visual_group_counts=group_counts,
           actual_training_count=sum(r.split=='train' for r in actual),old_replay_count=200,
           actual_training_counts=dict(Counter(r.food_label for r in actual if r.split=='train')),
           validation_count=sum(r.split=='val' for r in combined),old_test_count=295,
           new_test_count=sum(r.split=='test' for r in new_records),
           old_input_sha256={r.image_path:sha(r.image_path) for r in old},
           new_original_sha256={r['path']:r['sha256'] for r in rows},
           source_sha256={p.name:sha(p) for p in (OUT/'source_snapshot/oven_food').glob('*.py')},
           limits=['可颂200张无来源记录，含大量近似处理版本；人工保守视觉组不证明独立拍摄，仍可能漏掉同源。',
                   '土豆以黄皮/浅皮整薯和切面为主；薯条以浅色冷冻/预炸待加热形态为主；香肠以生香肠为主；可颂以未烘烤面团为主。',
                   '旧类测试295张已在此前开发过程中使用，其中277张来自历史测试、18张来自鸡翅补强测试；不是全新独立验收。',
                   '父模型与旧验证存在历史来源关联风险，本次维持隔离后的旧训练清单，但不能消除父模型已有知识。',
                   '本轮按有限样本回放，每旧类20张；没有蒸馏或参数隔离对照。',
                   '结果为网图/去背景数据，不是烤箱实拍或RK3506部署验收。'])
assert all(sha(r['path'])==r['sha256'] for r in rows)
for name,value in [('data_audit.json',audit),('review_and_split.json',review),('grouping_edges.json',edges),('new_split_summary.json',summary)]:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
plan='''# 第二次增量执行记录

用户提供四个目录并授权核对后开始第二次增量。PNG实际为未烘烤可颂，确认四类方案：土豆、香肠、薯条、可颂。
从最新鸡翅补强十类B加载，保留原十类下标，追加四类至14类。先加载旧模型，再扩展最后分类层，新增四行随机初始化。
每旧类20张训练图回放、全部新类训练图；20轮、lr=0.0001、AdamW、wd=0.0001、batch16、CPU四线程、seed42、gentle_plus、类别加权。
先按来源域及视觉近似分组，再固定划分；800张输入逐张预览，43张暂缓。原图只读，准备副本独立保存。
按验证准确率、同分损失选择模型；选模后才评估旧固定测试与新四类测试，不据测试追加训练或调参。
旧类在14个输出中竞争，比较父B在同一旧固定测试的成绩；同时报告初始六类累计变化，注明旧测试重复使用及来源局限。
保存实际回放、配置、完整日志、最佳模型、重新加载预测和独立代码包。保留A/B及所有历史交付，不自动替换旧版本。

Ruling: 沿用现有有限样本回放入口，不沿用上一次全旧数据补强策略——本轮是第二次增类，需要记录旧知识保持而非十类联合重训。
Ruling: 可颂无来源且存在处理簇，人工保守同组并暂缓38张严重像素化图片——优先避免已发现的同源跨集合，不声称已证明其余图片独立。
Ruling: 冷冻预炸薯条属于待最终加热输入，保留；不因标题有Pre-Fried或Oven-Baked就自动认定图像已熟制。香肠062来源与描述冲突，暂缓。
'''
(OUT/'执行计划与记录.md').write_text(plan,encoding='utf-8')
print(json.dumps({k:audit[k] for k in ['new_included','new_excluded','new_split','new_source_or_visual_group_counts','actual_training_count','validation_count','old_test_count','new_test_count']},ensure_ascii=False,indent=2))
