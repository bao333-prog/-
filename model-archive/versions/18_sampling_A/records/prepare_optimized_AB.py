"""仅修正抽样覆盖表和概率，不改原图、标签、来源事实或评测划分。"""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import sys

from PIL import Image

ROOT=Path.cwd()
RUN=Path((ROOT/'.superpowers/latest_optimized_AB.txt').read_text().strip())
BASE=Path((RUN/'baseline_run.txt').read_text().strip())
sys.path.insert(0,str(RUN))
from joint_support import hierarchical_weights,CoverageThenWeightedSampler


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


rows=read(RUN/'training_inventory.json')
baseline_rows=read(BASE/'training_inventory.json')
assert len(rows)==5734 and rows==baseline_rows
fixes={
 '土豆__sausage-have-hiah_0031.jpg':('土豆','session-20261111-60dfe4d862ad|potato-have-high','文件沿用香肠前缀，图为土豆；同session高位有锡纸组，仅修订抽样归组'),
 'bacon-mid-with_foil_0010.jpg':('培根','session-20261104-4fe35341516d|bacon-mid-with_foil','缺JSON；同目录相邻序号和视觉组推断，仅修订抽样归组，source_session仍missing'),
 'pumkin_no_high_data_0015.jpg':('南瓜','session-20261105-783733b73476|pumkin_no_high_data','缺JSON；同目录相邻序号和视觉组推断，仅修订抽样归组，source_session仍missing_metadata'),
}
changes=[]
for r in rows:
    r['sampling_group_original']=r['sampling_group']
    r['sampling_group_inferred']=False
    r['sampling_group_reason']=''
    filename=Path(r['source']).name
    if filename in fixes:
        label,group,reason=fixes[filename]
        assert r['label']==label
        assert any(x['sampling_group']==group and x['label']==label and x['era']==r['era'] for x in baseline_rows)
        r.update(sampling_group=group,sampling_group_inferred=True,sampling_group_reason=reason)
        changes.append(dict(source=r['source'],label=label,source_session_unchanged=r['source_session'],old_group=r['sampling_group_original'],new_group=group,reason=reason))
assert len(changes)==3
def pixel(r):
    assert sha(r['actual_input'])==r['input_sha256']
    with Image.open(r['actual_input']) as im:return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
with ThreadPoolExecutor(max_workers=4) as pool:pixels=list(pool.map(pixel,rows))
for r,h in zip(rows,pixels):r['pixel_sha256']=h
duplicates=defaultdict(list)
for r in rows:duplicates[r['pixel_sha256']].append(r)
pairs=[v for v in duplicates.values() if len(v)>1]
assert len(pairs)==1 and len(pairs[0])==2 and {r['label'] for r in pairs[0]}=={'可颂'}
assert len({r['sampling_group'] for r in pairs[0]})==1
# 来源事实和实际训练文件严格不变，修订仅用于抽样。
for before,after in zip(baseline_rows,rows):
    for field in ['actual_input','source','label','era','environment','source_session','source_group','raw_sha256','input_sha256']:
        assert before[field]==after[field]
weights=hierarchical_weights(rows,target_share=.75)
baseline_weights=read(BASE/'sampling_weights.json')
branches=defaultdict(list)
for i,r in enumerate(rows):branches[r['label'],r['environment'],r['era']].append(i)
branch_checks=[]
for k,indices in branches.items():
    old_mass=sum(baseline_weights[i] for i in indices)
    new_mass=sum(weights[i] for i in indices)
    assert abs(old_mass-new_mass)<1e-10
    assert max(weights[i] for i in indices)<=4/len(rows)+1e-12
    branch_checks.append(dict(branch=list(k),images=len(indices),old_mass=old_mass,new_mass=new_mass,max_expected=max(weights[i] for i in indices)*len(rows)))
spec=importlib.util.spec_from_file_location('baseline_sampler',BASE/'joint_support.py')
old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
overlay_only=old_module.hierarchical_weights(rows,target_share=.75)
sampler=CoverageThenWeightedSampler(weights,seed=42,coverage_interval=4)
counts=Counter()
for _ in range(20):counts.update(rows[i]['actual_input'] for i in sampler)
assert min(counts.values())>=5 and sum(counts.values())==5734*20
dump(RUN/'expected_sampling_counts.json',dict(counts))
dump(RUN/'training_inventory.json',rows)
dump(RUN/'sampling_weights.json',weights)
dump(RUN/'grouping_overrides.json',changes)
csv_write(RUN/'抽样分组修订.csv',changes)
csv_write(RUN/'训练逐图来源.csv',rows)
old_counts=read(BASE/'sample_counts.json')
stats=dict(baseline_run=str(BASE),image_pool_unchanged=True,grouping_fixes=changes,duplicate_groups=1,duplicate_extra_files=1,
           maximum_expected_draws_baseline=max(baseline_weights)*len(rows),maximum_expected_draws_after_group_fixes=max(overlay_only)*len(rows),
           maximum_expected_draws_final=max(weights)*len(rows),coverage_epochs=[1,5,9,13,17],weighted_epochs=sampler.weighted_epochs,
           baseline_min_reads=min(old_counts.values()),baseline_max_reads=max(old_counts.values()),baseline_once_only=sum(v==1 for v in old_counts.values()),
           preview_min_reads=min(counts.values()),preview_max_reads=max(counts.values()),branch_checks=branch_checks,
           source_facts_and_metadata_unchanged=True,cap_is_expectation_not_hard_read_limit=True)
dump(RUN/'sampling_optimization_audit.json',stats)
policy=dict(name='bounded_hierarchical_periodic_coverage_v2',target_share=.75,max_expected_draws_per_weighted_epoch=4,
            coverage_interval=4,coverage_epochs=[1,5,9,13,17],weighted_epochs=15,duplicate_aware_within_group=True,
            original_sessions_unchanged=True,public_validator_unchanged=True)
dump(RUN/'sampling_policy.json',policy)
audit=read(RUN/'data_audit.json');audit.update(optimization_baseline=str(BASE),sampling_policy=policy,
       grouping_fix_count=3,minimum_guaranteed_reads=5,remaining_epochs_weighted=15)
dump(RUN/'data_audit.json',audit)
assert sha(RUN/'official_initialization.pt')==sha(BASE/'official_initialization.pt')
for name in ['manifest.csv','training_manifest.csv','validation_A.csv','validation_B.csv','test_A.csv','test_B.csv']:
    assert sha(RUN/name)==sha(BASE/name)
note='# 抽样与数据分组优化执行记录\n\n仅运行一个已预先确定的优化方案。图像池5734条、标签、原始会话、官方初始权重、20轮配置及A83/84、B193/167划分与上轮相同。牛排仍只有第二批205张，不恢复第一批或网图牛排。\n\n3条推断覆盖仅改变sampling_group，原始source_session与JSON不改；缺失信息仍标为缺失，不作为独立性确认。修订依据和前后值见抽样分组修订.csv。一对重复可颂保留，概率由副本分摊。\n\n期望最高每轮抽样30.72→归组修正后7.68→限幅4次，不是实际次数硬上限。第1/5/9/13/17轮全覆盖，每图至少5次。训练前确定性预演次数范围'+str(min(counts.values()))+'—'+str(max(counts.values()))+'；真实日志将再核对。\n\n验证按原总体准确率及损失选模，额外记录每类及类平均准确率，日志不参与新的选模规则。训练结束、权重固定后才评分测试，不根据测试调参或追加训练。A南瓜同源、B测试包含A验证等限制保持开发对照口径。\n'
(RUN/'执行记录.md').write_text(note,encoding='utf-8')
print(json.dumps(dict(run=str(RUN),source_groups_fixed=3,pool_images=5734,max_expected_before=stats['maximum_expected_draws_baseline'],max_expected_after=stats['maximum_expected_draws_final'],preview_min=min(counts.values()),preview_max=max(counts.values()),init_unchanged=True),ensure_ascii=False))
