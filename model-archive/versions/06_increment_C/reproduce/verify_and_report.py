"""选定C后做新进程加载、原图一致性、父B旧类对照及14输出完整竞争评测。"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,time
import torch
from PIL import Image
from torch.utils.data import DataLoader
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest,validate_manifest
from predict import load_classifier,image_tensor
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];torch.set_num_threads(4)
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read_json(OUT/'data_audit.json');e=read_json(OUT/'execution.json')
assert e['status']=='completed' and e['protected_models_unchanged']
history=read_json(OUT/'training/history.json')
assert len(history)==20 and all(math.isfinite(v) for h in history for v in h.values())
model,names,cp=load_classifier(OUT/'training/best.pt')
parent,old_names,parent_cp=load_classifier(a['parent_checkpoint'])
assert names==a['class_names']==old_names+a['new_class_names'] and len(names)==14
assert cp['incremental']['parent_sha256']==a['parent_sha256']
assert cp['incremental']['replay_counts']=={n:20 for n in old_names}
assert all(torch.isfinite(v).all() for v in cp['model_state'].values() if v.is_floating_point())
best=min(history,key=lambda h:(-h['validation_accuracy'],h['validation_loss']))
assert cp['epoch']==best['epoch']
assert read_json(OUT/'initialization_audit.json')['all_existing_parameters_and_buffers_unchanged_during_expansion']
records=validate_manifest(read_manifest(OUT/'manifest.csv'),check_files=True)
actual=read_manifest(OUT/'training/training_manifest.csv')
assert actual==read_manifest(OUT/'expected_training_manifest.csv')
train=[r for r in actual if r.split=='train'];val=[r for r in records if r.split=='val'];test=[r for r in records if r.split=='test']
assert len(train)==729 and len(val)==407 and len(test)==409
assert Counter(r.food_label for r in train)==Counter(a['actual_training_counts'])
assert not {r.image_path for r in train}&{r.image_path for r in val+test}
assert not {r.session_id for r in train}&{r.session_id for r in val+test}
parent_train={r.image_path for r in read_manifest(Path(a['parent_dir'])/'manifest.csv') if r.split=='train'}
assert all(r.image_path in parent_train for r in train if r.food_label in old_names)
assert Counter(r.food_label for r in train if r.food_label in old_names)==Counter({n:20 for n in old_names})
review=read_json(OUT/'review_and_split.json');originals={r['prepared_path']:r['path'] for r in review if r['included']}
# 全部新类准备副本与原始图片经同一确定性入口处理，逐张核对输入。
new_records=[r for r in records if r.food_label in a['new_class_names']]
dataset=FoodImageDataset(new_records,data_root=OUT,class_names=names,resize_mode='letterbox')
for i,r in enumerate(new_records):
    with Image.open(originals[r.image_path]) as image:assert torch.equal(image_tensor(image,cp)[0],dataset[i][0])
initial_paths={r.image_path for r in read_manifest(ROOT/'runs/formal_A_20261025_113459/manifest.csv') if r.split=='test'}
wings_fresh_paths={r.image_path for r in read_manifest(Path(a['parent_dir'])/'frozen_new_chicken_test.csv')}

def evaluate(network,network_names,selected,folder):
    data=FoodImageDataset(selected,data_root=OUT,class_names=network_names,resize_mode='letterbox')
    predictions=[];loss_sum=0.;matrix=[[0]*len(network_names) for _ in network_names]
    with torch.inference_mode():
        for images,labels in DataLoader(data,batch_size=16,shuffle=False,num_workers=0):
            logits=network(images);assert torch.isfinite(logits).all()
            loss_sum+=float(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
            probability,predicted=logits.softmax(1).max(1)
            for y,p,prob in zip(labels.tolist(),predicted.tolist(),probability.tolist()):
                r=selected[len(predictions)];matrix[y][p]+=1
                predictions.append(dict(image_path=r.image_path,original_path=originals.get(r.image_path,r.image_path),
                    group=r.session_id,true_label=network_names[y],predicted_label=network_names[p],correct=y==p,
                    top1_probability=prob,cohort='old' if r.food_label in old_names else 'new',
                    initial_six_original_test=r.image_path in initial_paths,previous_wings_test=r.image_path in wings_fresh_paths))
    stats=dict(loss=loss_sum/len(selected),per_class={},confusion_matrix=matrix)
    for name in network_names:
        group=[r for r in predictions if r['true_label']==name]
        if group:stats['per_class'][name]=dict(correct=sum(r['correct'] for r in group),total=len(group),accuracy=sum(r['correct'] for r in group)/len(group))
    for scope in ['all','old','new','initial_six_original_test','previous_wings_test']:
        group=predictions if scope=='all' else ([r for r in predictions if r['cohort']==scope] if scope in {'old','new'} else [r for r in predictions if r[scope]])
        if group:stats[scope]=dict(correct=sum(r['correct'] for r in group),total=len(group),accuracy=sum(r['correct'] for r in group)/len(group))
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    with (folder/'confusion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['true/predicted',*network_names]);w.writerows([[n,*r] for n,r in zip(network_names,matrix)])
    print(json.dumps(dict(evaluation=folder.name,scores={k:v for k,v in stats.items() if k not in {'confusion_matrix','per_class','loss'}}),ensure_ascii=False),flush=True)
    return stats,predictions

started=time.perf_counter()
evaluations={};all_predictions={}
for label,network,network_names,chosen in [
    ('parent_old_validation',parent,old_names,[r for r in val if r.food_label in old_names]),
    ('validation_C',model,names,val),
    ('parent_old_test',parent,old_names,[r for r in test if r.food_label in old_names]),
    ('test_C',model,names,test)]:
    evaluations[label],all_predictions[label]=evaluate(network,network_names,chosen,OUT/label)
assert evaluations['parent_old_validation']['all']['correct']==292
assert evaluations['parent_old_test']['previous_wings_test']['correct']==16
assert evaluations['validation_C']['all']['accuracy']==cp['validation_accuracy']
assert abs(evaluations['validation_C']['loss']-cp['validation_loss'])<1e-5
old_parent=evaluations['parent_old_test']['all'];old_C=evaluations['test_C']['old'];new_C=evaluations['test_C']['new']
assert old_C['total']==old_parent['total']==295 and new_C['total']==114
drop=100*(old_parent['accuracy']-old_C['accuracy'])
first_six=evaluations['test_C']['initial_six_original_test'];assert first_six['total']==159
A_accuracy=read_json(ROOT/'runs/formal_A_20261025_113459/final_report.json')['evaluations']['test']['overall_accuracy']
assert all(sha(p)==d for p,d in a['protected_models'].items())
assert all(sha(p)==d for p,d in a['old_input_sha256'].items())
assert all(sha(p)==d for p,d in a['new_original_sha256'].items())
assert all(sha(OUT/'source_snapshot/oven_food'/n)==d for n,d in a['source_sha256'].items())
report=dict(stage='second_increment_C_10_to_14',class_names=names,new_class_names=a['new_class_names'],
    parent_sha256=a['parent_sha256'],checkpoint_sha256=sha(OUT/'training/best.pt'),best_epoch=cp['epoch'],epochs=20,
    learning_rate=.0001,training_seconds=e['training_seconds'],verification_seconds=time.perf_counter()-started,
    new_input_count=800,new_included=757,new_excluded=43,new_split=a['new_split'],old_replay_per_class=20,
    actual_training_count=729,actual_training_counts=a['actual_training_counts'],validation_count=407,
    evaluations=evaluations,old_accuracy_drop_percentage_points=drop,
    original_initial_six_A_accuracy=A_accuracy,initial_six_cumulative_drop_percentage_points=100*(A_accuracy-first_six['accuracy']),
    target_checks=dict(old_ten_accuracy_at_least_90pct=old_C['accuracy']>=.9,old_ten_drop_at_most_2pp=drop<=2,new_four_accuracy_at_least_86pct=new_C['accuracy']>=.86),
    initial_six_cumulative_drop_at_most_2pp=100*(A_accuracy-first_six['accuracy'])<=2,
    new_process_reload=True,raw_new_inputs_match_prepared=True,original_images_unchanged=True,
    protected_models_unchanged=True,actual_replay_matches_plan=True,old_test_is_reused_benchmark=True,
    test_errors=[r for r in all_predictions['test_C'] if not r['correct']],limits=a['limits'])
(OUT/'final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 第二次增量：十四类模型C', '', '2026年10月29日。加载鸡翅补强十类B后扩展至十四类，追加土豆、香肠、薯条、可颂。', '',
       '## 数据核对与训练', '',
       '输入800张，逐张预览；保留757张，暂缓43张（1土豆、1香肠、3薯条、38低清可颂）。土豆目录4张拼图预览不计作样本。',
       '可颂无来源记录且有大量近似处理版本，按人工保守视觉组及自动相似性合组；其他类别结合来源域分组，种子42整组划分。', '',
       '| 新类 | 训练 | 验证 | 测试 |','|---|---:|---:|---:|']
for name,s in a['new_split'].items():lines.append(f"| {name} | {s['train']} | {s['val']} | {s['test']} |")
lines += ['', '新类训练529张，旧十类每类20张回放共200张，实际训练729张；验证旧293+新114=407；测试旧295+新114=409。',
          '从原B严格加载后扩展最后分类层，旧十类参数和缓冲区保持一致，新四行随机初始化，再更新全模型。',
          f"训练20轮，lr=0.0001，AdamW，wd=0.0001，batch16，CPU四线程，gentle_plus补边增强、类别加权。最佳第{cp['epoch']}轮，训练{e['training_seconds']:.1f}秒。", '',
          '## 验证集开发指标', '', '| 范围 | 正确/总数 | 准确率 |', '|---|---:|---:|']
for key,title in [('old','旧十类'),('new','新四类'),('all','累计十四类')]:
    s=evaluations['validation_C'][key];lines.append(f"| {title} | {s['correct']}/{s['total']} | {s['accuracy']:.2%} |")
lines += ['', '## 固定测试对照（模型选定后评测）', '', '| 范围 | 正确/总数 | 准确率 |', '|---|---:|---:|',
          f"| 父B旧十类基准 | {old_parent['correct']}/{old_parent['total']} | {old_parent['accuracy']:.2%} |"]
for key,title in [('old','C旧十类'),('new','C新增四类'),('all','C累计十四类')]:
    s=evaluations['test_C'][key];lines.append(f"| {title} | {s['correct']}/{s['total']} | {s['accuracy']:.2%} |")
lines += ['', f'旧十类相对本轮父B下降{drop:.2f}个百分点（负数表示提高）。C旧类在完整14个输出中竞争，没有截取旧十类输出抬高成绩。',
          '旧295张是此前已使用的阶段对照基准，包含历史277张和鸡翅补强测试18张；不能称全新独立验收。新四类114张此前未用于训练、回放或选模，但来源独立性仍有局限。', '',
          '| 食材 | 正确/总数 | 准确率 |','|---|---:|---:|']
for name,s in evaluations['test_C']['per_class'].items():lines.append(f"| {name} | {s['correct']}/{s['total']} | {s['accuracy']:.2%} |")
lines += ['', '## 目标检查', '']
for name,passed in report['target_checks'].items():lines.append(f"- {name}：{'达到' if passed else '未达到'}。")
lines += ['', f"初始六类原159张测试上，C为{first_six['correct']}/159（{first_six['accuracy']:.2%}），原A为{A_accuracy:.2%}，累计下降{report['initial_six_cumulative_drop_percentage_points']:.2f}个百分点；累计保持2个百分点目标{'达到' if report['initial_six_cumulative_drop_at_most_2pp'] else '未达到'}。这与本轮旧十类相对B的下降是不同口径。", '',
          '## 逐轮记录', '', '| 轮次 | 训练损失 | 验证损失 | 十四类验证准确率 |','|---|---:|---:|---:|']
for h in history:lines.append(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['validation_loss']:.5f} | {h['validation_accuracy']:.2%} |")
lines += ['', '## 核对与边界', '',
          '实际训练/回放与预期清单逐条一致；训练不含验证/测试路径或来源组。所有757张新类原图输入与准备副本一致，严格重新加载14类权重和类别映射通过。',
          '800张新原图、1943张旧输入和受保护A/B权重、源码快照哈希检查通过。模型选定后才评测测试集，没有按测试追加训练。', '']
lines += ['- '+s for s in a['limits']]
(OUT/'第二次增量C_结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(best_epoch=cp['epoch'],old_drop_pp=drop,target_checks=report['target_checks'],initial_six_cumulative_drop_pp=report['initial_six_cumulative_drop_percentage_points']),ensure_ascii=False,indent=2))
