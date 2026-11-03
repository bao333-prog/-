"""独立进程验证补强结果；选定权重后才评估18张新鸡翅测试，旧测试不评测。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest, validate_manifest
from predict import load_classifier, image_tensor

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit = read_json(OUT/'data_audit.json')
execution = read_json(OUT/'execution.json')
config = read_json(OUT/'train_config.json')
assert execution['status']=='completed' and execution['protected_models_unchanged']
assert not config['new_classes'] and config.get('target_fraction') is None
assert config['checkpoint']==audit['parent_checkpoint']
assert sha(Path(config['checkpoint']))==audit['parent_sha256']
history = read_json(OUT/'training/history.json')
assert len(history)==20 and all(math.isfinite(v) for h in history for v in h.values())
candidate,names,checkpoint = load_classifier(OUT/'training/best.pt')
parent,parent_names,parent_checkpoint = load_classifier(audit['parent_checkpoint'])
assert names==parent_names and len(names)==10
assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values() if value.is_floating_point())
assert candidate.classifier[-1].out_features==parent.classifier[-1].out_features==10
assert checkpoint['incremental'] is None
assert checkpoint['config']['checkpoint']==audit['parent_checkpoint']
best = min(history,key=lambda h:(-h['validation_accuracy'],h['validation_loss']))
assert checkpoint['epoch']==best['epoch']
records = read_manifest(OUT/'manifest.csv')
assert records==read_manifest(OUT/'training/training_manifest.csv')
assert Counter(r.split for r in records)=={'train':1355,'val':293}
assert Counter(r.food_label for r in records if r.split=='train')==Counter(audit['training_counts'])
frozen = read_manifest(OUT/'frozen_new_chicken_test.csv')
validate_manifest(records+frozen,check_files=True)
assert len(frozen)==18 and not {r.image_path for r in frozen}&{r.image_path for r in records}
assert not {r.session_id for r in frozen}&{r.session_id for r in records}
assert not set(r['path'] for r in audit['removed_old_train']) & {r.image_path for r in records if r.split=='train'}
used_new = {r['prepared_path']:r for r in audit['new_used']}
baseline_dataset = FoodImageDataset([r for r in records if r.image_path in used_new],data_root=OUT,
                                  class_names=names,resize_mode='letterbox')
for i,r in enumerate(baseline_dataset.records):
    with Image.open(used_new[r.image_path]['original_path']) as image:
        assert torch.equal(image_tensor(image,checkpoint)[0],baseline_dataset[i][0])


def evaluate(network, selected):
    dataset = FoodImageDataset(selected,data_root=OUT,class_names=names,resize_mode='letterbox')
    loader = DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    predictions = []
    loss_sum = 0.
    matrix = [[0]*10 for _ in names]
    with torch.inference_mode():
        for images,labels in loader:
            logits = network(images)
            assert torch.isfinite(logits).all()
            loss_sum += float(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
            probabilities,predicted = logits.softmax(1).max(1)
            for true,pred,prob in zip(labels.tolist(),predicted.tolist(),probabilities.tolist()):
                r = selected[len(predictions)]
                matrix[true][pred] += 1
                fresh = used_new.get(r.image_path)
                subset = ('new_chicken_val' if fresh else 'historical_chicken_val') if r.food_label=='鸡翅' else 'other_nine_val'
                if r.split=='test':
                    subset='fresh_chicken_test'
                predictions.append(dict(image_path=r.image_path,original_path=fresh['original_path'] if fresh else r.image_path,
                        true_label=names[true],predicted_label=names[pred],correct=true==pred,top1_probability=prob,subset=subset))
    stats = dict(loss=loss_sum/len(selected),per_class={},confusion_matrix=matrix)
    for name in names:
        group = [r for r in predictions if r['true_label']==name]
        if group:
            stats['per_class'][name]=dict(correct=sum(r['correct'] for r in group),total=len(group),
                                         accuracy=sum(r['correct'] for r in group)/len(group))
    for subset in ['all','historical_chicken_val','new_chicken_val','other_nine_val','fresh_chicken_test','historical_all_val']:
        group = (predictions if subset=='all' else
                 [r for r in predictions if r['subset']!='new_chicken_val'] if subset=='historical_all_val' and selected[0].split=='val' else
                 [r for r in predictions if r['subset']==subset])
        if group:
            stats[subset]=dict(correct=sum(r['correct'] for r in group),total=len(group),
                               accuracy=sum(r['correct'] for r in group)/len(group))
    return stats,predictions


started = time.perf_counter()
evaluations = {}
for split,selected in [('val',[r for r in records if r.split=='val']),('fresh_chicken_test',frozen)]:
    evaluations[split] = {}
    for model_label,network in [('parent_augmented_B',parent),('chicken_supplement_candidate',candidate)]:
        stats,predictions = evaluate(network,selected)
        evaluations[split][model_label]=stats
        dest = OUT/split/model_label
        dest.mkdir(parents=True)
        (dest/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
        with (dest/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        with (dest/'confusion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['true/predicted',*names]);writer.writerows([[name,*row] for name,row in zip(names,stats['confusion_matrix'])])
        print(json.dumps(dict(split=split,model=model_label,counts=stats['all']),ensure_ascii=False),flush=True)
val = evaluations['val']
assert val['chicken_supplement_candidate']['all']['accuracy']==checkpoint['validation_accuracy']
assert abs(val['chicken_supplement_candidate']['loss']-checkpoint['validation_loss'])<1e-5
assert val['parent_augmented_B']['historical_all_val']['correct']==269
# 旧原图路径已失效时，不声称重新核对旧原图；核对本轮实际使用和封存的准备副本。
assert all(sha(Path(path))==digest for path,digest in audit['old_prepared_sha256'].items())
assert all(sha(Path(path))==digest for path,digest in audit['new_original_sha256'].items())
assert all(sha(Path(path))==digest for path,digest in audit['protected_models'].items())
assert all(sha(OUT/'source_snapshot/oven_food'/name)==digest for name,digest in audit['source_sha256'].items())
report = dict(stage='chicken_supplement_same_ten_classes',class_names=names,parent_sha256=audit['parent_sha256'],
              best_epoch=checkpoint['epoch'],epochs=20,learning_rate=.00005,
              training_seconds=execution['training_seconds'],verification_seconds=time.perf_counter()-started,
              training_count=1355,training_counts=audit['training_counts'],validation_count=293,
              fresh_chicken_test_count=18,evaluations=evaluations,old_test_evaluated=False,
              full_ten_class_independent_test_available=False,original_old_accuracy_retention_target='not_evaluated_this_run',
              raw_new_images_match_prepared=True,new_process_reload=True,
              protected_models_unchanged=True,new_originals_unchanged=True,old_prepared_images_unchanged=True,
              checkpoint_sha256=sha(OUT/'training/best.pt'),limits=audit['limits'])
(OUT/'final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 十类模型B：鸡翅补充数据训练', '', '2026年10月29日。用户授权开始训练；从最新增强B候选继续十类同标签微调。', '',
       '## 数据和训练', '',
       '从本批200张中筛选出的119张候选，使用83张训练、18张验证，18张测试冻结至选模后。77张旧图处理版本不作新增，3张质量暂缓，1张旧评估来源关联未训练。',
       '原1279张十类训练图中隔离7张新增发现的旧鸡翅来源关联图，余1272张加83张新增图，共1355张；鸡翅134张，其余九类1221张。',
       '旧验证275张加新鸡翅验证18张，共293张。本轮使用全部可用旧训练图，不再限制每旧类20张；这是数据补强，不是第二次增类。',
       '20轮、学习率0.00005、AdamW、权重衰减0.0001、batch16、CPU四线程、seed42、gentle_plus补边增强、训练类别加权；更新全模型。',
       f"最佳第{checkpoint['epoch']}轮；训练{execution['training_seconds']:.1f}秒。按293张验证准确率、同分验证损失选模，无测试驱动追加训练。", '',
       '## 验证对照（开发指标）', '', '| 范围 | 增强B父模型 | 本轮补强候选 |','|---|---:|---:|']
for scope,title in [('all','累计十类293张'),('historical_all_val','历史验证275张'),('historical_chicken_val','历史鸡翅验证'),('new_chicken_val','新增鸡翅验证'),('other_nine_val','其他九类验证')]:
    a,b=val['parent_augmented_B'][scope],val['chicken_supplement_candidate'][scope]
    lines.append(f"| {title} | {a['correct']}/{a['total']} ({a['accuracy']:.2%}) | {b['correct']}/{b['total']} ({b['accuracy']:.2%}) |")
lines += ['', '| 食材 | 父模型正确/总数 | 补强候选正确/总数 |','|---|---:|---:|']
for name in names:
    a,b=val['parent_augmented_B']['per_class'][name],val['chicken_supplement_candidate']['per_class'][name]
    lines.append(f"| {name} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} |")
lines += ['', '## 18张新增鸡翅测试（选模后一次评估）', '', '| 模型 | 正确/总数 | 准确率 |', '|---|---:|---:|']
for model_label,title in [('parent_augmented_B','增强B父模型'),('chicken_supplement_candidate','补强候选')]:
    score=evaluations['fresh_chicken_test'][model_label]['all']
    lines.append(f"| {title} | {score['correct']}/{score['total']} | {score['accuracy']:.2%} |")
lines += ['', '这18张只涉及鸡翅、样本量较小，不能作为十类整体独立成绩，也不能据此判断旧六类相对A下降是否在2个百分点以内。旧277张测试本轮没有评测。', '',
          '## 逐轮记录', '', '| 轮次 | 训练损失 | 验证损失 | 验证准确率 |','|---|---:|---:|---:|']
for h in history:
    lines.append(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['validation_loss']:.5f} | {h['validation_accuracy']:.2%} |")
lines += ['', '## 核对和解释边界', '',
          '新进程严格加载十类权重；101张新增训练/验证原图与准备副本的输入张量逐张一致；200张新原图、1831张旧准备副本及A、原B、增强B哈希核对通过。',
          '数据清单与实际配置一致；训练/验证/新测试来源组隔离，已知旧测试不参与训练；封存来源不明或历史同源风险仍须如实说明。', '',
          '父模型可能已学过本轮剔除的7张图，隔离不能抹去历史知识；旧验证仍为开发对照。',
          '训练样本总量和回放策略都变了，不能把全部成绩变化仅归因于新增83张鸡翅，不能与有限回放增量算法作公平优劣比较。',
          '保持独立候选，原版本保留；没有自动追加轮数或开始第二次增类。', '']
lines += ['- '+item for item in audit['limits']]
(OUT/'鸡翅补强训练_结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(best_epoch=report['best_epoch'],report=str(OUT/'鸡翅补强训练_结果报告.md')),ensure_ascii=False))
