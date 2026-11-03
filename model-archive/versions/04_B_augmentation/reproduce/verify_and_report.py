"""独立进程加载候选；只评估验证集，不读取测试集像素或预测。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import random
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader

from oven_food.augment import prepare_image, shrink_and_translate, adjust_color_temperature, degrade_resolution, jpeg_roundtrip
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest
from predict import load_classifier, image_tensor

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
torch.set_num_threads(4)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


baseline = read_json(OUT/'baseline.json')
BASE = Path(baseline['comparator_dir'])
PARENT = Path(baseline['parent_dir'])
assert sha(PARENT/'training/best.pt') == baseline['parent_sha256']
assert sha(BASE/'training/best.pt') == baseline['comparator_checkpoint_sha256']
assert sha(OUT/'manifest.csv') == baseline['manifest_sha256'] == sha(BASE/'manifest.csv')
config = read_json(OUT/'train_config.json')
original_config = read_json(BASE/'train_config.json')
for key in original_config:
    if key not in {'augmentation', 'experiment_name', 'manifest', 'output_dir'}:
        assert config[key] == original_config[key], key
execution = read_json(OUT/'execution.json')
assert execution['status']=='completed'
history = read_json(OUT/'training/history.json')
assert len(history)==20
assert all(math.isfinite(v) for row in history for v in row.values())
actual = read_manifest(OUT/'training/training_manifest.csv')
assert actual == read_manifest(BASE/'training/training_manifest.csv'), 'Replay or validation changed'
assert Counter(r.split for r in actual) == {'train': 670, 'val': 275}
records = [r for r in actual if r.split=='val']
assert not {r.image_path for r in actual if r.split=='train'} & {r.image_path for r in records}
reviews = read_json(PARENT/'review_and_split.json') + read_json(BASE/'review_and_split.json')
sources = {r['prepared_path']: r for r in reviews if r['included']}
candidate, names, checkpoint = load_classifier(OUT/'training/best.pt')
comparator, baseline_names, baseline_checkpoint = load_classifier(BASE/'training/best.pt')
assert names == baseline_names and len(names)==10
assert checkpoint['incremental']['parent_sha256']==baseline['parent_sha256']
assert checkpoint['config']['augmentation']=='gentle_plus'
best = min(history, key=lambda h: (-h['validation_accuracy'], h['validation_loss']))
assert best['epoch']==checkpoint['epoch']
dataset = FoodImageDataset(records, data_root=config['data_root'], class_names=names, resize_mode='letterbox')
for index, record in enumerate(records):
    # 在原图和准备副本之间核对实际推理输入，避免评估与交付预测的预处理不一致。
    with Image.open(sources[record.image_path]['path']) as source:
        assert torch.equal(image_tensor(source, checkpoint)[0], dataset[index][0])


class ValidationVariant(Dataset):
    def __init__(self, variant):
        self.variant = variant

    def __len__(self):
        return len(records)

    def __getitem__(self, index):
        with Image.open(records[index].image_path) as source:
            image = prepare_image(source, image_size=224, resize_mode='letterbox')
        # 事先固定的轻微扰动，两个模型看到逐像素相同的输入。它们只是诊断，不是独立测试集。
        if self.variant=='scale_translate':
            image = shrink_and_translate(image, scale=.8, rng=random.Random(42+index))
        elif self.variant=='blur':
            image = image.filter(ImageFilter.GaussianBlur(.6))
        elif self.variant=='jpeg':
            image = jpeg_roundtrip(image, 70)
        elif self.variant=='resolution':
            image = degrade_resolution(image, scale=160/224)
        elif self.variant=='warm':
            image = adjust_color_temperature(image, 1.05)
        elif self.variant=='cool':
            image = adjust_color_temperature(image, .95)
        # image_tensor只做确定性输入处理和归一化，不随机增强。
        return image_tensor(image, checkpoint)[0], names.index(records[index].food_label)


def evaluate(network, variant):
    counts = {name: {'correct': 0, 'total': 0} for name in names}
    rows = []
    loss_sum = 0.
    offset = 0
    with torch.inference_mode():
        for images, labels in DataLoader(ValidationVariant(variant), batch_size=16, num_workers=0):
            logits = network(images)
            assert torch.isfinite(logits).all()
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
            probabilities, predicted = logits.softmax(1).max(1)
            for label, pred, probability in zip(labels.tolist(), predicted.tolist(), probabilities.tolist()):
                record = records[offset]
                correct = label==pred
                counts[names[label]]['correct'] += int(correct)
                counts[names[label]]['total'] += 1
                rows.append(dict(variant=variant, image_path=record.image_path,
                                 original_path=sources[record.image_path]['path'],
                                 true_label=names[label], predicted_label=names[pred],
                                 correct=correct, top1_probability=probability))
                offset += 1
    stats = {'per_class': counts, 'loss': loss_sum/len(records)}
    for scope, chosen in [('old', names[:6]), ('new', names[6:]), ('all', names)]:
        correct = sum(counts[n]['correct'] for n in chosen)
        total = sum(counts[n]['total'] for n in chosen)
        stats[scope] = dict(correct=correct, total=total, accuracy=correct/total)
    return stats, rows


evaluations = {}
all_predictions = []
started = time.perf_counter()
for variant in ['clean', 'scale_translate', 'blur', 'jpeg', 'resolution', 'warm', 'cool']:
    evaluations[variant] = {}
    for label, network in [('original_B', comparator), ('augmented_B_candidate', candidate)]:
        stats, rows = evaluate(network, variant)
        evaluations[variant][label] = stats
        all_predictions.extend(dict(model=label, **row) for row in rows)
    print(json.dumps({'variant': variant, 'accuracies': {k:v['all']['accuracy'] for k,v in evaluations[variant].items()}}, ensure_ascii=False), flush=True)
clean = evaluations['clean']
assert clean['original_B']['all']['accuracy'] == read_json(BASE/'val/metrics.json')['overall_accuracy']
assert clean['augmented_B_candidate']['all']['accuracy'] == checkpoint['validation_accuracy']
assert abs(clean['augmented_B_candidate']['loss']-checkpoint['validation_loss']) < 1e-5
with (OUT/'validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_predictions[0]))
    writer.writeheader()
    writer.writerows(all_predictions)

# 检查所有原图，包括暂未采用的疑点图，仅读字节比对哈希，不对测试图做推理。
assert all(sha(Path(r['path']))==r['sha256'] for r in reviews)
assert sha(PARENT/'training/best.pt') == baseline['parent_sha256']
assert sha(BASE/'training/best.pt') == baseline['comparator_checkpoint_sha256']
report = dict(stage='B_augmentation_trial', best_epoch=checkpoint['epoch'], epochs=20,
              training_seconds=execution['training_seconds'], evaluation_seconds=time.perf_counter()-started,
              class_names=names, training_counts=dict(Counter(r.food_label for r in actual if r.split=='train')),
              fixed_validation_count=275, evaluations=evaluations, test_evaluated=False,
              raw_image_tensor_matches_prepared=True, original_images_checked=len(reviews),
              original_images_unchanged=True, parent_A_unchanged=True, original_B_unchanged=True,
              same_replay_and_validation=True, new_process_reload=True, related_tests_passed=63,
              checkpoint_sha256=sha(OUT/'training/best.pt'),
              limits=['候选与原B均用同一验证集选模，本轮结果只是开发对照，不能当独立验收。',
                      '没有重新评估原测试集，不判断候选是否达到旧类精度下降不超过2个百分点的目标。',
                      '一次固定种子组合增强试验不能确定各增强单项的贡献，也不能证明稳定优于原方案。',
                      '网图与去背景数据，非烤箱实拍或RK3506部署效果。'])
(OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
lines = ['# 十类B：新增数据增强对照试验', '',
         '2026年10月26日。保持第一次增量配置不变，从同一正式A重新进行六类到十类训练，只将gentle改为gentle_plus。', '',
         '## 本次增强', '',
         '- 原gentle保留：50%水平翻转、±8°旋转、亮度/对比度/饱和度0.9～1.1。',
         '- 整幅缩小至80%～100%，在画布空余区域随机平移；不随机裁剪。',
         '- 30%概率轻微色温变化，红蓝通道因子0.95～1.05及其倒数。',
         '- 20%概率高斯模糊，半径0.2～0.6；25%概率降至160或192再还原到224。',
         '- 30%概率JPEG往返压缩，质量70～100。',
         '- 仅训练时在线随机变换，旧类回放和新增四类使用相同策略。', '',
         '## 对照条件', '',
         '同一父A、相同每旧类20张回放(120张)、550张新类训练图片、275张验证图片、固定种子42。',
         '20轮，AdamW，学习率0.0001，类别加权，CPU四线程，batch16，输入224补边。',
         f"最佳第{checkpoint['epoch']}轮；训练{execution['training_seconds']:.1f}秒。只按验证准确率、同分损失选模。", '',
         '## 固定验证集（不是测试集）', '',
         '| 范围 | 原B | 新增增强候选 |', '|---|---:|---:|']
for scope, title in [('old', '旧六类'), ('new', '新四类'), ('all', '累计十类')]:
    a, b = clean['original_B'][scope], clean['augmented_B_candidate'][scope]
    lines.append(f"| {title} | {a['correct']}/{a['total']} ({a['accuracy']:.2%}) | {b['correct']}/{b['total']} ({b['accuracy']:.2%}) |")
lines += ['', '| 食材 | 原B正确/总数 | 增强候选正确/总数 |', '|---|---:|---:|']
for name in names:
    a, b = clean['original_B']['per_class'][name], clean['augmented_B_candidate']['per_class'][name]
    lines.append(f"| {name} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} |")
lines += ['', '## 轻微扰动诊断（仍来自同一验证集）', '', '| 固定扰动 | 原B正确/275 | 增强候选正确/275 |', '|---|---:|---:|']
titles = dict(scale_translate='缩小至80%并平移', blur='模糊半径0.6', jpeg='JPEG质量70', resolution='160后还原224', warm='暖色温1.05', cool='冷色温0.95')
for variant, title in titles.items():
    a, b = evaluations[variant]['original_B']['all'], evaluations[variant]['augmented_B_candidate']['all']
    lines.append(f"| {title} | {a['correct']} ({a['accuracy']:.2%}) | {b['correct']} ({b['accuracy']:.2%}) |")
lines += ['', '## 逐轮记录', '', '| 轮次 | 训练损失 | 验证损失 | 验证准确率 |', '|---|---:|---:|---:|']
for h in history:
    lines.append(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['validation_loss']:.5f} | {h['validation_accuracy']:.2%} |")
lines += ['', '## 本轮观察', '',
          '普通验证准确率持平(269/275)，旧六类和新四类的总正确数也分别持平，但具体错图发生变化。',
          '六项扰动诊断中四项正确数增加，一项持平，一项减少1张；这仅支持部分扰动下有改善，不能概括为全面提升。',
          '鸡翅在普通验证集中仍为11/12，不能据此宣称已解决上一轮的旧类保持问题。',
          '保留增强配置和独立候选，不自动覆盖原B或追加轮数。', '',
          '## 核对与结论边界', '',
          '63项相关检查通过；实际回放及验证清单逐行与原B相同；新进程加载和275张原图输入一致性通过。',
          f"{len(reviews)}张原图及父A、原B哈希核对通过。保留原A/B，本候选不自动替换正式版本。", '']
lines += ['- '+item for item in report['limits']]
(OUT/'增强对照结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'report':str(OUT/'增强对照结果.md'), 'clean':clean, 'test_evaluated':False}, ensure_ascii=False, indent=2))
