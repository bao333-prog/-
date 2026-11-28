"""固定最佳权重后，新进程评测全部类别、检查输入并输出交付包。

南瓜6张验证与训练同源，仅为诊断；选模使用另外77张。原测试不训练。
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / 'source_snapshot'))
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import ManifestRecord, read_manifest, validate_manifest
from oven_food.model import build_model, checkpoint_class_names, checkpoint_resize_mode


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_write(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(r):
    return f'{r["correct"]}/{r["total"]}（{r["correct"]/r["total"]:.2%}）' if r['total'] else '缺数据'


def evaluate(checkpoint, records, tag, names):
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert checkpoint_class_names(cp, observed_labels=[r.food_label for r in records]) == names
    assert checkpoint_resize_mode(cp) == 'letterbox'
    model = build_model(cp['model_name'], num_classes=len(names), pretrained=False)
    model.load_state_dict(cp['model_state'], strict=True)
    model.to('cuda:0').eval()
    dataset = FoodImageDataset(records, data_root=ROOT, class_names=names,
                              augmentation_profile='baseline', image_size=224, resize_mode='letterbox')
    matrix = np.zeros((14, 14), dtype=np.int64)
    predictions = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=32, num_workers=0):
            logits = model(images.to('cuda:0'))
            assert torch.isfinite(logits).all()
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels.to('cuda:0'), reduction='sum'))
            probabilities = logits.softmax(1).cpu()
            for target, predicted, prob in zip(labels.tolist(), probabilities.argmax(1).tolist(), probabilities):
                r = records[len(predictions)]
                matrix[target, predicted] += 1
                predictions.append(dict(image=r.image_path, expected=names[target], predicted=names[predicted], probability=float(prob[predicted])))
    assert len(predictions) == len(records)
    csv_write(RUN / (tag + '_predictions.csv'), predictions)
    per = {n: dict(correct=int(matrix[i, i]), total=int(matrix[i].sum())) for i, n in enumerate(names)}
    weakest = sorted([n for n in names if per[n]['total']], key=lambda n: (per[n]['correct']/per[n]['total'], names.index(n)))[:3]
    worst = [dict(label=n, **per[n], misclassified_as={names[j]: int(v) for j, v in enumerate(matrix[names.index(n)]) if v and names[j] != n}) for n in weakest]
    return dict(correct=int(matrix.trace()), total=int(matrix.sum()), loss=loss_sum/len(records), per_class=per,
                confusion_matrix=matrix.tolist(), worst_three=worst)


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audit = read(RUN / 'data_audit.json')
    config = read(RUN / 'train_config.json')
    names = audit['class_names']
    assert len(names) == 14 and config['augmentation'] == 'oven_geometry'
    records = read_manifest(RUN / 'manifest.csv')
    training = [r for r in records if r.split == 'train']
    selection = [r for r in records if r.split == 'val']
    validation = read_manifest(RUN / 'validation_all_manifest.csv')
    diagnostic = read_manifest(RUN / 'diagnostic_pumpkin_manifest.csv')
    tests = read_manifest(RUN / 'test_manifest.csv')
    assert tuple(map(len, [training, selection, validation, diagnostic, tests])) == (3651, 77, 83, 6, 84)
    assert all(r.food_label == '南瓜' for r in diagnostic) and all(r.food_label != '南瓜' for r in selection)
    validate_manifest(records, data_root=ROOT, check_files=True)
    assert {r.session_id for r in diagnostic} <= {r.session_id for r in training}
    assert all(r.split != 'test' for r in records)
    history = read(RUN / 'training/history.json')
    execution = read(RUN / 'execution.json')
    counts = read(RUN / 'sample_counts.json')
    assert execution['status'] == 'completed' and len(history) == 10
    assert [r['epoch'] for r in history] == list(range(1, 11))
    assert all(np.isfinite(r[k]) for r in history for k in ['train_loss', 'validation_loss', 'validation_accuracy'])
    assert set(counts) == {r.image_path for r in training} and set(counts.values()) == {10}
    inventory = read(RUN / 'training_inventory.json')
    support = read(RUN / 'training_snapshot.json')
    eval_inventory = read(RUN / 'eval_inventory.json')
    assert len(inventory) == 2811 and len(support) == 3651 and len(eval_inventory) == 167
    assert {r['prepared'] for r in inventory} == {r.image_path for r in training if r.environment == 'target'}
    assert len([r for r in training if r.environment == 'proxy']) == 840
    assert not {r['image'] for r in inventory} & set(audit['removed_sources'])
    for r in inventory:
        assert sha(r['image']) == sha(r['fixed_image']) == r['raw_sha256']
        assert sha(r['prepared']) == r['prepared_sha256']
        assert sha(r['fixed_metadata']) == r['fixed_metadata_sha256']
        if r['metadata_sha256']:
            assert sha(r['metadata']) == r['metadata_sha256']
    for r in support:
        assert sha(r['image']) == r['sha256']
    for r in eval_inventory:
        assert sha(r['source']) == sha(r['copy']) == r['sha256']
        assert sha(r['metadata_source']) == sha(r['metadata_copy']) == r['metadata_sha256']
        assert sha(r['prepared']) == r['prepared_sha256']
    assert not {r['raw_sha256'] for r in inventory} & {r['sha256'] for r in eval_inventory}
    assert not {r['sha256'] for r in support} & {r['prepared_sha256'] for r in eval_inventory}
    assert len([r for r in inventory if r['label'] == '可颂']) == 143
    assert len([r for r in inventory if r['label'] == '南瓜']) == 172
    cp_path = RUN / 'training/best.pt'
    parent_path = Path(config['checkpoint'])
    assert sha(parent_path) == audit['parent_sha256']
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    assert cp['class_names'] == parent['class_names'] == names
    assert cp['model_name'] == 'mobilenet_v3_small' and not cp['incremental'] and not cp['distillation']
    assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
    assert {k:tuple(v.shape) for k,v in cp['model_state'].items()} == {k:tuple(v.shape) for k,v in parent['model_state'].items()}
    assert any(not torch.equal(v, parent['model_state'][k]) for k,v in cp['model_state'].items() if k.endswith('weight'))
    best = sorted(history, key=lambda r: (-r['validation_accuracy'], r['validation_loss']))[0]
    assert cp['epoch'] == best['epoch'] and cp['validation_accuracy'] == best['validation_accuracy']
    hashes = dict(parent=sha(parent_path), trained=sha(cp_path))
    print('10轮、3651图每图10次、来源指纹及14类映射通过；固定权重后评分。', flush=True)

    old_snapshot = read(ROOT / 'runs/food_verify_20261107/input_snapshot.json')
    old_test = []
    for r in old_snapshot:
        if r['split'] == 'test':
            assert sha(r['image']) == r['sha256']
            old_test.append(ManifestRecord(r['prepared'], r['label'], 'target', r['source_group'], 'fixed', 'oven', 'none', 'none', 'test'))
    assert len(old_test) == 213
    result = dict(class_names=names, checkpoint_sha256=hashes['trained'], parent_sha256=hashes['parent'], epochs=10,
                  best_epoch=cp['epoch'], execution=execution, training_images=3651, real_training_images=2811, replay_images=840,
                  selected_validation_images=77, validation_report_images=83, pumpkin_validation_diagnostic_only=6,
                  test_images=84, independent_test=False, missing_evaluation_classes=['整鸡'], data_delivery=audit['data_delivery'],
                  gpu=dict(name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda))
    result['selection_after'] = evaluate(cp_path, selection, 'selection_validation77', names)
    assert result['selection_after']['correct']/77 == cp['validation_accuracy']
    assert abs(result['selection_after']['loss'] - cp['validation_loss']) < 1e-5
    for label, checkpoint in [('after', cp_path), ('before', parent_path)]:
        result['validation_' + label] = evaluate(checkpoint, validation, label + '_all_validation83', names)
        result['test_' + label] = evaluate(checkpoint, tests, label + '_new_test84', names)
        result['old_test_' + label] = evaluate(checkpoint, old_test, label + '_old_test213', names)
    assert sha(cp_path) == hashes['trained'] and sha(parent_path) == hashes['parent']

    # 使用实际交付的预测入口在CPU读取原图，检查模型可直接使用。
    predictor_path = ROOT / 'runs/C_retention_v2_20261029_231408/predict.py'
    spec = importlib.util.spec_from_file_location('delivery_predict', predictor_path)
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    cpu_model, cpu_names, cpu_cp = predictor.load_classifier(cp_path)
    sample = next(r for r in eval_inventory if r['split'] == 'test' and r['label'] == '可颂')
    raw_prediction = predictor.predict_image(cpu_model, cpu_names, cpu_cp, sample['copy'])
    with (RUN / 'after_new_test84_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        expected = next(r for r in csv.DictReader(f) if r['image'] == sample['prepared'])
    assert raw_prediction['predicted_class'] == expected['predicted']
    result['raw_cpu_prediction_smoke'] = raw_prediction
    dump(RUN / 'final_report.json', result)
    lines = ['# 南瓜、可颂补充训练结果', '',
             '先交付339张清理后数据，再从最新几何增强14类权重继续训练10轮。去掉8张手部遮挡南瓜、新增143张可颂；172张南瓜已在原训练中，不重复追加。',
             '实际训练3651张＝2811实拍＋840历史图回放，每张读取10次；RTX 5060 Laptop GPU，头学习率0.00005，骨干比例0.2，批量32，AdamW及余弦调度，标签平滑0.05。',
             '仅缩放、水平翻转和±6°旋转，不增加光照或颜色变化。可颂成品范围按用户最新说明允许；1对重复训练图保留。',
             '最佳第' + str(cp['epoch']) + '轮，总训练耗时' + f'{execution["seconds"]:.2f}' + '秒。', '',
             '选模仅使用77张验证。完整验证83张中，另6张南瓜与训练同源，训练后仅作诊断；完整测试84张，其中南瓜6张也与训练同源。验证/测试还有连续照片相关性，均为开发对照，不能作独立验收。整鸡无评测数据，模型仍有14类输出。',
             '原测试没有用于梯度训练。原图与旧模型未改写；父模型此前曾训练8张被剔除的图，本次只是不再读取它们，未宣称消除历史影响。', '',
             '| 固定评测范围 | 补充前 | 补充后 |', '|---|---:|---:|']
    for label, key in [('完整新验证83（含6张南瓜诊断）','validation'), ('新开发测试84','test'), ('旧开发测试213','old_test')]:
        lines.append('| ' + label + ' | ' + fmt(result[key + '_before']) + ' | ' + fmt(result[key + '_after']) + ' |')
    lines += ['', '以上每行在相同图片上比较，不将旧72张的97.22%直接与新84张总分比较。旧213也已公开复用，仅用于观察旧场景变化，测试错误不触发追加训练。', '',
              '| 食材 | 实拍训练 | 历史回放 | 完整新验证 | 新开发测试 | 旧开发测试 |', '|---|---:|---:|---:|---:|---:|']
    rows = []
    real_counts = Counter(r['label'] for r in inventory)
    for n in names:
        v = result['validation_after']['per_class'][n]
        t = result['test_after']['per_class'][n]
        o = result['old_test_after']['per_class'][n]
        lines.append('| ' + n + ' | ' + str(real_counts[n]) + ' | 60 | ' + fmt(v) + ('，仅诊断' if n == '南瓜' else '') + ' | ' + fmt(t) + ' | ' + fmt(o) + ' |')
        rows.append(dict(食材=n, 实拍训练=real_counts[n], 历史回放=60, 新验证=fmt(v), 新测试=fmt(t), 旧测试=fmt(o), 南瓜验证仅诊断=n=='南瓜'))
    for key, label in [('test_after','新84张测试最低三类'), ('old_test_after','旧213张测试最低三类')]:
        lines += ['', label + '（缺数据不排名；同分按类别顺序）：', '']
        for r in result[key]['worst_three']:
            wrong = '、'.join(n + str(c) + '张' for n,c in r['misclassified_as'].items()) or '无误判（100%并列）'
            lines.append('- ' + r['label'] + '：' + fmt(r) + '；' + wrong + '。')
    lines += ['', '训练数值均有限，保存与新进程严格重载、14类映射、原图CPU预测、训练抽样次数及来源指纹均通过核验。原始图、历史配置/权重均保留。']
    (RUN / '补充训练结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    csv_write(RUN / '每类准确率.csv', rows)
    csv_write(RUN / '每轮训练指标.csv', history)
    OUT = ROOT / 'outputs' / ('C南瓜可颂补充_' + RUN.name.removeprefix('C_pumpkin_croissant_'))
    OUT.mkdir(exist_ok=False)
    package = OUT / '十四类食材识别_南瓜可颂补充_代码与权重.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.md', '# 南瓜可颂补充14类分类模型\n\n安装：pip install -e .\n预测：python predict.py --model models/model_C_pumpkin_croissant.pt --image 图片路径\n\n逐类成绩和局限见records/补充训练结果.md。训练3651张，验证77选模＋6南瓜诊断，测试84。都是开发对照，南瓜评测同训练来源，整鸡缺评测。完整原始数据另行交付，模型包不附训练原图；历史清单使用本机路径，复训需调整配置副本。\n')
        z.write(cp_path, 'models/model_C_pumpkin_croissant.pt')
        z.write(ROOT / 'pyproject.toml', 'pyproject.toml')
        z.write(predictor_path, 'predict.py')
        for p in (RUN / 'source_snapshot/oven_food').glob('*.py'):
            z.write(p, 'src/oven_food/' + p.name)
        for name in ['manifest.csv','validation_all_manifest.csv','diagnostic_pumpkin_manifest.csv','test_manifest.csv','training_inventory.json','training_snapshot.json','eval_inventory.json','data_audit.json','train_config.json','sample_counts.json','training.log','execution.json','training/history.json','training/config.json','training/summary.json','final_report.json','补充训练结果.md','每类准确率.csv','每轮训练指标.csv','run_training.py','verify_and_package.py','方案与执行记录.md']:
            z.write(RUN / name, 'records/' + name)
        for p in RUN.glob('*_predictions.csv'):
            z.write(p, 'records/' + p.name)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('models/model_C_pumpkin_croissant.pt')).hexdigest() == hashes['trained']
        for p in (RUN / 'source_snapshot/oven_food').glob('*.py'):
            assert hashlib.sha256(z.read('src/oven_food/' + p.name)).hexdigest() == sha(p)
    for name in ['补充训练结果.md','每类准确率.csv','每轮训练指标.csv','final_report.json']:
        shutil.copy2(RUN / name, OUT / name)
    checks = dict(status='passed', training_images=3651, each_image_reads=10, epochs=10, class_count=14,
                  model_reloaded_in_new_process=True, source_images_unchanged=True, parent_sha256=hashes['parent'], checkpoint_sha256=hashes['trained'],
                  selection_only77=True, pumpkin6_validation_only_diagnostic=True, test84_not_training=True,
                  independent_test=False, package=str(package), package_sha256=sha(package), report=str(OUT / '补充训练结果.md'))
    dump(RUN / 'final_checks.json', checks)
    dump(OUT / '核验记录.json', checks)
    (ROOT / 'runs/latest_C_pumpkin_croissant.txt').write_text(str(RUN), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), package=str(package), best_epoch=cp['epoch'], selected_validation=result['selection_after']['correct'],
                          validation=result['validation_after']['correct'], test=result['test_after']['correct'], old_test=result['old_test_after']['correct'],
                          worst_three=result['test_after']['worst_three']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
