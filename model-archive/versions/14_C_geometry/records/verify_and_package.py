"""训练结束后在新进程中重载、评分和打包；测试结果不参与选模。

验证与测试虽没有完全相同的图片，但11个拍摄组跨集合，因此属于开发对照。
保留这项限制，不通过改组名或重新命名图片掩盖来源关系。
"""
from pathlib import Path
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
from oven_food.augment import prepare_image
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest, validate_manifest
from oven_food.model import build_model, checkpoint_class_names, checkpoint_resize_mode

NAMES = ['南瓜', '吐司', '牛排', '玉米', '西兰花', '鸡翅', '芦笋', '培根', '三文鱼', '整鸡', '土豆', '香肠', '薯条', '可颂']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(result):
    return f'{result["correct"]}/{result["total"]}（{result["correct"]/result["total"]:.2%}）' if result['total'] else '缺数据'


def evaluate(checkpoint, records, tag):
    """所有14类共同竞争；推理不增强，记录每张预测和混淆矩阵。"""
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    names = checkpoint_class_names(cp, observed_labels=[r.food_label for r in records])
    assert names == NAMES and checkpoint_resize_mode(cp) == 'letterbox'
    model = build_model(cp['model_name'], num_classes=len(names), pretrained=False)
    model.load_state_dict(cp['model_state'], strict=True)
    model.to('cuda:0').eval()
    data = FoodImageDataset(records, data_root=ROOT, class_names=names,
                           augmentation_profile='baseline', image_size=224, resize_mode='letterbox')
    matrix = np.zeros((14, 14), dtype=np.int64)
    predictions = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in DataLoader(data, batch_size=32, num_workers=0):
            logits = model(images.to('cuda:0'))
            assert torch.isfinite(logits).all()
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels.to('cuda:0'), reduction='sum'))
            probabilities = logits.softmax(1).cpu()
            for target, predicted, prob in zip(labels.tolist(), probabilities.argmax(1).tolist(), probabilities):
                record = records[len(predictions)]
                matrix[target, predicted] += 1
                predictions.append(dict(image=record.image_path, expected=names[target], predicted=names[predicted],
                                        probability=float(prob[predicted])))
    assert len(predictions) == len(records)
    csv_write(RUN / (tag + '_predictions.csv'), predictions)
    del model
    return dict(correct=int(matrix.trace()), total=int(matrix.sum()), loss=loss_sum / len(records),
                per_class={name: dict(correct=int(matrix[i, i]), total=int(matrix[i].sum())) for i, name in enumerate(names)},
                confusion_matrix=matrix.tolist())


def main():
    assert torch.cuda.is_available(), '本轮在GPU上验证，不自动回退CPU'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read(RUN / 'train_config.json')
    audit = read(RUN / 'data_audit.json')
    history = read(RUN / 'training/history.json')
    execution = read(RUN / 'execution.json')
    records = read_manifest(RUN / 'manifest.csv')
    train = [r for r in records if r.split == 'train']
    validation = [r for r in records if r.split == 'val']
    test = read_manifest(RUN / 'test_manifest.csv')
    diagnostics = read_manifest(RUN / 'old_diagnostic_manifest.csv')
    old_validation = [r for r in diagnostics if r.environment == 'target']
    web = [r for r in diagnostics if r.environment == 'proxy']
    validate_manifest(records, data_root=ROOT, check_files=True)
    validate_manifest(test, data_root=ROOT, check_files=True)
    assert (len(train), len(validation), len(test), len(old_validation), len(web)) == (3516, 71, 72, 214, 407)
    assert all(r.split != 'test' for r in records)
    assert len({r.session_id for r in validation} & {r.session_id for r in test}) == 11
    assert audit['independent_test'] is False and audit['missing_evaluation_classes'] == ['南瓜', '整鸡', '可颂']
    assert execution['status'] == 'completed' and config['epochs'] == 10 and config['augmentation'] == 'oven_geometry'
    assert len(history) == 10 and [r['epoch'] for r in history] == list(range(1, 11))
    assert all(np.isfinite(r[k]) for r in history for k in ['train_loss', 'validation_loss', 'validation_accuracy'])
    counts = read(RUN / 'sample_counts.json')
    assert set(counts) == {r.image_path for r in train} and set(counts.values()) == {10}

    # 先确认最佳权重只由新验证集决定，然后才打开测试评分。
    checkpoint = RUN / 'training/best.pt'
    parent = Path(config['checkpoint'])
    parent_sha = sha(parent)
    checkpoint_sha = sha(checkpoint)
    assert parent_sha == audit['parent_sha256']
    old = torch.load(parent, map_location='cpu', weights_only=True)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert cp['class_names'] == old['class_names'] == NAMES
    assert cp['model_name'] == 'mobilenet_v3_small' and not cp['incremental'] and not cp['distillation']
    assert {k: tuple(v.shape) for k, v in cp['model_state'].items()} == {k: tuple(v.shape) for k, v in old['model_state'].items()}
    assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
    assert any(not torch.equal(v, old['model_state'][k]) for k, v in cp['model_state'].items() if k.endswith('weight'))
    best = sorted(history, key=lambda r: (-r['validation_accuracy'], r['validation_loss']))[0]
    assert cp['epoch'] == best['epoch'] and cp['validation_accuracy'] == best['validation_accuracy']

    # 全部来源图、副本、JSON与输入缓存复核，既防止误改，也检查精确重复。
    inventory = read(RUN / 'training_inventory.json')
    eval_inventory = read(RUN / 'eval_inventory.json')
    support = read(RUN / 'training_snapshot.json')
    assert len(inventory) == 2676 and len(eval_inventory) == 143 and len(support) == 3516
    assert {r['prepared'] for r in inventory} == {r.image_path for r in train if r.environment == 'target'}
    assert len([r for r in train if r.environment == 'proxy']) == 840
    assert {r['image'] for r in support} == {r.image_path for r in train}
    for r in inventory:
        assert sha(r['image']) == sha(r['fixed_image']) == r['raw_sha256']
        assert sha(r['prepared']) == r['prepared_sha256']
        if r['metadata_sha256']:
            assert sha(r['metadata']) == r['metadata_sha256']
        assert sha(r['fixed_metadata']) == r['fixed_metadata_sha256']
    for r in support:
        assert sha(r['image']) == r['sha256']
    for r in eval_inventory:
        assert sha(r['source']) == sha(r['fixed_source']) == sha(r['copy']) == r['sha256']
        assert sha(r['metadata_source']) == sha(r['metadata_copy']) == r['metadata_sha256']
        assert sha(r['prepared']) == r['prepared_sha256']
        with Image.open(r['source']) as raw, Image.open(r['prepared']) as cached:
            assert np.array_equal(np.asarray(prepare_image(raw, image_size=224, resize_mode='letterbox')),
                                  np.asarray(prepare_image(cached, image_size=224, resize_mode='letterbox')))
    assert not {r['raw_sha256'] for r in inventory} & {r['sha256'] for r in eval_inventory}
    assert not {r['sha256'] for r in support} & {r['prepared_sha256'] for r in eval_inventory}
    assert not {r['session'] + '|' + r['label'] for r in inventory} & {r['source_session'] + '|' + r['label'] for r in eval_inventory}
    assert len({r['sha256'] for r in eval_inventory}) == len({r['pixel_sha256'] for r in eval_inventory}) == 143
    old_inputs = read(ROOT / 'runs/food_verify_20261107/input_snapshot.json')
    for r in old_inputs:
        if r['split'] == 'val':
            assert sha(r['image']) == r['sha256']
    print('输入及10轮训练核验通过；固定权重后开始评分。', flush=True)

    gpu = dict(name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda)
    result = dict(stage='second_batch_geometry_new_development_split', class_names=NAMES, gpu=gpu,
                  epochs=10, best_epoch=cp['epoch'], parent_sha256=parent_sha, checkpoint_sha256=checkpoint_sha,
                  training_images=3516, real_training_images=2676, replay_images=840, execution=execution,
                  independent_test=False, validation_test_shared_source_groups=11,
                  missing_evaluation_classes=audit['missing_evaluation_classes'], split_directory=audit['split_directory'],
                  selection='new validation71 only; test72 after checkpoint fixed; old214/web407 diagnostics only',
                  validation_after=evaluate(checkpoint, validation, 'new_validation'),
                  test_after=evaluate(checkpoint, test, 'new_test'),
                  validation_before=evaluate(parent, validation, 'parent_new_validation'),
                  test_before=evaluate(parent, test, 'parent_new_test'),
                  old_validation_after=evaluate(checkpoint, old_validation, 'old_validation_diagnostic'),
                  web_after=evaluate(checkpoint, web, 'web_diagnostic'),
                  old_validation_before=evaluate(parent, old_validation, 'parent_old_validation_diagnostic'),
                  web_before=evaluate(parent, web, 'parent_web_diagnostic'))
    assert result['validation_after']['correct'] / 71 == cp['validation_accuracy']
    assert abs(result['validation_after']['loss'] - cp['validation_loss']) < 1e-5
    # 包内预测入口以原图在CPU执行，另核对它与评测缓存的类别结果一致。
    spec = importlib.util.spec_from_file_location('package_predict', ROOT / 'runs/C_retention_v2_20261029_231408/predict.py')
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    cpu_model, names, cpu_cp = predictor.load_classifier(checkpoint)
    sample = next(r for r in eval_inventory if r['split'] == 'test')
    raw_prediction = predictor.predict_image(cpu_model, names, cpu_cp, sample['copy'])
    with (RUN / 'new_test_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
        predictions = list(csv.DictReader(handle))
    expected = next(r for r in predictions if r['image'] == sample['prepared'])
    assert raw_prediction['predicted_class'] == expected['predicted']
    result['raw_cpu_predict_smoke'] = raw_prediction
    per = result['test_after']['per_class']
    matrix = result['test_after']['confusion_matrix']
    weakest = sorted([n for n in NAMES if per[n]['total']], key=lambda n: (per[n]['correct']/per[n]['total'], NAMES.index(n)))[:3]
    result['worst_three_test'] = [dict(label=n, **per[n], misclassified_as={NAMES[j]: int(v) for j, v in enumerate(matrix[NAMES.index(n)]) if v and NAMES[j] != n}) for n in weakest]
    dump(RUN / 'final_report.json', result)
    rows = []
    report = ['# 第二批纯几何增强训练结果', '',
              'GPU完成10轮，最佳第' + str(cp['epoch']) + '轮；3516张训练图（2676实拍＋840旧网图），每张每轮读取一次。总耗时' + f'{execution["seconds"]:.2f}' + '秒。',
              '从南瓜补强14类权重重新开始本轮微调，未接着上一次CPU第二批权重继续。架构和14类顺序保持一致。',
              '批量32，头学习率0.00005、骨干比例0.2，AdamW/权重衰减0.0001/余弦调度/标签平滑0.05。224等比例缩放补边。',
              '增强仅水平翻转50%、±6°旋转及适配全画面后的95–100%缩放；不增加光照、颜色、噪声或模糊扰动。', '',
              '清理后143张候选按每类近1:1固定分为71验证、72测试；种子42。测试不在训练清单中，仅模型固定后评分。旧214实拍验证与407网图只作训练后诊断。',
              '**11个拍摄组跨验证/测试，不能作为独立验收。** 两边无相同字节/像素的图片，但相近连续照片仍相关；71/72张仅覆盖11类，南瓜、整鸡、待烤可颂缺数据。',
              '原始图片未改动，暂缓366张不恢复使用。训练图已全部解码/指纹核验；人工只抽查分组代表图，未宣称逐图完成食品审核。205张新增牛排来自同一拍摄会话。', '',
              '| 同一评测范围 | 父模型 | 本轮模型 |', '|---|---:|---:|']
    for label, before, after in [('新验证', 'validation_before', 'validation_after'), ('新开发测试', 'test_before', 'test_after'), ('旧实拍验证诊断', 'old_validation_before', 'old_validation_after'), ('旧网图诊断', 'web_before', 'web_after')]:
        report.append('| ' + label + ' | ' + fmt(result[before]) + ' | ' + fmt(result[after]) + ' |')
    report += ['', '上次CPU第二批模型的87.79%来自旧213张测试，不能直接与本次72张成绩比较。本表父模型是本轮实际初始化的南瓜补强模型。', '',
               '| 食材 | 新验证 | 新开发测试 |', '|---|---:|---:|']
    for n in NAMES:
        v = result['validation_after']['per_class'][n]
        t = per[n]
        report.append('| ' + n + ' | ' + fmt(v) + ' | ' + fmt(t) + ' |')
        rows.append(dict(食材=n, 验证正确=v['correct'], 验证总数=v['total'], 验证准确率=v['correct']/v['total'] if v['total'] else '',
                         测试正确=t['correct'], 测试总数=t['total'], 测试准确率=t['correct']/t['total'] if t['total'] else ''))
    report += ['', '测试中最低三类（缺数据不参与，同准确率按既定类别顺序）：', '']
    for r in result['worst_three_test']:
        report.append('- ' + r['label'] + '：' + fmt(r) + '；误判：' + ('、'.join(n + str(c) + '张' for n, c in r['misclassified_as'].items()) or '无') + '。')
    report += ['', '训练损失使用类别权重及标签平滑；验证损失是普通交叉熵，两者数值不能直接作同口径大小比较。',
               '本轮仅训练与报告，不依据测试错误追加训练。新143张已经用于开发，不能再称封存盲测。旧模型、旧配置及原图保留。',
               '独立验收仍需另采每类独立食材/摆盘批次。未新增检测、计数或层位功能，未进行板卡部署验收。']
    (RUN / '第二批几何增强训练结果.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    csv_write(RUN / '每类准确率.csv', rows)
    csv_write(RUN / '每轮训练指标.csv', history)
    OUT = ROOT / 'outputs' / ('C第二批几何增强_' + RUN.name.removeprefix('C_geometry_eval2_'))
    OUT.mkdir(exist_ok=False)
    package = OUT / '十四类食材识别_第二批几何增强_代码与权重.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('README.md', '# 第二批纯几何增强14类分类模型\n\n安装：pip install -e .\n预测：python predict.py --model models/model_C_geometry_eval2.pt --image 图片路径\n\nrecords/第二批几何增强训练结果.md 包含每类成绩与误判。新验证71、测试72，同源组跨集合，仅作开发对照，不代表独立验收。缺南瓜、整鸡、待烤可颂新评测。历史清单含本机路径，不附原始图片；复训需另备数据并调整配置副本。\n')
        archive.write(checkpoint, 'models/model_C_geometry_eval2.pt')
        archive.write(ROOT / 'pyproject.toml', 'pyproject.toml')
        archive.write(ROOT / 'runs/C_retention_v2_20261029_231408/predict.py', 'predict.py')
        for p in (RUN / 'source_snapshot/oven_food').glob('*.py'):
            archive.write(p, 'src/oven_food/' + p.name)
        for name in ['manifest.csv', 'test_manifest.csv', 'old_diagnostic_manifest.csv', 'training_inventory.json', 'training_snapshot.json', 'eval_inventory.json', 'data_audit.json', 'train_config.json', 'sample_counts.json', 'training.log', 'execution.json', 'training/history.json', 'training/config.json', 'training/summary.json', 'final_report.json', '第二批几何增强训练结果.md', '每类准确率.csv', '每轮训练指标.csv', 'run_training.py', '方案与划分说明.md', 'verify_and_package.py']:
            archive.write(RUN / name, 'records/' + name)
        for p in RUN.glob('*_predictions.csv'):
            archive.write(p, 'records/' + p.name)
        for name in ['每类数量.csv', '逐图划分清单.csv', '划分说明.json', '使用说明.md']:
            archive.write(Path(audit['split_directory']) / name, 'data_split/' + name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert hashlib.sha256(archive.read('models/model_C_geometry_eval2.pt')).hexdigest() == checkpoint_sha
        assert len([n for n in archive.namelist() if n.startswith('src/oven_food/')]) == 16
    assert sha(checkpoint) == checkpoint_sha and sha(parent) == parent_sha
    for name in ['第二批几何增强训练结果.md', '每类准确率.csv', '每轮训练指标.csv', 'final_report.json']:
        shutil.copy2(RUN / name, OUT / name)
    checks = dict(status='passed', epochs=10, training_images=3516, each_image_reads=10, class_count=14,
                  source_images_unchanged=True, model_reloaded_in_new_process=True, raw_cpu_predict_checked=True,
                  validation_matches_training=True, independent_test=False, validation_test_shared_source_groups=11,
                  parent_sha256=parent_sha, checkpoint_sha256=checkpoint_sha, package=str(package), package_sha256=sha(package),
                  report=str(OUT / '第二批几何增强训练结果.md'), split_directory=audit['split_directory'])
    dump(RUN / 'final_checks.json', checks)
    dump(OUT / '核验记录.json', checks)
    (ROOT / 'runs/latest_C_geometry_eval2.txt').write_text(str(RUN), encoding='utf-8')
    print(json.dumps(dict(run=str(RUN), output=str(OUT), best_epoch=cp['epoch'], validation=result['validation_after']['correct'],
                          test=result['test_after']['correct'], total_test=72, package=str(package), worst_three=result['worst_three_test']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
