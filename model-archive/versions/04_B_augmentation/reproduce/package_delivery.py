"""打包中文注释源码、候选权重和验证对照记录，并解压后实际预测。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
report = json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
baseline = json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
parent = Path(baseline['parent_dir'])
dest = ROOT/'outputs/B新增增强试验_20261026'
dest.mkdir(parents=True, exist_ok=True)
archive = dest/'十类B_新增增强候选_代码与权重.zip'
clean = report['evaluations']['clean']
readme = f"""# 十类B：新增数据增强候选

2026年10月26日。这是一次从正式六类A重新进行第一次增量的对照试验。
保持原B的训练图片、旧类回放、验证划分、训练参数与种子不变，仅改变训练增强。
20轮，学习率0.0001，AdamW，batch16，CPU四线程；最佳第{report['best_epoch']}轮。

原gentle增强上增加：整幅缩小80%～100%并在空白内平移；轻微色温；低概率轻微模糊、分辨率下降、JPEG压缩。
不随机裁剪主体；每个类别使用相同策略。验证和预测保持确定性输入。详细概率和范围见records/增强对照结果.md。

固定验证集275张（不是独立测试集）：
- 原B：{clean['original_B']['all']['correct']}/275，{clean['original_B']['all']['accuracy']:.2%}。
- 增强候选：{clean['augmented_B_candidate']['all']['correct']}/275，{clean['augmented_B_candidate']['all']['accuracy']:.2%}。
- 本轮未重新评估原测试集，不能据此判断旧类下降不超过2个百分点的验收目标是否通过。

一次组合增强、单一种子不能证明每种增强的单独贡献，或证明换一个训练种子仍会提升。
候选独立保存，没有覆盖原A或原B；不自动晋升为后续增量的父模型。

## 使用

Python3.10以上，在解压根目录执行：

    python -m pip install -e .
    python predict.py --model models/model_B_augmentation_candidate.pt --image "你的图片.jpg"

src/oven_food/是本次运行封存的源码，模型、训练、数据增强等有中文注释。
models/parent_A.pt为六类父模型，models/model_B_augmentation_candidate.pt为十类候选。
十类顺序：南瓜、吐司、牛排、玉米、西兰花、鸡翅、芦笋、培根、三文鱼、整鸡。
predict.py自动恢复权重中的类别、补边和归一化设置。

records/含配置、训练和回放清单、逐轮记录、验证预测、对照诊断、哈希审计。
reproduce/含本机执行脚本；调整配置和脚本中的绝对路径并另取图片后才能重训。
本包不含原图及准备副本；预览示例位于records/增强示例.jpg，只取训练集图片。

本模型为单标签分类，不支持检测计数、未知类别拒识。网图分数不是烤箱实测或RK3506性能。
"""
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('README.md', readme)
    z.write(ROOT/'pyproject.toml', 'pyproject.toml')
    z.write(OUT/'predict.py', 'predict.py')
    for path in (OUT/'source_snapshot/oven_food').glob('*.py'):
        z.write(path, 'src/oven_food/'+path.name)
    z.write(OUT/'training/best.pt', 'models/model_B_augmentation_candidate.pt')
    z.write(parent/'training/best.pt', 'models/parent_A.pt')
    for name in ['baseline.json', 'trial_protocol.json', 'train_config.json', 'manifest.csv', 'execution.json',
                 'initialization_audit.json', 'comparison.json', '增强对照结果.md',
                 'training.log', 'validation_predictions.csv', 'preview_sources.json', '增强示例.jpg']:
        z.write(OUT/name, 'records/'+name)
    for path in (OUT/'training').glob('*'):
        if path.suffix in {'.json', '.csv'}:
            z.write(path, 'records/training/'+path.name)
    for name in ['run_training.py', 'verify_and_report.py', 'preview.py', 'package_delivery.py']:
        z.write(OUT/name, 'reproduce/'+name)
    for path in (ROOT/'tests').glob('*.py'):
        if path.name.startswith(('test_gentle_plus', 'test_incremental', 'test_training_improvements',
                                 'test_augment', 'test_config', 'test_checkpoint_contract', 'test_dataset',
                                 'test_train_guard', 'test_manifest', 'test_metrics', 'test_cli', 'test_split',
                                 'test_report', '__init__')):
            z.write(path, 'tests/'+path.name)

with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('models/model_B_augmentation_candidate.pt')).hexdigest()==report['checkpoint_sha256']
    assert hashlib.sha256(z.read('models/parent_A.pt')).hexdigest()==baseline['parent_sha256']
    for path in (OUT/'source_snapshot/oven_food').glob('*.py'):
        assert z.read('src/oven_food/'+path.name)==path.read_bytes()
    with tempfile.TemporaryDirectory(prefix='B_augmentation_delivery_') as temp:
        z.extractall(temp)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(temp)/'src')
        with (OUT/'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            expected = next(row for row in csv.DictReader(f)
                            if row['model']=='augmented_B_candidate' and row['variant']=='clean')
        value = subprocess.check_output([sys.executable, '-X', 'utf8', '-B', 'predict.py', '--model',
                        'models/model_B_augmentation_candidate.pt', '--image', expected['original_path']],
                        cwd=temp, env=env, text=True, encoding='utf-8')
        result = json.loads(value)
        assert result['predicted_class']==expected['predicted_label']
        assert len(result['probabilities'])==10
check = dict(archive=str(archive), size_bytes=archive.stat().st_size,
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             integrity='passed', source_snapshot_matches='passed', extracted_prediction='passed')
(OUT/'delivery_check.json').write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(check, ensure_ascii=False, indent=2))
