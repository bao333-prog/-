"""交付独立补强候选；保留原权重，检查包内源码和新进程预测。"""
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
report = json.loads((OUT/'final_report.json').read_text(encoding='utf-8'))
audit = json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
dest = ROOT/'outputs/鸡翅补强模型B_20261029'
dest.mkdir(parents=True,exist_ok=True)
archive = dest/'十类B_鸡翅补强_中文注释代码与权重.zip'
if archive.exists():
    raise FileExistsError(archive)
val = report['evaluations']['val']
fresh = report['evaluations']['fresh_chicken_test']
readme = f"""# 十类B：鸡翅补充数据训练候选

2026年10月29日。从最新增强B候选继续十类同标签微调，不增加类别，不是第二次增类。
结构仍为MobileNetV3-Small；保留原A、原B和增强B。

## 数据和训练

新增鸡翅83张训练、18张验证、18张测试预留。使用全部可用旧训练数据，而不是每类20张回放。
隔离新来源核查发现的7张旧鸡翅训练图；实际训练1355张(鸡翅134、其余九类1221)，验证293张。
20轮，lr=0.00005，AdamW，batch16，CPU四线程，seed42，全参数更新，gentle_plus、补边与类别加权。
最佳第{report['best_epoch']}轮，按验证准确率和同分损失选择；模型已在新进程重新加载并预测。

## 本轮结果

- 开发验证293张：增强B父模型{val['parent_augmented_B']['all']['correct']}/293（{val['parent_augmented_B']['all']['accuracy']:.2%}），补强候选{val['chicken_supplement_candidate']['all']['correct']}/293（{val['chicken_supplement_candidate']['all']['accuracy']:.2%}）。
- 新鸡翅测试18张，选模后一次评估：父模型{fresh['parent_augmented_B']['all']['correct']}/18，补强候选{fresh['chicken_supplement_candidate']['all']['correct']}/18。
- 旧277张测试未重测；本轮没有十类整体独立测试成绩，不据此宣称原旧类保持目标通过。

父模型可能已学过本轮剔除的旧来源关联图，当前隔离不能消除历史知识；旧验证仅作开发对照。
数据量和旧类使用方式都变了，不能把全部分数变化仅归因于新鸡翅。网图结果不等于真实烤箱或RK3506效果。

## 预测

Python3.10或更新版本，在解压根目录运行：

    python -m pip install -e .
    python predict.py --model models/model_B_chicken_supplement.pt --image "你的图片.jpg"

models/model_B_chicken_supplement.pt为本次十类候选，models/parent_augmented_B.pt为十类父模型。
类别顺序：南瓜、吐司、牛排、玉米、西兰花、鸡翅、芦笋、培根、三文鱼、整鸡。
predict.py从权重自动恢复类别映射与输入处理；src/oven_food/为训练时封存的中文注释源码。

## 文件和复现

records/有数据隔离、训练实际清单、配置、逐轮记录、验证和新测试逐张预测及混淆矩阵。
reproduce/为执行脚本，依赖原电脑其他运行目录和数据，供查阅。
本包不含训练图片及准备副本；其他电脑重训需要另取数据并修改路径。预测不需下载预训练权重。
本轮没有修改公共模型代码，没有增加网络层或额外类别。单标签分类不支持计数和未知食品拒识。
"""
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('README.md',readme)
    z.write(ROOT/'pyproject.toml','pyproject.toml')
    z.write(OUT/'predict.py','predict.py')
    z.write(OUT/'training/best.pt','models/model_B_chicken_supplement.pt')
    z.write(audit['parent_checkpoint'],'models/parent_augmented_B.pt')
    for path in (OUT/'source_snapshot/oven_food').glob('*.py'):
        z.write(path,'src/oven_food/'+path.name)
    for name in ['鸡翅补强训练_结果报告.md','final_report.json','data_audit.json','execution.json','preflight_checks.json',
                 'incoming_data_review_summary.json','train_config.json','manifest.csv','frozen_new_chicken_test.csv',
                 'training.log','verification.log','执行计划与记录.md']:
        z.write(OUT/name,'records/'+name)
    for folder in ['training','val','fresh_chicken_test']:
        for path in (OUT/folder).rglob('*'):
            if path.is_file() and path.suffix in {'.json','.csv'}:
                z.write(path,'records/'+path.relative_to(OUT).as_posix())
    for name in ['prepare_run.py','run_training.py','verify_and_report.py','package_delivery.py']:
        z.write(OUT/name,'reproduce/'+name)
    for name in ['usage_summary.json','reviewed_inventory.json','逐张复核与使用建议.csv','新增候选建议划分.csv']:
        z.write(ROOT/'runs/chicken_wings_review_20261029_171649'/name,'records/incoming_review/'+name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('models/model_B_chicken_supplement.pt')).hexdigest()==report['checkpoint_sha256']
    assert hashlib.sha256(z.read('models/parent_augmented_B.pt')).hexdigest()==audit['parent_sha256']
    for path in (OUT/'source_snapshot/oven_food').glob('*.py'):
        assert z.read('src/oven_food/'+path.name)==path.read_bytes()
    with tempfile.TemporaryDirectory(prefix='B_wings_supplement_delivery_') as temp:
        assert Path(temp).resolve().parent==Path(tempfile.gettempdir()).resolve()
        z.extractall(temp)
        env=os.environ.copy();env['PYTHONPATH']=str(Path(temp)/'src')
        with (OUT/'fresh_chicken_test/chicken_supplement_candidate/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            expected=next(csv.DictReader(f))
        completed=subprocess.run([sys.executable,'-X','utf8','-B','predict.py','--model','models/model_B_chicken_supplement.pt',
                                  '--image',expected['image_path']],cwd=temp,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
        result=json.loads(completed.stdout)
        assert result['predicted_class']==expected['predicted_label']
        assert list(result['probabilities'])==report['class_names']
checks=dict(archive=str(archive),size_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            integrity='passed',weights_match='passed',source_snapshot_matches='passed',extracted_prediction='passed')
(OUT/'delivery_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
