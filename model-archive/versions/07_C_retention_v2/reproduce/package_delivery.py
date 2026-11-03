"""独立交付改进C：只保存学生部署权重，另附教师供训练复现查阅。"""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,tempfile,zipfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
r=json.loads((OUT/'final_report.json').read_text(encoding='utf-8'))
a=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
clean=r['evaluations']['clean'];old=clean['original_C'];new=clean['retention_C_v2']
dest=ROOT/'outputs/C框架与增强改进_v2_20261029';dest.mkdir(parents=True,exist_ok=True)
archive=dest/'十四类C_v2_知识保持与增强_中文注释代码与权重.zip'
if archive.exists():raise FileExistsError(archive)
readme=f"""# 十四类C v2：训练框架和温和增强改进

2026年10月29日。推理网络仍为MobileNetV3-Small，{r['parameter_count']:,}参数，结构和原C相同。
本轮从同一十类B重新做第二次增量，不从已经遗忘的原C继续训练，不增加食品类别。

## 改动

- 冻结十类教师，只蒸馏教师判断正确的旧训练样本；学生在全14类上归一化，约束新类抢占旧图概率。
- CE+1.0×KL；温度2并乘T²。教师无梯度、BN不更新，不保存进学生权重。
- 骨干学习率是分类头20%；头初始0.0001，余弦下降到初始的5%。
- gentle_retention：保留主体的镜像、小角度旋转和90%～100%缩小平移；轻微色温、gamma、噪声、高质量JPEG。
- 对低清图不继续叠加模糊和降采样。旧profile保持，新增训练选项默认关闭，兼容旧配置。

## 对照与取舍

同一数据和回放：每旧类20张，总729张训练，407张验证，种子42，20轮，CPU四线程。
最佳第{r['best_epoch']}轮，按验证准确率、同分损失选模。

| 普通验证 | 原C | 本候选 |
|---|---:|---:|
| 旧十类 | {old['old']['correct']}/293（{old['old']['accuracy']:.2%}） | {new['old']['correct']}/293（{new['old']['accuracy']:.2%}） |
| 新四类 | {old['new']['correct']}/114（{old['new']['accuracy']:.2%}） | {new['new']['correct']}/114（{new['new']['accuracy']:.2%}） |
| 总体 | {old['all']['correct']}/407（{old['all']['accuracy']:.2%}） | {new['all']['correct']}/407（{new['all']['accuracy']:.2%}） |

旧类验证明显改善，新类验证下降，不能声称各类全面提升。
相对父B的旧验证下降从{r['old_validation_drop_before_pp']:.2f}变为{r['old_validation_drop_after_pp']:.2f}个百分点。
本轮未重新给409张测试图评分；验证口径满足2个百分点不代表测试验收通过。
原C测试4.07个百分点下降的历史记录保留；候选尚需新的独立测试。

## 使用

Python3.10以上，在解压根目录执行：

    python -m pip install -e .
    python predict.py --model models/model_C_v2.pt --image "你的图片.jpg"

models/model_C_v2.pt是部署用的十四类学生；models/teacher_B.pt只在复现训练时使用，推理不需要它。
src/oven_food/有中文注释；continual.py封装蒸馏和分组学习率，train.py复用原入口完成训练。
训练配置新增distillation_weight、distillation_temperature、backbone_lr_scale、lr_schedule；默认分别为0、2、1、none。
推理预处理仍是EXIF/透明白底/等比例补边224/ImageNet归一化，验证和预测无随机增强。

## 记录与边界

records/含配置、训练及回放清单、验证逐张预测、固定扰动诊断和逐轮CE/KL/LR记录。
reproduce/为本机脚本，训练复现需另取完整图片并修改绝对路径；本包不含原始或准备图片，仅包含训练图增强预览。
69项相关检查和只读代码复核通过；老师全程冻结、回放不含评估图片、旧模型及原图哈希核对通过。
保留所有历史权重，本候选不自动覆盖原C。多个训练因素共同变化，缺少逐项消融，不能把收益仅归因于蒸馏。
可颂缺来源且近似图多，历史来源隔离存在风险；单一种子、反复使用的验证集不代表烤箱泛化或RK3506部署。
本模型仍为单标签分类，不支持计数和未知类别拒识。
"""
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('README.md',readme);z.write(ROOT/'pyproject.toml','pyproject.toml');z.write(OUT/'predict.py','predict.py')
    z.write(OUT/'training/best.pt','models/model_C_v2.pt');z.write(a['parent_checkpoint'],'models/teacher_B.pt')
    for p in (OUT/'source_snapshot/oven_food').glob('*.py'):z.write(p,'src/oven_food/'+p.name)
    for name in ['训练框架与增强改进_结果报告.md','final_report.json','data_audit.json','checks_before_training.json','代码复核记录.md',
                 'train_config.json','manifest.csv','expected_training_manifest.csv','execution.json','training.log','verification.log',
                 'preview_sources.json','增强方案对照.jpg','数据复核与划分说明.md','new_split_summary.json']:
        z.write(OUT/name,'records/'+name)
    for folder in ['training','validation']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.csv','.json'}:z.write(p,'records/'+p.relative_to(OUT).as_posix())
    for name in ['run_training.py','preview.py','verify_and_report.py','package_delivery.py']:
        z.write(OUT/name,'reproduce/'+name)
    for name in ['test_retention_training.py','test_gentle_plus.py','test_incremental.py','test_training_improvements.py','test_augment.py',
                 'test_config.py','test_checkpoint_contract.py','test_dataset.py','test_train_guard.py','test_manifest.py','test_metrics.py','test_cli.py','test_split.py','test_report.py','__init__.py']:
        z.write(ROOT/'tests'/name,'tests/'+name)
    z.write(ROOT/'docs/superpowers/specs/2026-10-29-retention-training-design.md','records/设计说明.md')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('models/model_C_v2.pt')).hexdigest()==r['checkpoint_sha256']
    assert hashlib.sha256(z.read('models/teacher_B.pt')).hexdigest()==a['parent_sha256']
    for p in (OUT/'source_snapshot/oven_food').glob('*.py'):assert z.read('src/oven_food/'+p.name)==p.read_bytes()
    with tempfile.TemporaryDirectory(prefix='retention_C_v2_delivery_') as temp:
        assert Path(temp).resolve().parent==Path(tempfile.gettempdir()).resolve()
        z.extractall(temp);env=os.environ.copy();env['PYTHONPATH']=str(Path(temp)/'src')
        with (OUT/'validation/clean/retention_C_v2/predictions.csv').open(encoding='utf-8-sig',newline='') as f:expected=next(csv.DictReader(f))
        text=subprocess.check_output([sys.executable,'-X','utf8','-B','predict.py','--model','models/model_C_v2.pt','--image',expected['image_path']],cwd=temp,env=env,text=True,encoding='utf-8')
        predicted=json.loads(text)
        assert predicted['predicted_class']==expected['predicted_label'] and list(predicted['probabilities'])==r['class_names']
check=dict(archive=str(archive),size_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           integrity='passed',source_and_weights_match='passed',extracted_prediction='passed')
(OUT/'delivery_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
