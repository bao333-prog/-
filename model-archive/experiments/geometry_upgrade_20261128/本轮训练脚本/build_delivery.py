"""从已冻结最终结果建立独立交付及ZIP；此脚本绝不再次评分测试集。"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ('A224','B320','C320')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def copy(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def verify_arm_checkpoint(path,arm,selection):
    """在复制前后都以CPU核对SHA和分支契约，不做推理或测试评分。"""
    import torch
    from predict import validate_arm_payload
    if sha(path)!=selection['checkpoint_sha256']:
        raise RuntimeError('交付权重副本SHA与选定者不一致: '+str(path))
    payload=torch.load(path,map_location='cpu',weights_only=False)
    validate_arm_payload(payload,arm,selection)
    return dict(arm=arm,image_size=payload['image_size'],variant=payload['variant'],epoch=payload['epoch'],
                class_names=payload['class_names'],resize_mode=payload['preprocessing']['resize_mode'],
                checkpoint_sha256=selection['checkpoint_sha256'])


def make_archive(folder, archive):
    folder,archive=Path(folder),Path(archive)
    files=sorted(p for p in folder.rglob('*') if p.is_file())
    forbidden={'data','inputs','foreground_data','foreground_replacements','foreground_representatives','__pycache__'}
    for path in files:
        parts=path.relative_to(folder).parts
        if any(part in forbidden for part in parts):
            raise ValueError('交付禁止包含大原图/训练缓存目录: '+str(path))
    if archive.exists():raise FileExistsError('拒绝覆盖已有ZIP: '+str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in files:stream.write(path,path.relative_to(folder).as_posix())
    verified={}
    with zipfile.ZipFile(archive) as stream:
        bad=stream.testzip()
        if bad is not None:raise RuntimeError('ZIP CRC失败: '+bad)
        if len(stream.namelist())!=len(files) or len(set(stream.namelist()))!=len(files):
            raise RuntimeError('ZIP文件数量/名称不一致')
        for path in files:
            name=path.relative_to(folder).as_posix()
            expected=sha(path);actual=hashlib.sha256(stream.read(name)).hexdigest()
            if actual!=expected:raise RuntimeError('ZIP SHA失败: '+name)
            verified[name]=dict(sha256=expected,crc32=stream.getinfo(name).CRC,bytes=path.stat().st_size)
    return dict(passed=True,verified_files=len(files),archive=str(archive.resolve()),
                archive_sha256=sha(archive),files=verified)


def write_tables(destination,evaluation,selection):
    results=destination/'评测结果'
    results.mkdir(exist_ok=True)
    with (results/'每类验证与最终测试准确率.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['类别','验证正确','验证总数','验证准确率','最终测试正确','最终测试总数','最终测试准确率'])
        for name,val in evaluation['validation']['per_class'].items():
            test=evaluation['test']['per_class'][name]
            writer.writerow([name,val['correct'],val['total'],'' if val['accuracy'] is None else val['accuracy'],
                              test['correct'],test['total'],'' if test['accuracy'] is None else test['accuracy']])
    with (results/'三分支验证对照.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['分支','输入分辨率','选用变体','选用轮次','训练轮次','macro','最差类','整体准确率','loss','checkpoint_SHA256'])
        for arm,item in selection['arms'].items():
            writer.writerow([arm,224 if arm=='A224' else 320,item['variant'],item['epoch'],item['completed_epochs'],
                             *item['key'][:3],item['loss'],item['checkpoint_sha256']])
    dump(results/'最差3类及混淆.json',{split:dict(worst3=evaluation[split]['worst3'],
          top_confusions=evaluation[split]['top_confusions'],confusion_matrix=evaluation[split]['confusion_matrix'])
          for split in ('validation','test')})


def readme_text(evaluation,selection):
    chosen=selection['chosen'];val=evaluation['validation'];test=evaluation['test']
    original_run=Path(chosen['path']).resolve().parents[2]
    original_root=original_run.parents[1]
    return f'''# 模型框架与增强优化交付

最终模型：{chosen['arm']}，输入 {evaluation['image_size']}×{evaluation['image_size']}，变体 {evaluation['variant']}，选用第 {chosen['epoch']} 轮。
验证：{val['correct']}/{val['total']}；最终测试：{test['correct']}/{test['total']}。
只按验证集 macro、最差有样本类、总体准确率、负loss的顺序选定最终模型，三分支全部完成且文件指纹核对通过后才测试最终者一次。

## 单张调用

需要 Python 3.10+、torch、torchvision、Pillow、numpy。权重为 MobileNetV3 Small，固定官方14类；随包 code 提供冻结代码，运行时不会下载预训练权重。

```text
python predict.py 图片路径 --checkpoint model.pt
python predict.py --image 图片路径 --checkpoint model.pt --device cpu
python predict.py 图片路径 --checkpoint model.pt --device cuda:0
```

默认CPU；请求GPU而不可用时明确停止。预测自动使用权重内224/320分辨率、letterbox补边、ImageNet归一化与类别顺序。输入应为完整原图；不需要自行裁剪ROI。
`CPU独立加载验证.json`记录交付副本预测脚本在全新CPU进程中实际加载model.pt并预测一张验证原图的结果。

## 训练复现所需原始位置与额外依赖

本包 requirements.txt 仅推理依赖；本包是非完整自包含训练复现包。可独立运行的是 model.pt、predict.py 与 code 中的预测代码。
“本轮训练脚本”用于归档审查：私有训练/准备脚本按原RUN及 source_snapshot 目录结构工作，配置和来源清单仍保留原电脑的绝对数据、掩码、缓存、初始化权重路径；不能直接在交付归档目录内重新训练。
原RUN：`{original_run.as_posix()}`；训练冻结源码：`{(original_run/'source_snapshot').as_posix()}`。若复现实验，需要保留或恢复这些原始位置和每项SHA对应的数据，另建实验时须明确迁移路径，不能把路径改写后的实验冒充原冻结执行。
离线 foreground.py 额外依赖历史背景修补模块：`{(original_root/'runs/masked_samples_20261123_135614/background_fill.py').as_posix()}`（提供 erase_food_background）；该历史目录没有随本交付打包。
SciPy 从独立依赖目录 `{(original_root/'runs/segmentation_20261122_203914/dependencies').as_posix()}` 加载。离线前景生成还需要 OpenCV（cv2），原工作区使用 `.venv` 环境；训练/推理使用 `.venv-gpu`。这些环境及额外SciPy/OpenCV依赖不包含在推理 requirements.txt 内。
源照片、完整食材掩码、历史背景修补代码和这些独立依赖必须另行备齐；ZIP里的来源及审核清单只提供路径、证据和SHA，不替代数据与运行环境。

## 最终评测恢复边界

验证逐类计数和loss重载一致性、CPU/GPU同一验证原图batch=1诊断、A/B验证小批测速均在测试开始前完成。GPU单张与GPU批量验证top1另行记录，避免混淆两种口径。
测试开始标记紧邻唯一测试调用。测试结束立即保存 test_predictions.json 与 final_evaluation.json，之后才读旧版报告。若旧报告等后处理失败，在原RUN运行 `python finalize_models.py --run 原RUN路径 --postprocess` 可只重试已保存结果的后处理，绝不再次评分测试。
如预检失败且还没有测试开始标记，可在原RUN使用 `--resume-selected` 沿用原冻结选择重试；不会重选模型。普通入口发现已有测试结果或开始标记时会拒绝重新评分。

## 对照含义和限制

A224/B320仅改变输入分辨率；A相对旧版额外加入50% identity概率，因此A与旧版不是仅分辨率相同的严格复现。
C320同时加入前景旋转、场景均衡，并停止target图片整体几何变换；C相对B的变化不能分别归因给其中任一项。
EMA沿同一次live训练轨迹并行维护，没有另训一个EMA模型；最终EMA BN重估仅使用5590个训练原图family，未用验证或测试图片更新BN。
59个新前景是训练来源的派生视图，不是59个独立实物。完整来源和使用计数见来源审核；只有实际使用计数大于0者才可称为参与训练。
foreground_manifest.json 与 foreground_visual_review.json 保留生成候选阶段的历史 pending / training_eligible=false 状态，仅用于溯源。
C320实际唯一训练资格依据是随后主线程逐张查看后生成并冻结的 foreground_reviewed.json：root_visual_review_complete=true，59条均为 review_status=approved、training_eligible=true。旧候选记录不会覆盖最终资格文件，也不表示这59项未经后续核验；该主线程审核不等同于用户逐项人工签字。
场景分组只标识时间相邻且像素近似的保守摆放组，组间不保证物理个体独立。
本次复用既有验证/测试集合，不是新的盲测。牛排与整鸡缺验证样本、整鸡缺测试样本；缺类准确率留空/null，不应作泛化结论。
本机CPU/GPU小批推理测速只用验证样本，不含磁盘及预处理，不是板卡实测，也不能直接代表部署延迟。

## 包内记录

model.pt是最终权重；分支权重目录另存A224/B320/C320各自验证选定者，共3份，保留各自raw/ema/ema_bn变体名。
评测结果包含逐类准确率、最差3类和混淆、三分支验证表、逐轮raw/EMA验证曲线、CPU/GPU诊断及旧版只读对照。
来源与审核保留完整JSON/CSV来源清单、前景审核、59项前景预览索引和实际使用计数、原始文件未改变证据。
本轮训练脚本保留训练与数据准备代码；code目录是预测实际使用的冻结源码。原始照片和大训练缓存不放入ZIP，路径与SHA清单可追溯原位置。
文件指纹.json记录包内每项SHA256；ZIP外的交付校验.json记录全部CRC/SHA复核及ZIP指纹。
'''


def build(run,destination=None):
    run=Path(run).resolve()
    # 所有必要结果先检查，再创建交付目录；不导入训练/最终评测入口。
    selection=load(run/'final_selection.json');evaluation=load(run/'final_evaluation.json')
    integrity=load(run/'final_integrity_check.json');diagnostic=load(run/'cpu_gpu_diagnostic.json')
    if (evaluation.get('test_inference_passes')!=1 or evaluation.get('test_used_for_selection') is not False or
        not integrity.get('passed') or not diagnostic.get('top1_equal') or
        evaluation.get('final_selection_sha256')!=sha(run/'final_selection.json') or
        selection.get('integrity_check_sha256')!=sha(run/'final_integrity_check.json')):
        raise RuntimeError('缺少唯一最终评测/完整性/CPU一致性证据，禁止交付')
    chosen=selection['chosen']
    if (evaluation.get('checkpoint_sha256')!=chosen['checkpoint_sha256'] or
        evaluation.get('arm',chosen['arm'])!=chosen['arm'] or evaluation.get('variant')!=chosen['variant'] or
        evaluation.get('image_size')!=(224 if chosen['arm']=='A224' else 320)):
        raise RuntimeError('最终评测和选择权重不一致')
    for arm,item in selection['arms'].items():
        verify_arm_checkpoint(item['path'],arm,item)
    for path,digest in selection['code_and_configuration_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('冻结后代码/配置/来源清单改变: '+path)
    if destination is None:
        destination=run.parents[1]/'outputs/模型框架与增强优化_20261128_092947'
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('独立交付目录已存在，拒绝覆盖: '+str(destination))
    destination.mkdir(parents=True)
    copy(selection['chosen']['path'],destination/'model.pt')
    final_contract=verify_arm_checkpoint(destination/'model.pt',chosen['arm'],chosen)
    dump(destination/'最终模型元数据复核.json',final_contract)
    copy(run/'predict.py',destination/'predict.py')
    for path in (run/'source_snapshot').rglob('*.py'):
        copy(path,destination/'code'/path.relative_to(run/'source_snapshot'))
    for path in run.glob('*.py'):
        copy(path,destination/'本轮训练脚本'/path.name)
    for arm in ARMS:
        item=selection['arms'][arm]
        archived=destination/'分支权重'/f"{arm}_{item['variant']}.pt"
        copy(item['path'],archived)
        verify_arm_checkpoint(archived,arm,item)
        for name in ['config.json','options.json','selection.json','execution.json','best_candidates.json',
                     'validation_history.json','sample_counts.json','bn_audit.json','training/history.json']:
            copy(run/arm/name,destination/'分支记录'/arm/name)
        for name in ['training.json','raw_training.json','validation.json','test.json','weights.json',
                     'train_validation.csv','test_only.csv','preflight.json','data_audit.json']:
            copy(run/arm/name,destination/'来源与审核'/arm/name)
    required=['final_selection.json','final_evaluation.json','final_integrity_check.json','test_predictions.json',
              'validation_predictions.json','cpu_gpu_diagnostic.json','local_inference_benchmark.json','previous_version_comparison.json']
    for name in required:copy(run/name,destination/'评测结果'/name)
    for path in run.glob('*.json'):
        if path.name.startswith(('foreground_','scene_','frozen_external')) or path.name=='official_class_names.json':
            copy(path,destination/'来源与审核'/path.name)
    # 审核缩略接触表允许交付；不复制源照片或全尺寸训练缓存。
    for directory in [run/'preview',run]:
        patterns=['*.jpg','*.png'] if directory.name=='preview' else ['scene_contact_*.jpg','augmentation_preserve_preview.jpg']
        for pattern in patterns:
            for path in directory.glob(pattern):copy(path,destination/'审核预览'/path.name)
    usage=selection['training_read_audit']['C320']['foreground_usage']
    foreground_rows=[r for r in load(run/'C320/training.json') if r['kind']=='foreground_rotation']
    if len(foreground_rows)!=59:raise RuntimeError('前景视图索引应为59项')
    dump(destination/'来源与审核/59前景预览与使用索引.json',dict(
         independent_physical_samples=False,derived_views=59,
         actually_used=sum(x['count']>0 for x in usage),views=foreground_rows,usage=usage,
         preview_evidence=load(run/'foreground_preview_manifest.json')))
    write_tables(destination,evaluation,selection)
    (destination/'使用说明.md').write_text(readme_text(evaluation,selection),encoding='utf-8')
    (destination/'requirements.txt').write_text('# Inference only; offline foreground generation additionally needs historical background_fill, SciPy and OpenCV.\ntorch>=2.2\ntorchvision>=0.17\nPillow>=9.5\nnumpy>=1.24\n',encoding='utf-8')
    # 用交付副本而非RUN内脚本，证明相对code/model路径在新CPU进程可独立加载。
    validation=load(run/selection['chosen']['arm']/'validation.json')
    sample=validation[0]
    process=subprocess.run([sys.executable,str(destination/'predict.py'),sample['source'],
              '--checkpoint',str(destination/'model.pt'),'--device','cpu'],
              cwd=destination,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True,
              env=dict(__import__('os').environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'))
    prediction=json.loads(process.stdout)
    expected=diagnostic.get('gpu_top1',load(run/'validation_predictions.json')[0]['predicted'])
    standalone=dict(passed=prediction['label']==expected and prediction['image_size']==final_contract['image_size']
                       and prediction['variant']==final_contract['variant'],device='cpu',image_source=sample['source'],
             predicted=prediction['label'],saved_gpu_single_prediction=expected,stdout=prediction,
             predictor_sha256=sha(destination/'predict.py'),checkpoint_sha256=sha(destination/'model.pt'),
             test_set_read=False,new_process=True)
    dump(destination/'CPU独立加载验证.json',standalone)
    if not standalone['passed']:raise RuntimeError('交付副本CPU预测与已保存验证GPU预测不一致')
    verify_arm_checkpoint(destination/'model.pt',chosen['arm'],chosen)
    # 清除单张CPU调用产生的字节码；它不属于交付且不能随ZIP进入包内。
    for cache in destination.rglob('__pycache__'):
        if not cache.resolve().is_relative_to(destination):raise RuntimeError('缓存路径越界')
        shutil.rmtree(cache)
    fingerprints={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    dump(destination/'文件指纹.json',fingerprints)
    verification=make_archive(destination,destination.with_suffix('.zip'))
    # 指纹只覆盖稳定交付文件；自描述的文件指纹.json不递归引用自身。
    for name,digest in fingerprints.items():
        if sha(destination/name)!=digest:raise RuntimeError('打包后文件指纹变化: '+name)
    dump(run/'delivery.json',dict(directory=str(destination),archive=verification['archive'],
            archive_sha256=verification['archive_sha256'],files=verification['verified_files'],
            cpu_standalone_load_passed=True,test_inference_repeated=False))
    dump(destination.parent/(destination.name+'_交付校验.json'),verification)
    return dict(directory=str(destination),archive=verification['archive'],files=verification['verified_files'])


def main():
    parser=argparse.ArgumentParser(description='从已保存的最终结果建立独立模型交付；不再运行测试')
    parser.add_argument('--run',type=Path,default=HERE)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.run,args.output),ensure_ascii=False))


if __name__=='__main__':main()
