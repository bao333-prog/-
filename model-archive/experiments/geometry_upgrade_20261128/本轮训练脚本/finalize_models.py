"""三分支完成后先冻结验证选模，再以全新进程对最终模型做一次测试。"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARMS = ('A224', 'B320', 'C320')
HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value, *, exclusive=False):
    path = Path(path)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validation_key(stats, loss):
    counts = list(stats['per_class'].values())
    for entry in counts + [stats]:
        if any(type(entry.get(k)) is not int for k in ('correct','total')) or not 0 <= entry['correct'] <= entry['total']:
            raise ValueError('损坏的验证计数')
    if sum(x['correct'] for x in counts) != stats['correct'] or sum(x['total'] for x in counts) != stats['total']:
        raise ValueError('逐类和总体计数不一致')
    rates = [x['correct']/x['total'] for x in counts if x['total']]
    if not rates or not math.isfinite(loss) or loss < 0:
        raise ValueError('无效的验证统计/loss')
    return [math.fsum(rates)/len(rates), min(rates), stats['correct']/stats['total'], -loss]


def choose_from_completed_arms(run):
    import torch
    from predict import validate_arm_payload
    run = Path(run)
    if (run/'final_evaluation.json').exists() or (run/'test_inference_started.json').exists():
        raise RuntimeError('已有最终评测或测试推理标记，拒绝再次评分测试集')
    arms = {}
    for arm in ARMS:
        directory = run/arm
        execution = load(directory/'execution.json')
        if execution.get('status') != 'completed' or execution.get('test_used_for_selection', False):
            raise RuntimeError(arm + '未完成或选模使用了测试数据')
        selected = load(directory/'selection.json')
        if selected.get('test_used') is not False:
            raise ValueError(arm + '缺少test_used:false证明')
        expected = validation_key(selected['stats'], selected['loss'])
        if len(selected['key']) != 4 or any(abs(a-b)>1e-12 for a,b in zip(expected,selected['key'])):
            raise ValueError(arm + '保存的选模键与验证计数不一致')
        candidates = load(directory/'best_candidates.json')
        for candidate in candidates.values():
            key = validation_key(candidate['stats'], candidate['loss'])
            if key != candidate['key']:
                raise ValueError(arm + '候选选模键损坏')
        if tuple(selected['key']) != max(tuple(x['key']) for x in candidates.values()):
            raise ValueError(arm + '选定者不是raw/EMA/BN候选中的验证最优')
        path = Path(selected['path'])
        if not path.is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('选定权重必须是本分支内的绝对路径')
        digest = sha(path)
        if digest != selected['checkpoint_sha256'] or digest != execution.get('checkpoint_sha256'):
            raise ValueError(arm + '权重SHA与完成记录不一致')
        if Path(execution['selected']).resolve() != path.resolve():
            raise ValueError(arm + '完成记录选定路径不一致')
        variants = [name for name,x in candidates.items() if Path(x['path']).resolve()==path.resolve() and x['key']==selected['key']]
        if len(variants) != 1:
            raise ValueError('无法唯一对应选定的raw/EMA/BN变体')
        arms[arm] = dict(selected, arm=arm, variant=variants[0], completed_epochs=execution['epochs'])
        # 控制进程只在CPU读取元数据，不做模型推理，也不接触测试结果。
        payload=torch.load(path,map_location='cpu',weights_only=False)
        validate_arm_payload(payload,arm,arms[arm])
    # 固定ARMS顺序在完全相同验证键时确定性打破平局，不访问测试或旧报告。
    chosen = max(arms.values(), key=lambda item: tuple(item['key']))
    return dict(chosen), arms


def check_file_integrity(sources):
    files = {}
    for source in sources:
        for path, digest in source.items():
            path = str(Path(path))
            if not Path(path).is_absolute() or not isinstance(digest,str) or len(digest)!=64:
                raise ValueError('完整性清单必须使用绝对路径及SHA256')
            if path in files and files[path] != digest:
                raise ValueError('冻结清单对同一路径给出了冲突SHA: ' + path)
            files[path] = digest
    def check(item):
        path, expected = item
        try:
            actual = sha(path)
            return None if actual == expected else {'path':path,'expected':expected,'actual':actual}
        except OSError as exc:
            return {'path':path,'expected':expected,'error':str(exc)}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = [x for x in pool.map(check, files.items()) if x is not None]
    return dict(passed=not errors, unique_files=len(files), errors=errors,
                threads=4, seconds=time.perf_counter()-started, file_sha256=files)


def metrics_from_predictions(names, actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError('预测与标签数量不符或为空')
    matrix = [[0]*len(names) for _ in names]
    for a,p in zip(actual,predicted):
        if not 0 <= a < len(names) or not 0 <= p < len(names):
            raise ValueError('预测下标越界')
        matrix[a][p] += 1
    per = {name: dict(correct=matrix[i][i], total=sum(matrix[i]),
            accuracy=matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None) for i,name in enumerate(names)}
    rates = [v['accuracy'] for v in per.values() if v['total']]
    worst = sorted([dict(label=n,**v) for n,v in per.items() if v['total']],
                   key=lambda x:(x['accuracy'],-x['total'],x['label']))[:3]
    for item in worst:
        i = names.index(item['label'])
        item['confusions'] = [dict(predicted=names[j],count=count) for j,count in
                sorted(enumerate(matrix[i]), key=lambda x:-x[1]) if j!=i and count][:3]
    confusions = [dict(true=names[i],predicted=names[j],count=matrix[i][j])
                  for i in range(len(names)) for j in range(len(names)) if i!=j and matrix[i][j]]
    confusions.sort(key=lambda x:(-x['count'],x['true'],x['predicted']))
    correct = sum(matrix[i][i] for i in range(len(names)))
    return dict(correct=correct,total=len(actual),accuracy=correct/len(actual),
                macro_accuracy=math.fsum(rates)/len(rates),min_class_accuracy=min(rates),
                per_class=per,confusion_matrix=matrix,worst3=worst,top_confusions=confusions[:3],
                missing_classes=[n for n,v in per.items() if not v['total']])


def freeze_selection(run):
    run = Path(run).resolve()
    chosen, arms = choose_from_completed_arms(run)
    if (run/'final_selection.json').exists():
        raise RuntimeError('final_selection已存在；先审核已有冻结结果，禁止自动覆盖重选')
    sources = [load(run/'frozen_external_files.json'), load(run/'foreground_frozen_external.json')]
    if len(sources[0])!=17853 or len(sources[1])!=177:
        raise RuntimeError('源/缓存冻结清单数量不符，拒绝不完整核对')
    read_audit, frozen = {}, {}
    for arm in ARMS:
        directory = run/arm
        preflight = load(directory/'preflight.json')
        if preflight.get('approved') is not True or preflight.get('errors'):
            raise RuntimeError(arm+' preflight未批准')
        sources.append(preflight['file_sha256'])
        rows, raw, counts = load(directory/'training.json'), load(directory/'raw_training.json'), load(directory/'sample_counts.json')
        families = {r['family_id'] for r in raw}
        read_families = {r['family_id'] for r in rows if counts.get(r['actual_input'],0)>0}
        missing_original = sorted({r['actual_input'] for r in raw if counts.get(r['actual_input'],0)<=0})
        if len(families)!=5590 or families-read_families or missing_original:
            raise RuntimeError(arm+'原图/family实际读取覆盖不完整')
        foreground = [r for r in rows if r['kind']=='foreground_rotation']
        read_audit[arm] = dict(training_families=len(families),read_families=len(read_families),
            original_unique_paths=len({r['actual_input'] for r in raw}),all_original_paths_read=True,
            reads=sum(counts.values()),foreground_views=len(foreground),
            foreground_views_actually_used=sum(counts.get(r['actual_input'],0)>0 for r in foreground),
            foreground_usage=[dict(family_id=r['family_id'],actual_input=r['actual_input'],
                count=counts.get(r['actual_input'],0),label=r['label']) for r in foreground])
        bn = load(directory/'bn_audit.json')
        if bn.get('split')!='train' or bn.get('images')!=5590 or bn.get('validation_or_test_used_for_bn') is not False:
            raise RuntimeError(arm+' EMA BN仅训练原图证据不符')
        for name in ['config.json','options.json','training.json','raw_training.json','validation.json','test.json',
                     'weights.json','train_validation.csv','test_only.csv','preflight.json',
                     'selection.json','execution.json','best_candidates.json','sample_counts.json','bn_audit.json',
                     'validation_history.json','training/history.json']:
            path = directory/name
            frozen[str(path)] = sha(path)
    for path in sorted(run.glob('*.py')) + sorted((run/'source_snapshot').rglob('*.py')):
        frozen[str(path)] = sha(path)
    for name in ['official_class_names.json','scene_index.json','scene_audit.json','foreground_reviewed.json',
                 'foreground_verification.json','foreground_visual_review.json','frozen_external_files.json','foreground_frozen_external.json']:
        frozen[str(run/name)] = sha(run/name)
    sources.append(frozen)
    sources.append({item['path']:item['checkpoint_sha256'] for item in arms.values()})
    integrity = check_file_integrity(sources)
    dump(run/'final_integrity_check.json', integrity)
    if not integrity['passed']:
        raise RuntimeError('冻结源文件/缓存/代码完整性失败，禁止评测；见final_integrity_check.json')
    frozen_selection = dict(version=1,chosen=chosen,arms=arms,test_used_for_selection=False,
         selection_rule='macro_accuracy,min_class_accuracy,overall_accuracy,negative_loss',
         tie_break='A224,B320,C320固定顺序',integrity_check_sha256=sha(run/'final_integrity_check.json'),
         code_and_configuration_sha256=frozen,training_read_audit=read_audit,
         previous_report_read_for_selection=False,source_preflight_sha256={a:sha(run/a/'preflight.json') for a in ARMS})
    dump(run/'final_selection.json', frozen_selection, exclusive=True)
    return frozen_selection


def evaluate_rows(model, payload, rows, *, device, code_root, batch_size=32):
    import torch
    from predict import CLASS_NAMES, prepare_tensor
    predictions, actual, predicted, loss_sum = [], [], [], 0.
    with torch.inference_mode():
        for offset in range(0,len(rows),batch_size):
            batch = rows[offset:offset+batch_size]
            images = torch.stack([prepare_tensor(r['actual_input'],payload,code_root=code_root) for r in batch]).to(device)
            labels = torch.tensor([CLASS_NAMES.index(r['label']) for r in batch],device=device)
            logits = model(images)
            if logits.shape != (len(batch),14) or not torch.isfinite(logits).all():
                raise FloatingPointError('无效的14类推理输出')
            loss_sum += torch.nn.functional.cross_entropy(logits,labels,reduction='sum').item()
            probabilities = logits.softmax(1).cpu()
            for row, label, prob in zip(batch,labels.cpu().tolist(),probabilities):
                p = int(prob.argmax())
                actual.append(label); predicted.append(p)
                predictions.append(dict(source=row['source'],actual_input=row['actual_input'],
                     family_id=row['family_id'],true=row['label'],predicted=CLASS_NAMES[p],
                     correct=label==p,confidence=float(prob[p]),probabilities=prob.tolist()))
    metrics = metrics_from_predictions(CLASS_NAMES,actual,predicted)
    metrics['loss'] = loss_sum/len(rows)
    return metrics,predictions


def local_benchmark(run, arms, *, device, samples=4, repeats=8):
    import torch
    from predict import load_predictor, prepare_tensor, validate_arm_payload
    results = {}
    for arm in ('A224','B320'):
        rows = load(run/arm/'validation.json')[:samples]
        for bench_device in ('cpu',device):
            model,payload = load_predictor(arms[arm]['path'],device=bench_device,code_root=run/'source_snapshot')
            validate_arm_payload(payload,arm,arms[arm])
            images = torch.stack([prepare_tensor(r['actual_input'],payload,code_root=run/'source_snapshot') for r in rows]).to(bench_device)
            def sync():
                if bench_device.startswith('cuda'): torch.cuda.synchronize(bench_device)
            with torch.inference_mode():
                for _ in range(3): model(images)
                sync(); started=time.perf_counter()
                for _ in range(repeats): model(images)
                sync(); seconds=time.perf_counter()-started
            results[arm+'_'+bench_device] = dict(image_size=payload['image_size'],batch_size=len(rows),
                repeats=repeats,ms_per_batch=1000*seconds/repeats,ms_per_image=1000*seconds/(repeats*len(rows)),
                includes_preprocessing_or_io=False,validation_samples_only=True)
            del model,images
            if bench_device.startswith('cuda'): torch.cuda.empty_cache()
    return dict(results=results,cpu_threads=torch.get_num_threads(),
         gpu=torch.cuda.get_device_name(torch.device(device)),
         note='本机小批验证样本推理耗时；不含磁盘/预处理，非板卡实际测速，不能代表部署延迟。')


def validate_reloaded_validation(metrics, chosen, *, loss_rtol=1e-5, loss_atol=1e-6):
    """验证失败在测试marker创建前终止，损失允许浮点求和次序带来的微差。"""
    expected=chosen['stats']
    if (metrics.get('correct')!=expected['correct'] or metrics.get('total')!=expected['total'] or
        set(metrics.get('per_class',{}))!=set(expected['per_class']) or
        any((v['correct'],v['total']) != (expected['per_class'][n]['correct'],expected['per_class'][n]['total'])
            for n,v in metrics['per_class'].items())):
        raise RuntimeError('重载权重完整图验证逐类计数与选模时不一致；未评分测试')
    loss=metrics.get('loss')
    if not isinstance(loss,(int,float)) or not math.isfinite(loss) or not math.isclose(loss,chosen['loss'],rel_tol=loss_rtol,abs_tol=loss_atol):
        raise RuntimeError('重载权重验证loss超过容差；未评分测试')


def single_image_diagnostic(run, chosen, row, batch_prediction, *, device):
    """两个设备都以batch=1读取同一完整验证原图；另记GPU批量口径差异。"""
    from predict import predict_image
    image=row['source']
    cpu=predict_image(chosen['path'],image,device='cpu',code_root=run/'source_snapshot')
    gpu=predict_image(chosen['path'],image,device=device,code_root=run/'source_snapshot')
    diagnostic=dict(source=image,actual_input=row['actual_input'],cpu_batch_size=1,gpu_batch_size=1,
        cpu_top1=cpu['label'],gpu_top1=gpu['label'],gpu_batch_top1=batch_prediction['predicted'],
        top1_equal=cpu['label']==gpu['label'],
        gpu_single_vs_batch_top1_equal=gpu['label']==batch_prediction['predicted'],
        validation_sample_only=True,same_full_original_source=True)
    if not diagnostic['top1_equal']:
        dump(run/'cpu_gpu_diagnostic.json',diagnostic)
        raise RuntimeError('同一验证原图CPU/GPU单张batch=1 top1不一致；尚未评分测试')
    return diagnostic


def postprocess_saved_results(run):
    """只用正式保存的结果补旧版对照；可重试，不加载模型、不再次评分测试。"""
    run=Path(run).resolve()
    try:
        result=load(run/'final_evaluation.json')
        predictions=load(run/'test_predictions.json')
        if (result.get('test_inference_passes')!=1 or result.get('test_used_for_selection') is not False or
            result.get('final_selection_sha256')!=sha(run/'final_selection.json') or
            len(predictions)!=result['test']['total'] or
            result.get('test_predictions_sha256')!=sha(run/'test_predictions.json')):
            raise RuntimeError('已保存的正式结果或预测指纹不符；后处理禁止重算')
        previous_path=run.parent/'asparagus_half_20261126_172248/final_evaluation.json'
        previous=load(previous_path)
        dump(run/'previous_version_comparison.json',dict(previous_report=str(previous_path),
            previous_report_sha256=sha(previous_path),read_after_final_selection=True,
            read_after_saved_final_evaluation=True,prior_report=previous,
            current_validation=result['validation'],current_test=result['test'],
            caution='同一固定测试集的复用对照，不是新的盲测；未用旧结果重选/调参。'))
        status=dict(status='completed',final_evaluation_sha256=sha(run/'final_evaluation.json'),
                    test_inference_repeated=False,models_loaded=False)
        dump(run/'postprocess_status.json',status)
        return status
    except Exception as exc:
        dump(run/'postprocess_status.json',dict(status='failed',error=repr(exc),
                  retry='finalize_models.py --run RUN --postprocess',test_inference_repeated=False))
        raise


def score_test_once(run, selection, *, device, val_metrics, val_predictions, diagnostic, benchmark, test_evaluator):
    """唯一测试事务边界：所有验证诊断已通过后才占用marker，立即保存正式结果。"""
    run=Path(run).resolve()
    if (run/'final_evaluation.json').exists() or (run/'test_inference_started.json').exists():
        raise RuntimeError('已有最终结果或唯一测试尝试标记，禁止重新评分；仅允许--postprocess')
    chosen=selection['chosen']
    validate_reloaded_validation(val_metrics,chosen)
    if len(val_predictions)!=val_metrics['total']:
        raise RuntimeError('验证预测数量不完整；未评分测试')
    if (diagnostic.get('top1_equal') is not True or diagnostic.get('cpu_batch_size')!=1 or
        diagnostic.get('gpu_batch_size')!=1):
        raise RuntimeError('需要通过CPU/GPU同batch=1单张验证诊断；未评分测试')
    selection_sha=sha(run/'final_selection.json')
    # 这些磁盘写入和可失败的验证测速均先于marker，不消耗唯一测试机会。
    dump(run/'validation_predictions.json',val_predictions)
    dump(run/'cpu_gpu_diagnostic.json',diagnostic)
    dump(run/'local_inference_benchmark.json',benchmark)
    dump(run/'test_inference_started.json',dict(selection_sha256=selection_sha,
        checkpoint_sha256=chosen['checkpoint_sha256'],arm=chosen['arm'],device=device,
        test_attempt=1,test_selection_complete=True,all_pretest_checks_completed=True),exclusive=True)
    # marker紧邻真正测试调用，期间不运行验证、测速、CPU诊断或读旧报告。
    test_metrics,test_predictions=test_evaluator()
    dump(run/'test_predictions.json',test_predictions,exclusive=True)
    result=dict(arm=chosen['arm'],variant=chosen['variant'],image_size=224 if chosen['arm']=='A224' else 320,
        device=device,checkpoint=chosen['path'],checkpoint_sha256=chosen['checkpoint_sha256'],
        final_selection_sha256=selection_sha,test_used_for_selection=False,test_inference_passes=1,
        test_predictions_sha256=sha(run/'test_predictions.json'),
        validation=val_metrics,test=test_metrics,arms_validation_comparison=selection['arms'],
        cpu_gpu_diagnostic=diagnostic,local_benchmark=benchmark,
        validation_reload_loss_tolerance=dict(relative=1e-5,absolute=1e-6),
        limitations=['验证缺牛排和整鸡；测试缺整鸡，缺类准确率为null，不能据此声称这些类泛化。',
          '测试集复用，不是新的盲测。','A/B只改变分辨率；A相对旧版额外50% identity。',
          'C同时改变前景旋转、场景均衡及target停止全图变换，不能拆分归因。',
          'EMA沿同一次live训练轨迹并行；最终BN只用5590训练原图。',
          '59个新前景是派生视图，非独立实物；只有使用计数>0者才实际用于训练。'])
    # 测试完成后没有任何外部读操作或耗时任务插入正式结果保存之前。
    dump(run/'final_evaluation.json',result,exclusive=True)
    postprocess_saved_results(run)
    return result


def evaluation_worker(run, *, device, selection_sha):
    import torch
    from predict import CLASS_NAMES, load_predictor, require_device, validate_arm_payload
    run = Path(run).resolve()
    require_device(device,torch)
    if not str(device).startswith('cuda'):
        raise RuntimeError('正式最终评测必须使用可用GPU；不会静默回退CPU')
    if (sha(run/'final_selection.json') != selection_sha or (run/'final_evaluation.json').exists() or
        (run/'test_inference_started.json').exists()):
        raise RuntimeError('冻结选择改变或已完成评测')
    selection = load(run/'final_selection.json')
    if sha(run/'final_integrity_check.json') != selection['integrity_check_sha256']:
        raise RuntimeError('完整性核对报告改变')
    chosen = selection['chosen']; arm = chosen['arm']
    if sha(chosen['path']) != chosen['checkpoint_sha256']:
        raise RuntimeError('最终权重发生变化')
    if load(run/'official_class_names.json') != CLASS_NAMES:
        raise ValueError('官方类别顺序不一致')
    for path,digest in selection['code_and_configuration_sha256'].items():
        if sha(path)!=digest: raise RuntimeError('最终冻结后的代码/配置变化: '+path)
    for candidate_arm,item in selection['arms'].items():
        if sha(item['path'])!=item['checkpoint_sha256']:raise RuntimeError('选定分支权重改变')
        validate_arm_payload(torch.load(item['path'],map_location='cpu',weights_only=False),candidate_arm,item)
    test,validation = load(run/arm/'test.json'),load(run/arm/'validation.json')
    for rows,split,size in [(test,'test',167),(validation,'val',193)]:
        if len(rows)!=size or any(r['kind']!='original' or r['split']!=split or r.get('cache_crop_xyxy') is not None for r in rows):
            raise RuntimeError('必须使用完整图原始验证193/测试167清单')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,payload=load_predictor(chosen['path'],device=device,code_root=run/'source_snapshot')
    validate_arm_payload(payload,arm,chosen)
    val_metrics,val_predictions=evaluate_rows(model,payload,validation,device=device,code_root=run/'source_snapshot')
    validate_reloaded_validation(val_metrics,chosen)
    diagnostic=single_image_diagnostic(run,chosen,validation[0],val_predictions[0],device=device)
    benchmark=local_benchmark(run,selection['arms'],device=device)
    result=score_test_once(run,selection,device=device,val_metrics=val_metrics,val_predictions=val_predictions,
        diagnostic=diagnostic,benchmark=benchmark,
        test_evaluator=lambda:evaluate_rows(model,payload,test,device=device,code_root=run/'source_snapshot'))
    print(json.dumps(dict(arm=arm,validation_correct=val_metrics['correct'],
                          test_correct=result['test']['correct'],test_total=result['test']['total'])))


def main():
    parser=argparse.ArgumentParser(description='先完成三arm完整性与验证选模冻结，再唯一测试最终者')
    parser.add_argument('--run',type=Path,default=HERE)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--postprocess',action='store_true',help='仅从已保存正式结果补旧版对照；不加载模型，不重跑测试')
    parser.add_argument('--resume-selected',action='store_true',help='尚未开始测试时沿用已冻结选择重试预检；不重新选模')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--selection-sha',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if sum([args.worker,args.postprocess,args.resume_selected])>1:
        parser.error('--worker、--postprocess、--resume-selected不可同时使用')
    if args.postprocess:
        print(json.dumps(postprocess_saved_results(args.run),ensure_ascii=False))
    elif args.worker:
        evaluation_worker(args.run,device=args.device,selection_sha=args.selection_sha)
    else:
        import torch
        from predict import require_device
        require_device(args.device,torch)
        if not args.device.startswith('cuda'): raise RuntimeError('正式评测需要GPU')
        if args.resume_selected:
            if (args.run/'final_evaluation.json').exists() or (args.run/'test_inference_started.json').exists():
                raise RuntimeError('已开始或完成唯一测试；仅能--postprocess，禁止重评分')
            load(args.run/'final_selection.json')
        else:
            freeze_selection(args.run)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run',str(args.run.resolve()),
             '--device',args.device,'--worker','--selection-sha',sha(args.run/'final_selection.json')],check=True)


if __name__=='__main__':
    main()
