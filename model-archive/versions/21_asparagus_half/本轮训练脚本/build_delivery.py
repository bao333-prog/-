"""训练结束后的独立核对、逐类对照与交付；不根据测试结果再改训练。"""
import html
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from inspect_new import RUN, ROOT, PREVIOUS, load, dump, sha


def pct(x): return '缺数据' if x is None else f'{100*x:.2f}%'


def main():
    execution=load(RUN/'execution.json'); assert execution['status']=='completed'
    evaluation=load(RUN/'final_evaluation.json');old=load(PREVIOUS/'final_evaluation.json')
    assert evaluation['checkpoint_sha256']==execution['checkpoint_sha256']==sha(RUN/'training/best.pt')
    history=load(RUN/'training/history.json')
    assert len(history)==execution['epochs'] and all(math.isfinite(r[k]) for r in history for k in ['train_loss','validation_loss','validation_accuracy'])
    best=max(history,key=lambda r:(r['validation_accuracy'],-r['validation_loss']))
    assert best['epoch']==evaluation['best_epoch']
    from continuation import should_stop
    assert should_stop(history)
    for name,digest in load(RUN/'integration_preflight_review.json')['file_sha256'].items():assert sha(RUN/name)==digest,name
    def check(item):
        p,digest=item
        assert sha(p)==digest,p
    frozen=load(RUN/'frozen_external_files.json')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,frozen.items()))
    counts=load(RUN/'sample_counts.json')
    assert counts==load(RUN/'expected_sampling_counts.json')
    new_rows=[r for r in load(RUN/'training_base.json') if r.get('added_this_run')]
    new_reads=[counts.get(r['actual_input'],0) for r in new_rows]
    assert len(new_reads)==121 and min(new_reads)>0
    # 从保存的逐图预测重算每类计数，并保证新旧模型用同一批原始评测图。
    differences={}
    for split in ['validation','test']:
        new_preds=load(RUN/f'{split}_predictions.json');old_preds=load(PREVIOUS/f'{split}_predictions.json')
        assert [(r['source_sha256'],r['true_label']) for r in new_preds]==[(r['source_sha256'],r['true_label']) for r in old_preds]
        for name,stats in evaluation[split]['per_class'].items():
            rows=[r for r in new_preds if r['true_label']==name]
            assert len(rows)==stats['total'] and sum(r['correct'] for r in rows)==stats['correct']
            assert dict(Counter(r['predicted_label'] for r in rows if not r['correct']))==stats['misclassified_as']
        differences[split]=[dict(source=n['source'],true_label=n['true_label'],
              previous_prediction=o['predicted_label'],new_prediction=n['predicted_label'],
              previous_confidence=o['confidence'],new_confidence=n['confidence'])
              for o,n in zip(old_preds,new_preds) if o['predicted_label']!=n['predicted_label']]
    dump(RUN/'prediction_changes.json',differences)
    out=ROOT/'outputs'/('半截芦笋补充模型_'+RUN.name.removeprefix('asparagus_half_'))
    out.mkdir(exist_ok=False)
    shutil.copy2(RUN/'training/best.pt',out/'model.pt');shutil.copy2(RUN/'predict.py',out/'predict.py')
    shutil.copytree(RUN/'source_snapshot/oven_food',out/'code/oven_food',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts=out/'本轮训练脚本';scripts.mkdir()
    for name in ['inspect_new.py','prepare_data.py','prepare_update.py','train_integrated.py','evaluate_final.py',
                 'family_sampler.py','joint_support.py','validation_monitor.py','continuation.py',
                 'test_family_sampler.py','test_continuation.py','verify_membership.py','build_delivery.py']:
        shutil.copy2(RUN/name,scripts/name)
    records=out/'数据与隔离记录';records.mkdir()
    for name in ['training_base.json','validation_base.json','test_base.json','training_integrated.json',
                 'new_inventory_approved.json','new_data_initial_audit.json','prepared_data_audit.json',
                 'integration_preflight_review.json','continuation_decision.json','execution.json',
                 'prediction_changes.json','frozen_external_files.json']:
        shutil.copy2(RUN/name,records/name)
    for a,b in [('train_config.json','训练配置.json'),('training/history.json','逐轮训练记录.json'),
                ('final_evaluation.json','评测结果.json'),('validation_predictions.json','验证逐图预测.json'),
                ('test_predictions.json','测试逐图预测.json')]:shutil.copy2(RUN/a,out/b)
    shutil.copytree(RUN/'review',out/'新增图片核验索引')
    contact_html='<!doctype html><meta charset="utf-8"><title>121张新增半截芦笋</title><style>body{font:16px Microsoft YaHei,sans-serif;max-width:1300px;margin:auto}img{max-width:100%}</style><h1>121张新增半截芦笋核验索引</h1><p>高位60张、中位31张、低位30张，均进训练；源图与JSON未修改。接触表仅供浏览，不是训练输入。</p>'
    contact_html+=''.join(f'<p>第{i}页</p><img src="contact_{i:02d}.jpg">' for i in range(1,8))
    (out/'新增图片核验索引/索引.html').write_text(contact_html,encoding='utf-8')
    names=evaluation['class_names'];lines=[];table=[]
    for name in names:
        v=evaluation['validation']['per_class'][name];t=evaluation['test']['per_class'][name];o=old['test']['per_class'][name]
        row=[name,f"{v['correct']}/{v['total']}",pct(v['accuracy']),f"{o['correct']}/{o['total']}",pct(o['accuracy']),
             f"{t['correct']}/{t['total']}",pct(t['accuracy'])]
        table.append(row)
    total=lambda result:sum(s['correct'] for s in result['per_class'].values())
    lines=[f"半截芦笋补充模型：完成{execution['epochs']}轮，验证选定第{evaluation['best_epoch']}轮，训练耗时{execution['seconds']:.1f}秒。",
       '新增121张：高位60、中位31、低位30，来自一个训练会话；存在连续近似照片，未拆到验证/测试，不等于121个独立实物。',
       '当前5591原图条目（5590个不同文件内容来源）+2011 ROI+94核验合成，共7696训练视图；旧视图逐项不变。',
       '芦笋原图359→480，其中实拍221→342，网图138保持；不额外增加芦笋在加权轮的类别总权重。',
       '从同一官方epoch0预训练骨干/随机最后分类层独立初始化，未继承旧任务权重；保持几何增强，无光照或颜色增强。',
       f"验证{total(evaluation['validation'])}/193（{pct(evaluation['validation']['overall_accuracy'])}）；测试{total(evaluation['test'])}/167（{pct(evaluation['test']['overall_accuracy'])}），上一版测试{total(old['test'])}/167（{pct(old['test']['overall_accuracy'])}）。",'',
       '食材｜验证正确/总数｜验证准确率｜上一版测试正确/总数｜上一版测试准确率｜本版测试正确/总数｜本版测试准确率']
    lines+=['｜'.join(r) for r in table]
    lines+=['','本版测试最低三类（同分按固定类别顺序；缺测试类不参与）：']
    for name in evaluation['test']['worst_three']:
        t=evaluation['test']['per_class'][name]
        errors='；'.join(f'{k}{n}张' for k,n in t['misclassified_as'].items()) or '无误判'
        lines.append(f"{name}：{t['correct']}/{t['total']}，{pct(t['accuracy'])}；{errors}")
    limits='无文件、输入像素或已知采集会话跨集合交叉；实物个体ID不完整，不能证明不同会话完全不同实物。现有测试集已用于历史开发，本次是复用对照，不是新盲测。整鸡缺实拍测试；牛排和整鸡缺验证。单次训练变化不能证明统计显著或完全由补充数据导致。'
    lines+=['',limits,'',f'完整运行目录：{RUN}',f'新增原图路径：{load(RUN/"new_inventory_approved.json")[0]["source"]}',
       '停止判定：'+json.dumps(load(RUN/'continuation_decision.json'),ensure_ascii=False)]
    (out/'训练结果.txt').write_text('\n'.join(lines),encoding='utf-8')
    style='body{font:16px Microsoft YaHei,sans-serif;max-width:1200px;margin:30px auto;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}'
    h='<!doctype html><meta charset="utf-8"><title>半截芦笋补充模型</title><style>'+style+'</style><h1>半截芦笋补充模型</h1>'
    h+=''.join('<p>'+html.escape(x)+'</p>' for x in lines[:6])
    h+='<table><tr>'+''.join('<th>'+x+'</th>' for x in ['食材','验证正确/总数','验证准确率','旧测试正确/总数','旧测试准确率','新测试正确/总数','新测试准确率'])+'</tr>'
    h+=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in table)+'</table>'
    h+=''.join('<p>'+html.escape(x)+'</p>' for x in lines[8+len(table):])
    h+='<p><a href="model.pt">模型权重</a> · <a href="使用说明.txt">使用说明</a> · <a href="训练结果.txt">完整报告</a> · <a href="新增图片核验索引/索引.html">121张图片核验索引</a></p>'
    (out/'训练报告.html').write_text(h,encoding='utf-8')
    (out/'使用说明.txt').write_text('model.pt为14类单标签分类模型，输入完整相机照片，不需要人工画框。\npredict.py与code为对应推理代码。\n本机用项目.venv-gpu环境运行：python predict.py 图片完整路径 --checkpoint model.pt；默认CPU，可加 --device cuda:0。\n中文注释的训练脚本和来源清单已附，重现训练需要保留原运行目录、上一版缓存与原图路径，并非附带全部照片的自包含数据包。\n原始照片和历史模型未覆盖；未上传GitHub。\n'+limits,encoding='utf-8')
    # 交付包作为独立入口，以完整验证原图在CPU上实际加载并预测一次。
    example=load(RUN/'validation_base.json')[0]
    process=subprocess.run([sys.executable,str(out/'predict.py'),example['source'],'--checkpoint',str(out/'model.pt')],
                            capture_output=True,text=True,encoding='utf-8',check=True)
    prediction=json.loads(process.stdout)
    archived=load(RUN/'validation_predictions.json')[0]
    assert prediction['top3'][0]['label']==archived['predicted_label']
    final_checks={'errors':[],'epochs':execution['epochs'],'best_epoch':evaluation['best_epoch'],
       'frozen_source_metadata_and_input_files_unchanged':len(frozen),
       'old_model_unchanged':True,'frozen_code_manifests_unchanged':True,
       'sampling_counts_equal_expected':True,'new_old_eval_same_sources_labels':True,
       'added_training_images_actually_sampled':len(new_reads),
       'added_training_images_min_reads':min(new_reads),'added_training_images_max_reads':max(new_reads),
       'added_training_images_total_reads':sum(new_reads),
       'packaged_checkpoint_sha256':sha(out/'model.pt'),'standalone_cpu_full_image_prediction_agrees':True,
       'cpu_example':prediction}
    dump(RUN/'final_checks.json',final_checks);dump(out/'最终核对.json',final_checks)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    dump(out/'文件指纹.json',files)
    package=out/'新模型_权重代码与报告.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p!=package and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,p.relative_to(out))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for name,digest in files.items():
            import hashlib
            assert hashlib.sha256(z.read(name.replace('\\','/'))).hexdigest()==digest,name
    dump(RUN/'delivery.json',dict(output=str(out),package=str(package),zip_sha256=sha(package),
         checkpoint_sha256=sha(out/'model.pt')))
    print('\n'.join(lines),flush=True);print('交付目录：',out,flush=True)


if __name__=='__main__':main()
