"""在训练结束后另起进程重载模型，核对输入并输出逐类成绩和交付包。"""
from pathlib import Path
from collections import Counter
import csv,hashlib,importlib.util,json,shutil,sys,zipfile
import numpy as np
import torch
ROOT=Path.cwd();RUN=Path((ROOT/'.superpowers/latest_batch2_training.txt').read_text(encoding='utf-8').strip())
sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.manifest import read_manifest,ManifestRecord

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fmt(r):return f'{r["correct"]}/{r["total"]}（{r["correct"]/r["total"]:.2%}）' if r['total'] else '缺实拍数据'

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 audit=json.loads((RUN/'data_audit.json').read_text(encoding='utf-8'))
 cfg=json.loads((RUN/'train_config.json').read_text(encoding='utf-8'))
 history=json.loads((RUN/'training/history.json').read_text(encoding='utf-8'))
 inventory=json.loads((RUN/'inventory.json').read_text(encoding='utf-8'))
 support=json.loads((RUN/'support_snapshot.json').read_text(encoding='utf-8'))
 counts=json.loads((RUN/'sample_counts.json').read_text(encoding='utf-8'))
 records=read_manifest(RUN/'manifest.csv');train=[r for r in records if r.split=='train']
 assert len(history)==10 and all(np.isfinite(r[k]) for r in history for k in ['train_loss','validation_loss','validation_accuracy'])
 assert len(train)==3516 and len(inventory)==2676
 assert set(counts)=={r.image_path for r in train} and set(counts.values())=={10}
 checkpoint=RUN/'training/best.pt';parent=Path(cfg['checkpoint'])
 assert sha(parent)==audit['parent_sha256']
 old=torch.load(parent,map_location='cpu',weights_only=True);cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert cp['class_names']==old['class_names'] and len(cp['class_names'])==14
 assert cp['model_name']=='mobilenet_v3_small' and not cp['incremental'] and not cp['distillation']
 assert {k:tuple(v.shape) for k,v in cp['model_state'].items()}=={k:tuple(v.shape) for k,v in old['model_state'].items()}
 assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
 assert any(not torch.equal(v,old['model_state'][k]) for k,v in cp['model_state'].items() if k.endswith('weight'))
 best=sorted(history,key=lambda r:(-r['validation_accuracy'],r['validation_loss']))[0]
 assert cp['epoch']==best['epoch'] and cp['validation_accuracy']==best['validation_accuracy']
 for r in inventory:
  assert sha(r['image'])==sha(r['fixed_image'])==r['raw_sha256']
  assert sha(r['prepared'])==r['prepared_sha256']
  if r['metadata_sha256']:assert sha(r['metadata'])==r['metadata_sha256']
  assert sha(r['fixed_metadata'])==r['fixed_metadata_sha256']
 for r in support:assert sha(r['image'])==r['sha256']
 snapshot=json.loads((ROOT/'runs/food_verify_20261107/input_snapshot.json').read_text(encoding='utf-8'))
 for r in snapshot:assert sha(r['image'])==r['sha256']
 assert not {r['raw_sha256'] for r in inventory} & {r['sha256'] for r in snapshot}
 validation=[r for r in records if r.split=='val' and r.environment=='target']
 web=[r for r in records if r.split=='val' and r.environment=='proxy']
 test=[ManifestRecord(r['prepared'],r['label'],'target','verify_'+r['source_group'],'fixed','oven','none','none','test') for r in snapshot if r['split']=='test']
 assert len(validation)==214 and len(test)==213 and len(web)==407
 train_hashes={sha(r.image_path) for r in train}
 assert not train_hashes & {sha(r.image_path) for r in validation+test+web}
 evaluator_path=ROOT/'runs/C_oven_real_v2_20261107/verify_and_package.py'
 spec=importlib.util.spec_from_file_location('fixed_evaluator',evaluator_path);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);e.RUN=RUN
 previous=json.loads((ROOT/'runs/C_pumpkin_supplement_20261107/final_report.json').read_text(encoding='utf-8'))
 result=dict(stage='second_real_batch',epochs=10,best_epoch=cp['epoch'],class_names=cp['class_names'],parent_sha256=sha(parent),checkpoint_sha256=sha(checkpoint),
  execution=json.loads((RUN/'execution.json').read_text(encoding='utf-8')),validation_before=previous['validation_after'],test_before=previous['test_after'],web_before=previous['web_after'],
  validation_after=e.evaluate(checkpoint,validation,'real_validation'),test_after=e.evaluate(checkpoint,test,'reused_test'),web_after=e.evaluate(checkpoint,web,'web_validation'),
  independent_test=False,new_test_candidate_scored=False,new_test_candidate_count=143,training_images=3516,real_training_images=2676,replay_images=840)
 names=cp['class_names'];matrix=result['test_after']['confusion_matrix'];per=result['test_after']['per_class']
 weakest=sorted([n for n in names if per[n]['total']],key=lambda n:(per[n]['correct']/per[n]['total'],names.index(n)))[:3]
 result['worst_three_reused_test']=[dict(label=n,**per[n],misclassified_as={names[j]:v for j,v in enumerate(matrix[names.index(n)]) if v and names[j]!=n}) for n in weakest]
 dump(RUN/'final_report.json',result)
 report=['# 第二批实拍训练结果','',
  '从南瓜补强14类模型继续训练10轮，最佳第'+str(cp['epoch'])+'轮；保持14类单标签识别。2676张实拍（含205张新增牛排）加840张旧网图回放，共3516张，每张每轮读取一次。',
  '沿用CPU4线程、批量32、头学习率0.00005、骨干比例0.2、AdamW及余弦调度。oven_real温和增强：等比例缩放补边、镜像、亮度/对比度/轻微颜色与光照梯度、轻微伽马/色温/噪声/JPEG扰动；标签平滑0.05。',
  '214张实拍验证用于选模。213张测试是此前多次公开复用、且与验证存在连续照片同源的开发基准，不能作为独立验收。新143张测试候选本轮未评测。',
  '本批缺整鸡和符合当前要求的待烤可颂实拍，暂缓366张保持不使用；旧样本回放仍含这两类的历史数据。全部2676图已解码及指纹核验，人工仅检查分组代表图，未逐张人工审核。','',
  '本批2676张中249张与历史实拍训练图完全相同，另有10张曾用于11月12日的公开现场开发测试；该现场测试不再作为独立评测。它们均未混入本轮214/213张验证及复用测试。205张新增牛排来自同一拍摄会话，覆盖6种层位/锡纸组合，不等于205个独立实物。','',
  '| 范围 | 训练前 | 训练后 |','|---|---:|---:|',
  '| 实拍验证 | '+fmt(result['validation_before'])+' | '+fmt(result['validation_after'])+' |',
  '| 复用实拍测试 | '+fmt(result['test_before'])+' | '+fmt(result['test_after'])+' |',
  '| 旧网图验证诊断 | '+fmt(result['web_before'])+' | '+fmt(result['web_after'])+' |','',
  '| 食材 | 实拍验证 | 复用实拍测试 |','|---|---:|---:|']
 rows=[]
 for n in names:
  v=result['validation_after']['per_class'][n];t=per[n]
  report.append('| '+n+' | '+fmt(v)+' | '+fmt(t)+' |')
  rows.append(dict(食材=n,验证正确=v['correct'],验证总数=v['total'],验证准确率=v['correct']/v['total'] if v['total'] else '',测试正确=t['correct'],测试总数=t['total'],测试准确率=t['correct']/t['total'] if t['total'] else ''))
 report+=['','复用测试中最差三类（整鸡无数据，不参与排名）：','']
 for r in result['worst_three_reused_test']:
  report.append('- '+r['label']+'：'+fmt(r)+'；误判为'+('、'.join(n+str(c)+'张' for n,c in r['misclassified_as'].items()) or '无')+'。')
 report+=['','总耗时：'+f'{result["execution"]["seconds"]:.2f}'+'秒。训练损失及验证损失均有限；新进程严格加载、14类映射、父模型和输入图像/JSON指纹检查通过。',
  '本轮为实拍域适配，不是新增类别。仍是MobileNetV3-Small分类模型，未新增YOLO、计数或层位识别，未做RK3506板卡验收。']
 (RUN/'第二批训练结果.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 with (RUN/'每类准确率.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 OUT=ROOT/'outputs'/('C第二批实拍训练_'+RUN.name.removeprefix('C_batch2_'));OUT.mkdir(exist_ok=False)
 package=OUT/'十四类食材识别_第二批实拍_代码与权重.zip'
 with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('README.md','# 第二批实拍14类分类模型\n\n安装依赖后执行 pip install -e .；预测：python predict.py --model models/model_C_batch2.pt --image 图片路径。\n每类成绩及最差三类误判见records/第二批训练结果.md。旧测试已多次复用，仅作开发对比，新143候选未评测。\n历史清单含本机路径，不附带原始图片；复训需备齐数据并调整配置副本。\n')
  z.write(checkpoint,'models/model_C_batch2.pt');z.write(ROOT/'pyproject.toml','pyproject.toml');z.write(ROOT/'runs/C_retention_v2_20261029_231408/predict.py','predict.py')
  for p in (RUN/'source_snapshot/oven_food').glob('*.py'):z.write(p,'src/oven_food/'+p.name)
  for name in ['manifest.csv','inventory.json','support_snapshot.json','data_audit.json','train_config.json','sample_counts.json','training.log','execution.json','training/history.json','final_report.json','第二批训练结果.md','每类准确率.csv','run_training.py','方案与执行记录.md','publication_record.json']:
   z.write(RUN/name,'records/'+name)
  z.write(Path(__file__),'records/verify_and_package.py');z.write(evaluator_path,'records/previous_evaluator.py')
  for p in RUN.glob('*_predictions.csv'):z.write(p,'records/'+p.name)
 with zipfile.ZipFile(package) as z:assert z.testzip() is None and hashlib.sha256(z.read('models/model_C_batch2.pt')).hexdigest()==sha(checkpoint)
 for name in ['第二批训练结果.md','每类准确率.csv','final_report.json']:shutil.copy2(RUN/name,OUT/name)
 checks=dict(status='passed',epochs=10,training_images=3516,each_image_reads=10,class_count=14,source_images_unchanged=True,model_reloaded_in_new_process=True,
  parent_sha256=sha(parent),checkpoint_sha256=sha(checkpoint),package=str(package),package_sha256=sha(package),report=str(OUT/'第二批训练结果.md'),new_test_candidate_scored=False)
 dump(RUN/'final_checks.json',checks);dump(OUT/'核验记录.json',checks)
 (ROOT/'runs/latest_C_batch2.txt').write_text(str(RUN),encoding='utf-8')
 print(json.dumps(dict(run=str(RUN),validation=result['validation_after']['correct'],test=result['test_after']['correct'],total_test=213,best_epoch=cp['epoch'],package=str(package),worst_three=result['worst_three_reused_test']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
