"""只在相同验证集比较原C与改进C；禁止以已有测试集重新选方案。"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,random,time
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from oven_food.augment import prepare_image,adjust_gamma,add_sensor_noise,shrink_and_translate,jpeg_roundtrip
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest,validate_manifest
from predict import load_classifier,image_tensor
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];torch.set_num_threads(4)
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read_json(OUT/'data_audit.json');base=Path(a['comparator_dir']);e=read_json(OUT/'execution.json')
assert e['status']=='completed' and e['teacher_parameters_and_bn_unchanged'] and e['teacher_has_no_gradients']
history=read_json(OUT/'training/history.json')
assert len(history)==20 and all(math.isfinite(v) for h in history for v in h.values())
assert all(h['distillation_examples']>0 for h in history)
candidate,names,cp=load_classifier(OUT/'training/best.pt')
original,original_names,original_cp=load_classifier(base/'training/best.pt')
parent,old_names,parent_cp=load_classifier(a['parent_checkpoint'])
assert names==original_names==a['class_names'] and names[:len(old_names)]==old_names
assert cp['distillation']['teacher_sha256']==a['parent_sha256']
assert cp['distillation']['student_normalization']=='all_outputs'
assert cp['incremental']['replay_counts']=={n:20 for n in old_names}
assert not any('teacher' in key for key in cp['model_state'])
assert {k:tuple(v.shape) for k,v in cp['model_state'].items()}=={k:tuple(v.shape) for k,v in original_cp['model_state'].items()}
assert sum(p.numel() for p in candidate.parameters())==sum(p.numel() for p in original.parameters())
assert all(torch.isfinite(v).all() for v in cp['model_state'].values() if v.is_floating_point())
best=min(history,key=lambda h:(-h['validation_accuracy'],h['validation_loss']))
assert cp['epoch']==best['epoch']
actual=read_manifest(OUT/'training/training_manifest.csv')
assert actual==read_manifest(OUT/'expected_training_manifest.csv')==read_manifest(base/'training/training_manifest.csv')
assert Counter(r.split for r in actual)=={'train':729,'val':407}
records=[r for r in actual if r.split=='val'];validate_manifest(actual,check_files=True)
assert not {r.image_path for r in actual if r.split=='train'}&{r.image_path for r in records}
reviews=read_json(OUT/'review_and_split.json');raw_sources={r['prepared_path']:r['path'] for r in reviews if r['included']}
new_val=[r for r in records if r.food_label in a['new_class_names']]
data=FoodImageDataset(new_val,data_root=OUT,class_names=names,resize_mode='letterbox')
for i,r in enumerate(new_val):
    with Image.open(raw_sources[r.image_path]) as image:assert torch.equal(image_tensor(image,cp)[0],data[i][0])

class Variant(Dataset):
    def __init__(self,variant):self.variant=variant
    def __len__(self):return len(records)
    def __getitem__(self,index):
        with Image.open(records[index].image_path) as image:image=prepare_image(image,image_size=224,resize_mode='letterbox')
        if self.variant=='gamma_dark':image=adjust_gamma(image,1.1)
        elif self.variant=='gamma_light':image=adjust_gamma(image,.9)
        elif self.variant=='sensor_noise':image=add_sensor_noise(image,1.,random.Random(42+index))
        elif self.variant=='scale_translate':image=shrink_and_translate(image,scale=.9,rng=random.Random(42+index))
        elif self.variant=='jpeg':image=jpeg_roundtrip(image,85)
        return image_tensor(image,cp)[0],names.index(records[index].food_label)

def evaluate(network,variant):
    predictions=[];loss=0.;matrix=[[0]*14 for _ in names]
    with torch.inference_mode():
        for images,labels in DataLoader(Variant(variant),batch_size=16,shuffle=False,num_workers=0):
            logits=network(images);assert torch.isfinite(logits).all()
            loss+=float(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
            probabilities,indices=logits.softmax(1).max(1)
            for y,p,prob in zip(labels.tolist(),indices.tolist(),probabilities.tolist()):
                r=records[len(predictions)];matrix[y][p]+=1
                predictions.append(dict(image_path=r.image_path,original_path=raw_sources.get(r.image_path,r.image_path),
                    true_label=names[y],predicted_label=names[p],correct=y==p,top1_probability=prob,
                    cohort='old' if r.food_label in old_names else 'new'))
    stats=dict(loss=loss/len(records),per_class={},confusion_matrix=matrix)
    for name in names:
        group=[r for r in predictions if r['true_label']==name]
        stats['per_class'][name]=dict(correct=sum(r['correct'] for r in group),total=len(group),accuracy=sum(r['correct'] for r in group)/len(group))
    for scope in ['all','old','new']:
        group=predictions if scope=='all' else [r for r in predictions if r['cohort']==scope]
        stats[scope]=dict(correct=sum(r['correct'] for r in group),total=len(group),accuracy=sum(r['correct'] for r in group)/len(group))
    return stats,predictions

evaluations={};started=time.perf_counter()
for variant in ['clean','gamma_dark','gamma_light','sensor_noise','scale_translate','jpeg']:
    evaluations[variant]={}
    for label,network in [('original_C',original),('retention_C_v2',candidate)]:
        stats,predictions=evaluate(network,variant);evaluations[variant][label]=stats
        dest=OUT/'validation'/variant/label;dest.mkdir(parents=True,exist_ok=False)
        (dest/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
        with (dest/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    print(json.dumps(dict(variant=variant,counts={label:s['all'] for label,s in evaluations[variant].items()}),ensure_ascii=False),flush=True)
clean=evaluations['clean']
assert clean['original_C']['all']['correct']==387
assert clean['original_C']['old']['correct']==275 and clean['original_C']['new']['correct']==112
assert clean['retention_C_v2']['all']['accuracy']==cp['validation_accuracy']
assert abs(clean['retention_C_v2']['loss']-cp['validation_loss'])<1e-5
parent_old_val=read_json(base/'final_report.json')['evaluations']['parent_old_validation']['all']
drop_before=100*(parent_old_val['accuracy']-clean['original_C']['old']['accuracy'])
drop_after=100*(parent_old_val['accuracy']-clean['retention_C_v2']['old']['accuracy'])
assert all(sha(p)==s for p,s in a['protected_models'].items())
assert all(sha(p)==s for p,s in a['old_input_sha256'].items())
assert all(sha(p)==s for p,s in a['new_original_sha256'].items())
assert all(sha(OUT/'source_snapshot/oven_food'/n)==s for n,s in a['source_sha256'].items())
report=dict(stage='C_retention_framework_v2',class_names=names,epochs=20,best_epoch=cp['epoch'],
    training_seconds=e['training_seconds'],verification_seconds=time.perf_counter()-started,
    distillation=cp['distillation'],replay_per_class=20,training_count=729,validation_count=407,
    related_tests_passed=69,code_review='no_blocking_findings',same_actual_manifest=True,
    parameter_count=sum(p.numel() for p in candidate.parameters()),inference_structure_unchanged=True,
    evaluations=evaluations,parent_B_old_validation=parent_old_val,old_validation_drop_before_pp=drop_before,
    old_validation_drop_after_pp=drop_after,old_validation_drop_at_most_2pp=drop_after<=2,
    teacher_frozen_verified=True,new_process_reload=True,protected_models_unchanged=True,
    original_images_unchanged=True,raw_new_validation_matches_prepared=True,
    checkpoint_sha256=sha(OUT/'training/best.pt'),test_scored=False,test_retention_target='not_evaluated_this_run',
    limits=a['limits'])
(OUT/'final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 十四类C：训练框架与增强改进', '', '2026年10月29日。联合改进训练约束、分组学习率、调度和数据增强，未更换推理网络。', '',
       '## 改了什么','',
       '- 冻结十类B教师，仅对教师判断正确的旧训练样本蒸馏。教师旧类概率扩展新类零质量，学生在全部14输出归一化，约束新类抢占旧类概率。',
       '- CE+1.0×KL；温度2，KL乘T²。教师仅用于训练，无梯度、BN不变，不进入部署权重。',
       '- 骨干学习率为分类头的20%；分类头初始0.0001，余弦衰减至初始的5%。',
       '- gentle_retention：镜像、小角度旋转、90%～100%缩小平移、轻微亮度/对比度/饱和度、色温、gamma、传感噪声和高质量JPEG。保持完整主体，不再叠加模糊/降采样。',
       '- 新配置默认关闭以上训练约束，旧配置/旧增强兼容；中文注释及模块化训练辅助函数已加入。','',
       '## 对照条件','',
       '与原C从同一鸡翅补强B重新扩至14类；每旧类仍20张回放，总729张训练图，407张验证。来源分组、文件和类别顺序不变。',
       f"训练20轮，最佳第{cp['epoch']}轮，耗时{e['training_seconds']:.1f}秒。实际训练清单逐行与原C相同。",'',
       '## 普通验证对照（不是测试成绩）','', '| 范围 | 原C | 改进候选 |','|---|---:|---:|']
for scope,title in [('old','旧十类'),('new','新四类'),('all','十四类总体')]:
    x,y=clean['original_C'][scope],clean['retention_C_v2'][scope]
    lines.append(f"| {title} | {x['correct']}/{x['total']} ({x['accuracy']:.2%}) | {y['correct']}/{y['total']} ({y['accuracy']:.2%}) |")
lines += ['',f"同一父B旧验证为{parent_old_val['correct']}/293（{parent_old_val['accuracy']:.2%}）；旧验证下降从{drop_before:.2f}个百分点变为{drop_after:.2f}个百分点。仅验证口径，2个百分点检查{'满足' if drop_after<=2 else '不满足'}，不能据此宣布测试保持目标通过。",'',
          '| 食材 | 原C正确/总数 | 改进候选正确/总数 |','|---|---:|---:|']
for name in names:
    x,y=clean['original_C']['per_class'][name],clean['retention_C_v2']['per_class'][name]
    lines.append(f"| {name} | {x['correct']}/{x['total']} | {y['correct']}/{y['total']} |")
lines += ['', '## 固定扰动诊断（同一验证集）','', '| 扰动 | 原C正确/407 | 改进候选正确/407 |','|---|---:|---:|']
for variant,title in [('gamma_dark','gamma=1.1'),('gamma_light','gamma=0.9'),('sensor_noise','噪声sigma=1'),('scale_translate','缩小90%并平移'),('jpeg','JPEG质量85')]:
    lines.append(f"| {title} | {evaluations[variant]['original_C']['all']['correct']} | {evaluations[variant]['retention_C_v2']['all']['correct']} |")
lines += ['', '## 逐轮记录','', '| 轮次 | CE损失 | 蒸馏损失 | 有效蒸馏旧样本 | 验证准确率 |','|---|---:|---:|---:|---:|']
for h in history:lines.append(f"| {h['epoch']} | {h['supervised_loss']:.5f} | {h['distillation_loss']:.5f} | {h['distillation_examples']} | {h['validation_accuracy']:.2%} |")
lines += ['', '## 核对及局限','',
          f"69项相关检查及只读复核通过；教师参数/BN/梯度检查通过，部署模型仍为{report['parameter_count']:,}参数，与原C逐层形状相同。", 
          '新进程严格加载、114张新类验证原图输入一致性、旧版本/原图/代码快照哈希及实际回放隔离检查通过。',
          '本轮没有重新给409张已公开测试图片评分。原C测试上的4.07个百分点下降结论保留；此改进候选尚无新的独立测试结论。',
          '验证反复参与方案开发且旧来源有局限，不能代表新场景泛化或烤箱效果。联合修改缺少单项消融，不将收益归因于蒸馏一项。','',
          '技术依据：[蒸馏原论文](https://arxiv.org/abs/1503.02531)、[PyTorch KLDivLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)。本项目的旧样本筛选和全输出归一化为具体实现设计。','']
lines += ['- '+limit for limit in a['limits']]
(OUT/'训练框架与增强改进_结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(best_epoch=cp['epoch'],clean=clean,old_validation_drop_after_pp=drop_after,test_scored=False),ensure_ascii=False))
