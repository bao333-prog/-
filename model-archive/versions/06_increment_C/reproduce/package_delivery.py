"""打包第二次增量的十四类C、十类父B、注释代码及可核查记录。"""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys,tempfile,zipfile
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
r=json.loads((OUT/'final_report.json').read_text(encoding='utf-8'))
a=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
dest=ROOT/'outputs/第二次增量C_20261029';dest.mkdir(parents=True,exist_ok=True)
archive=dest/'十四类模型C_第二次增量_中文注释代码与权重.zip'
if archive.exists():raise FileExistsError(archive)
old=r['evaluations']['test_C']['old'];new=r['evaluations']['test_C']['new'];overall=r['evaluations']['test_C']['all']
target='达到本轮旧十类/新四类固定网图基准目标' if all(r['target_checks'].values()) else '至少一项本轮固定网图基准目标未达到，详情见报告'
readme=f"""# 第二次增量：十四类模型C

2026年10月29日。从鸡翅补强后的十类B加载并扩展，新增土豆、香肠、薯条、可颂；网络为MobileNetV3-Small单标签分类。
先严格加载B，再复制保留原十类分类行并新增四行；扩层时旧参数和缓冲区一致检查通过，随后更新全部参数。

## 数据与训练

新增800张，逐张预览后暂缓43张，保留757张（529训练、114验证、114测试）。
可颂含大量相似处理版本，200张均无来源记录；按人工视觉组和相似性保守合组，不证明其余文件均独立实拍。
每旧类20张训练图回放共200张，总训练729张；验证旧293+新114=407；测试旧295+新114=409。
20轮、lr=0.0001、AdamW、batch16、CPU四线程、seed42、gentle_plus和类别加权；最佳第{r['best_epoch']}轮。

## 固定测试结果

- 旧十类：{old['correct']}/{old['total']}，{old['accuracy']:.2%}；相对本轮父B下降{r['old_accuracy_drop_percentage_points']:.2f}个百分点。
- 新四类：{new['correct']}/{new['total']}，{new['accuracy']:.2%}。
- 累计十四类：{overall['correct']}/{overall['total']}，{overall['accuracy']:.2%}。
- {target}。负的下降值表示提高。

旧295张测试已在此前阶段开发中使用；阶段对照不是全新独立验收。旧类在完整14个输出中竞争，无截取旧类输出。
测试仅在验证集选模后评分，未据测试自动追加训练。首次六类相对A的累计变化另列在详细报告。
旧来源关联风险、无来源可颂及转载关系遗漏仍存在，网图高分不代表烤箱或RK3506效果。

## 使用

Python3.10以上，在解压根目录执行：

    python -m pip install -e .
    python predict.py --model models/model_C.pt --image "你的图片.jpg"

models/model_C.pt是14类C；models/parent_B_chicken_supplement.pt是本轮10类父B。
类别顺序：{ '、'.join(r['class_names']) }。
predict.py按权重元数据处理原图并恢复类别下标；src/oven_food/是本轮运行封存的中文注释代码。
单标签分类不支持目标计数或未知食品拒识；输入其他食品也会被分进已知类别之一。

## 复核和复现

records/含筛选理由、来源与视觉组、固定划分、实际回放清单、逐轮记录、父模型指纹和逐张预测。
reproduce/为本机执行脚本，依赖原电脑数据目录，供查阅；异机重训要另取数据和调整路径。
本包不含原图或训练准备副本。保留所有历史模型，没有覆盖A或B。
"""
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('README.md',readme);z.write(ROOT/'pyproject.toml','pyproject.toml');z.write(OUT/'predict.py','predict.py')
    z.write(OUT/'training/best.pt','models/model_C.pt');z.write(a['parent_checkpoint'],'models/parent_B_chicken_supplement.pt')
    for p in (OUT/'source_snapshot/oven_food').glob('*.py'):z.write(p,'src/oven_food/'+p.name)
    for name in ['第二次增量C_结果报告.md','final_report.json','data_audit.json','inventory_summary.json','review_and_split.json',
                 'grouping_edges.json','new_split_summary.json','manifest.csv','expected_training_manifest.csv','train_config.json',
                 'execution.json','initialization_audit.json','执行计划与记录.md','training.log','verification.log',
                 'old_visual_overlap_candidates.json','cross_check.log','数据复核与划分说明.md']:
        z.write(OUT/name,'records/'+name)
    for folder in ['training','parent_old_validation','validation_C','parent_old_test','test_C']:
        for p in (OUT/folder).glob('*'):
            if p.suffix in {'.csv','.json'}:z.write(p,'records/'+folder+'/'+p.name)
    for name in ['inventory_and_preview.py','check_old_visual_overlap.py','prepare_training.py','run_training.py','verify_and_report.py','package_delivery.py']:
        z.write(OUT/name,'reproduce/'+name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('models/model_C.pt')).hexdigest()==r['checkpoint_sha256']
    assert hashlib.sha256(z.read('models/parent_B_chicken_supplement.pt')).hexdigest()==a['parent_sha256']
    for p in (OUT/'source_snapshot/oven_food').glob('*.py'):assert z.read('src/oven_food/'+p.name)==p.read_bytes()
    with tempfile.TemporaryDirectory(prefix='second_increment_C_delivery_') as temp:
        assert Path(temp).resolve().parent==Path(tempfile.gettempdir()).resolve()
        z.extractall(temp);env=os.environ.copy();env['PYTHONPATH']=str(Path(temp)/'src')
        with (OUT/'test_C/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            expected=next(row for row in csv.DictReader(f) if row['cohort']=='new')
        completed=subprocess.run([sys.executable,'-X','utf8','-B','predict.py','--model','models/model_C.pt',
                 '--image',expected['original_path']],cwd=temp,env=env,capture_output=True,text=True,encoding='utf-8',check=True)
        predicted=json.loads(completed.stdout)
        assert predicted['predicted_class']==expected['predicted_label']
        assert list(predicted['probabilities'])==r['class_names']
checks=dict(archive=str(archive),size_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            integrity='passed',weights_match='passed',source_snapshot_matches='passed',extracted_prediction='passed')
(OUT/'delivery_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
