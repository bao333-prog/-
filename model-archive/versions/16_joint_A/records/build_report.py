"""生成两套联合模型的逐类报告和可直接预测的代码权重包。"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import zipfile


def fmt(r):
    return f'{r["correct"]}/{r["total"]}（{r["correct"]/r["total"]:.2%}）' if r['total'] else '缺数据'


def create_report_and_packages(run, root, result):
    def read(p):
        return json.loads(Path(p).read_text(encoding='utf-8'))

    def dump(p,v):
        Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def csv_write(p,rows):
        with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    audit=read(run/'data_audit.json')
    names=result['class_names']
    output=root/'outputs'/('联合训练_AB_'+run.name.removeprefix('joint_AB_'))
    output.mkdir(exist_ok=False)
    # 新进程中调用实际交付预测入口，用第二批牛排原图检查CPU预测与GPU评分一致。
    predictor_path=root/'runs/C_retention_v2_20261029_231408/predict.py'
    spec=importlib.util.spec_from_file_location('delivery_predict',predictor_path)
    predictor=importlib.util.module_from_spec(spec);spec.loader.exec_module(predictor)
    second=read(run/'second_eval_inventory.json')
    sample=next(r for r in second if r['label']=='牛排' and r['original_half']=='test')
    smokes={}
    for protocol,tag in [('A','A_test84'),('B','B_common84')]:
        model,classes,cp=predictor.load_classifier(run/f'selection_{protocol}/best.pt')
        raw=predictor.predict_image(model,classes,cp,sample['source'])
        with (run/(tag+'_predictions.csv')).open(encoding='utf-8-sig',newline='') as f:
            row=next(r for r in csv.DictReader(f) if r['image']==sample['actual_input'])
        assert raw['predicted_class']==row['predicted']
        smokes[protocol]=raw
    result['raw_cpu_predict_smoke']=smokes
    dump(run/'final_report.json',result)

    report=['# 联合训练与两套验证方案结果','',
            '两版均为14类MobileNetV3-Small联合模型，共用一次20轮训练和相同参数更新；A、B仅最佳轮次选择规则不同。',
            '训练5734张＝3990实拍＋1744历史网图；其中恢复1180张未重复旧实拍，使用完整历史选用网图训练池，去掉所有140张牛排网图。第一批79张牛排训练图及21张牛排验证均排除；本轮牛排仅205张第二批实拍。',
            '另去掉9张旧遮挡图、1张本批已知遮挡薯条图；前轮8张遮挡南瓜不恢复。新增1189旧候选全部浏览缩略图，29疑点放大；当前已选数据沿用前轮清理，不宣称全部原尺寸逐图验收。原图、旧模型不改动。','',
            '初始化来自官方ImageNet预训练骨干，14类输出头随机初始化。官方权重SHA256：'+audit['official_sha256']+'；本轮不使用旧模型参数、教师或优化器状态，避免继承第一批牛排的项目训练权重。',
            'GPU20轮，头lr0.0003、骨干比例0.2，AdamW/weight decay0.0001、余弦、标签平滑0.05；batch32、CPU4线程/workers0。仅oven_geometry：224补边、水平翻转、±6°和缩放，无光照/颜色扰动。',
            '类别→实拍/网图→新旧批次→来源组→图片分层平衡；同时有两域时75%实拍/25%网图，缺域则给现有域。首轮全图一次，其后19轮有放回抽样，每轮5734次；总读取'+str(result['sampling']['total_reads'])+'次，各图读取'+str(result['sampling']['minimum_reads'])+'—'+str(result['sampling']['maximum_reads'])+'次，全部图至少一次。类别已均衡采样，不再重复加类别损失权重。',
            '总训练耗时'+f'{result["execution"]["seconds"]:.2f}'+'秒。A最佳第'+str(result['best_epochs']['A'])+'轮，B最佳第'+str(result['best_epochs']['B'])+'轮。','',
            '| 方案 | 选模验证 | 最终测试 |', '|---|---|---|',
            '| A | 第二批按类别近1:1的83张 | 另一半84张 |',
            '| B | 第一批验证剔除牛排后193张 | 第二批全部可用167张 |','',
            '第二批167张来自原food_test02的182张：保留南瓜/可颂，按已有修正版仍暂缓白玉米12张及手/筷子干扰3张。原两类各12张现各6验证/6测试；A完整83张均参与选模，不沿用前轮77+6诊断规则。',
            'A的验证与训练存在1个南瓜来源组交叉，验证/测试也有连续同源照片。B与训练无已知同类别同采集session交叉，但未确认所有食品个体独立。B的167张中含83张A验证，两套协议按要求共享图片，不能当作相互独立盲测；B轮次只由第一批验证决定。所有结果定位开发对照，不能称独立验收。',
            '整鸡缺实拍评测；B验证还缺牛排，牛排只由第二批测试检查。没有用测试结果改变轮次、超参数或追加训练。','',
            '| 范围 | 成绩 |', '|---|---:|',
            '| A验证83 | '+fmt(result['A']['validation'])+' |',
            '| A测试84 | '+fmt(result['A']['test'])+' |',
            '| B验证193 | '+fmt(result['B']['validation'])+' |',
            '| B测试167 | '+fmt(result['B']['test'])+' |','',
            '| 相同84张第二批测试 | 成绩 |', '|---|---:|',
            '| 补充前现版9db30ce0（仅固定参考） | '+fmt(result['baseline']['common84'])+' |',
            '| 联合A | '+fmt(result['A']['test'])+' |',
            '| 联合B | '+fmt(result['B']['common84'])+' |','',
            '9db旧版在完整167张上的参考成绩：'+fmt(result['baseline']['full167'])+'。旧版不参与新模型初始化、教师、选模。两方案主测试数量不同，不能仅凭各自百分比断定谁更好；共同84张才是相同图片对照。数据、初始化、采样与学习率多项同时变化，不能把收益归给单一因素。','',
            '| 食材 | A验证83 | A测试84 | B验证193 | B测试167 | B共同84 |', '|---|---:|---:|---:|---:|---:|']
    rows=[]
    for n in names:
        values=[result['A']['validation']['per_class'][n],result['A']['test']['per_class'][n],result['B']['validation']['per_class'][n],result['B']['test']['per_class'][n],result['B']['common84']['per_class'][n]]
        report.append('| '+n+' | '+' | '.join(fmt(v) for v in values)+' |')
        rows.append(dict(食材=n,A验证=fmt(values[0]),A测试=fmt(values[1]),B验证=fmt(values[2]),B测试=fmt(values[3]),B共同84=fmt(values[4])))
    for title,score in [('A测试84最低三类',result['A']['test']),('B测试167最低三类',result['B']['test']),('B共同84最低三类',result['B']['common84'])]:
        report += ['',title+'（缺数据不排名；同分按既定类别顺序）：','']
        for r in score['worst_three']:
            wrong='、'.join(n+str(c)+'张' for n,c in r['misclassified_as'].items()) or '无误判（100%并列）'
            report.append('- '+r['label']+'：'+fmt(r)+'；'+wrong+'。')
    report += ['','两份模型参数是否完全相同：'+str(result['same_model_parameters'])+'。','',
               '检查通过：官方初始化与14类映射、第一批和网图牛排排除、训练/评测精确交叉为0、采样覆盖、20轮有限数值、A/B各自最佳规则、原图指纹、新进程重载、第二批牛排原图CPU预测和模型包完整性。']
    (run/'联合训练结果.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    csv_write(run/'每类准确率.csv',rows)
    for p in ['A','B']:
        csv_write(run/f'每轮指标_{p}.csv',read(run/f'selection_{p}/history.json'))
    packages={}
    common_records=['manifest.csv','training_manifest.csv','validation_A.csv','validation_B.csv','test_A.csv','test_B.csv',
                    'training_inventory.json','second_eval_inventory.json','first_validation_inventory.json','data_audit.json',
                    'initialization_audit.json','pixel_overlap_audit.json','preflight.json','train_config.json','sampling_weights.json',
                    'sample_counts.json','sampling_execution.json','execution.json','training.log','final_report.json',
                    '训练逐图来源.csv','质量剔除清单.csv','每类数量.csv','每类准确率.csv','每轮指标_A.csv','每轮指标_B.csv',
                    '联合训练结果.md','执行记录.md','joint_support.py','run_training.py','test_joint_support.py','verify_and_package.py','build_report.py']
    for p,description in [('A','第二批验证'),('B','第一批验证')]:
        checkpoint=run/f'selection_{p}/best.pt'
        model_name=f'model_joint_{p}.pt'
        shutil.copy2(checkpoint,output/model_name)
        package=output/f'联合模型{p}_{description}_代码与权重.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('README.md','# 联合模型'+p+'\n\n安装：pip install -e .\n预测：python predict.py --model models/'+model_name+' --image 图片路径\n\n本包为14类单标签分类。选模规则'+p+'，详见records/联合训练结果.md。牛排仅第二批实拍；初始化为官方预训练，未沿用旧训练模型。A验证83/测试84，B验证193/测试167；均为开发对照，整鸡缺实拍评测。\n\nrecords内保留共享训练与双验证程序、配置、来源和指标。历史数据路径为本机路径，不含全部原图；复训需另备数据并调整路径。不能直接用普通单验证入口代替双验证程序来复现A/B选模。\n')
            z.write(checkpoint,'models/'+model_name)
            z.write(run/'official_initialization.pt','records/official_initialization.pt')
            z.write(root/'.superpowers/prepare_joint_AB_20261118.py','records/prepare_joint_AB.py')
            z.write(root/'pyproject.toml','pyproject.toml');z.write(predictor_path,'predict.py')
            for f in (run/'source_snapshot/oven_food').glob('*.py'): z.write(f,'src/oven_food/'+f.name)
            for name in common_records: z.write(run/name,'records/'+name)
            for protocol in ['A','B']:
                for name in ['history.json','summary.json']: z.write(run/f'selection_{protocol}'/name,f'records/selection_{protocol}/'+name)
            for f in run.glob('*_predictions.csv'): z.write(f,'records/'+f.name)
        with zipfile.ZipFile(package) as z:
            assert z.testzip() is None
            assert hashlib.sha256(z.read('models/'+model_name)).hexdigest()==result['checkpoint_sha256'][p]
            for f in (run/'source_snapshot/oven_food').glob('*.py'):
                assert hashlib.sha256(z.read('src/oven_food/'+f.name)).hexdigest()==sha(f)
        packages[p]=dict(package=str(package),package_sha256=sha(package),model=str(output/model_name),model_sha256=sha(checkpoint))
    for name in ['联合训练结果.md','每类准确率.csv','每类数量.csv','每轮指标_A.csv','每轮指标_B.csv','final_report.json','data_audit.json']:
        shutil.copy2(run/name,output/name)
    dump(output/'交付核验.json',dict(status='passed',packages=packages,model_reloaded=True,source_images_unchanged=True))
    return dict(output=str(output),report=str(output/'联合训练结果.md'),packages=packages)
