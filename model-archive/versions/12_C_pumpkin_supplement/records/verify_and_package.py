"""核对南瓜补强模型，复用固定模型评测逻辑并生成交付包。"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0,str(RUN / "source_snapshot"))
from oven_food.manifest import read_manifest, ManifestRecord
from oven_food.augment import prepare_image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value):
    return f'{value["correct"]}/{value["total"]}（{value["correct"]/value["total"]:.2%}）' if value["total"] else "缺数据"


def main():
    torch.set_num_threads(4)
    config=json.loads((RUN/"train_config.json").read_text())
    audit=json.loads((RUN/"data_audit.json").read_text(encoding="utf-8"))
    inventory=json.loads((RUN/"inventory.json").read_text(encoding="utf-8"))
    counts=json.loads((RUN/"sample_counts.json").read_text(encoding="utf-8"))
    records=read_manifest(RUN/"manifest.csv")
    training=[r for r in records if r.split=="train"]
    assert len(training)==2357 and len(inventory)==1517
    assert set(counts)=={r.image_path for r in training} and set(counts.values())=={10}
    history=json.loads((RUN/"training/history.json").read_text())
    assert len(history)==10 and all(np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"]) for row in history)
    parent=Path(config["checkpoint"]); student=RUN/"training/best.pt"
    assert sha(parent)==audit["parent_sha256"]
    old=torch.load(parent,map_location="cpu",weights_only=True)
    new=torch.load(student,map_location="cpu",weights_only=True)
    assert old["class_names"]==new["class_names"] and len(new["class_names"])==14
    assert {k:tuple(v.shape) for k,v in old["model_state"].items()}=={k:tuple(v.shape) for k,v in new["model_state"].items()}
    for row in inventory:
        assert sha(Path(row["image"]))==row["raw_sha256"]
        assert sha(Path(row["prepared"]))==row["prepared_sha256"]
    snapshot=json.loads((ROOT/"runs/food_verify_20261107/input_snapshot.json").read_text(encoding="utf-8"))
    for row in snapshot:
        assert sha(Path(row["image"]))==row["sha256"]
    assert not {row["sha256"] for row in snapshot} & {row["raw_sha256"] for row in inventory}
    for row in [r for r in inventory if r["label"]=="南瓜"]:
        with Image.open(row["image"]) as raw, Image.open(row["prepared"]) as cache:
            assert np.array_equal(np.asarray(prepare_image(raw,image_size=224,resize_mode="letterbox")),
                                  np.asarray(prepare_image(cache,image_size=224,resize_mode="letterbox")))
    evaluation_path=ROOT/"runs/C_oven_real_v2_20261107/verify_and_package.py"
    spec=importlib.util.spec_from_file_location("previous_evaluator",evaluation_path)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    evaluator.RUN=RUN
    validation=[r for r in records if r.split=="val" and r.environment=="target"]
    web=[r for r in records if r.split=="val" and r.environment=="proxy"]
    test=[ManifestRecord(row["prepared"],row["label"],"target","verify_"+row["source_group"],
                         "fixed","oven","none","none","test") for row in snapshot if row["split"]=="test"]
    assert len(validation)==214 and len(test)==213 and len(web)==407
    assert not {r.image_path for r in training} & {r.image_path for r in validation+test}
    previous=json.loads((ROOT/"runs/C_oven_real_v2_20261107/final_report.json").read_text(encoding="utf-8"))
    result=dict(stage="pumpkin_real_supplement",epochs=10,best_epoch=new["epoch"],
                class_names=new["class_names"],parent_sha256=sha(parent),checkpoint_sha256=sha(student),
                execution=json.loads((RUN/"execution.json").read_text()),
                validation_before=previous["validation_after"],test_before=previous["test_after"],
                web_before=previous["web_validation_after"],
                validation_after=evaluator.evaluate(student,validation,"real_validation"),
                test_after=evaluator.evaluate(student,test,"verify_test"),
                web_after=evaluator.evaluate(student,web,"web_validation"),
                independent_test=False,all_training_images_read_per_epoch=True)
    pumpkin=[r for r in training if r.environment=="target" and r.food_label=="南瓜"]
    result["pumpkin_train_diagnostic"]=evaluator.evaluate(student,pumpkin,"pumpkin_training")
    (RUN/"final_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# 南瓜实拍补强结果","",
            "将用户指定的89张南瓜训练图加入原1428张实拍，并保留840张旧图，合计2357张训练图。全部训练图片每轮完整读取，运行10轮，最佳第"+str(new["epoch"])+"轮。",
            "沿用oven_real光照增强、0.05标签平滑、AdamW，头学习率0.00005，骨干为头的20%，余弦调度，批量32，CPU4线程。输出保持十四类食材。",
            "新增南瓜数据覆盖有无锡纸和高中低三档。其中1张缺同名JSON，按用户提供的文件夹标签保留。代表图已检查，89张输入与缓存预处理逐张一致。",
            "验证214张用于选模，测试213张在权重确定后评测。测试为此前公开的复用基准，且同来源连续照片跨验证/测试集合。新增南瓜与该评测集的精确原图指纹无重合。","",
            "| 范围 | 补强前 | 补强后 |","|---|---:|---:|",
            "| 实拍验证 | "+fmt(result["validation_before"])+" | "+fmt(result["validation_after"])+" |",
            "| 复用实拍测试 | "+fmt(result["test_before"])+" | "+fmt(result["test_after"])+" |",
            "| 旧网图验证 | "+fmt(result["web_before"])+" | "+fmt(result["web_after"])+" |","",
            "| 食材 | 验证 | 测试 |","|---|---:|---:|"]
    for name in new["class_names"]:
        report.append("| "+name+" | "+fmt(result["validation_after"]["per_class"][name])+" | "+fmt(result["test_after"]["per_class"][name])+" |")
    report+=["","新增89张南瓜训练图回代为"+fmt(result["pumpkin_train_diagnostic"])+"。整鸡仍缺实拍。","",
             "原图及模型指纹、每图10次读取记录、有限损失、模型严格重载及十四类映射检查通过。逐图预测CSV和混淆矩阵JSON随包提供。"]
    (RUN/"南瓜补强结果.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    out=ROOT/"outputs/C南瓜实拍补强_20261107";out.mkdir(exist_ok=False)
    package=out/"十四类食材识别_南瓜实拍补强_代码与权重.zip"
    with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md","# 南瓜实拍补强十四类模型\n\n安装依赖后在本目录执行 pip install -e .。\n预测命令为 python predict.py --model models/model_C_pumpkin.pt --image 食物.jpg。\n结果和逐类成绩见records/南瓜补强结果.md。历史配置含本机路径，复训前需准备对应数据并调整配置副本。\n")
        archive.write(student,"models/model_C_pumpkin.pt")
        archive.write(ROOT/"pyproject.toml","pyproject.toml")
        archive.write(ROOT/"runs/C_retention_v2_20261029_231408/predict.py","predict.py")
        for path in (RUN/"source_snapshot/oven_food").glob("*.py"):
            archive.write(path,"src/oven_food/"+path.name)
        for name in ["manifest.csv","inventory.json","data_audit.json","train_config.json","sample_counts.json","training.log","execution.json","training/history.json","final_report.json","南瓜补强结果.md","run_training.py","verify_and_package.py"]:
            archive.write(RUN/name,"records/"+name)
        archive.write(evaluation_path,"records/previous_evaluator.py")
        for path in RUN.glob("*_predictions.csv"):
            archive.write(path,"records/"+path.name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None and hashlib.sha256(archive.read("models/model_C_pumpkin.pt")).hexdigest()==sha(student)
    shutil.copyfile(RUN/"南瓜补强结果.md",out/"南瓜补强结果.md")
    checks=dict(status="passed",epochs=10,class_count=14,raw_real_images_unchanged=1517,new_pumpkin_images=89,
                package=str(package),package_sha256=sha(package),checkpoint_sha256=sha(student))
    (RUN/"final_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:dict(correct=result[key]["correct"],total=result[key]["total"])
                      for key in ["validation_after","test_after","web_after","pumpkin_train_diagnostic"]}),flush=True)


if __name__=="__main__":
    main()
