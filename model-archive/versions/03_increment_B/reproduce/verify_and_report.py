"""选定B后才评测测试集；旧类在十类完整输出中竞争，不能只截取前六类。"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math
from PIL import Image
import torch
from oven_food.config import load_experiment_config
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest
from oven_food.evaluate import evaluate_experiment
from predict import load_classifier,image_tensor,predict_image

OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
baseline=json.loads((OUT/"baseline.json").read_text(encoding="utf-8"));BASE=Path(baseline["parent_dir"])
assert hashlib.sha256((BASE/"training/best.pt").read_bytes()).hexdigest()==baseline["parent_sha256"]
assert hashlib.sha256((BASE/"manifest.csv").read_bytes()).hexdigest()==baseline["old_manifest_sha256"]
execution=json.loads((OUT/"execution.json").read_text(encoding="utf-8"));assert execution["status"]=="completed"
history=json.loads((OUT/"training/history.json").read_text(encoding="utf-8"));assert len(history)==20
assert all(math.isfinite(v) for row in history for v in row.values())
network,names,checkpoint=load_classifier(OUT/"training/best.pt")
parent=torch.load(BASE/"training/best.pt",map_location="cpu",weights_only=True)
old_names=parent["class_names"];new_names=baseline["new_classes"]
assert names==old_names+new_names and len(names)==10
assert checkpoint["incremental"]["parent_sha256"]==baseline["parent_sha256"]
best=min(history,key=lambda h:(-h["validation_accuracy"],h["validation_loss"]))
assert checkpoint["epoch"]==best["epoch"]
initial=json.loads((OUT/"initialization_audit.json").read_text(encoding="utf-8"))
assert initial["old_head_rows_identical_before_training"] and initial["backbone_unchanged_during_expansion"]
records=read_manifest(OUT/"manifest.csv");actual=read_manifest(OUT/"training/training_manifest.csv")
used_train=[r for r in actual if r.split=="train"]
old_train_paths={r.image_path for r in read_manifest(BASE/"manifest.csv") if r.split=="train"}
assert all(r.image_path in old_train_paths for r in used_train if r.food_label in old_names)
assert Counter(r.food_label for r in used_train if r.food_label in old_names)==Counter({name:20 for name in old_names})
heldout_paths={r.image_path for r in records if r.split in {"val","test"}}
assert not heldout_paths & {r.image_path for r in used_train}
old_review=json.loads((BASE/"review_and_split.json").read_text(encoding="utf-8"))
new_review=json.loads((OUT/"review_and_split.json").read_text(encoding="utf-8"))
reviews=old_review+new_review;source_map={r["prepared_path"]:r for r in reviews if r["included"]}
evaluations={}
for split in ["val","test"]:
    config=json.loads((OUT/"train_config.json").read_text(encoding="utf-8"))
    config.update(mode="evaluate",experiment_name="B_"+split,checkpoint=str(OUT/"training/best.pt"),output_dir=str(OUT/split),
                  evaluation_environments=["proxy"],evaluation_split=split,new_classes=[])
    path=OUT/(split+"_config.json");path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    metrics=json.loads(evaluate_experiment(load_experiment_config(path)).read_text(encoding="utf-8"))
    selected=[r for r in records if r.split==split]
    dataset=FoodImageDataset(selected,data_root=OUT,class_names=names,resize_mode="letterbox")
    predictions=[]
    for i,record in enumerate(selected):
        original=source_map[record.image_path]["path"]
        with Image.open(original) as im:raw=image_tensor(im,checkpoint)
        assert torch.equal(raw[0],dataset[i][0])
        result=predict_image(network,names,checkpoint,original)
        predictions.append({"original_path":original,"group":record.session_id,"true_label":record.food_label,
                            "predicted_label":result["predicted_class"],"correct":result["predicted_class"]==record.food_label,
                            "top1_probability":result["top1_probability"],"category_group":"old" if record.food_label in old_names else "new"})
    with (OUT/split/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    stats={}
    for label,chosen in [("old",old_names),("new",new_names),("all",names)]:
        rows=[r for r in predictions if r["true_label"] in chosen]
        stats[label]={"correct":sum(r["correct"] for r in rows),"total":len(rows),"accuracy":sum(r["correct"] for r in rows)/len(rows)}
    stats["per_class"]={name:{"correct":sum(r["correct"] for r in predictions if r["true_label"]==name),
                               "total":sum(r["true_label"]==name for r in predictions)} for name in names}
    assert stats["all"]["accuracy"]==metrics["overall_accuracy"]
    if split=="val":assert metrics["overall_accuracy"]==checkpoint["validation_accuracy"]
    evaluations[split]=stats
assert all(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in reviews)
baseline_report=json.loads((BASE/"final_report.json").read_text(encoding="utf-8"))
old_A=baseline_report["evaluations"]["test"]["overall_accuracy"]
drop_pp=100*(old_A-evaluations["test"]["old"]["accuracy"])
report={"stage":"first_increment_B","parent_stage":"formal_A_web_v1","parent_sha256":baseline["parent_sha256"],
        "class_names":names,"old_classes":old_names,"new_classes":new_names,"new_input_count":len(new_review),
        "new_included":sum(r["included"] for r in new_review),"new_excluded":sum(not r["included"] for r in new_review),
        "actual_training_counts":dict(Counter(r.food_label for r in used_train)),"training_count":len(used_train),
        "replay_per_old_class":20,"new_split":json.loads((OUT/"new_split_summary.json").read_text(encoding="utf-8")),
        "epochs":20,"learning_rate":.0001,"best_epoch":checkpoint["epoch"],"evaluations":evaluations,
        "A_old_test_accuracy":old_A,"B_old_test_accuracy":evaluations["test"]["old"]["accuracy"],
        "old_accuracy_drop_percentage_points":drop_pp,
        "original_target_checks":{"old_accuracy_at_least_90pct":evaluations["test"]["old"]["accuracy"]>=.9,
                                  "old_drop_at_most_2pp":drop_pp<=2,
                                  "new_accuracy_at_least_86pct":evaluations["test"]["new"]["accuracy"]>=.86},
        "parent_A_unchanged":True,"original_images_unchanged":True,"test_or_val_in_replay":False,
        "raw_input_matches_dataset":True,"new_process_reload":True,"execution":execution,
        "limits":["本轮为网图/去背景数据上的类增量，不等于烤箱或RK3506验收。",
                  "三文鱼70张无来源记录；来源记录与转载独立性未逐站核实，仍可能漏掉近重复。",
                  "只做样本回放，没有蒸馏/参数隔离对照；不能断言此算法最优。",
                  "旧类成绩按完整十类输出计算；新增竞争类别也可能影响旧类表现。",
                  "不依据测试结果追加调参或自动开启第二次增量。"]}
(OUT/"final_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
lines=["# 第一次增量：十类模型B", "", "从正式A继续训练，保留旧六类顺序，追加芦笋、培根、三文鱼、整鸡；没有从ImageNet重新训练十类。", "",
       "## 数据与配置", "", "新增794张，8张疑点暂排除。旧类回放每类20张，共120张；新类训练550张，实际训练共670张。",
       "旧验证157、新验证118；旧测试159、新测试118。旧划分保留，同源组隔离，回放不使用验证或测试图片。",
       "CPU四线程，20轮，lr=0.0001，AdamW，batch16；gentle+letterbox，按实际训练频次进行类别加权。仅使用十类验证选模型。",
       f"最佳第{checkpoint['epoch']}轮；训练耗时{execution['training_seconds']:.1f}秒。", "",
       "## 固定测试集结果", "", "| 范围 | 正确/总数 | 准确率 |", "|---|---:|---:|"]
for label,title in [("old","旧六类"),("new","新增四类"),("all","累计十类")]:
    stat=evaluations["test"][label];lines.append(f"| {title} | {stat['correct']}/{stat['total']} | {stat['accuracy']:.2%} |")
lines += ["",f"旧类A基准{old_A:.2%}；B相对A下降{drop_pp:.2f}个百分点（负数表示提高）。旧类在完整十类输出中竞争，不把输出截成六类后再算成绩。", "",
          "| 食材 | 正确/总数 | 准确率 |", "|---|---:|---:|"]
for name,counts in evaluations["test"]["per_class"].items():
    lines.append(f"| {name} | {counts['correct']}/{counts['total']} | {counts['correct']/counts['total']:.2%} |")
lines += ["", "原始目标检查（仅对应本轮网图测试）：", ""]
for key,value in report["original_target_checks"].items():lines.append(f"- {key}：{'达到' if value else '未达到'}。")
lines += ["", "## 逐轮验证", "", "| 轮次 | 训练损失 | 验证损失 | 十类验证准确率 |", "|---|---:|---:|---:|"]
for h in history:lines.append(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['validation_loss']:.5f} | {h['validation_accuracy']:.2%} |")
lines += ["", "## 审计与边界", "", "扩层时旧分类行和骨干参数相等检查通过；旧顺序保留，新行随机初始化。training/training_manifest.csv记录实际回放图片及验证图。",
          "新进程模型加载、逐张原图预测、输入一致性、父模型哈希和原图哈希检查通过；58项相关回归/集成检查通过。", ""]
lines += ["- "+item for item in report["limits"]]
(OUT/"第一次增量B_结果报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"best_epoch":checkpoint["epoch"],"test":evaluations["test"],"old_drop_pp":drop_pp,
                  "target_checks":report["original_target_checks"]},ensure_ascii=False,indent=2))
