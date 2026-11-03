"""训练结束后在新进程中加载模型；测试集只用于此次最终评测，不用于调参。"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import sys

import numpy as np
import torch
from oven_food.config import load_experiment_config
from oven_food.dataset import FoodImageDataset
from oven_food.evaluate import evaluate_experiment
from oven_food.manifest import read_manifest, validate_manifest
from predict import load_classifier, image_tensor, predict_image

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
execution = json.loads((OUT / "execution.json").read_text(encoding="utf-8"))
assert execution["status"] == "completed"
history = json.loads((OUT / "training/history.json").read_text(encoding="utf-8"))
assert len(history) == 20 and all(math.isfinite(v) for row in history for v in row.values())
review = json.loads((OUT / "review_and_split.json").read_text(encoding="utf-8"))
records = validate_manifest(read_manifest(OUT / "manifest.csv"), check_files=True)
model,names,checkpoint = load_classifier(OUT / "training/best.pt")
assert set(names) == {"牛排","鸡翅","南瓜","吐司","玉米","西兰花"}
assert checkpoint["manifest_sha256"] == hashlib.sha256((OUT / "manifest.csv").read_bytes()).hexdigest()

# 再次检查整组隔离及原图重复，不能只相信文件夹名称或划分比例。
split_groups = defaultdict(set)
split_hashes = defaultdict(set)
for row in review:
    if row["included"]:
        split_groups[row["group"]].add(row["split"])
        split_hashes[row["sha256"]].add(row["split"])
assert all(len(s)==1 for s in split_groups.values())
assert all(len(s)==1 for s in split_hashes.values())
assert all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"] for row in review)
source_map = {row["prepared_path"]: row for row in review if row["included"]}
evaluations = {}
for split in ["val","test"]:
    config = json.loads((OUT / "train_config.json").read_text(encoding="utf-8"))
    config.update(mode="evaluate",experiment_name="formal_A_"+split,
                  checkpoint=str(OUT/"training/best.pt"),output_dir=str(OUT/split),
                  evaluation_environments=["proxy"],evaluation_split=split,batch_size=16)
    config_path = OUT / (split+"_config.json")
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    metrics = json.loads(evaluate_experiment(load_experiment_config(config_path)).read_text(encoding="utf-8"))
    selected = [record for record in records if record.split==split]
    dataset = FoodImageDataset(selected,data_root=OUT,class_names=names,augmentation_profile="baseline")
    predictions = []
    for i,record in enumerate(selected):
        original = source_map[record.image_path]["path"]
        # 用户直接输入原图的结果必须与训练准备副本的预处理一致。
        raw = image_tensor(original,checkpoint["preprocessing"])
        prepared,_ = dataset[i]
        assert torch.equal(raw[0], prepared)
        result = predict_image(model,names,checkpoint["preprocessing"],original)
        assert math.isfinite(result["top1_probability"])
        predictions.append({"original_path":original,"group":record.session_id,
                            "true_label":record.food_label,"predicted_label":result["predicted_class"],
                            "correct":result["predicted_class"]==record.food_label,
                            "top1_probability":result["top1_probability"]})
    counts = {name: {"correct":sum(p["correct"] for p in predictions if p["true_label"]==name),
                     "total":sum(p["true_label"]==name for p in predictions)} for name in names}
    correct = sum(p["correct"] for p in predictions)
    assert correct/len(predictions) == metrics["overall_accuracy"]
    if split == "val":
        assert metrics["overall_accuracy"] == checkpoint["validation_accuracy"]
    with (OUT/split/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]))
        writer.writeheader();writer.writerows(predictions)
    evaluations[split] = {"correct":correct,"total":len(predictions),
                          "overall_accuracy":metrics["overall_accuracy"],"per_class":counts,
                          "macro_class_accuracy":sum(v["correct"]/v["total"] for v in counts.values())/len(names)}

# 确认模型确实在预训练起点上更新过骨干参数。
weights_path = Path(torch.hub.get_dir())/"checkpoints/mobilenet_v3_small-047dcff4.pth"
official = torch.load(weights_path,map_location="cpu",weights_only=True)
key = "features.0.0.weight"
delta = float((checkpoint["model_state"][key]-official[key]).abs().max())
assert delta > 0
report = {"status":"passed","model_stage":"formal_A_web_v1","data_version":OUT.name,
          "input_count":len(review),"included_count":sum(row["included"] for row in review),
          "excluded_count":sum(not row["included"] for row in review),
          "split_summary":json.loads((OUT/"split_summary.json").read_text(encoding="utf-8")),
          "epochs_completed":20,"learning_rate":0.0003,"best_epoch":checkpoint["epoch"],
          "evaluations":evaluations,"new_process_reload":True,"raw_input_preprocessing_verified":True,
          "group_overlap":False,"exact_duplicate_cross_split":False,"original_image_hashes_unchanged":True,
          "backbone_weight_max_change":delta,"execution":execution,
          "limits":["网页/去背景数据上的基准，不代表目标烤箱或RK3506性能。",
                    "鸡翅无来源元数据；感知哈希和保守来源分组不能保证所有转载、近重复已识别。",
                    "来源说明由数据包提供，尚未逐站核实真实性、图像生成方式或来源独立性。",
                    "单标签六类分类，对其他食品仍会给出六类之一；不支持检测、计数或未知类拒识。",
                    "没有设精度门槛，不因测试结果追加调参；后续增量应固定本次旧类测试集合。"]}
(OUT/"final_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
lines = ["# 正式六类模型A：网页数据基准", "", "本轮训练与新进程加载检查已完成。正式A指固定数据版本后的初类基线，不代表目标烤箱验收通过。", "",
         "## 训练设置", "", "MobileNetV3-Small官方预训练，全模型更新；CPU四线程；20轮，学习率0.0003，AdamW，batch=16，seed=42。",
         "输入1090张，排除45张疑点，实际使用1045张。原图不改，生成224×224等比例白底补边副本；透明图片合成白底。",
         "没有沿用60张小样本模型。按验证准确率选择模型，平分时保留最早一轮；测试结果不参与选择。", "",
         f"最佳轮次：第{checkpoint['epoch']}轮；训练进程耗时：{execution['training_seconds']:.1f}秒（不含前期数据复核与准备）。", "",
         "## 数据划分", "", "| 类别 | 训练 | 验证 | 测试 |", "|---|---:|---:|---:|"]
for name,counts in report["split_summary"].items():
    lines.append(f"| {name} | {counts['train']} | {counts['val']} | {counts['test']} |")
for split,title in [("val","验证集"),("test","固定网页测试集")]:
    result=evaluations[split]
    lines += ["",f"## {title}","",f"总体：{result['correct']}/{result['total']}，{result['overall_accuracy']:.2%}；各类准确率平均：{result['macro_class_accuracy']:.2%}。", "",
              "| 类别 | 正确/总数 | 准确率 |", "|---|---:|---:|"]
    for name,c in result["per_class"].items():
        lines.append(f"| {name} | {c['correct']}/{c['total']} | {c['correct']/c['total']:.2%} |")
lines += ["","## 逐轮训练","","| 轮次 | 训练损失 | 验证损失 | 验证准确率 |","|---|---:|---:|---:|"]
for h in history:
    lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['validation_loss']:.4f} | {h['validation_accuracy']:.2%} |")
lines += ["","## 复核与边界","",
          "按来源域名、旧数据来源整体、近重复和人工疑似同源归组；旧来源标注test的吐司整组只进测试。固定清单及哈希保存在manifest.csv与review_and_split.json。",
          "排除包含鸡腿部位疑点、过曝/文字标识、插画来源疑点、熟制状态疑点；逐项理由在review_and_split.json。未删除原图。",
          "新进程加载、逐张原图预测与副本输入一致性、同组隔离、原图哈希检查均通过。",
          "模型、训练、数据读取和评测代码已添加中文注释；33项相关测试通过，已核对仅注释/说明变化。", ""]
lines += ["- "+item for item in report["limits"]]
lines += ["","## 主要文件","","- training/best.pt：正式A权重，含类别顺序与预处理约定。",
          "- predict.py：有中文注释的原图预测示例。",
          "- test/predictions.csv、test/confusion_matrix.csv：逐张测试结果与混淆矩阵。",
          "- train_config.json、manifest.csv、review_and_split.json：配置、固定划分、来源及排除理由。",
          "- training/history.json、training.log：逐轮记录。"]
(OUT/"正式模型A_结果报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
with (OUT/"progress.md").open("a",encoding="utf-8") as handle:
    handle.write("\n训练20轮完成；新进程原图预测、验证与测试统计、原图哈希及整组隔离检查通过。详见final_report.json。\n")
print(json.dumps({"best_epoch":report["best_epoch"],"evaluations":evaluations,"training_seconds":execution["training_seconds"]},ensure_ascii=False,indent=2))
