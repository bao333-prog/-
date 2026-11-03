"""只用固定验证集及预先规定的扰动检查候选，不重新查看旧测试集成绩选模型。"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math

from PIL import Image,ImageEnhance,ImageOps
import torch
from oven_food.config import load_experiment_config
from oven_food.dataset import FoodImageDataset
from oven_food.evaluate import evaluate_experiment
from oven_food.manifest import read_manifest
from predict import load_classifier,image_tensor,predict_image

OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
baseline=json.loads((OUT/"baseline.json").read_text(encoding="utf-8"));BASE=Path(baseline["baseline_dir"])
assert hashlib.sha256((BASE/"training/best.pt").read_bytes()).hexdigest()==baseline["baseline_checkpoint_sha256"]
assert hashlib.sha256((BASE/"manifest.csv").read_bytes()).hexdigest()==baseline["manifest_sha256"]
history=json.loads((OUT/"training/history.json").read_text(encoding="utf-8"))
assert len(history)==20 and all(math.isfinite(v) for h in history for v in h.values())
expected=min(history,key=lambda h:(-h["validation_accuracy"],h["validation_loss"]))
model,names,checkpoint=load_classifier(OUT/"training/best.pt")
assert checkpoint["epoch"]==expected["epoch"]
assert checkpoint["class_loss_weights"] is not None
assert checkpoint["config"]["checkpoint"] is None and checkpoint["config"]["pretrained"]
config=json.loads((OUT/"train_config.json").read_text(encoding="utf-8"))
config.update(mode="evaluate",experiment_name="v2_validation",checkpoint=str(OUT/"training/best.pt"),
              output_dir=str(OUT/"validation"),evaluation_environments=["proxy"],evaluation_split="val")
(OUT/"validation_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
metrics=json.loads(evaluate_experiment(load_experiment_config(OUT/"validation_config.json")).read_text(encoding="utf-8"))
assert metrics["overall_accuracy"]==checkpoint["validation_accuracy"]
review=json.loads((BASE/"review_and_split.json").read_text(encoding="utf-8"))
sources={r["prepared_path"]:r for r in review if r["included"]}
records=[r for r in read_manifest(BASE/"manifest.csv") if r.split=="val"]
dataset=FoodImageDataset(records,data_root=BASE,class_names=names,resize_mode="letterbox")
rows=[]
for i,record in enumerate(records):
    original=sources[record.image_path]["path"]
    with Image.open(original) as image:
        raw=image_tensor(image,checkpoint)
    assert torch.equal(raw[0],dataset[i][0])
    result=predict_image(model,names,checkpoint,original)
    rows.append({"original_path":original,"true_label":record.food_label,"predicted_label":result["predicted_class"],
                 "correct":result["predicted_class"]==record.food_label,"top1_probability":result["top1_probability"]})
with (OUT/"validation/predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert sum(r["correct"] for r in rows)/len(rows)==metrics["overall_accuracy"]

# 固定的验证集扰动，仅诊断敏感性；不按这张表追加调参，也不能当独立测试成绩。
transforms={"horizontal_flip":ImageOps.mirror,
            "brightness_0.7":lambda image:ImageEnhance.Brightness(image).enhance(0.7),
            "brightness_1.3":lambda image:ImageEnhance.Brightness(image).enhance(1.3),
            "rotation_15":lambda image:image.rotate(15,resample=Image.Resampling.BILINEAR,expand=True,fillcolor="white")}
stress={}
for version,path in [("A_v1",BASE/"training/best.pt"),("A_candidate_v2",OUT/"training/best.pt")]:
    network,labels,cp=load_classifier(path)
    stress[version]={}
    for case,transform in transforms.items():
        correct=0
        for start in range(0,len(records),16):
            tensors=[]
            for record in records[start:start+16]:
                with Image.open(record.image_path) as image:
                    tensors.append(image_tensor(transform(image.convert("RGB")),cp))
            with torch.inference_mode():
                predictions=network(torch.cat(tensors)).argmax(1).tolist()
            correct+=sum(labels[pred]==r.food_label for pred,r in zip(predictions,records[start:start+16]))
        stress[version][case]={"correct":correct,"total":len(records),"accuracy":correct/len(records)}
assert all(hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in review)
execution=json.loads((OUT/"execution.json").read_text(encoding="utf-8"))
report={"candidate":"A_v2","epochs":20,"learning_rate":0.0003,"best_epoch":checkpoint["epoch"],
        "validation_accuracy":metrics["overall_accuracy"],"validation_loss":checkpoint["validation_loss"],
        "last_epoch_validation_accuracy":history[-1]["validation_accuracy"],
        "per_class_validation":metrics["per_class_accuracy"],
        "baseline_validation_accuracy":1.0,"baseline_validation_loss":0.028884543247586176,
        "class_weights":dict(zip(names,checkpoint["class_loss_weights"])),"validation_stress":stress,
        "training_seconds":execution["training_seconds"],"original_A_unchanged":True,
        "original_images_unchanged":True,"raw_and_dataset_tensors_equal":True,
        "test_not_evaluated_for_v2":True,
        "training_loss_log_definition":"本轮运行启动于日志汇总修正前，train_loss为按样本数平均的加权批次均值；不影响梯度、未加权验证损失或模型选择。交付代码已改为按标签权重总量汇总。",
        "limits":["验证集用于选模型，扰动诊断不等同独立测试。","代码修正已验证，不保证所有新图片更准。",
                  "缺少来源、近重复遗漏和数据范围不足不能靠这些代码修改消除。"]}
(OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
lines=["# A候选v2：构建与训练改进","",
       "保留MobileNetV3-Small结构，旧A权重和原图不变。20轮、学习率0.0003，从官方预训练重新训练，相同729张训练和157张验证。", "",
       "## 修改", "", "1. 补边、透明合成白底、EXIF方向校正集中到prepare_image，训练与预测共用；权重保存输入约定。",
       "2. 温和增强只用于训练：镜像、±8度旋转（扩展画布）、亮度/对比度/饱和度±10%，不随机裁掉食材。",
       "3. 按训练集频次计算逆频率交叉熵权重，补偿鸡翅样本偏少；不会产生新的独立样本。",
       "4. 最佳模型先比较验证准确率，同分再选验证损失较低者；非有限损失立即停止。", "",
       "复核另修正了standard/targeted与letterbox组合的透明背景处理，以及类别加权训练损失的日志汇总。当前候选使用gentle，训练参数及验证指标不受这两项修正影响。", "",
       "## 结果", "", f"最佳第{checkpoint['epoch']}轮，验证准确率{metrics['overall_accuracy']:.2%}，未加权验证损失{checkpoint['validation_loss']:.6f}。",
       "旧A验证准确率100%，损失0.028885。验证损失降低并不直接证明真实泛化或置信度校准改善。", "",
       "## 固定验证集扰动诊断", "", "此表用于诊断，不是独立测试，不据此追加训练或调参。", "",
       "| 扰动 | 旧A | 候选v2 |", "|---|---:|---:|"]
for case in transforms:
    lines.append(f"| {case} | {stress['A_v1'][case]['correct']}/157 | {stress['A_candidate_v2'][case]['correct']}/157 |")
lines += ["", "## 逐轮记录", "", "训练后半段有明显验证波动，本次保留的是最佳验证权重，不是最后一轮。未继续追加训练。", "",
          "| 轮次 | 训练日志损失 | 验证损失 | 验证准确率 |", "|---|---:|---:|---:|"]
for h in history:
    lines.append(f"| {h['epoch']} | {h['train_loss']:.5f} | {h['validation_loss']:.5f} | {h['validation_accuracy']:.2%} |")
lines += ["", "## 验证与使用边界", "", "新进程加载、逐张原图与数据集输入一致性、旧A及原图哈希检查通过。",
          "旧测试集已经展示过成绩，本次不使用它挑选候选版本；没有报告v2的新独立测试成绩。",
          "新增与既有回归检查通过，另含真实网络的小数据训练和重新加载评测检查。", "",
          "日志说明：本轮长训练在日志汇总修正前已启动，train_loss保留当时按样本数平均的加权批次均值；交付代码已按标签权重总量正确汇总。该差异只影响训练日志，不影响本轮梯度、验证损失或模型选择。", "",
          "旧配置默认保持center_crop和不加权损失；旧权重仍可加载。新配置显式选择gentle、letterbox、class_balanced_loss=true。",
          "现有代码仍未实现新增类别的扩层、回放或蒸馏；本次不是增量B/C。"]
(OUT/"改进说明与验证结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
