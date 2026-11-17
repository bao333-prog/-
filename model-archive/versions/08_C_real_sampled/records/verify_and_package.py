"""新进程检查适配模型，记录训练关联诊断与旧验证，生成交付包。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / "source_snapshot"))
from oven_food.augment import prepare_image
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import ManifestRecord, read_manifest
from oven_food.model import build_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(checkpoint, records, tag):
    cp = torch.load(checkpoint, map_location="cpu", weights_only=True)
    names = cp["class_names"]
    model = build_model(cp["model_name"], num_classes=len(names), pretrained=False)
    model.load_state_dict(cp["model_state"], strict=True)
    model.eval()
    data = FoodImageDataset(records, data_root=ROOT, class_names=names,
                            augmentation_profile="baseline", image_size=224, resize_mode="letterbox")
    matrix = np.zeros((len(names), len(names)), dtype=int)
    predictions = []
    offset = 0
    with torch.inference_mode():
        for images, labels in DataLoader(data, batch_size=16, num_workers=0):
            probabilities = model(images).softmax(1)
            assert torch.isfinite(probabilities).all()
            indices = probabilities.argmax(1)
            for label, index, prob in zip(labels.tolist(), indices.tolist(), probabilities):
                matrix[label, index] += 1
                predictions.append(dict(image=records[offset].image_path, expected=names[label],
                                        predicted=names[index], probability=float(prob[index])))
                offset += 1
    result = dict(correct=int(matrix.trace()), total=int(matrix.sum()),
                  per_class={n: dict(correct=int(matrix[i,i]), total=int(matrix[i].sum()))
                             for i,n in enumerate(names)}, confusion_matrix=matrix.tolist())
    with (RUN / (tag + "_predictions.csv")).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return result


def main():
    torch.set_num_threads(4)
    config = json.loads((RUN / "train_config.json").read_text())
    inventory = json.loads((RUN / "inventory.json").read_text(encoding="utf-8"))
    audit = json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))
    history = json.loads((RUN / "training/history.json").read_text())
    assert len(history) == 20 and all(np.isfinite(r["train_loss"]) and np.isfinite(r["validation_loss"]) for r in history)
    parent = Path(config["checkpoint"])
    student = RUN / "training/best.pt"
    assert sha(parent) == audit["parent_sha256"]
    old_cp = torch.load(parent, map_location="cpu", weights_only=True)
    cp = torch.load(student, map_location="cpu", weights_only=True)
    assert cp["class_names"] == old_cp["class_names"] and len(cp["class_names"]) == 14
    assert {k:tuple(v.shape) for k,v in cp["model_state"].items()} == {k:tuple(v.shape) for k,v in old_cp["model_state"].items()}
    assert not cp.get("incremental") and not cp.get("distillation")
    assert cp["config"]["new_classes"] == ()
    # 全部输入只读指纹复核，同一字节图像不能带不同食材标签。
    labels_by_hash = {}
    for row in inventory:
        assert sha(Path(row["image"])) == row["raw_sha256"]
        assert sha(Path(row["prepared"])) == row["prepared_sha256"]
        assert labels_by_hash.setdefault(row["raw_sha256"],row["label"]) == row["label"]
    # 每类选取首末两帧，比较原图与缓存预处理的像素。
    for label in sorted({r["label"] for r in inventory}):
        subset = [r for r in inventory if r["label"] == label]
        for row in (subset[0], subset[-1]):
            with Image.open(row["image"]) as raw, Image.open(row["prepared"]) as cache:
                assert np.array_equal(np.asarray(prepare_image(raw,image_size=224,resize_mode="letterbox")),
                                      np.asarray(prepare_image(cache,image_size=224,resize_mode="letterbox")))
    real_records = [ManifestRecord(r["prepared"], r["label"], "target",
                                  r["label"]+"_"+r["session"], "fixed", "oven", "none", "none", "train")
                    for r in inventory]
    validation = [r for r in read_manifest(RUN / "manifest.csv") if r.split == "val"]
    result = dict(stage="C_real_food_adaptation", independent_target_test=False,
                  real_scope="Training-associated frames; includes selected and correlated unselected frames.",
                  parent_sha256=sha(parent), checkpoint_sha256=sha(student), epochs=20, best_epoch=cp["epoch"],
                  execution=json.loads((RUN / "execution.json").read_text()), class_names=cp["class_names"],
                  raw_hashes_checked=len(inventory), preprocessing_checks=24,
                  diagnostic_before=evaluate(parent,real_records,"real_before"),
                  diagnostic_after=evaluate(student,real_records,"real_after"),
                  validation_before=evaluate(parent,validation,"old_validation_before"),
                  validation_after=evaluate(student,validation,"old_validation_after"))
    (RUN / "final_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    def rate(record):
        return f'{record["correct"]}/{record["total"]}（{record["correct"]/record["total"]:.2%}）'
    text = ["# 十四类食材识别实拍适配", "",
            "从C v2继续训练20轮，最佳第"+str(cp["epoch"])+"轮。推理仍采用十四类MobileNetV3-Small。",
            "本次仅训练食材分类，实际训练使用521张实拍代表帧与每旧类60张共840张回放图。",
            "整鸡和南瓜缺少实拍，本轮仅靠各60张旧训练图片维持其识别。",
            "父模型C v2和适配版用于真实场景改进，不能将本轮结果单独解释为初类到最终模型的增量效果。", "",
            "## 结果口径", "",
            "| 范围 | 适配前 | 适配后 |", "|---|---:|---:|",
            "| 实拍训练关联诊断，1428帧 | "+rate(result["diagnostic_before"])+" | "+rate(result["diagnostic_after"])+" |",
            "| 固定旧网图验证，407帧 | "+rate(result["validation_before"])+" | "+rate(result["validation_after"])+" |", "",
            "实拍同一序列和同一会话内的图像相关，全部用于训练或训练关联诊断。当前没有独立实拍测试成绩，表中实拍结果用于检查是否学会本批外观。选模仅使用原固定网图验证，未重测历史测试集。", "",
            "## 实拍逐类诊断", "", "| 食材 | 适配前 | 适配后 |", "|---|---:|---:|"]
    for name,record in result["diagnostic_before"]["per_class"].items():
        after = result["diagnostic_after"]["per_class"][name]
        text.append("| "+name+" | "+(rate(record) if record["total"] else "缺实拍")+" | "+(rate(after) if after["total"] else "缺实拍")+" |")
    text += ["", "## 数据和限制", "",
             "检查了12类各命名序列的首中末代表帧，未逐张视觉复核全部1428张。原图全部成功解码且有元数据指纹的均匹配。剔除一张明显手部遮挡的玉米训练帧，其余少量手部、工具及锡纸反光仍可能影响表现。",
             "按文件夹和代表图确认食材标签，忽略采集JSON里沿用的候选pumpkin类别。可颂本批为熟制外观，牛排本批主要有锡纸，采样覆盖仍有限。原始JSON和图片保持原文件。",
             "固定配置为AdamW、头学习率0.00005、骨干学习率为头的20%、余弦调度、gentle_retention增强、目标域抽样占比60%、批量16、CPU4线程。未开展新的类别增量。",
             "权重保存、独立进程加载、类别顺序、网络形状、有限数值、1428张原图与副本指纹和24张输入一致性检查通过。"]
    report = "\n".join(text)+"\n"
    (RUN / "实拍适配结果.md").write_text(report,encoding="utf-8")
    delivery = ROOT / "outputs/C十四类实拍适配_20261105"
    delivery.mkdir(exist_ok=False)
    package = delivery / "十四类食材识别_实拍适配_代码与权重.zip"
    readme = ("# 十四类食材识别实拍适配\n\n"
              "安装Python依赖后在此目录执行 pip install -e .。\n"
              "预测命令为 python predict.py --model models/model_C_real.pt --image 实拍图片.jpg。\n"
              "输出十四类之一及已知类别相对分数，仅识别食材。"
              "南瓜和整鸡仍保留在类别中，本批缺其实际图片。\n"
              "实拍诊断包含训练图和同源图，实际泛化需要新的独立拍摄验证。\n"
              "详细结果见records/实拍适配结果.md。配置和清单含开发机原始路径，复训前需准备数据和修改路径副本。\n")
    with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md",readme)
        archive.write(student,"models/model_C_real.pt")
        archive.write(ROOT / "pyproject.toml","pyproject.toml")
        archive.write(ROOT / "runs/C_retention_v2_20261029_231408/predict.py","predict.py")
        for path in (RUN / "source_snapshot/oven_food").glob("*.py"):
            archive.write(path,"src/oven_food/"+path.name)
        for relative in ["final_report.json","实拍适配结果.md","data_audit.json","inventory.json","train_config.json","manifest.csv","training.log","execution.json","training/history.json","run_training.py","verify_and_package.py"]:
            archive.write(RUN / relative,"records/"+relative)
        for path in RUN.glob("*_predictions.csv"):
            archive.write(path,"records/"+path.name)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert hashlib.sha256(archive.read("models/model_C_real.pt")).hexdigest() == sha(student)
    shutil.copyfile(RUN / "实拍适配结果.md",delivery / "实拍适配结果.md")
    checks = dict(status="passed",package=str(package),package_sha256=sha(package),
                  checkpoint_sha256=sha(student),raw_images_unchanged=1428,
                  class_count=14,epochs=20,independent_real_test=False)
    (RUN / "final_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{field:v[field] for field in ("correct","total")}
                      for k,v in result.items() if k.startswith(("diagnostic","validation"))},
                     ensure_ascii=False),flush=True)
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
