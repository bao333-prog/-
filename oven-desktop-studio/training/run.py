"""Fine-tune the archived model with confirmed desktop samples, then export ONNX."""
import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "upstream"))
from oven_food.manifest import ManifestRecord, write_manifest


def prepare(package, output, splits):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    hashes = {}
    with zipfile.ZipFile(package) as archive:
        labels = [json.loads(line) for line in archive.read("labels.jsonl").decode("utf-8-sig").splitlines() if line.strip()]
        for index, item in enumerate(labels):
            batch = (item.get("batch") or "").strip()
            if batch not in splits or splits[batch] not in ("train", "val"):
                raise ValueError("请在批次划分文件中为每个拍摄批次指定 train 或 val：" + repr(batch))
            data = archive.read(item["image"])
            digest = hashlib.sha256(data).hexdigest()
            split = splits[batch]
            if digest in hashes:
                if hashes[digest] != (split, item["label"]):
                    raise ValueError("相同图片存在跨集合或标签冲突")
                continue
            hashes[digest] = (split, item["label"])
            name = f"images/{index}.image"
            (output / "images").mkdir(exist_ok=True)
            (output / name).write_bytes(data)
            rows.append(ManifestRecord(
                image_path=name, food_label=item["label"], environment="target",
                session_id=batch + "/" + item["label"], camera_position=item.get("device") or "unknown",
                lighting_condition="unknown", steam_level="none", occlusion_level="none", split=split,
            ))
    if {row.split for row in rows} != {"train", "val"}:
        raise ValueError("需要互不重叠的训练批次和验证批次")
    write_manifest(output / "manifest.csv", rows)
    (output / "source-labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "upstream/model_C_v2.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("训练轮数必须大于零")
    rows = prepare(args.samples, args.output, json.loads(args.splits.read_text(encoding="utf-8-sig")))
    import numpy as np
    import torch
    import onnxruntime as ort
    from PIL import Image
    from oven_food.config import ExperimentConfig
    from oven_food.model import build_model, checkpoint_resize_mode, checkpoint_class_names
    from oven_food.augment import prepare_image
    from oven_food.train import train_experiment

    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_count = sum(row.split == "train" for row in rows)
    if train_count < 2:
        raise ValueError("训练至少需要两张不同图片，以满足模型训练时的批归一化计算")
    batch_size = next((n for n in range(min(8, train_count), 1, -1) if train_count % n != 1), train_count)
    names = checkpoint_class_names(parent, observed_labels=[row.food_label for row in rows])
    size = parent["image_size"]
    resize = checkpoint_resize_mode(parent)
    mean = np.asarray(parent["preprocessing"]["mean"], dtype=np.float32)
    std = np.asarray(parent["preprocessing"]["std"], dtype=np.float32)
    from oven_food.dataset import IMAGENET_MEAN, IMAGENET_STD
    if not np.allclose(mean, IMAGENET_MEAN) or not np.allclose(std, IMAGENET_STD):
        raise ValueError("此训练工程使用 ImageNet 归一化，所选父权重的输入约定不一致")
    config = ExperimentConfig(
        mode="train", experiment_name="desktop_confirmed",
        manifest=str((args.output / "manifest.csv").resolve()), data_root=str(args.output.resolve()),
        output_dir=str((args.output / "training").resolve()), checkpoint=str(args.checkpoint.resolve()),
        train_environments=("target",), validation_environments=("target",),
        augmentation="gentle", model_name=parent["model_name"], image_size=size,
        batch_size=batch_size, epochs=args.epochs, learning_rate=0.0001, device="cpu",
        pretrained=False, resize_mode=resize, freeze_backbone=True, balance_sessions=True,
    )
    best = train_experiment(config)

    def load(checkpoint):
        model = build_model(checkpoint["model_name"], num_classes=len(names), pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        return model.eval()

    old = load(parent)
    candidate = torch.load(best, map_location="cpu", weights_only=True)
    new = load(candidate)
    results = []
    example = None
    with torch.inference_mode():
        for row in rows:
            if row.split != "val":
                continue
            with Image.open(args.output / row.image_path) as image:
                pixels = np.asarray(prepare_image(image, image_size=size, resize_mode=resize), dtype=np.float32) / 255
            array = ((pixels - mean) / std).transpose(2, 0, 1)[None].copy()
            example = torch.from_numpy(array)
            before = names[old(example).argmax(1).item()]
            after = names[new(example).argmax(1).item()]
            results.append({"image": row.image_path, "label": row.food_label, "before": before, "after": after})
    report = {
        "evaluation": "validation_used_for_model_selection",
        "parentSha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "count": len(results),
        "unassessedClasses": [name for name in names if not any(r["label"] == name for r in results)],
        "beforeAccuracy": sum(r["before"] == r["label"] for r in results) / len(results),
        "afterAccuracy": sum(r["after"] == r["label"] for r in results) / len(results),
        "perClass": {name: {
            "count": sum(r["label"] == name for r in results),
            "beforeCorrect": sum(r["label"] == name and r["before"] == name for r in results),
            "afterCorrect": sum(r["label"] == name and r["after"] == name for r in results),
        } for name in names},
        "results": results,
    }
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model_path = args.output / "food.onnx"
    example = example.clone()
    torch.onnx.export(new, example, model_path, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        np.testing.assert_allclose(session.run(None, {"image": example.numpy()})[0], new(example).numpy(), rtol=1e-3, atol=1e-4)
    metadata = {"names": names, "size": size, "resize_mode": resize, "mean": mean.tolist(), "std": std.tolist(),
                "version": "confirmed-" + datetime.now().strftime("%Y%m%d-%H%M%S")}
    (args.output / "food.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    heldout = args.samples.parent / "heldout.json"
    if heldout.exists():
        import os
        from replay import evaluate
        current = Path(os.environ.get("OVENDESK_COMPARE_MODEL", str(Path(__file__).resolve().parents[1] / "model")))
        evaluate(heldout, current, args.output)
    with zipfile.ZipFile(args.output / "model-package.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("food.onnx", "food.json", "comparison.json"):
            archive.write(args.output / name, name)
    print("训练和导出完成。查看 comparison.json 后，可通过模型管理导入 model-package.zip。", flush=True)


if __name__ == "__main__":
    main()

