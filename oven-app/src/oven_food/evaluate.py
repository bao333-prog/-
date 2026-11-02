from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .config import ExperimentConfig
from .dataset import FoodImageDataset, records_for_usage
from .manifest import read_manifest, validate_manifest
from .metrics import classification_metrics
from .model import build_model, checkpoint_class_names, checkpoint_resize_mode
from .train import _require_torch, _resolve_device


def _measure_inference_latency_ms(
    model,
    *,
    device,
    image_size: int,
    torch,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> float:
    sample = torch.zeros((1, 3, image_size, image_size), device=device)
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(timed_runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000 / timed_runs


def evaluate_experiment(config: ExperimentConfig) -> Path:
    """重新加载已选定模型，在指定集合上计算准确率、混淆矩阵及本机耗时。"""
    if config.mode != "evaluate":
        raise ValueError("evaluate_experiment requires an evaluate configuration")
    torch, _, DataLoader, _ = _require_torch()
    device = _resolve_device(torch, config.device)
    checkpoint_path = Path(config.checkpoint or "")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    records = validate_manifest(
        read_manifest(config.manifest), data_root=config.data_root, check_files=True
    )
    evaluation_records = records_for_usage(
        records,
        split=config.evaluation_split,
        environments=set(config.evaluation_environments),
    )
    class_names = checkpoint_class_names(
        checkpoint,
        observed_labels={record.food_label for record in evaluation_records},
    )
    dataset = FoodImageDataset(
        evaluation_records,
        data_root=config.data_root,
        class_names=class_names,
        augmentation_profile="baseline",
        image_size=checkpoint.get("image_size", config.image_size),
        seed=config.seed,
        # 以权重中的输入约定为准，防止默认配置把补边模型误当中心裁剪模型。
        resize_mode=checkpoint_resize_mode(checkpoint),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    # 先重建相同网络结构，再加载训练参数；此处不重新下载预训练权重。
    model = build_model(
        checkpoint.get("model_name", config.model_name),
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            true_labels.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    inference_latency = _measure_inference_latency_ms(
        model,
        device=device,
        image_size=checkpoint.get("image_size", config.image_size),
        torch=torch,
    )
    # 混淆矩阵行是真实类别、列是预测类别，可定位哪些食材相互混淆。
    # 电脑端耗时不是RK3506板端指标，两者必须分别实测。
    metrics = classification_metrics(
        true_labels, predicted_labels, class_names=class_names
    )
    metrics.update(
        {
            "experiment_name": config.experiment_name,
            "split": config.evaluation_split,
            "environments": list(config.evaluation_environments),
            "checkpoint": str(checkpoint_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "mean_end_to_end_latency_ms_per_image": elapsed * 1000 / len(dataset),
            "mean_inference_latency_ms_per_image": inference_latency,
            "evaluation_batch_size": config.batch_size,
            "inference_timing_batch_size": 1,
            "inference_warmup_runs": 5,
            "inference_timed_runs": 20,
            "device": str(device),
            "peak_device_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        class_names,
        metrics["confusion_matrix"],
    )
    return metrics_path


def _write_confusion_matrix(path: Path, class_names: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *row])
