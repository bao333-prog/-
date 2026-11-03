from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .dataset import FoodImageDataset, IMAGENET_MEAN, IMAGENET_STD, records_for_usage
from .manifest import read_manifest, validate_manifest
from .model import build_model, checkpoint_class_names, checkpoint_resize_mode


def _class_loss_weights(records, class_names) -> list[float]:
    """仅按训练集频次计算逆频率权重；不读取验证或测试集的类别比例。"""
    counts = Counter(record.food_label for record in records)
    if any(counts[name] == 0 for name in class_names):
        raise ValueError("all classes need training samples for balanced loss")
    total = sum(counts.values())
    return [total / (len(class_names) * counts[name]) for name in class_names]


def _is_better_checkpoint(accuracy, loss, best_accuracy, best_loss) -> bool:
    """先比较验证准确率，同分时选择验证损失更低的权重，拒绝非有限数值。"""
    return (math.isfinite(accuracy) and math.isfinite(loss)
            and (accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss)))


def _loss_totals(loss, labels, class_weights) -> tuple[float, float]:
    # 加权交叉熵的mean分母是标签权重之和，而不是图片数；用于准确汇总日志。
    mass = float(class_weights[labels].sum()) if class_weights is not None else labels.size(0)
    return loss.item() * mass, mass


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            "Training requires PyTorch and torchvision. Install with: pip install -e ."
        ) from exc
    return torch, nn, DataLoader, WeightedRandomSampler


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(torch, seed: int) -> None:
    # 固定随机种子，使初始化、打乱顺序等尽量可复现；跨设备仍可能有数值差异。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weighted_sampler(records, target_fraction: float, torch, sampler_type):
    counts = {
        environment: sum(record.environment == environment for record in records)
        for environment in ("proxy", "target")
    }
    if not counts["proxy"] or not counts["target"]:
        raise ValueError("target_fraction requires non-empty proxy and target training records")
    environment_mass = {"proxy": 1 - target_fraction, "target": target_fraction}
    weights = [environment_mass[record.environment] / counts[record.environment] for record in records]
    return sampler_type(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
    )


def _seed_worker(worker_id: int) -> None:
    import torch

    worker = torch.utils.data.get_worker_info()
    if worker is not None and hasattr(worker.dataset, "reseed"):
        worker.dataset.reseed(worker.seed)


def train_experiment(config: ExperimentConfig) -> Path:
    """按配置完成训练，用验证集选择最佳权重；测试集不参与模型选择。"""
    if config.mode != "train":
        raise ValueError("train_experiment requires a train configuration")
    if config.train_split != "train":
        raise ValueError("train_split must be 'train'")
    if config.validation_split != "val":
        raise ValueError("validation_split must be 'val'")
    torch, nn, DataLoader, WeightedRandomSampler = _require_torch()
    _seed_everything(torch, config.seed)
    device = _resolve_device(torch, config.device)

    # 清单明确记录图片、标签、同源组和集合；同源组不能跨训练/验证/测试。
    records = validate_manifest(
        read_manifest(config.manifest), data_root=config.data_root, check_files=True
    )
    train_records = records_for_usage(
        records,
        split=config.train_split,
        environments=set(config.train_environments),
        training=True,
    )
    validation_records = records_for_usage(
        records,
        split=config.validation_split,
        environments=set(config.validation_environments),
    )

    # 载入已有模型时保留其类别下标；重新训练正式A时checkpoint留空。
    checkpoint = None
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        class_names = checkpoint_class_names(
            checkpoint,
            observed_labels={item.food_label for item in train_records + validation_records},
        )
    else:
        class_names = sorted(
            {item.food_label for item in train_records + validation_records}
        )
    if checkpoint is not None and checkpoint_resize_mode(checkpoint) != config.resize_mode:
        raise ValueError("checkpoint resize_mode does not match training configuration")

    train_dataset = FoodImageDataset(
        train_records,
        data_root=config.data_root,
        class_names=class_names,
        augmentation_profile=config.augmentation,
        image_size=config.image_size,
        seed=config.seed,
        resize_mode=config.resize_mode,
    )
    validation_dataset = FoodImageDataset(
        validation_records,
        data_root=config.data_root,
        class_names=class_names,
        augmentation_profile="baseline",
        image_size=config.image_size,
        seed=config.seed,
        resize_mode=config.resize_mode,
    )

    sampler = None
    if config.target_fraction is not None:
        sampler = _weighted_sampler(
            train_records, config.target_fraction, torch, WeightedRandomSampler
        )
    # 训练批次打乱，验证批次顺序固定；验证图片不做随机增强。
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker if config.num_workers > 0 else None,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        config.model_name,
        num_classes=len(class_names),
        pretrained=config.pretrained and checkpoint is None,
        freeze_backbone=config.freeze_backbone,
    )
    if checkpoint is not None:
        if checkpoint.get("model_name", config.model_name) != config.model_name:
            raise ValueError("checkpoint model_name does not match configuration")
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # 交叉熵用于单标签分类；AdamW按梯度更新可训练参数，权重衰减抑制过拟合。
    class_weights = _class_loss_weights(train_records, class_names) if config.class_balanced_loss else None
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    # 验证损失始终不加类别权重，保证不同训练方案的模型选择指标可比较。
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_accuracy = -1.0
    best_loss = math.inf
    best_path = output_dir / "best.pt"

    # 一轮(epoch)遍历一次训练加载器；验证只检查表现，不更新模型参数。
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_loss_mass = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 清空旧梯度 → 前向计算 → 计算损失 → 反向求梯度 → 更新参数。
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss; stopped before parameter update")
            loss.backward()
            optimizer.step()
            batch_total, batch_mass = _loss_totals(loss, labels, weight_tensor)
            train_loss_sum += batch_total
            train_loss_mass += batch_mass

        validation_loss, validation_accuracy = _validate_epoch(
            model, validation_loader, validation_criterion, device, torch
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_loss_mass,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        # 只用验证数据选模型；准确率同分时进一步比较验证损失。
        if _is_better_checkpoint(validation_accuracy, validation_loss, best_accuracy, best_loss):
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    # 权重本身不包含类别名称，必须一起保存才能正确解释预测下标。
                    "class_names": class_names,
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "class_loss_weights": class_weights,
                    "preprocessing": {"resize_mode": config.resize_mode,
                                      "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist()},
                    "config": asdict(config),
                },
                best_path,
            )

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_name": config.experiment_name,
                "best_validation_accuracy": best_accuracy,
                "best_validation_loss": best_loss,
                "checkpoint": str(best_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path


def _validate_epoch(model, loader, criterion, device, torch) -> tuple[float, float]:
    # eval切换Dropout/BatchNorm的行为；inference_mode关闭梯度记录以节省资源。
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count

