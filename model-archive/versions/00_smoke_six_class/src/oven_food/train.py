from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .dataset import FoodImageDataset, records_for_usage
from .manifest import read_manifest, validate_manifest
from .model import build_model, checkpoint_class_names


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
    if config.mode != "train":
        raise ValueError("train_experiment requires a train configuration")
    if config.train_split != "train":
        raise ValueError("train_split must be 'train'")
    if config.validation_split != "val":
        raise ValueError("validation_split must be 'val'")
    torch, nn, DataLoader, WeightedRandomSampler = _require_torch()
    _seed_everything(torch, config.seed)
    device = _resolve_device(torch, config.device)

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

    train_dataset = FoodImageDataset(
        train_records,
        data_root=config.data_root,
        class_names=class_names,
        augmentation_profile=config.augmentation,
        image_size=config.image_size,
        seed=config.seed,
    )
    validation_dataset = FoodImageDataset(
        validation_records,
        data_root=config.data_root,
        class_names=class_names,
        augmentation_profile="baseline",
        image_size=config.image_size,
        seed=config.seed,
    )

    sampler = None
    if config.target_fraction is not None:
        sampler = _weighted_sampler(
            train_records, config.target_fraction, torch, WeightedRandomSampler
        )
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_count += labels.size(0)

        validation_loss, validation_accuracy = _validate_epoch(
            model, validation_loader, criterion, device, torch
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "model_name": config.model_name,
                    "image_size": config.image_size,
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
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
                "checkpoint": str(best_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_path


def _validate_epoch(model, loader, criterion, device, torch) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, correct / count

