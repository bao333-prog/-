from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .augment import ProxyAugmenter
from .manifest import ALLOWED_ENVIRONMENTS, ManifestRecord, validate_manifest


# 输入归一化与官方预训练模型保持一致；不是从验证/测试图片估计这些常数。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def records_for_usage(
    records: Iterable[ManifestRecord],
    *,
    split: str,
    environments: set[str] | None = None,
    training: bool = False,
) -> list[ManifestRecord]:
    if training and split != "train":
        raise ValueError("training data must use split='train'")
    if environments is not None:
        invalid = environments - ALLOWED_ENVIRONMENTS
        if invalid:
            raise ValueError(f"unknown environments: {', '.join(sorted(invalid))}")

    materialized = validate_manifest(records)
    selected = [
        record
        for record in materialized
        if record.split == split
        and (environments is None or record.environment in environments)
    ]
    if not selected:
        environment_text = "all environments" if environments is None else ", ".join(sorted(environments))
        raise ValueError(f"no records found for split={split!r} in {environment_text}")
    return selected


def build_class_to_index(
    records: Iterable[ManifestRecord],
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, int]:
    labels = {record.food_label for record in records}
    if class_names is None:
        ordered = sorted(labels)
    else:
        ordered = list(class_names)
        if len(ordered) != len(set(ordered)):
            raise ValueError("class_names contains duplicates")
        unknown = labels - set(ordered)
        if unknown:
            raise ValueError("unknown food labels: " + ", ".join(sorted(unknown)))
    return {name: index for index, name in enumerate(ordered)}


class FoodImageDataset:
    """A torch-compatible dataset with torch imported only when samples are read."""

    def __init__(
        self,
        records: Iterable[ManifestRecord],
        *,
        data_root: str | Path,
        class_names: Sequence[str] | None = None,
        augmentation_profile: str = "baseline",
        image_size: int = 224,
        seed: int = 42,
        resize_mode: str = "center_crop",
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("dataset needs at least one record")
        self.data_root = Path(data_root)
        self.augmentation_profile = augmentation_profile
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.class_to_index = build_class_to_index(self.records, class_names=class_names)
        self.class_names = [
            name for name, _ in sorted(self.class_to_index.items(), key=lambda item: item[1])
        ]
        self.augmenter = ProxyAugmenter(
            self.augmentation_profile,
            image_size=self.image_size,
            seed=seed,
            resize_mode=self.resize_mode,
        )

    def reseed(self, seed: int) -> None:
        """Reset augmentation randomness for a DataLoader worker."""
        self.augmenter = ProxyAugmenter(
            self.augmentation_profile,
            image_size=self.image_size,
            seed=seed,
            resize_mode=self.resize_mode,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on runtime setup
            raise RuntimeError(
                "PyTorch is required to read training samples; install the project dependencies"
            ) from exc

        record = self.records[index]
        image_path = Path(record.image_path)
        resolved = image_path if image_path.is_absolute() else self.data_root / image_path
        # 旧配置仍中心裁剪；新候选显式使用letterbox，原图直接补边，预测使用同一函数。
        with Image.open(resolved) as source:
            image = self.augmenter(source)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        # Pillow数组为高×宽×通道，PyTorch卷积需要通道×高×宽。
        tensor = torch.from_numpy(np.transpose(pixels, (2, 0, 1)).copy())
        return tensor, self.class_to_index[record.food_label]
