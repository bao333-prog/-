from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from .augment import PROFILES, RESIZE_MODES
from .manifest import ALLOWED_ENVIRONMENTS, ALLOWED_SPLITS


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    mode: str
    experiment_name: str
    manifest: str
    data_root: str
    output_dir: str
    checkpoint: str | None = None
    train_environments: tuple[str, ...] = ()
    validation_environments: tuple[str, ...] = ()
    evaluation_environments: tuple[str, ...] = ()
    train_split: str = "train"
    validation_split: str = "val"
    evaluation_split: str = "test"
    augmentation: str = "baseline"
    model_name: str = "mobilenet_v3_small"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    freeze_backbone: bool = False
    target_fraction: float | None = None
    # 旧配置保持原有行为；新候选显式启用补边和按训练样本数计算的类别权重。
    resize_mode: str = "center_crop"
    class_balanced_loss: bool = False


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a JSON object")

    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError("unknown configuration keys: " + ", ".join(unknown))

    required = ("mode", "experiment_name", "manifest", "data_root", "output_dir")
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ConfigError("missing required configuration keys: " + ", ".join(missing))

    for name in ("train_environments", "validation_environments", "evaluation_environments"):
        value = raw.get(name, ())
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{name} must be a JSON array")
        raw[name] = tuple(value)

    try:
        config = ExperimentConfig(**raw)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.mode not in {"train", "evaluate"}:
        raise ConfigError("mode must be 'train' or 'evaluate'")
    if config.augmentation not in PROFILES:
        raise ConfigError("augmentation must be one of: " + ", ".join(sorted(PROFILES)))
    if config.resize_mode not in RESIZE_MODES:
        raise ConfigError("resize_mode must be one of: " + ", ".join(sorted(RESIZE_MODES)))
    if config.augmentation == "gentle" and config.resize_mode != "letterbox":
        raise ConfigError("gentle augmentation requires resize_mode='letterbox'")
    if not isinstance(config.class_balanced_loss, bool):
        raise ConfigError("class_balanced_loss must be a boolean")
    if config.train_split != "train":
        raise ConfigError("train_split must be 'train'")
    for name in ("validation_split", "evaluation_split"):
        if getattr(config, name) not in ALLOWED_SPLITS:
            raise ConfigError(f"{name} must be train, val, or test")

    environment_fields = (
        "train_environments",
        "validation_environments",
        "evaluation_environments",
    )
    for name in environment_fields:
        values = set(getattr(config, name))
        invalid = values - ALLOWED_ENVIRONMENTS
        if invalid:
            raise ConfigError(f"{name} contains unknown values: {', '.join(sorted(invalid))}")

    if config.mode == "train":
        if config.validation_split != "val":
            raise ConfigError("validation_split must be 'val'")
        if not config.train_environments:
            raise ConfigError("train mode requires train_environments")
        if not config.validation_environments:
            raise ConfigError("train mode requires validation_environments")
    else:
        if not config.checkpoint:
            raise ConfigError("evaluate mode requires checkpoint")
        if not config.evaluation_environments:
            raise ConfigError("evaluate mode requires evaluation_environments")

    if config.target_fraction is not None:
        if not 0 < config.target_fraction < 1:
            raise ConfigError("target_fraction must be between 0 and 1")
        if set(config.train_environments) != {"proxy", "target"}:
            raise ConfigError(
                "target_fraction requires both proxy and target in train_environments"
            )

    positive_values = {
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ConfigError(f"{name} must be positive")
    if config.num_workers < 0 or config.weight_decay < 0:
        raise ConfigError("num_workers and weight_decay must be nonnegative")

