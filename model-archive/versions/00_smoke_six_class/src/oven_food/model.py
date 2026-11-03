from __future__ import annotations

from collections.abc import Iterable, Mapping


class CheckpointContractError(ValueError):
    """Raised when a checkpoint cannot safely be reused."""


def checkpoint_class_names(
    checkpoint: Mapping,
    *,
    observed_labels: Iterable[str] = (),
) -> list[str]:
    raw_names = checkpoint.get("class_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise CheckpointContractError("checkpoint class_names must be a non-empty list")
    names = list(raw_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise CheckpointContractError("checkpoint class_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise CheckpointContractError("checkpoint class_names contains duplicates")
    unknown = set(observed_labels) - set(names)
    if unknown:
        raise CheckpointContractError("unknown labels: " + ", ".join(sorted(unknown)))
    return names


def build_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool = False,
):
    try:
        from torch import nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            "PyTorch and torchvision are required for model training; install project dependencies"
        ) from exc

    if model_name != "mobilenet_v3_small":
        raise ValueError("only mobilenet_v3_small is supported in the first version")
    if num_classes <= 1:
        raise ValueError("num_classes must be at least 2")

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model

