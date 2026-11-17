from __future__ import annotations

from collections.abc import Iterable, Mapping


class CheckpointContractError(ValueError):
    """Raised when a checkpoint cannot safely be reused."""


def checkpoint_resize_mode(checkpoint: Mapping) -> str:
    """从权重恢复输入约定，也识别正式A第一版的手工补边元数据。"""
    from .augment import RESIZE_MODES

    metadata = checkpoint.get("preprocessing", {})
    mode = metadata.get("resize_mode")
    if mode is None:
        legacy_resize = metadata.get("resize")
        if legacy_resize is not None and legacy_resize != "aspect_ratio_pad":
            raise ValueError("unsupported checkpoint resize: " + str(legacy_resize))
        mode = "letterbox" if legacy_resize == "aspect_ratio_pad" else "center_crop"
    if mode not in RESIZE_MODES:
        raise ValueError("unsupported checkpoint resize_mode: " + str(mode))
    return mode


def checkpoint_class_names(
    checkpoint: Mapping,
    *,
    observed_labels: Iterable[str] = (),
) -> list[str]:
    """读取权重文件中的类别顺序，保证预测下标与训练标签一一对应。"""
    raw_names = checkpoint.get("class_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise CheckpointContractError("checkpoint class_names must be a non-empty list")
    names = list(raw_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise CheckpointContractError("checkpoint class_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise CheckpointContractError("checkpoint class_names contains duplicates")
    # 评测图片不能包含模型从未学习过的类别；增类时须另行扩展分类层。
    unknown = set(observed_labels) - set(names)
    if unknown:
        raise CheckpointContractError("unknown labels: " + ", ".join(sorted(unknown)))
    return names


def incremental_class_names(checkpoint: Mapping, new_classes: Iterable[str], *,
                            observed_labels: Iterable[str] = ()) -> list[str]:
    """保留旧类别顺序，只在末尾追加新类别，不把旧分类层下标打乱。"""
    old_names = checkpoint_class_names(checkpoint)
    added = list(new_classes)
    if (not added or any(not isinstance(name, str) or not name for name in added)
            or len(set(added)) != len(added) or set(added) & set(old_names)):
        raise CheckpointContractError("new_classes must be non-empty, unique and disjoint from old classes")
    return checkpoint_class_names({"class_names": old_names + added}, observed_labels=observed_labels)


def expand_classifier(model, *, num_classes: int):
    """扩展最后一层：旧类别的权重和偏置逐行原样复制，新行随机初始化。"""
    import torch
    from torch import nn

    previous = model.classifier[-1]
    if num_classes <= previous.out_features:
        raise ValueError("expanded num_classes must exceed the previous class count")
    expanded = nn.Linear(previous.in_features, num_classes, bias=previous.bias is not None,
                         device=previous.weight.device, dtype=previous.weight.dtype)
    with torch.no_grad():
        expanded.weight[:previous.out_features].copy_(previous.weight)
        if previous.bias is not None:
            expanded.bias[:previous.out_features].copy_(previous.bias)
    expanded.train(previous.training)
    model.classifier[-1] = expanded
    return model


def build_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool = False,
):
    """构建分类网络：预训练骨干提取特征，最后一层输出指定类别的分数。

    pretrained=True 从官方 ImageNet 权重开始；加载自己的检查点时用 False，
    随后再调用 load_state_dict，避免下载或覆盖已训练参数。
    返回的是网络结构，训练后的参数由训练程序单独保存。
    """
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

    # 迁移学习复用通用视觉特征；正式模型A不加载此前的小样本试跑权重。
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # 官方模型原本识别ImageNet类别，这里仅替换最后一层，以适配食材类别数。
    # 输出为 [批量大小, 类别数] 的原始分数；交叉熵内部处理归一化。
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    # 可选冻结features特征提取部分；classifier中的层仍可训练。
    # False表示更新整个网络。本选项仅控制梯度，不冻结BatchNorm运行统计量。
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model

