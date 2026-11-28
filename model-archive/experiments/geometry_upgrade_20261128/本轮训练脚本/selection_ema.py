"""本轮训练专用的验证选模与 EMA 工具；本模块不读取任何数据文件。"""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from typing import Any

from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def _counts(record: Mapping[str, Any], label: str) -> tuple[int, int]:
    """校验计数，拒绝布尔值、浮点数以及 correct 超过 total 的损坏统计。"""
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} 必须包含 correct、total 整数计数")
    values = []
    for name in ("correct", "total"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{label}.{name} 必须是非负整数")
        values.append(int(value))
    correct, total = values
    if correct > total:
        raise ValueError(f"{label}.correct 不能超过 total")
    return correct, total


def selection_key(
    stats: Mapping[str, Any], loss: float
) -> tuple[float, float, float, float]:
    """返回可直接用 ``>`` 比较的四项选模键，越大越好。

    输入采用 validation_monitor 的 ``correct/total/per_class`` 结构；
    per_class 的键为类别名称，值仍为 ``correct/total``。计数必须为
    非负整数，且逐类之和必须等于全局计数。loss 必须是有限非负实数。
    统计缺失、计数损坏或验证集没有样本时抛出 ValueError。

    排序顺序为：有样本类别的平均正确率、这些类别中的最差正确率、
    总体正确率、负 loss。准确率均为 [0, 1] 比例；total=0 的类别
    不参与平均和最小值，绝不按 100% 处理。忽略缓存的派生统计字段。
    """
    if (
        isinstance(loss, bool)
        or not isinstance(loss, Real)
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise ValueError("loss 必须是有限非负实数")
    correct, total = _counts(stats, "stats")
    if total == 0:
        raise ValueError("验证集没有样本")
    per_class = stats.get("per_class")
    if not isinstance(per_class, Mapping) or not per_class:
        raise ValueError("per_class 必须包含逐类计数")

    rates = []
    class_correct = 0
    class_total = 0
    for name, record in per_class.items():
        item_correct, item_total = _counts(record, f"per_class[{name!r}]")
        class_correct += item_correct
        class_total += item_total
        if item_total > 0:
            rates.append(item_correct / item_total)
    if class_correct != correct or class_total != total:
        raise ValueError("逐类计数之和与全局 correct/total 不一致")
    if not rates:
        raise ValueError("没有任何有验证样本的类别")
    return (
        float(math.fsum(rates) / len(rates)),
        float(min(rates)),
        float(correct / total),
        -float(loss),
    )


def create_ema(model: nn.Module, decay: float = 0.99) -> AveragedModel:
    """创建独立模型副本，绝不修改传入的 live 模型。

    decay 必须是 [0, 1] 内有限实数。创建时 n_averaged=0；第一次
    update_ema 原样复制当时的 live 参数，之后按
    ``ema = decay * ema + (1 - decay) * live`` 更新。

    use_buffers=False：每次更新直接复制最新 live buffers，包括 BN 的
    running_mean/running_var/num_batches_tracked；这些值不参与 EMA。
    最终 BN 重估由训练入口用仅含训练集的 loader 显式完成。
    """
    if (
        isinstance(decay, bool)
        or not isinstance(decay, Real)
        or not math.isfinite(decay)
        or not 0.0 <= decay <= 1.0
    ):
        raise ValueError("EMA decay 必须是 [0, 1] 内的有限实数")
    return AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(float(decay)),
        use_buffers=False,
    )


def update_ema(ema: AveragedModel, model: nn.Module) -> None:
    """在 optimizer.step 成功之后调用；只更新 EMA 副本，不反向传播。

    使用混合精度时，调用者须避免在跳过的 optimizer step 后更新 EMA。
    本函数不执行优化器更新，也不读取数据或重估 BN。
    """
    ema.update_parameters(model)


def ema_model_state(ema: AveragedModel) -> Mapping[str, Any]:
    """导出原模型的干净 state_dict，可 strict=True 加载到同结构模型。

    只返回 ema.module 的参数与 buffers，不含 AveragedModel 的
    n_averaged 或额外 module. 前缀。与标准 state_dict 一样，张量是
    当前状态的引用；需要内存中的冻结快照时由调用者 deepcopy。
    类别到索引的映射由 checkpoint 保存入口单独记录。
    """
    return ema.module.state_dict()
