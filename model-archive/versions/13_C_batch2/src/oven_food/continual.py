"""仅训练时使用的旧知识约束；不为学生推理网络增加层或参数。"""
from __future__ import annotations

import math


def old_class_distillation_loss(student_logits, teacher_logits, labels, *, temperature: float):
    """只蒸馏教师判断正确的旧样本；返回损失和有效样本数。

    教师只有旧类输出，学生有旧类+新类。先对学生全部输出做log_softmax，再取旧类项，
    相当于将教师的新类概率补零；若先截取学生旧类再归一化，就无法约束新类抢占旧图概率。
    """
    import torch
    from torch.nn import functional as F

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if (student_logits.ndim != 2 or teacher_logits.ndim != 2
            or student_logits.shape[0] != teacher_logits.shape[0]
            or labels.ndim != 1 or len(labels) != len(student_logits)
            or not 0 < teacher_logits.shape[1] <= student_logits.shape[1]):
        raise ValueError('distillation logits/labels have incompatible shapes')
    old_count = teacher_logits.shape[1]
    detached_teacher = teacher_logits.detach()
    if not torch.isfinite(detached_teacher).all():
        raise FloatingPointError('non-finite teacher logits')
    keep = (labels >= 0) & (labels < old_count) & (detached_teacher.argmax(1) == labels)
    count = int(keep.sum())
    if count == 0:
        return student_logits.sum() * 0., 0
    targets = F.softmax(detached_teacher[keep] / temperature, dim=1)
    log_probabilities = F.log_softmax(student_logits[keep] / temperature, dim=1)[:, :old_count]
    loss = F.kl_div(log_probabilities, targets, reduction='batchmean') * temperature ** 2
    return loss, count


def optimizer_parameter_groups(model, *, learning_rate: float, backbone_lr_scale: float):
    """骨干慢更新、分类头按配置学习率更新；不包含冻结教师或冻结参数。"""
    if not math.isfinite(backbone_lr_scale) or not 0 < backbone_lr_scale <= 1:
        raise ValueError('backbone_lr_scale must be finite and in (0, 1]')
    groups = []
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    if backbone:
        groups.append(dict(params=backbone, lr=learning_rate * backbone_lr_scale))
    if head:
        groups.append(dict(params=head, lr=learning_rate))
    return groups
