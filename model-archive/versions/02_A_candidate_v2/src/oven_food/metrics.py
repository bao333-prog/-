from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    num_classes: int,
) -> list[list[int]]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true and predicted labels must have the same length")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        if not 0 <= true_label < num_classes or not 0 <= predicted_label < num_classes:
            raise ValueError("class index is outside the configured class range")
        matrix[true_label][predicted_label] += 1
    return matrix


def classification_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    *,
    class_names: Sequence[str],
) -> dict:
    names = list(class_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("class_names must be non-empty and unique")
    matrix = confusion_matrix(true_labels, predicted_labels, num_classes=len(names))
    total = len(true_labels)
    correct = sum(matrix[index][index] for index in range(len(names)))
    per_class: dict[str, float | None] = {}
    for index, name in enumerate(names):
        support = sum(matrix[index])
        per_class[name] = matrix[index][index] / support if support else None
    return {
        "num_samples": total,
        "overall_accuracy": correct / total if total else None,
        "per_class_accuracy": per_class,
        "class_names": names,
        "confusion_matrix": matrix,
    }

