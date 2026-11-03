from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .manifest import ManifestRecord, ManifestValidationError, validate_manifest


SPLIT_NAMES = ("train", "val", "test")


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    nonzero = sum(ratio > 0 for ratio in ratios)
    if total < nonzero:
        raise ManifestValidationError(
            f"each food label needs at least {nonzero} sessions for the requested nonempty splits"
        )

    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda item: raw[item] - counts[item], reverse=True):
        if sum(counts) == total:
            break
        counts[index] += 1

    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            donor = max(
                (candidate for candidate in range(3) if counts[candidate] > 1),
                key=lambda candidate: counts[candidate],
            )
            counts[donor] -= 1
            counts[index] += 1
    return counts


def split_by_session(
    records: Iterable[ManifestRecord],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ManifestRecord]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must be nonnegative and sum to 1")

    materialized = validate_manifest(records)
    sessions_by_label: dict[str, set[str]] = defaultdict(set)
    for record in materialized:
        sessions_by_label[record.food_label].add(record.session_id)

    rng = random.Random(seed)
    session_to_split: dict[str, str] = {}
    for label in sorted(sessions_by_label):
        sessions = sorted(sessions_by_label[label])
        rng.shuffle(sessions)
        counts = _allocate_counts(len(sessions), ratios)
        offset = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            for session in sessions[offset : offset + count]:
                session_to_split[session] = split_name
            offset += count

    return [replace(record, split=session_to_split[record.session_id]) for record in materialized]

