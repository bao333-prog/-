from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = (
    "image_path",
    "food_label",
    "environment",
    "session_id",
    "camera_position",
    "lighting_condition",
    "steam_level",
    "occlusion_level",
)
ALLOWED_ENVIRONMENTS = {"proxy", "target"}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high"}
ALLOWED_SPLITS = {"train", "val", "test"}


class ManifestValidationError(ValueError):
    """Raised when a data manifest violates the project contract."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    image_path: str
    food_label: str
    environment: str
    session_id: str
    camera_position: str
    lighting_condition: str
    steam_level: str
    occlusion_level: str
    split: str = ""


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestValidationError("manifest has no header")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ManifestValidationError(
                "manifest is missing required columns: " + ", ".join(missing)
            )

        records: list[ManifestRecord] = []
        for row in reader:
            values = {field: (row.get(field) or "").strip() for field in REQUIRED_FIELDS}
            records.append(
                ManifestRecord(
                    **values,
                    split=(row.get("split") or "").strip(),
                )
            )
    return records


def validate_manifest(
    records: Iterable[ManifestRecord],
    *,
    data_root: str | Path = ".",
    check_files: bool = False,
) -> list[ManifestRecord]:
    materialized = list(records)
    errors: list[str] = []
    session_labels: dict[str, set[str]] = {}
    session_splits: dict[str, set[str]] = {}
    root = Path(data_root)

    if not materialized:
        raise ManifestValidationError("manifest contains no data rows")

    for index, record in enumerate(materialized, start=2):
        for field in REQUIRED_FIELDS:
            if not getattr(record, field):
                errors.append(f"row {index}: {field} is empty")

        if record.environment and record.environment not in ALLOWED_ENVIRONMENTS:
            errors.append(
                f"row {index}: environment must be proxy or target, got {record.environment!r}"
            )
        for field in ("steam_level", "occlusion_level"):
            value = getattr(record, field)
            if value and value not in ALLOWED_SEVERITIES:
                allowed = ", ".join(sorted(ALLOWED_SEVERITIES))
                errors.append(f"row {index}: {field} must be one of {allowed}, got {value!r}")
        if record.split and record.split not in ALLOWED_SPLITS:
            errors.append(
                f"row {index}: split must be train, val, or test, got {record.split!r}"
            )

        session_labels.setdefault(record.session_id, set()).add(record.food_label)
        if record.split:
            session_splits.setdefault(record.session_id, set()).add(record.split)

        if check_files and record.image_path:
            image_path = Path(record.image_path)
            resolved = image_path if image_path.is_absolute() else root / image_path
            if not resolved.is_file():
                errors.append(f"row {index}: image does not exist: {resolved}")

    for session_id, labels in sorted(session_labels.items()):
        if len(labels) > 1:
            errors.append(
                f"session {session_id!r} contains multiple food labels: {', '.join(sorted(labels))}"
            )
    for session_id, splits in sorted(session_splits.items()):
        if len(splits) > 1:
            errors.append(
                f"session {session_id!r} appears in multiple splits: {', '.join(sorted(splits))}"
            )

    if errors:
        raise ManifestValidationError("manifest validation failed:\n- " + "\n- ".join(errors))
    return materialized


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    materialized = list(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_split = any(record.split for record in materialized)
    fieldnames = list(REQUIRED_FIELDS) + (["split"] if include_split else [])
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in materialized:
            values = asdict(record)
            if not include_split:
                values.pop("split", None)
            writer.writerow(values)
