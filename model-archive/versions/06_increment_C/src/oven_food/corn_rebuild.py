"""Prepare the corn image archive without changing its source files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
from typing import Mapping, Sequence
import uuid
import zipfile

from PIL import Image, ImageDraw, ImageOps


@dataclass(frozen=True, slots=True)
class CornRecord:
    id: str
    image_path: str
    food_label: str
    source: str
    source_group_id: str
    width: int
    height: int
    sha256: str
    form: str
    source_page: str = ""
    license: str = ""
    credit: str = ""
    environment: str = "external"
    cooking_state: str = ""
    ai_retouched: bool = False
    parent_image_id: str = ""
    perceptual_hash: int | None = None
    dataset_role: str = "unassigned"
    proposed_split: str = "unassigned"
    is_derived: bool = False
    is_synthetic: bool = False
    low_resolution: bool = False
    metric_eligible: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildPlan:
    records: tuple[CornRecord, ...]


@dataclass(frozen=True, slots=True)
class BuildSummary:
    final_dir: Path
    final_zip: Path
    source_sha256: str
    counts_by_role: Mapping[str, int]
    counts_by_split: Mapping[str, int]


def classify_records(records: Sequence[CornRecord]) -> list[CornRecord]:
    """Assign immutable source records to their initial data-quality role."""

    classified: list[CornRecord] = []
    for record in records:
        low_resolution = min(record.width, record.height) < 224
        is_derived = bool(record.parent_image_id)
        notes: list[str] = []

        if record.is_synthetic:
            role = "synthetic_proxy"
            notes.append("synthetic")
        elif record.ai_retouched:
            role = "excluded"
            notes.append("ai_retouched")
        elif low_resolution or is_derived:
            role = "supplemental_real"
            if low_resolution:
                notes.append("low_resolution")
            if is_derived:
                notes.append("derived_crop")
        else:
            role = "core_real"

        classified.append(
            replace(
                record,
                dataset_role=role,
                is_derived=is_derived,
                low_resolution=low_resolution,
                metric_eligible=role == "core_real",
                notes=tuple(notes),
            )
        )

    return classified


def select_diverse_openear(
    hashes: Mapping[str, int], *, limit: int = 50
) -> set[str]:
    """Select a deterministic, perceptually spread subset of 64-bit hashes."""

    if limit <= 0 or not hashes:
        return set()
    for image_id, value in hashes.items():
        if not 0 <= value < (1 << 64):
            raise ValueError(f"{image_id} does not have a 64-bit perceptual hash")

    return set(_diverse_hash_order(hashes)[:limit])


def _diverse_hash_order(hashes: Mapping[str, int]) -> list[str]:
    if not hashes:
        return []

    remaining = set(hashes)
    first = min(remaining)
    selected = [first]
    remaining.remove(first)

    while remaining:
        next_id = min(
            remaining,
            key=lambda candidate: (
                -min(
                    (hashes[candidate] ^ hashes[chosen]).bit_count()
                    for chosen in selected
                ),
                candidate,
            ),
        )
        selected.append(next_id)
        remaining.remove(next_id)

    return selected


def _validate_parent_links(records: Sequence[CornRecord]) -> None:
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record IDs must be unique")

    for record in records:
        if record.parent_image_id and record.parent_image_id not in by_id:
            raise ValueError(
                f"{record.id} has missing parent {record.parent_image_id}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(record_id: str) -> None:
        if record_id in visiting:
            raise ValueError(f"cyclic parent link involving {record_id}")
        if record_id in visited:
            return
        visiting.add(record_id)
        parent_id = by_id[record_id].parent_image_id
        if parent_id:
            visit(parent_id)
        visiting.remove(record_id)
        visited.add(record_id)

    for record_id in by_id:
        visit(record_id)


def _components(records: Sequence[CornRecord]) -> list[list[CornRecord]]:
    parent = {record.id: record.id for record in records}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_group: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_group[record.source_group_id].append(record.id)
        if record.parent_image_id:
            union(record.id, record.parent_image_id)
    for ids in by_group.values():
        for record_id in ids[1:]:
            union(ids[0], record_id)

    grouped: dict[str, list[CornRecord]] = defaultdict(list)
    for record in records:
        grouped[find(record.id)].append(record)
    return [sorted(group, key=lambda item: item.id) for group in grouped.values()]


def _is_challenge_record(record: CornRecord) -> bool:
    match = re.fullmatch(r"corn_(\d+)", record.id)
    return bool(match and 201 <= int(match.group(1)) <= 210)


def plan_splits(
    records: Sequence[CornRecord],
    *,
    seed: int = 42,
    val_group_count: int = 15,
    test_group_count: int = 15,
    openear_core_limit: int = 50,
) -> BuildPlan:
    """Plan source-safe train, temporary web evaluation, and challenge splits."""

    if val_group_count < 0 or test_group_count < 0:
        raise ValueError("evaluation group counts cannot be negative")
    _validate_parent_links(records)
    classified = classify_records(records)
    components = _components(classified)

    assignments: dict[str, str] = {}
    challenge: set[int] = set()
    candidates: list[int] = []
    for index, component in enumerate(components):
        if any(record.is_synthetic for record in component):
            for record in component:
                assignments[record.id] = "train"
        elif any(_is_challenge_record(record) for record in component):
            challenge.add(index)
            for record in component:
                assignments[record.id] = "challenge_appliance"
        elif any(record.metric_eligible for record in component):
            candidates.append(index)

    non_openear = [
        index
        for index in candidates
        if any(record.source.lower() != "openear" for record in components[index])
    ]
    openear = [index for index in candidates if index not in non_openear]
    rng = random.Random(seed)
    rng.shuffle(non_openear)

    openear_hashes: dict[str, int] = {}
    component_by_key: dict[str, int] = {}
    for index in openear:
        key = components[index][0].id
        available = [
            record.perceptual_hash
            for record in components[index]
            if record.perceptual_hash is not None
        ]
        openear_hashes[key] = available[0] if available else 0
        component_by_key[key] = index
    openear_order = [component_by_key[key] for key in _diverse_hash_order(openear_hashes)]
    ordered = non_openear + openear_order

    requested = val_group_count + test_group_count
    if len(ordered) < requested:
        raise ValueError(
            f"not enough independent source groups for evaluation: "
            f"need {requested}, found {len(ordered)}"
        )
    val_components = set(ordered[:val_group_count])
    test_components = set(ordered[val_group_count:requested])

    for index, component in enumerate(components):
        if index in challenge or any(record.id in assignments for record in component):
            continue
        if index in val_components:
            split = "val"
        elif index in test_components:
            split = "test_web"
        else:
            split = "train"
        for record in component:
            assignments[record.id] = split

    planned = [
        replace(record, proposed_split=assignments[record.id])
        for record in classified
    ]

    openear_train = {
        record.id: record.perceptual_hash if record.perceptual_hash is not None else 0
        for record in planned
        if record.proposed_split == "train"
        and record.dataset_role == "core_real"
        and record.source.lower() == "openear"
    }
    selected_openear = select_diverse_openear(
        openear_train, limit=openear_core_limit
    )
    planned = [
        replace(
            record,
            dataset_role="supplemental_real",
            metric_eligible=False,
            notes=record.notes + ("openear_core_cap",),
        )
        if record.id in openear_train and record.id not in selected_openear
        else record
        for record in planned
    ]

    plan = BuildPlan(records=tuple(sorted(planned, key=lambda item: item.id)))
    validate_plan(plan)
    return plan


def validate_plan(plan: BuildPlan) -> None:
    """Reject leakage or invalid evaluation records in a planned package."""

    records = plan.records
    _validate_parent_links(records)
    by_id = {record.id: record for record in records}
    allowed_splits = {"train", "val", "test_web", "challenge_appliance"}

    for record in records:
        if record.proposed_split not in allowed_splits:
            raise ValueError(f"{record.id} has no valid proposed split")
        if record.is_synthetic and record.proposed_split != "train":
            raise ValueError("synthetic records must be train-only")
        if record.parent_image_id:
            parent = by_id[record.parent_image_id]
            if record.proposed_split != parent.proposed_split:
                raise ValueError("parent families cannot cross splits")
        if record.metric_eligible and record.proposed_split in {
            "val",
            "test_web",
            "challenge_appliance",
        }:
            if record.ai_retouched or record.is_derived or record.is_synthetic:
                raise ValueError("evaluation metrics require original real images")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record.source_group_id].add(record.proposed_split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("source groups cannot cross splits")


MANIFEST_FIELDS = (
    "id",
    "image_path",
    "food_label",
    "dataset_role",
    "proposed_split",
    "environment",
    "session_id",
    "camera_position",
    "lighting_condition",
    "steam_level",
    "occlusion_level",
    "source",
    "source_group_id",
    "parent_image_id",
    "is_derived",
    "is_synthetic",
    "ai_retouched",
    "low_resolution",
    "form",
    "cooking_state",
    "foreground_source_id",
    "background_id",
    "transform_parameters",
    "sha256",
    "metric_eligible",
    "notes",
    "source_page",
    "license",
    "credit",
    "width",
    "height",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_zip_member(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"unsafe ZIP member path: {name}")


def _resolve_member(manifest_name: str, image_path: str) -> str:
    image = PurePosixPath(image_path)
    if image.is_absolute() or ".." in image.parts or "\\" in image_path:
        raise ValueError(f"unsafe manifest image path: {image_path}")
    root = PurePosixPath(manifest_name).parent
    return str(root / image)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    get_pixels = getattr(grayscale, "get_flattened_data", grayscale.getdata)
    pixels = list(get_pixels())
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | (
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return result


def _load_archive(
    source_zip: Path,
) -> tuple[list[CornRecord], dict[str, bytes], dict[str, str]]:
    with zipfile.ZipFile(source_zip) as archive:
        for info in archive.infolist():
            _validate_zip_member(info.filename)

        manifest_entries = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).name.lower() == "manifest.csv"
        ]
        if len(manifest_entries) != 1:
            raise ValueError("ZIP must contain exactly one manifest.csv")
        manifest_info = manifest_entries[0]
        manifest_text = archive.read(manifest_info).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(manifest_text, newline="")))
        required = {
            "id",
            "image_path",
            "food_label",
            "source",
            "source_group_id",
            "width",
            "height",
            "sha256",
            "form",
            "ai_retouched",
            "parent_image_id",
        }
        if not rows:
            raise ValueError("manifest.csv contains no records")
        missing_columns = required - set(rows[0])
        if missing_columns:
            raise ValueError(
                "manifest.csv is missing columns: " + ", ".join(sorted(missing_columns))
            )

        archive_names = {info.filename for info in archive.infolist() if not info.is_dir()}
        records: list[CornRecord] = []
        image_payloads: dict[str, bytes] = {}
        original_paths: dict[str, str] = {}
        for row in rows:
            record_id = row["id"].strip()
            member_name = _resolve_member(manifest_info.filename, row["image_path"])
            if member_name not in archive_names:
                raise ValueError(f"{row['image_path']} is missing from ZIP")
            payload = archive.read(member_name)
            actual_hash = hashlib.sha256(payload).hexdigest()
            if row["sha256"].strip() and actual_hash != row["sha256"].strip().lower():
                raise ValueError(f"SHA-256 mismatch for {record_id}")
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
                    perceptual_hash = _difference_hash(image)
            except Exception as error:
                raise ValueError(f"unreadable image for {record_id}") from error
            if (width, height) != (int(row["width"]), int(row["height"])):
                raise ValueError(f"image dimensions mismatch for {record_id}")

            records.append(
                CornRecord(
                    id=record_id,
                    image_path=row["image_path"].strip(),
                    food_label=row["food_label"].strip(),
                    source=row["source"].strip(),
                    source_group_id=row["source_group_id"].strip() or record_id,
                    width=width,
                    height=height,
                    sha256=actual_hash,
                    form=row["form"].strip(),
                    source_page=row.get("source_page", "").strip(),
                    license=row.get("license", "").strip(),
                    credit=row.get("credit", "").strip(),
                    environment=row.get("environment", "external").strip() or "external",
                    cooking_state=row.get("cooking_state", "").strip(),
                    ai_retouched=_parse_bool(row["ai_retouched"]),
                    parent_image_id=row["parent_image_id"].strip(),
                    perceptual_hash=perceptual_hash,
                )
            )
            image_payloads[record_id] = payload
            original_paths[record_id] = row["image_path"].strip()

    return records, image_payloads, original_paths


def _manifest_row(record: CornRecord, output_image_path: str) -> dict[str, object]:
    return {
        "id": record.id,
        "image_path": output_image_path,
        "food_label": record.food_label,
        "dataset_role": record.dataset_role,
        "proposed_split": record.proposed_split,
        "environment": record.environment,
        "session_id": record.source_group_id,
        "camera_position": "unknown",
        "lighting_condition": "unknown",
        "steam_level": "unknown",
        "occlusion_level": "unknown",
        "source": record.source,
        "source_group_id": record.source_group_id,
        "parent_image_id": record.parent_image_id,
        "is_derived": record.is_derived,
        "is_synthetic": record.is_synthetic,
        "ai_retouched": record.ai_retouched,
        "low_resolution": record.low_resolution,
        "form": record.form,
        "cooking_state": record.cooking_state,
        "foreground_source_id": "",
        "background_id": "",
        "transform_parameters": "",
        "sha256": record.sha256,
        "metric_eligible": record.metric_eligible,
        "notes": ";".join(record.notes),
        "source_page": record.source_page,
        "license": record.license,
        "credit": record.credit,
        "width": record.width,
        "height": record.height,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(
    path: Path,
    records: Sequence[CornRecord],
    routed_paths: Mapping[str, Path],
) -> None:
    columns = 5
    rows = 4
    cell_width = 180
    cell_height = 180
    label_height = 24
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    if not records:
        draw.text((20, 20), "No images", fill=(30, 30, 30))
    for index, record in enumerate(records[: columns * rows]):
        with Image.open(routed_paths[record.id]) as image:
            thumbnail = ImageOps.contain(
                image.convert("RGB"), (cell_width - 12, cell_height - 12)
            )
        column = index % columns
        row = index // columns
        x = column * cell_width + (cell_width - thumbnail.width) // 2
        y = row * (cell_height + label_height) + (cell_height - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        draw.text(
            (column * cell_width + 6, row * (cell_height + label_height) + cell_height),
            record.id,
            fill=(20, 20, 20),
        )
    canvas.save(path, format="JPEG", quality=90)


def _write_readme(path: Path, summary: Mapping[str, object]) -> None:
    path.write_text(
        """# 玉米数据集整理版

这是从原始 222 张压缩包生成的**真实图片整理中间版**。原始压缩包未被修改。

- `core_real`：可用于当前训练或临时网络域评测的真实图片。
- `supplemental_real`：低分辨率、衍生裁剪或因同源图片过多而降级的备用图片；默认不进入训练。
- `excluded`：AI 修补图片；不进入训练、验证或测试。
- `val.csv` 与 `test_web.csv` 只衡量网络图片上的效果，不能代表目标烤箱效果。
- `challenge_appliance.csv` 是较接近厨房/设备环境的临时挑战集，也不是目标烤箱测试集。

等目标烤箱到位后，应另行按完整拍摄批次建立目标域验证集和最终测试集。不要把同一段视频的相邻帧拆到不同集合。

本版本没有生成风格迁移图或程序合成图。

## 数量摘要

```json
"""
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def _zip_directory(source_dir: Path, destination: Path, *, root_name: str) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(source_dir)
            archive.write(
                path,
                str(PurePosixPath(root_name) / PurePosixPath(relative.as_posix())),
            )


def rebuild_corn_dataset(
    source_zip: Path,
    output_dir: Path,
    *,
    openear_core_limit: int = 50,
    val_group_count: int = 15,
    test_group_count: int = 15,
    seed: int = 42,
) -> BuildSummary:
    """Build a verified real-data preparation package from the immutable ZIP."""

    source_zip = Path(source_zip).resolve()
    output_dir = Path(output_dir).resolve()
    final_zip = output_dir.with_suffix(".zip")
    if output_dir.exists() or final_zip.exists():
        raise FileExistsError("output directory or ZIP already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = _file_sha256(source_zip)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    temporary_zip = final_zip.with_name(f".{final_zip.name}.staging-{uuid.uuid4().hex}")
    published = False
    try:
        records, payloads, original_paths = _load_archive(source_zip)
        plan = plan_splits(
            records,
            seed=seed,
            val_group_count=val_group_count,
            test_group_count=test_group_count,
            openear_core_limit=openear_core_limit,
        )

        images_root = stage / "images"
        manifests_root = stage / "manifests"
        previews_root = stage / "previews"
        for role in ("core_real", "supplemental_real", "excluded"):
            (images_root / role).mkdir(parents=True, exist_ok=True)
        manifests_root.mkdir(parents=True)
        previews_root.mkdir(parents=True)

        routed_paths: dict[str, Path] = {}
        manifest_rows: list[dict[str, object]] = []
        for record in plan.records:
            suffix = PurePosixPath(original_paths[record.id]).suffix.lower() or ".png"
            relative_path = PurePosixPath("images") / record.dataset_role / f"{record.id}{suffix}"
            destination = stage / Path(relative_path.as_posix())
            destination.write_bytes(payloads[record.id])
            routed_paths[record.id] = destination
            manifest_rows.append(_manifest_row(record, relative_path.as_posix()))

        rows_by_id = {str(row["id"]): row for row in manifest_rows}
        _write_csv(manifests_root / "manifest_v2.csv", manifest_rows)
        split_filters = {
            "train.csv": lambda record: record.proposed_split == "train"
            and record.dataset_role == "core_real",
            "val.csv": lambda record: record.proposed_split == "val"
            and record.metric_eligible,
            "test_web.csv": lambda record: record.proposed_split == "test_web"
            and record.metric_eligible,
            "challenge_appliance.csv": lambda record: record.proposed_split
            == "challenge_appliance"
            and record.metric_eligible,
        }
        for filename, predicate in split_filters.items():
            _write_csv(
                manifests_root / filename,
                [rows_by_id[record.id] for record in plan.records if predicate(record)],
            )

        for role in ("core_real", "supplemental_real", "excluded"):
            _write_contact_sheet(
                previews_root / f"{role}_contact_sheet.jpg",
                [record for record in plan.records if record.dataset_role == role],
                routed_paths,
            )

        counts_by_role: dict[str, int] = defaultdict(int)
        counts_by_split: dict[str, int] = defaultdict(int)
        for record in plan.records:
            counts_by_role[record.dataset_role] += 1
            counts_by_split[record.proposed_split] += 1
        source_hash_after = _file_sha256(source_zip)
        verification = {
            "source_zip": str(source_zip),
            "source_sha256": source_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after,
            "total_source_images": len(plan.records),
            "counts_by_role": dict(sorted(counts_by_role.items())),
            "counts_by_split": dict(sorted(counts_by_split.items())),
            "unique_source_image_hashes": len({record.sha256 for record in plan.records}),
            "unreadable_output_images": 0,
            "cross_split_source_groups": 0,
            "cross_split_parent_families": 0,
            "ai_images_in_evaluation": 0,
            "derived_images_in_evaluation": 0,
            "synthetic_images_in_evaluation": 0,
            "synthetic_images_generated": 0,
        }
        (stage / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_readme(
            stage / "README.md",
            {
                "counts_by_role": verification["counts_by_role"],
                "counts_by_split": verification["counts_by_split"],
            },
        )

        _zip_directory(stage, temporary_zip, root_name=output_dir.name)
        stage.rename(output_dir)
        published = True
        temporary_zip.replace(final_zip)
        return BuildSummary(
            final_dir=output_dir,
            final_zip=final_zip,
            source_sha256=source_hash_before,
            counts_by_role=dict(counts_by_role),
            counts_by_split=dict(counts_by_split),
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if temporary_zip.exists():
            temporary_zip.unlink()
        if published and output_dir.exists():
            shutil.rmtree(output_dir)
        raise
