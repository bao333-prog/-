from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable


COMPARISON_FIELDS = (
    "experiment_name",
    "split",
    "environments",
    "num_samples",
    "overall_accuracy",
    "mean_end_to_end_latency_ms_per_image",
    "mean_inference_latency_ms_per_image",
    "checkpoint_size_bytes",
    "parameter_count",
    "device",
    "metrics_path",
)
REQUIRED_METRIC_FIELDS = (
    "experiment_name",
    "num_samples",
    "overall_accuracy",
    "split",
    "environments",
)


def build_comparison(metrics_paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for raw_path in metrics_paths:
        path = Path(raw_path)
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read metrics file {path}: {exc}") from exc
        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in metrics]
        if missing:
            raise ValueError(f"metrics file {path} is missing: {', '.join(missing)}")
        environments = metrics["environments"]
        if not isinstance(environments, list):
            raise ValueError(f"metrics file {path} has non-list environments")
        row = {
            "experiment_name": metrics["experiment_name"],
            "split": metrics["split"],
            "environments": ",".join(environments),
            "num_samples": metrics["num_samples"],
            "overall_accuracy": metrics["overall_accuracy"],
            "mean_end_to_end_latency_ms_per_image": metrics.get(
                "mean_end_to_end_latency_ms_per_image"
            ),
            "mean_inference_latency_ms_per_image": metrics.get(
                "mean_inference_latency_ms_per_image"
            ),
            "checkpoint_size_bytes": metrics.get("checkpoint_size_bytes"),
            "parameter_count": metrics.get("parameter_count"),
            "device": metrics.get("device"),
            "metrics_path": str(path),
        }
        rows.append(row)
    if not rows:
        raise ValueError("at least one metrics file is required")
    return rows


def write_comparison(path: str | Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("comparison needs at least one row")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def select_best_proxy_checkpoint(
    summary_paths: Iterable[str | Path], output_checkpoint: str | Path
) -> dict:
    candidates: list[dict] = []
    for raw_path in summary_paths:
        path = Path(raw_path)
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read training summary {path}: {exc}") from exc
        missing = [
            field
            for field in ("experiment_name", "best_validation_accuracy", "checkpoint")
            if field not in summary
        ]
        if missing:
            raise ValueError(f"training summary {path} is missing: {', '.join(missing)}")
        accuracy = summary["best_validation_accuracy"]
        if not isinstance(accuracy, (int, float)):
            raise ValueError(f"training summary {path} has non-numeric validation accuracy")
        checkpoint = Path(summary["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint from training summary does not exist: {checkpoint}")
        candidates.append(
            {
                "selected_experiment": summary["experiment_name"],
                "best_validation_accuracy": accuracy,
                "source_checkpoint": str(checkpoint.resolve()),
                "source_summary": str(path.resolve()),
            }
        )
    if not candidates:
        raise ValueError("at least one training summary is required")

    selected = max(candidates, key=lambda candidate: candidate["best_validation_accuracy"])
    output = Path(output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["source_checkpoint"], output)
    selection = {**selected, "selected_checkpoint": str(output.resolve())}
    (output.parent / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection
