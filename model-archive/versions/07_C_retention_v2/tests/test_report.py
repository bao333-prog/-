import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.report import (  # noqa: E402
    build_comparison,
    select_best_proxy_checkpoint,
    write_comparison,
)


class ReportTests(unittest.TestCase):
    def test_builds_comparison_rows_from_metrics_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(
                    {
                        "experiment_name": "baseline",
                        "num_samples": 20,
                        "overall_accuracy": 0.7,
                        "split": "test",
                        "environments": ["proxy"],
                        "mean_end_to_end_latency_ms_per_image": 4.2,
                        "mean_inference_latency_ms_per_image": 2.1,
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "experiment_name": "targeted",
                        "num_samples": 20,
                        "overall_accuracy": 0.8,
                        "split": "test",
                        "environments": ["proxy"],
                        "mean_end_to_end_latency_ms_per_image": 4.4,
                        "mean_inference_latency_ms_per_image": 2.0,
                    }
                ),
                encoding="utf-8",
            )

            rows = build_comparison([first, second])

            self.assertEqual([item["experiment_name"] for item in rows], ["baseline", "targeted"])
            self.assertEqual(rows[1]["overall_accuracy"], 0.8)
            self.assertEqual(rows[0]["environments"], "proxy")
            self.assertEqual(rows[0]["mean_end_to_end_latency_ms_per_image"], 4.2)
            self.assertEqual(rows[1]["mean_inference_latency_ms_per_image"], 2.0)

    def test_writes_comparison_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            rows = [
                {
                    "experiment_name": "baseline",
                    "split": "test",
                    "environments": "proxy",
                    "num_samples": 10,
                    "overall_accuracy": 0.9,
                    "mean_end_to_end_latency_ms_per_image": 5.0,
                    "mean_inference_latency_ms_per_image": 2.5,
                    "checkpoint_size_bytes": 123,
                    "parameter_count": 456,
                    "device": "cpu",
                    "metrics_path": "metrics.json",
                }
            ]

            write_comparison(path, rows)

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["experiment_name"], "baseline")
            self.assertEqual(saved[0]["overall_accuracy"], "0.9")

    def test_rejects_metrics_without_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"experiment_name": "broken"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overall_accuracy"):
                build_comparison([path])

    def test_selects_proxy_checkpoint_by_validation_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_checkpoint = root / "first.pt"
            second_checkpoint = root / "second.pt"
            first_checkpoint.write_bytes(b"first")
            second_checkpoint.write_bytes(b"second")
            summaries = []
            for name, accuracy, checkpoint in (
                ("baseline", 0.70, first_checkpoint),
                ("targeted", 0.82, second_checkpoint),
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "experiment_name": name,
                            "best_validation_accuracy": accuracy,
                            "checkpoint": str(checkpoint),
                        }
                    ),
                    encoding="utf-8",
                )
                summaries.append(path)
            output = root / "selected" / "best.pt"

            selection = select_best_proxy_checkpoint(summaries, output)

            self.assertEqual(output.read_bytes(), b"second")
            self.assertEqual(selection["selected_experiment"], "targeted")
            self.assertEqual(selection["best_validation_accuracy"], 0.82)
            self.assertTrue((output.parent / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
