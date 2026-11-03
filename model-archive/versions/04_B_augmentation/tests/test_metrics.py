import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.metrics import classification_metrics, confusion_matrix  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_counts_true_rows_and_predicted_columns(self) -> None:
        matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)

        self.assertEqual(matrix, [[1, 1], [0, 2]])

    def test_classification_metrics_report_overall_and_per_class_accuracy(self) -> None:
        metrics = classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            class_names=["corn", "potato"],
        )

        self.assertEqual(metrics["num_samples"], 4)
        self.assertEqual(metrics["overall_accuracy"], 0.75)
        self.assertEqual(metrics["per_class_accuracy"], {"corn": 0.5, "potato": 1.0})
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])

    def test_class_without_samples_has_null_accuracy(self) -> None:
        metrics = classification_metrics([0], [0], class_names=["corn", "potato"])

        self.assertIsNone(metrics["per_class_accuracy"]["potato"])

    def test_rejects_mismatched_prediction_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            confusion_matrix([0, 1], [0], num_classes=2)

    def test_rejects_class_index_outside_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            confusion_matrix([0], [2], num_classes=2)


if __name__ == "__main__":
    unittest.main()
