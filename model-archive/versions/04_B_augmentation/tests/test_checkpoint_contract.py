import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.model import CheckpointContractError, checkpoint_class_names  # noqa: E402


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_order_is_returned_unchanged_for_known_labels(self) -> None:
        checkpoint = {
            "class_names": ["potato", "corn"],
            "model_name": "mobilenet_v3_small",
            "image_size": 224,
        }

        names = checkpoint_class_names(checkpoint, observed_labels={"corn"})

        self.assertEqual(names, ["potato", "corn"])

    def test_unknown_target_label_is_rejected(self) -> None:
        checkpoint = {"class_names": ["potato", "corn"]}

        with self.assertRaisesRegex(CheckpointContractError, "unknown labels: tomato"):
            checkpoint_class_names(checkpoint, observed_labels={"tomato"})

    def test_missing_or_duplicate_checkpoint_classes_are_rejected(self) -> None:
        with self.assertRaisesRegex(CheckpointContractError, "non-empty"):
            checkpoint_class_names({})
        with self.assertRaisesRegex(CheckpointContractError, "duplicates"):
            checkpoint_class_names({"class_names": ["corn", "corn"]})


if __name__ == "__main__":
    unittest.main()
