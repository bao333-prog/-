import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.dataset import (  # noqa: E402
    FoodImageDataset,
    build_class_to_index,
    records_for_usage,
)
from oven_food.manifest import ManifestRecord  # noqa: E402


def record(label: str, environment: str, split: str, session: str) -> ManifestRecord:
    return ManifestRecord(
        image_path=f"images/{session}.jpg",
        food_label=label,
        environment=environment,
        session_id=session,
        camera_position="top-center",
        lighting_condition="warm",
        steam_level="none",
        occlusion_level="none",
        split=split,
    )


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record("corn", "proxy", "train", "proxy-corn-train"),
            record("potato", "proxy", "train", "proxy-potato-train"),
            record("corn", "target", "train", "target-corn-train"),
            record("corn", "target", "test", "target-corn-test"),
        ]

    def test_training_selection_excludes_target_test_records(self) -> None:
        selected = records_for_usage(
            self.records,
            split="train",
            environments={"proxy", "target"},
            training=True,
        )

        self.assertEqual(len(selected), 3)
        self.assertNotIn("target-corn-test", {item.session_id for item in selected})

    def test_training_selection_rejects_non_train_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "training data must use split='train'"):
            records_for_usage(
                self.records,
                split="test",
                environments={"target"},
                training=True,
            )

    def test_environment_filter_is_exact(self) -> None:
        selected = records_for_usage(
            self.records,
            split="train",
            environments={"proxy"},
            training=True,
        )

        self.assertEqual({item.environment for item in selected}, {"proxy"})
        self.assertEqual(len(selected), 2)

    def test_checkpoint_class_order_is_preserved(self) -> None:
        mapping = build_class_to_index(
            [record("corn", "target", "train", "target-corn")],
            class_names=["potato", "corn"],
        )

        self.assertEqual(mapping, {"potato": 0, "corn": 1})

    def test_unknown_label_is_rejected_against_checkpoint_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown food labels"):
            build_class_to_index(
                [record("tomato", "target", "train", "target-tomato")],
                class_names=["potato", "corn"],
            )

    def test_reseeding_gives_reproducible_but_worker_distinct_augmentations(self) -> None:
        dataset = FoodImageDataset(
            [record("corn", "proxy", "train", "proxy-corn")],
            data_root=".",
            augmentation_profile="targeted",
            image_size=24,
            seed=1,
        )
        image = Image.fromarray(
            np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3),
            mode="RGB",
        )

        dataset.reseed(100)
        worker_one = np.asarray(dataset.augmenter(image))
        dataset.reseed(100)
        worker_one_repeat = np.asarray(dataset.augmenter(image))
        dataset.reseed(101)
        worker_two = np.asarray(dataset.augmenter(image))

        np.testing.assert_array_equal(worker_one, worker_one_repeat)
        self.assertFalse(np.array_equal(worker_one, worker_two))


if __name__ == "__main__":
    unittest.main()
