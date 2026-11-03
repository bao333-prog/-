import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.config import ExperimentConfig  # noqa: E402
from oven_food.train import train_experiment  # noqa: E402


class TrainingGuardTests(unittest.TestCase):
    def test_direct_config_cannot_use_test_as_validation_split(self) -> None:
        config = ExperimentConfig(
            mode="train",
            experiment_name="unsafe",
            manifest="not-used.csv",
            data_root=".",
            output_dir="not-used",
            train_environments=("proxy",),
            validation_environments=("target",),
            validation_split="test",
        )

        with self.assertRaisesRegex(ValueError, "validation_split must be 'val'"):
            train_experiment(config)

    def test_direct_config_cannot_reuse_train_as_validation_split(self) -> None:
        config = ExperimentConfig(
            mode="train",
            experiment_name="overlap",
            manifest="not-used.csv",
            data_root=".",
            output_dir="not-used",
            train_environments=("proxy",),
            validation_environments=("proxy",),
            validation_split="train",
        )

        with self.assertRaisesRegex(ValueError, "validation_split must be 'val'"):
            train_experiment(config)


if __name__ == "__main__":
    unittest.main()
