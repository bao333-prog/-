import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.config import ConfigError, load_experiment_config  # noqa: E402


def valid_train_config() -> dict:
    return {
        "mode": "train",
        "experiment_name": "proxy-baseline",
        "manifest": "data/manifest.csv",
        "data_root": ".",
        "output_dir": "runs/proxy-baseline",
        "train_environments": ["proxy"],
        "validation_environments": ["proxy"],
        "augmentation": "baseline",
    }


class ConfigTests(unittest.TestCase):
    def write_config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_defaults_for_training_config(self) -> None:
        config = load_experiment_config(self.write_config(valid_train_config()))

        self.assertEqual(config.mode, "train")
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 30)
        self.assertEqual(config.train_environments, ("proxy",))

    def test_evaluation_requires_checkpoint_and_environments(self) -> None:
        data = valid_train_config()
        data.update({"mode": "evaluate", "train_environments": []})

        with self.assertRaisesRegex(ConfigError, "checkpoint"):
            load_experiment_config(self.write_config(data))

    def test_target_fraction_requires_proxy_and_target_training_data(self) -> None:
        data = valid_train_config()
        data["target_fraction"] = 0.5

        with self.assertRaisesRegex(ConfigError, "proxy and target"):
            load_experiment_config(self.write_config(data))

    def test_rejects_unknown_configuration_keys(self) -> None:
        data = valid_train_config()
        data["mystery"] = True

        with self.assertRaisesRegex(ConfigError, "unknown configuration keys: mystery"):
            load_experiment_config(self.write_config(data))

    def test_rejects_target_test_as_training_split(self) -> None:
        data = valid_train_config()
        data["train_split"] = "test"

        with self.assertRaisesRegex(ConfigError, "train_split must be 'train'"):
            load_experiment_config(self.write_config(data))

    def test_training_requires_validation_split_val(self) -> None:
        data = valid_train_config()
        data["validation_split"] = "test"
        data["validation_environments"] = ["target"]

        with self.assertRaisesRegex(ConfigError, "validation_split must be 'val'"):
            load_experiment_config(self.write_config(data))


if __name__ == "__main__":
    unittest.main()
