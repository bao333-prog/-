import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.cli import main  # noqa: E402
from oven_food.config import load_experiment_config  # noqa: E402


class CliTests(unittest.TestCase):
    def test_validate_manifest_command_runs_without_ml_dependencies(self) -> None:
        fields = [
            "image_path",
            "food_label",
            "environment",
            "session_id",
            "camera_position",
            "lighting_condition",
            "steam_level",
            "occlusion_level",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": "private/not-available.jpg",
                        "food_label": "corn",
                        "environment": "proxy",
                        "session_id": "session-1",
                        "camera_position": "top-center",
                        "lighting_condition": "warm",
                        "steam_level": "none",
                        "occlusion_level": "none",
                    }
                )

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["validate-manifest", str(path), "--skip-files"])

            self.assertEqual(exit_code, 0)
            self.assertIn("1 records", output.getvalue())

    def test_shipped_configs_are_valid_and_cover_five_result_groups(self) -> None:
        paths = sorted((ROOT / "configs").glob("*.json"))

        configs = [load_experiment_config(path) for path in paths]

        self.assertEqual(len(configs), 9)
        self.assertEqual(sum(config.mode == "train" for config in configs), 4)
        self.assertEqual(sum(config.mode == "evaluate" for config in configs), 5)
        self.assertIn("04_target_zero_shot", {config.experiment_name for config in configs})
        self.assertIn("05_target_finetuned", {config.experiment_name for config in configs})


if __name__ == "__main__":
    unittest.main()
