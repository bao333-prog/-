import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.manifest import (  # noqa: E402
    ManifestValidationError,
    read_manifest,
    validate_manifest,
)


FIELDS = [
    "image_path",
    "food_label",
    "environment",
    "session_id",
    "camera_position",
    "lighting_condition",
    "steam_level",
    "occlusion_level",
]


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str] = FIELDS) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(image_path: str = "images/a.jpg", **overrides: str) -> dict[str, str]:
    value = {
        "image_path": image_path,
        "food_label": "corn",
        "environment": "proxy",
        "session_id": "corn-session-1",
        "camera_position": "top-center",
        "lighting_condition": "warm",
        "steam_level": "none",
        "occlusion_level": "low",
    }
    value.update(overrides)
    return value


class ManifestTests(unittest.TestCase):
    def test_reads_utf8_manifest_into_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            write_rows(manifest, [row(food_label="玉米")])

            records = read_manifest(manifest)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].food_label, "玉米")
            self.assertEqual(records[0].environment, "proxy")

    def test_rejects_missing_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            values = row()
            values.pop("food_label")
            write_rows(
                manifest,
                [values],
                fields=[field for field in FIELDS if field != "food_label"],
            )

            with self.assertRaisesRegex(ManifestValidationError, "food_label"):
                read_manifest(manifest)

    def test_rejects_invalid_environment_and_severity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            write_rows(manifest, [row(environment="other", steam_level="heavy")])

            with self.assertRaises(ManifestValidationError) as raised:
                validate_manifest(read_manifest(manifest))

            message = str(raised.exception)
            self.assertIn("environment", message)
            self.assertIn("steam_level", message)

    def test_rejects_one_session_with_multiple_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            write_rows(
                manifest,
                [row(food_label="corn"), row(image_path="images/b.jpg", food_label="potato")],
            )

            with self.assertRaisesRegex(ManifestValidationError, "multiple food labels"):
                validate_manifest(read_manifest(manifest))

    def test_rejects_missing_image_when_file_check_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            write_rows(manifest, [row()])

            with self.assertRaisesRegex(ManifestValidationError, "does not exist"):
                validate_manifest(read_manifest(manifest), data_root=root, check_files=True)

    def test_rejects_session_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            fields = FIELDS + ["split"]
            write_rows(
                manifest,
                [row(split="train"), row(image_path="images/b.jpg", split="test")],
                fields=fields,
            )

            with self.assertRaisesRegex(ManifestValidationError, "multiple splits"):
                validate_manifest(read_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
