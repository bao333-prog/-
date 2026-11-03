import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.manifest import ManifestRecord, ManifestValidationError  # noqa: E402
from oven_food.split import split_by_session  # noqa: E402


def record(label: str, session: str, image_number: int = 0) -> ManifestRecord:
    return ManifestRecord(
        image_path=f"images/{session}-{image_number}.jpg",
        food_label=label,
        environment="proxy",
        session_id=session,
        camera_position="top-center",
        lighting_condition="warm",
        steam_level="none",
        occlusion_level="none",
    )


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_session_safe(self) -> None:
        records = []
        for label in ("corn", "potato"):
            for number in range(5):
                session = f"{label}-{number}"
                records.extend([record(label, session, 1), record(label, session, 2)])

        first = split_by_session(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=17)
        second = split_by_session(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=17)

        self.assertEqual([item.split for item in first], [item.split for item in second])
        session_splits: dict[str, set[str]] = {}
        for item in first:
            session_splits.setdefault(item.session_id, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in session_splits.values()))

        for label in ("corn", "potato"):
            counts = Counter(
                next(iter(session_splits[session]))
                for session in session_splits
                if session.startswith(label)
            )
            self.assertEqual(counts, {"train": 3, "val": 1, "test": 1})

    def test_rejects_too_few_sessions_for_three_nonempty_splits(self) -> None:
        records = [record("corn", "corn-1"), record("corn", "corn-2")]

        with self.assertRaisesRegex(ManifestValidationError, "at least 3 sessions"):
            split_by_session(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=1)

    def test_rejects_ratios_that_do_not_sum_to_one(self) -> None:
        records = [record("corn", f"corn-{number}") for number in range(3)]

        with self.assertRaisesRegex(ValueError, "sum to 1"):
            split_by_session(records, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2, seed=1)


if __name__ == "__main__":
    unittest.main()
