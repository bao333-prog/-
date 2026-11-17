import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oven_food.augment import ProxyAugmenter, degrade_resolution  # noqa: E402


class AugmentationTests(unittest.TestCase):
    def test_all_profiles_return_requested_rgb_size(self) -> None:
        image = Image.new("RGB", (48, 32), (80, 120, 160))

        for profile in ("baseline", "standard", "targeted"):
            with self.subTest(profile=profile):
                output = ProxyAugmenter(profile, image_size=24, seed=9)(image)
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.size, (24, 24))

    def test_targeted_profile_changes_pixels_beyond_baseline_resize(self) -> None:
        values = np.tile(np.arange(48, dtype=np.uint8), (32, 1))
        image = Image.fromarray(np.stack([values, values, values], axis=-1), mode="RGB")

        baseline = np.asarray(ProxyAugmenter("baseline", image_size=24, seed=4)(image))
        targeted = np.asarray(ProxyAugmenter("targeted", image_size=24, seed=4)(image))

        self.assertFalse(np.array_equal(baseline, targeted))

    def test_seed_makes_targeted_augmentation_reproducible(self) -> None:
        image = Image.new("RGB", (40, 40), (100, 130, 160))

        first = np.asarray(ProxyAugmenter("targeted", image_size=24, seed=13)(image))
        second = np.asarray(ProxyAugmenter("targeted", image_size=24, seed=13)(image))

        np.testing.assert_array_equal(first, second)

    def test_resolution_degradation_keeps_canvas_size(self) -> None:
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[::2, ::2] = 255
        image = Image.fromarray(pixels, mode="RGB")

        output = degrade_resolution(image, scale=0.25)

        self.assertEqual(output.size, image.size)
        self.assertFalse(np.array_equal(np.asarray(output), pixels))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "augmentation profile"):
            ProxyAugmenter("unknown")


if __name__ == "__main__":
    unittest.main()
