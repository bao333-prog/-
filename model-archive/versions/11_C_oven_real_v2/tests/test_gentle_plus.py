"""保护食材完整性、旧增强兼容性及验证集不随机增强的实际约束。"""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from oven_food.augment import ProxyAugmenter, prepare_image, shrink_and_translate
from oven_food.config import ConfigError, load_experiment_config
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import ManifestRecord


class GentlePlusTests(unittest.TestCase):
    def test_translation_keeps_all_four_corner_markers(self):
        image = Image.new("RGB", (224, 224), "white")
        colors = [(220, 10, 10), (10, 220, 10), (10, 10, 220), (10, 10, 10)]
        for color, box in zip(colors, [(0, 0, 24, 24), (200, 0, 224, 24),
                                       (0, 200, 24, 224), (200, 200, 224, 224)]):
            image.paste(color, box)
        for seed in range(40):
            out = np.asarray(shrink_and_translate(image, scale=.8, rng=random.Random(seed)))
            self.assertEqual(out.shape, (224, 224, 3))
            for color in colors:
                self.assertTrue(np.all(out == color, axis=2).any())
        for scale in [0, 1.1]:
            with self.assertRaises(ValueError):
                shrink_and_translate(image, scale=scale, rng=random.Random(0))

    def test_complete_pipeline_keeps_subject_ends_and_is_reproducible(self):
        image = Image.new("RGBA", (300, 50), (0, 0, 0, 0))
        image.paste((255, 0, 0, 255), (0, 0, 45, 50))
        image.paste((0, 0, 255, 255), (255, 0, 300, 50))
        for seed in range(40):
            one = ProxyAugmenter("gentle_plus", seed=seed, resize_mode="letterbox")(image)
            two = ProxyAugmenter("gentle_plus", seed=seed, resize_mode="letterbox")(image)
            p = np.asarray(one)
            np.testing.assert_array_equal(p, np.asarray(two))
            self.assertEqual(one.size, (224, 224))
            self.assertTrue(((p[:, :, 0] > 180) & (p[:, :, 2] < 80)).any())
            self.assertTrue(((p[:, :, 2] > 180) & (p[:, :, 0] < 80)).any())

    def test_old_gentle_still_matches_original_random_sequence(self):
        image = Image.fromarray(np.arange(80*40*3, dtype=np.uint8).reshape(40, 80, 3))
        rng = random.Random(42)
        expected = prepare_image(image, image_size=80, resize_mode="letterbox")
        if rng.random() < .5:
            expected = ImageOps.mirror(expected)
        expected = expected.rotate(rng.uniform(-8, 8), resample=Image.Resampling.BILINEAR,
                                   expand=True, fillcolor="white")
        for enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
            expected = enhancer(expected).enhance(rng.uniform(.9, 1.1))
        expected = prepare_image(expected, image_size=80, resize_mode="letterbox")
        actual = ProxyAugmenter("gentle", image_size=80, seed=42, resize_mode="letterbox")(image)
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))

    def test_profile_config_requires_letterbox(self):
        data = dict(mode="train", experiment_name="plus", manifest="unused.csv", data_root=".",
                    output_dir="unused", train_environments=["proxy"], validation_environments=["proxy"],
                    augmentation="gentle_plus", resize_mode="letterbox")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_experiment_config(path).augmentation, "gentle_plus")
            data["resize_mode"] = "center_crop"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "gentle_plus.*letterbox"):
                load_experiment_config(path)
        with self.assertRaisesRegex(ValueError, "gentle_plus.*letterbox"):
            ProxyAugmenter("gentle_plus")

    def test_baseline_dataset_remains_deterministic_and_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.png"
            Image.new("RGBA", (80, 30), (200, 10, 30, 140)).save(path)
            original = path.read_bytes()
            record = ManifestRecord(str(path), "food", "proxy", "one", "unknown", "unknown", "none", "none", "val")
            dataset = FoodImageDataset([record], data_root=temp, class_names=["food"],
                                       augmentation_profile="baseline", resize_mode="letterbox")
            np.testing.assert_array_equal(dataset[0][0].numpy(), dataset[0][0].numpy())
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
