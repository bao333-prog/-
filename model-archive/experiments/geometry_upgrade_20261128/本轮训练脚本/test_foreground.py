"""Behavior tests; no model predictions or held-out image decoding."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np

MODULE = Path(__file__).with_name('foreground.py')


class ForegroundTests(unittest.TestCase):
    def api(self):
        self.assertTrue(MODULE.exists(), 'foreground rotation implementation is missing')
        spec = importlib.util.spec_from_file_location('foreground', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        # Asymmetric solid object, independently located about center (50, 50).
        image = np.full((101, 101, 3), (12, 24, 31), np.uint8)
        mask = np.zeros((101, 101), bool)
        mask[45:56, 30:71] = True
        mask[35:46, 30:41] = True
        image[mask] = (180, 90, 40)
        return image, mask

    def test_quarter_turn_removes_old_food_keeps_background_and_color(self):
        # Catches pasting onto the original or modifying the full frame.
        api = self.api()
        image, mask = self.fixture()
        original = image.copy()
        result = api.rotate_foreground(image, mask, 90, halo=3)
        out = result['image']
        self.assertEqual(out.shape, image.shape)
        self.assertTrue(np.array_equal(image, original), 'input was mutated')
        allowed = result['erase_mask'] | result['rotated_mask']
        self.assertTrue(np.array_equal(out[~allowed], image[~allowed]))
        self.assertLess(out[50, 65, 0], 30, 'original horizontal arm survived')
        self.assertTupleEqual(tuple(out[35, 50]), (180, 90, 40))
        self.assertEqual(int(result['rotated_mask'].sum()), int(mask.sum()))

    def test_warm_reflection_is_erased_without_changing_distant_background(self):
        # Catches the known food-colored reflection left at the old position.
        api = self.api()
        image = np.full((160, 240, 3), (12, 24, 31), np.uint8)
        mask = np.zeros((160, 240), bool)
        mask[45:90, 70:160] = True
        image[mask] = (180, 90, 40)
        image[95:115, 90:145] = (80, 45, 20)
        result = api.rotate_foreground(image, mask, 90)
        # The patch's outer corner is outside the new vertical food.
        self.assertLess(result['background'][110, 140, 0], 30)
        self.assertTrue(np.array_equal(result['image'][:15], image[:15]))

    def test_rejects_rotation_that_would_clip_food(self):
        api = self.api()
        image = np.full((80, 120, 3), 20, np.uint8)
        mask = np.zeros((80, 120), bool)
        mask[5:15, 8:110] = True
        image[mask] = 180
        with self.assertRaisesRegex(ValueError, 'clip|截断|bound'):
            api.rotate_foreground(image, mask, 90)

    def test_repaired_textured_tray_does_not_become_a_flat_blob(self):
        # Catches low-frequency-only interpolation wiping the foil's texture.
        api = self.api()
        yy, xx = np.indices((200, 300))
        stripe = ((xx // 4 + yy // 5) % 2 * 35).astype(np.uint8)
        image = np.dstack((stripe + 30, stripe + 40, stripe + 45))
        mask = np.zeros((200, 300), bool)
        mask[80:120, 90:210] = True
        image[mask] = (180, 90, 40)
        result = api.rotate_foreground(image, mask, 90)
        self.assertGreater(result['background'][85:115, 95:115, 0].std(), 8,
                           'same-frame tray texture was replaced by a flat blob')

    def test_source_gate_rejects_unverified_visible_and_heldout(self):
        api = self.api()
        record = {'source_sha256': 'a', 'split': 'train', 'mask_usage': 'full_food_only',
                  'mask_status': 'verified', 'label': 'sausage'}
        train = [{'raw_sha256': 'a', 'split': 'train', 'kind': 'original'}]
        self.assertTrue(api.source_is_eligible(record, train, set()))
        for changes in ({'mask_status': 'pending'}, {'mask_usage': 'visible_bbox_only'},
                        {'mask_usage': 'forbidden_for_rotation'}, {'split': 'validation'}):
            bad = copy.deepcopy(record)
            bad.update(changes)
            self.assertFalse(api.source_is_eligible(bad, train, set()))
        self.assertFalse(api.source_is_eligible(record, train, {'a'}))
        self.assertFalse(api.source_is_eligible(record, [], set()))

    def test_rejects_empty_or_border_mask_and_invalid_angle(self):
        api = self.api()
        image, mask = self.fixture()
        for bad_mask, angle in ((np.zeros_like(mask), 90), (np.ones_like(mask), 90), (mask, 0)):
            with self.assertRaises(ValueError):
                api.rotate_foreground(image, bad_mask, angle)

    def test_small_supplement_includes_both_real_batches(self):
        # Catches sorted strata consuming every slot from the first batch.
        api = self.api()
        pool = []
        for era in ('first_real', 'second_real'):
            for foil in ('with', 'without'):
                for height in ('high', 'medium', 'low'):
                    pool.append({'label': '三文鱼', 'era': era, 'foil': foil,
                                 'height': height, 'source_sha256': era + foil + height})
        chosen = api.select_candidates(pool)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(sum(r['era'] == 'first_real' for r in chosen), 3)
        self.assertEqual(sum(r['era'] == 'second_real' for r in chosen), 3)
        self.assertEqual({r['height'] for r in chosen}, {'high', 'medium', 'low'})
        self.assertEqual({r['foil'] for r in chosen}, {'with', 'without'})

    def test_generator_refuses_nonmember_even_if_marked_verified(self):
        # Catches a caller bypassing current train membership using flags.
        api = self.api()
        with tempfile.TemporaryDirectory(prefix='foreground_test_', dir=MODULE.parent) as tmp:
            record = {'source_sha256': 'outside-current-train', 'label': '香肠',
                      'source_path': str(Path(tmp) / 'must-not-be-opened.jpg'),
                      'mask_path': str(Path(tmp) / 'must-not-be-opened.png'),
                      'current_mask_sha256': 'not-a-real-mask', 'split': 'train',
                      'mask_usage': 'full_food_only', 'mask_status': 'verified'}
            result = api.generate_samples([record], output_dir=Path(tmp) / 'data',
                                          manifest_path=Path(tmp) / 'manifest.json')
            self.assertEqual(result['sources'], [])
            self.assertIn('current train', result['skipped'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
