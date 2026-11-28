"""选模与 EMA 的 CPU 合约测试；不访问图像、预测文件或 GPU。"""

import copy
import importlib
import math
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel


class SelectionEMATestCase(unittest.TestCase):
    def setUp(self):
        # 首次 RED 会明确失败为“功能尚未实现”，避免把导入错误误当测试结果。
        self.assertTrue(
            Path(__file__).with_name("selection_ema.py").is_file(),
            "尚未实现 selection_ema.py",
        )
        self.api = importlib.import_module("selection_ema")


def stats_for(*counts):
    """仅构造真实 monitor 的公开计数结构；预期排序值另行手算。"""
    names = ("饺子", "芦笋", "虾仁", "牛排")
    return {
        "correct": sum(correct for correct, _ in counts),
        "total": sum(total for _, total in counts),
        "per_class": {
            name: {"correct": correct, "total": total}
            for name, (correct, total) in zip(names, counts)
        },
    }


class SelectionKeyTests(SelectionEMATestCase):
    def test_class_balance_beats_majority_class_compensation(self):
        majority_only = stats_for((90, 100), (0, 10))
        balanced = stats_for((70, 100), (7, 10))
        self.assertGreater(
            self.api.selection_key(balanced, 0.8),
            self.api.selection_key(majority_only, 0.2),
        )

    def test_worst_class_breaks_equal_macro_accuracy(self):
        uneven = stats_for((9, 10), (1, 10))
        even = stats_for((6, 10), (4, 10))
        self.assertGreater(
            self.api.selection_key(even, 2.0),
            self.api.selection_key(uneven, 0.1),
        )

    def test_overall_accuracy_breaks_equal_macro_and_worst(self):
        lower_overall = stats_for((8, 10), (4, 10))
        higher_overall = stats_for((16, 20), (4, 10))
        self.assertGreater(
            self.api.selection_key(higher_overall, 2.0),
            self.api.selection_key(lower_overall, 0.1),
        )

    def test_lower_loss_breaks_all_accuracy_ties(self):
        stats = stats_for((6, 10), (4, 10))
        self.assertGreater(
            self.api.selection_key(stats, 0.2),
            self.api.selection_key(stats, 0.8),
        )

    def test_key_is_four_floats_in_the_documented_order(self):
        result = self.api.selection_key(stats_for((8, 10), (4, 10)), 0.25)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for actual, expected in zip(result, (0.6, 0.4, 0.6, -0.25)):
            self.assertIsInstance(actual, float)
            self.assertAlmostEqual(actual, expected)

    def test_zero_sample_class_is_excluded_instead_of_perfect(self):
        stats = stats_for((1, 2), (0, 0))
        stats["macro_accuracy"] = 1.0  # 派生字段不能覆盖原始计数。
        self.assertEqual(self.api.selection_key(stats, 0.4), (0.5, 0.5, 0.5, -0.4))

    def test_all_wrong_nonempty_validation_is_valid(self):
        self.assertEqual(
            self.api.selection_key(stats_for((0, 3), (0, 2)), 0.0),
            (0.0, 0.0, 0.0, -0.0),
        )

    def test_empty_validation_is_rejected(self):
        for stats in (stats_for(), stats_for((0, 0), (0, 0))):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                self.api.selection_key(stats, 0.3)

    def test_bad_global_counts_are_rejected(self):
        for field, invalid in (
            ("correct", -1), ("total", -1), ("correct", 4),
            ("correct", 1.5), ("total", True), ("correct", math.nan),
            ("total", math.inf), ("correct", "1"),
        ):
            stats = stats_for((1, 3))
            stats[field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                self.api.selection_key(stats, 0.3)

    def test_bad_per_class_counts_are_rejected(self):
        for field, invalid in (
            ("correct", -1), ("total", -1), ("correct", 4),
            ("correct", 0.5), ("total", False), ("correct", math.inf),
            ("total", "3"),
        ):
            stats = stats_for((1, 3))
            stats["per_class"]["饺子"][field] = invalid
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                self.api.selection_key(stats, 0.3)

    def test_nonzero_correct_with_zero_total_is_rejected(self):
        stats = stats_for((1, 2), (0, 0))
        stats["per_class"]["芦笋"]["correct"] = 1
        with self.assertRaises(ValueError):
            self.api.selection_key(stats, 0.3)

    def test_inconsistent_global_and_class_counts_are_rejected(self):
        for field, invalid in (("correct", 0), ("total", 4)):
            stats = stats_for((1, 3))
            stats[field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.api.selection_key(stats, 0.3)

    def test_invalid_loss_is_rejected(self):
        for loss in (math.nan, math.inf, -math.inf, -0.1, "0.3", None, True):
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                self.api.selection_key(stats_for((1, 3)), loss)

    def test_malformed_stats_are_rejected(self):
        for stats in (
            None, [], {}, {"correct": 1, "total": 3},
            {"correct": 1, "total": 3, "per_class": []},
            {"correct": 1, "total": 3, "per_class": {"饺子": None}},
            {"correct": 1, "total": 3, "per_class": {"饺子": {"total": 3}}},
        ):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                self.api.selection_key(stats, 0.3)


class EMATests(SelectionEMATestCase):
    @staticmethod
    def linear(value=0.0):
        model = nn.Linear(1, 1, bias=False).cpu()
        with torch.no_grad():
            model.weight.fill_(value)
        return model

    def test_ema_is_an_independent_averaged_model(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live)
        self.assertIsInstance(ema, AveragedModel)
        self.assertIsNot(ema.module, live)
        self.assertNotEqual(ema.module.weight.data_ptr(), live.weight.data_ptr())
        with torch.no_grad():
            ema.module.weight.fill_(99.0)
        self.assertEqual(live.weight.item(), 2.0)

    def test_update_never_changes_live_parameters(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live, decay=0.5)
        self.api.update_ema(ema, live)
        with torch.no_grad():
            live.weight.fill_(6.0)
        self.api.update_ema(ema, live)
        self.assertEqual(live.weight.item(), 6.0)
        self.assertEqual(ema.module.weight.item(), 4.0)
        self.assertIsNone(live.weight.grad)

    def test_first_update_copies_latest_model_instead_of_averaging_creation(self):
        live = self.linear(1.0)
        ema = self.api.create_ema(live, decay=0.5)
        self.assertEqual(ema.n_averaged.item(), 0)
        with torch.no_grad():
            live.weight.fill_(5.0)
        self.api.update_ema(ema, live)
        self.assertEqual(ema.n_averaged.item(), 1)
        self.assertEqual(ema.module.weight.item(), 5.0)

    def test_known_exponential_average_across_three_updates(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live, decay=0.75)
        self.api.update_ema(ema, live)
        with torch.no_grad():
            live.weight.fill_(6.0)
        self.api.update_ema(ema, live)
        self.assertEqual(ema.module.weight.item(), 3.0)
        with torch.no_grad():
            live.weight.fill_(11.0)
        self.api.update_ema(ema, live)
        self.assertEqual(ema.module.weight.item(), 5.0)
        self.assertEqual(ema.n_averaged.item(), 3)

    def test_default_decay_is_point_99(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live)
        self.api.update_ema(ema, live)
        with torch.no_grad():
            live.weight.fill_(4.0)
        self.api.update_ema(ema, live)
        self.assertAlmostEqual(ema.module.weight.item(), 2.02, places=6)

    def test_boundary_decay_zero_uses_latest_parameters(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live, decay=0.0)
        self.api.update_ema(ema, live)
        with torch.no_grad():
            live.weight.fill_(6.0)
        self.api.update_ema(ema, live)
        self.assertEqual(ema.module.weight.item(), 6.0)

    def test_boundary_decay_one_keeps_first_update_parameters(self):
        live = self.linear(2.0)
        ema = self.api.create_ema(live, decay=1.0)
        self.api.update_ema(ema, live)
        with torch.no_grad():
            live.weight.fill_(6.0)
        self.api.update_ema(ema, live)
        self.assertEqual(ema.module.weight.item(), 2.0)

    def test_invalid_decay_is_rejected(self):
        for decay in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.99"):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                self.api.create_ema(self.linear(), decay=decay)

    def test_update_after_optimizer_step_tracks_post_step_weights(self):
        live = self.linear(1.0)
        ema = self.api.create_ema(live, decay=0.5)
        optimizer = torch.optim.SGD(live.parameters(), lr=0.1)
        prediction = live(torch.tensor([[1.0]], device="cpu"))
        prediction.square().sum().backward()
        optimizer.step()
        self.api.update_ema(ema, live)
        self.assertAlmostEqual(live.weight.item(), 0.8, places=6)
        self.assertAlmostEqual(ema.module.weight.item(), 0.8, places=6)

    def test_batch_norm_buffers_copy_latest_values_at_every_update(self):
        live = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).cpu()
        ema = self.api.create_ema(live, decay=0.75)
        self.assertFalse(ema.use_buffers)
        for mean, variance, batches in ((2.0, 4.0, 3), (8.0, 9.0, 7)):
            with torch.no_grad():
                live[1].running_mean.fill_(mean)
                live[1].running_var.fill_(variance)
                live[1].num_batches_tracked.fill_(batches)
            self.api.update_ema(ema, live)
            torch.testing.assert_close(ema.module[1].running_mean, torch.full((2,), mean))
            torch.testing.assert_close(ema.module[1].running_var, torch.full((2,), variance))
            self.assertEqual(ema.module[1].num_batches_tracked.item(), batches)
            self.assertNotEqual(
                ema.module[1].running_mean.data_ptr(), live[1].running_mean.data_ptr()
            )
            torch.testing.assert_close(live[1].running_mean, torch.full((2,), mean))

    def test_clean_state_strictly_reloads_original_model(self):
        live = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).cpu()
        ema = self.api.create_ema(live)
        self.api.update_ema(ema, live)
        state = self.api.ema_model_state(ema)
        self.assertEqual(set(state), set(live.state_dict()))
        self.assertNotIn("n_averaged", state)
        self.assertTrue(all(not key.startswith("module.") for key in state))
        restored = copy.deepcopy(live)
        restored.load_state_dict(state, strict=True)
        restored.eval()
        ema.eval()
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], device="cpu")
        with torch.inference_mode():
            torch.testing.assert_close(restored(inputs), ema(inputs))

    def test_clean_state_strictly_reloads_mobilenet_v3_small_with_14_labels(self):
        # 不下载权重，也不前向训练；验证生产模型的键名与 BN buffer 都能原样加载。
        from torchvision.models import mobilenet_v3_small

        live = mobilenet_v3_small(weights=None, num_classes=14).cpu()
        ema = self.api.create_ema(live)
        self.api.update_ema(ema, live)
        restored = mobilenet_v3_small(weights=None, num_classes=14).cpu()
        result = restored.load_state_dict(self.api.ema_model_state(ema), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
