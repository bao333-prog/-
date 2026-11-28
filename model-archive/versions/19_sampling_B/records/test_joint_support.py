"""本轮行为检查：分层概率、所有图首轮覆盖、确定性及验证数据隔离。"""
import unittest
from joint_support import hierarchical_weights, CoverageThenWeightedSampler


class JointSamplingChecks(unittest.TestCase):
    def rows(self):
        rows = []
        for group, size, era in [('old1', 1, 'first_real'), ('old2', 5, 'first_real'), ('new1', 4, 'second_real')]:
            rows += [dict(label='A', environment='target', era=era, sampling_group=group, split='train')] * size
        rows += [dict(label='A', environment='proxy', era='web', sampling_group='webA', split='train')] * 3
        rows += [dict(label='B', environment='proxy', era='web', sampling_group='webB', split='train')] * 2
        return rows

    def test_class_domain_era_group_mass_is_independent_of_image_counts(self):
        rows = self.rows()
        weights = hierarchical_weights(rows, target_share=0.75)
        mass = lambda predicate: sum(w for w, r in zip(weights, rows) if predicate(r))
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(mass(lambda r: r['label'] == 'A'), .5)
        self.assertAlmostEqual(mass(lambda r: r['label'] == 'B'), .5)
        self.assertAlmostEqual(mass(lambda r: r['label'] == 'A' and r['environment'] == 'target'), .375)
        self.assertAlmostEqual(mass(lambda r: r['era'] == 'first_real'), .1875)
        self.assertAlmostEqual(mass(lambda r: r['sampling_group'] == 'old1'), .09375)
        self.assertAlmostEqual(mass(lambda r: r['sampling_group'] == 'old2'), .09375)

    def test_first_epoch_covers_every_record_and_sequences_are_reproducible(self):
        weights = hierarchical_weights(self.rows(), target_share=.75)
        a = CoverageThenWeightedSampler(weights, seed=42)
        b = CoverageThenWeightedSampler(weights, seed=42)
        self.assertEqual(sorted(list(a)), list(range(len(weights))))
        list(b)  # 同样走完首轮，再比较加权轮次。
        second_a, second_b = list(a), list(b)
        self.assertEqual(second_a, second_b)
        self.assertEqual(len(second_a), len(weights))
        self.assertGreaterEqual(min(second_a), 0)
        self.assertLess(max(second_a), len(weights))

    def test_validation_cannot_be_used_to_build_training_weights(self):
        rows = self.rows()
        rows[0] = dict(rows[0], split='val')
        with self.assertRaises(ValueError):
            hierarchical_weights(rows, target_share=.75)


if __name__ == '__main__':
    unittest.main()
