"""采样改进的行为检查：概率、重复数据与全覆盖下界。"""
import unittest
from collections import Counter
from joint_support import hierarchical_weights, CoverageThenWeightedSampler


def row(group, fingerprint=None, label='A'):
    item=dict(label=label,environment='target',era='second_real',sampling_group=group,split='train')
    if fingerprint is not None: item['pixel_sha256']=fingerprint
    return item


class SamplingLimitChecks(unittest.TestCase):
    def test_singleton_group_does_not_take_half_the_training_probability(self):
        rows=[row('singleton')]+[row('large') for _ in range(99)]
        weights=hierarchical_weights(rows,target_share=.75)
        self.assertAlmostEqual(sum(weights),1)
        self.assertLessEqual(max(weights),4/len(rows)+1e-12)
        self.assertAlmostEqual(weights[0],.04)

    def test_duplicate_copies_share_the_unique_image_probability(self):
        rows=[row('same','photo_a'),row('same','photo_a'),row('same','photo_b')]
        weights=hierarchical_weights(rows,target_share=.75)
        self.assertAlmostEqual(weights[0],weights[1])
        self.assertAlmostEqual(weights[0]+weights[1],weights[2])
        self.assertAlmostEqual(weights[2],.5)

    def test_small_class_retains_its_mass_when_global_cap_is_infeasible(self):
        rows=[row('normal',label='A') for _ in range(100)]+[row('rare',label='B')]
        weights=hierarchical_weights(rows,target_share=.75)
        self.assertAlmostEqual(sum(weights[:-1]),.5)
        self.assertAlmostEqual(weights[-1],.5)

    def test_every_image_is_seen_at_least_five_times_over_twenty_epochs(self):
        weights=[.4]+[.6/99]*99
        a=CoverageThenWeightedSampler(weights,seed=42,coverage_interval=4)
        b=CoverageThenWeightedSampler(weights,seed=42,coverage_interval=4)
        counts=Counter()
        for epoch in range(1,21):
            first,second=list(a),list(b)
            self.assertEqual(first,second)
            self.assertEqual(len(first),100)
            if (epoch-1)%4==0: self.assertEqual(sorted(first),list(range(100)))
            counts.update(first)
        self.assertEqual(sum(counts.values()),2000)
        self.assertGreaterEqual(min(counts.values()),5)
        self.assertEqual(a.coverage_epochs,[1,5,9,13,17])


if __name__=='__main__': unittest.main()
