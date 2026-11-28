"""防止一个来源的多份增强图增加该来源在覆盖轮的训练次数。"""
import unittest
from collections import Counter
from family_sampler import FamilyCoverageSampler


class SourceFamilyTests(unittest.TestCase):
    def test_each_coverage_epoch_visits_one_view_per_source(self):
        rows = [{'family_id': f, 'actual_input': p, 'label': '食物'} for f,p in
                [('a','a_full'),('a','a_roi'),('a','a_synth1'),('a','a_synth2'),('b','b_full')]]
        sampler = FamilyCoverageSampler(rows, [.13,.05,.01,.01,.8], seed=42, coverage_interval=1)
        seen = set()
        self.assertEqual(len(sampler), 2)
        for _ in range(5):
            indices = list(sampler)
            self.assertEqual(Counter(rows[i]['family_id'] for i in indices), {'a':1,'b':1})
            seen.update(rows[i]['actual_input'] for i in indices)
        self.assertEqual(len(seen),5)

    def test_duplicate_source_paths_do_not_add_family_or_view(self):
        rows=[{'family_id': 'a','actual_input':'same','label':'a'},
              {'family_id': 'a','actual_input':'same','label':'a'},
              {'family_id': 'b','actual_input':'other','label':'b'}]
        sampler=FamilyCoverageSampler(rows,[.25,.25,.5],seed=1)
        self.assertEqual(len(sampler),2)
        self.assertEqual(len(list(sampler)),2)

    def test_family_with_conflicting_labels_is_rejected(self):
        rows=[{'family_id':'x','actual_input':'x','label':'a'},
              {'family_id':'x','actual_input':'x_roi','label':'b'}]
        with self.assertRaises(ValueError):FamilyCoverageSampler(rows,[.5,.5],seed=1)


if __name__=='__main__':unittest.main()
