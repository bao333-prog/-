"""场景均衡采样的可执行契约；所有图片均为临时训练夹具。"""
import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

try:
    module = importlib.import_module('scene_sampler')
except ModuleNotFoundError:
    module = None


def row(family, *, label='food', group='g', kind='original', path=None,
        environment='target', era='real', session='session', split='train'):
    return dict(family_id=family, label=label, environment=environment,
                era=era, sampling_group=group, source_session=session,
                actual_input=path or family + '_' + kind + '.png',
                kind=kind, split=split)


class SceneSamplerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module, '尚未实现场景采样模块')

    def test_branch_probability_preserves_supplied_mass(self):
        # 捕获把所有scene放到全局均匀池、破坏环境/类别/时期权重的错误。
        rows = [row('a'), row('b'), row('c', environment='proxy', era='web')]
        sampler = module.SceneBalancedSampler(rows, [.1, .2, .7],
                  {'a': 's1', 'b': 's2', 'c': 's3'}, num_samples=50000)
        self.assertAlmostEqual(sampler.branch_probabilities[('food', 'target', 'real', 'g')], .3)
        self.assertAlmostEqual(sampler.branch_probabilities[('food', 'proxy', 'web', 'g')], .7)
        list(sampler)  # 第一轮强制原图覆盖，不用于估计稳定分布。
        counts = Counter(rows[i]['environment'] for i in sampler)
        self.assertAlmostEqual(counts['target'] / 50000, .3, delta=.01)

    def test_more_frames_in_one_scene_do_not_increase_scene_probability(self):
        # 捕获按family数给scene分配质量的错误；分支总质量保持相同。
        rows = [row('a'), row('b'), row('c'), row('d')]
        sampler = module.SceneBalancedSampler(rows, [.25] * 4,
                  {'a': 'many', 'b': 'many', 'c': 'many', 'd': 'single'},
                  num_samples=30000)
        all_draws = list(sampler)
        second = list(sampler)
        all_draws.extend(second)
        counts = Counter('single' if rows[i]['family_id'] == 'd' else 'many' for i in second)
        self.assertAlmostEqual(counts['single'] / 30000, .5, delta=.015)
        # 轮换状态跨epoch继续；整段累计差<=1，任意截取epoch可在两端各余一次。
        family = Counter(rows[i]['family_id'] for i in all_draws)
        self.assertLessEqual(max(family[k] for k in 'abc') - min(family[k] for k in 'abc'), 1)

    def test_duplicate_path_split_weights_have_identical_draws(self):
        # 捕获同一路径作为额外视图增加采样机会的错误。
        rows = [row('a'), row('a', kind='roi'), row('b')]
        scenes = {'a': 'same', 'b': 'other'}
        one = module.SceneBalancedSampler(rows, [.2, .3, .5], scenes, seed=7)
        duplicate = rows + [dict(rows[1])]
        two = module.SceneBalancedSampler(duplicate, [.2, .15, .5, .15], scenes, seed=7)
        for _ in range(15):
            self.assertEqual([rows[i]['actual_input'] for i in one],
                             [duplicate[i]['actual_input'] for i in two])

    def test_first_epoch_reads_each_original_then_has_no_periodic_coverage(self):
        rows = [row(f) for f in 'abcd'] + [row(f, kind='roi') for f in 'abcd']
        sampler = module.SceneBalancedSampler(rows, [.125] * 8,
                  {'a': 'many', 'b': 'many', 'c': 'many', 'd': 'single'}, seed=42)
        self.assertEqual(len(sampler), 4)
        first = list(sampler)
        self.assertEqual({rows[i]['family_id'] for i in first}, set('abcd'))
        self.assertTrue(all(rows[i]['kind'] == 'original' for i in first))
        later = [list(sampler) for _ in range(12)]
        self.assertTrue(any(rows[i]['kind'] == 'roi' for epoch in later for i in epoch))
        self.assertTrue(any(len({rows[i]['family_id'] for i in epoch}) < 4 for epoch in later))
        # 旧采样器第5、9、13轮重新覆盖会使这些轮必有四个family。
        self.assertTrue(any(len({rows[i]['family_id'] for i in later[k]}) < 4 for k in (3, 7, 11)))

    def test_restore_mid_epoch_and_next_epochs_exactly(self):
        rows = [row(f) for f in 'abcde'] + [row('a', kind='roi')]
        scenes = {'a': 's', 'b': 's', 'c': 's', 'd': 't', 'e': 'u'}
        first = module.SceneBalancedSampler(rows, [1] * 6, scenes, num_samples=31)
        list(first)
        active = iter(first)
        [next(active) for _ in range(13)]
        state = json.loads(json.dumps(first.state_dict()))
        resumed = module.SceneBalancedSampler(rows, [1] * 6, scenes, seed=999, num_samples=31)
        resumed.load_state_dict(state)
        self.assertEqual(list(active), list(resumed))
        for _ in range(5):
            self.assertEqual(list(first), list(resumed))

    def test_restore_during_first_original_coverage(self):
        rows = [row(f) for f in 'abcdef']
        scenes = {f: f for f in 'abcdef'}
        sampler = module.SceneBalancedSampler(rows, [1] * 6, scenes)
        active = iter(sampler)
        next(active)
        other = module.SceneBalancedSampler(rows, [1] * 6, scenes)
        other.load_state_dict(sampler.state_dict())
        self.assertEqual(list(active), list(other))
        self.assertEqual(list(sampler), list(other))

    def test_rejects_nontrain_conflicts_missing_original_and_invalid_weights(self):
        cases = [([row('a', split='validation')], [1], {'a': 's'}),
                 ([row('a'), row('a', label='other')], [1, 1], {'a': 's'}),
                 ([row('a', kind='roi')], [1], {'a': 's'}),
                 ([row('a')], [float('nan')], {'a': 's'}),
                 ([row('a')], [0], {'a': 's'}),
                 ([row('a')], [1], {})]
        for rows, weights, scenes in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.SceneBalancedSampler(rows, weights, scenes)

    def test_rejects_scene_crossing_branch_or_session(self):
        for other in (row('b', label='else'), row('b', group='different'),
                      row('b', session='different')):
            with self.subTest(other=other), self.assertRaises(ValueError):
                module.SceneBalancedSampler([row('a'), other], [1, 1], {'a': 's', 'b': 's'})

    def test_restore_rejects_different_input_or_sample_count(self):
        rows = [row('a'), row('b')]
        first = module.SceneBalancedSampler(rows, [1, 1], {'a': 'a', 'b': 'b'})
        other = module.SceneBalancedSampler(rows, [1, 2], {'a': 'a', 'b': 'b'})
        with self.assertRaises(ValueError):
            other.load_state_dict(first.state_dict())

    def test_original_coverage_is_independent_of_view_row_order(self):
        # 同一缓存路径既有派生别名又有original时，仍须返回original记录。
        rows = [row('a', kind='roi', path='same.png'), row('a', path='same.png'), row('b')]
        sampler = module.SceneBalancedSampler(rows, [.1, .4, .5], {'a': 'a', 'b': 'b'})
        self.assertTrue(all(rows[i]['kind'] == 'original' for i in sampler))

    def test_proxy_families_cannot_share_a_scene(self):
        rows = [row('a', environment='proxy'), row('b', environment='proxy')]
        with self.assertRaises(ValueError):
            module.SceneBalancedSampler(rows, [1, 1], {'a': 'same', 'b': 'same'})


class SceneGroupingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module, '尚未实现场景分组模块')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, family, time, value=50, **kwargs):
        r = row(family, **kwargs)
        p = self.root / (family + '.png')
        Image.fromarray(np.full((72, 96, 3), value, dtype=np.uint8)).save(p)
        sidecar = p.with_suffix('.json')
        sidecar.write_text(json.dumps({'camera_context': {
            'captured_at_host_approx': time, 'session_id': r['source_session']}}), encoding='utf-8')
        r.update(source=str(p), metadata=str(sidecar))
        return r

    def test_time_pixel_and_representative_prevent_chain_drift(self):
        # a≈b、b≈c而a不≈c；禁止链式连通分量吞并c。
        rows = [self.source('a', '2026-11-28T10:00:00+08:00', 50),
                self.source('b', '2026-11-28T10:00:01+08:00', 52),
                self.source('c', '2026-11-28T10:00:02+08:00', 54),
                self.source('d', '2026-11-28T10:00:10+08:00', 50)]
        index, audit = module.build_scene_index(rows)
        ids = index['family_to_scene']
        self.assertEqual(ids['a'], ids['b'])
        self.assertNotEqual(ids['a'], ids['c'])
        self.assertNotEqual(ids['a'], ids['d'])
        self.assertFalse(audit['physical_individual_independence_proven'])

    def test_proxy_and_missing_time_remain_singletons(self):
        rows = [self.source('a', None), self.source('b', None),
                self.source('c', '2026-11-28T10:00:00+08:00', environment='proxy'),
                self.source('d', '2026-11-28T10:00:01+08:00', environment='proxy')]
        index, _ = module.build_scene_index(rows)
        self.assertEqual(len(set(index['family_to_scene'].values())), 4)

    def test_grouping_never_crosses_session_or_sampling_group(self):
        rows = [self.source('a', '2026-11-28T10:00:00+08:00'),
                self.source('b', '2026-11-28T10:00:01+08:00', group='another'),
                self.source('c', '2026-11-28T10:00:01+08:00', session='another')]
        index, _ = module.build_scene_index(rows)
        self.assertEqual(len(set(index['family_to_scene'].values())), 3)

    def test_grouping_rejects_nontrain_before_opening_any_path(self):
        r = row('a', split='test')
        r.update(source='MUST_NOT_OPEN.png', metadata='MUST_NOT_OPEN.json')
        with self.assertRaises(ValueError):
            module.build_scene_index([r])

    def test_large_hash_change_is_rejected_even_with_small_rgb_mean(self):
        rows = [self.source('a', '2026-11-28T10:00:00+08:00', 50),
                self.source('b', '2026-11-28T10:00:01+08:00', 50)]
        a = np.tile(np.arange(96, dtype=np.float64) * .02 + 50, (72, 1))
        b = np.fliplr(a)
        # 小幅水平梯度使全局均差较小、但差分哈希方向相反。
        Image.fromarray(np.stack([a] * 3, axis=-1).astype('uint8')).save(rows[0]['source'])
        Image.fromarray(np.stack([b] * 3, axis=-1).astype('uint8')).save(rows[1]['source'])
        index, _ = module.build_scene_index(rows)
        self.assertNotEqual(index['family_to_scene']['a'], index['family_to_scene']['b'])


if __name__ == '__main__':
    unittest.main()
