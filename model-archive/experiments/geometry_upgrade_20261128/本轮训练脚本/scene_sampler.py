"""仅用训练原图识别近似摆放组，并保持既有层级质量进行场景均衡抽样。

scene不是物理独立个体的证据。分组不使用ROI、模型、验证或测试图片；
索引以family_id为键，不绑定最终训练尺寸或缓存路径。
"""
import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


BRANCH_FIELDS = ('label', 'environment', 'era', 'sampling_group')


def _branch(row):
    try:
        result = tuple(str(row[k]) for k in BRANCH_FIELDS)
    except KeyError as exc:
        raise ValueError('缺少层级字段: ' + str(exc)) from exc
    if any(not x for x in result):
        raise ValueError('层级字段不可为空')
    return result


def _session(row):
    return str(row.get('source_session') or row.get('session_id') or row.get('recorded_session') or '')


def _train_only(rows):
    # 必须在读取任何路径前完成集合检查，绝不打开非训练图片。
    for row in rows:
        if row.get('split', 'train') != 'train':
            raise ValueError('仅接受train记录')


def _cdf(values):
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('权重总质量必须有限且为正')
    result, acc = [], 0.
    for value in values:
        acc += value / total
        result.append(acc)
    result[-1] = 1.
    return result


def _tuples(value):
    # 允许检查点通过JSON中转后恢复Python RNG的tuple结构。
    return tuple(_tuples(x) for x in value) if isinstance(value, (tuple, list)) else value


class SceneBalancedSampler:
    """分支按输入质量，分支内scene等概率，scene内family无放回轮换。

    同family/path的权重已由调用方拆分，因此合并时相加、保留一个index。
    第一次完整迭代先各读一次family原图，之后没有周期性全覆盖。
    num_samples不得小于family数，以确保第一轮覆盖契约。
    检查点记录已发出index的位置；DataLoader预取时需在epoch边界保存，
    或由调用方同步保存已消费批次及预取队列，才能恢复真实训练读取进度。
    """

    def __init__(self, rows, weights, scene_ids: dict[str, str], *, seed=42, num_samples=None):
        rows, weights = list(rows), list(weights)
        if not rows or len(rows) != len(weights):
            raise ValueError('rows/weights非空且长度必须一致')
        _train_only(rows)
        self.families = {}
        for index, (row, weight) in enumerate(zip(rows, weights)):
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('weights必须为有限正数')
            family = str(row['family_id'])
            scene = scene_ids.get(family)
            if not family or not isinstance(scene, str) or not scene:
                raise ValueError('每个family必须有非空scene_id')
            branch, session = _branch(row), _session(row)
            info = self.families.setdefault(family, dict(branch=branch, session=session,
                        scene=scene, views={}, original=None))
            if (info['branch'], info['session']) != (branch, session):
                raise ValueError('同family出现类别/分支/session冲突')
            path = os.path.normcase(os.path.normpath(str(row['actual_input'])))
            view = info['views'].setdefault(path, {'index': index, 'weight': 0., 'kind': row['kind']})
            view['weight'] += weight
            if row['kind'] == 'original' and info['original'] is None:
                info['original'] = index
        self.keys = list(self.families)
        self.num_samples = len(self.keys) if num_samples is None else int(num_samples)
        if self.num_samples < len(self.keys) or (num_samples is not None and self.num_samples != num_samples):
            raise ValueError('num_samples必须为不少于family数的整数')
        self.scenes, branch_mass = {}, defaultdict(float)
        for family, info in self.families.items():
            if info['original'] is None:
                raise ValueError('每个family至少需要一个original视图')
            scene = info['scene']
            entry = self.scenes.setdefault(scene, {'branch': info['branch'],
                                'session': info['session'], 'families': []})
            if (entry['branch'], entry['session']) != (info['branch'], info['session']):
                raise ValueError('同scene不可跨类别/分支/session')
            if info['branch'][1] == 'proxy' and entry['families']:
                raise ValueError('proxy必须每个family独立scene')
            entry['families'].append(family)
            views = list(info['views'].values())
            info['indices'] = [v['index'] for v in views]
            info['cdf'] = _cdf([v['weight'] for v in views])
            branch_mass[info['branch']] += math.fsum(v['weight'] for v in views)
        self.branch_mass = dict(branch_mass)
        total = math.fsum(branch_mass.values())
        self.branch_probabilities = {k: v / total for k, v in branch_mass.items()}
        self.branches = list(branch_mass)
        self.branch_cdf = _cdf(list(branch_mass.values()))
        self.branch_scenes = {k: [] for k in self.branches}
        for scene, info in self.scenes.items():
            self.branch_scenes[info['branch']].append(scene)
        self.scene_probabilities = {s: self.branch_probabilities[x['branch']] /
                  len(self.branch_scenes[x['branch']]) for s, x in self.scenes.items()}
        self._rng = random.Random(seed)
        self._epoch, self._position = 0, 0
        self._coverage_order = []
        self._rotation = {s: {'order': [], 'position': 0} for s in self.scenes}
        signature = {'rows': [(r['family_id'], r['actual_input'], r['kind'], _branch(r), _session(r), float(w),
                              scene_ids[r['family_id']]) for r, w in zip(rows, weights)],
                     'num_samples': self.num_samples}
        self._signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return self.num_samples

    def _weighted_draw(self):
        branch = self.branches[bisect.bisect_left(self.branch_cdf, self._rng.random())]
        scene = self._rng.choice(self.branch_scenes[branch])
        rotation = self._rotation[scene]
        if rotation['position'] >= len(rotation['order']):
            rotation['order'] = list(self.scenes[scene]['families'])
            self._rng.shuffle(rotation['order'])
            rotation['position'] = 0
        family = rotation['order'][rotation['position']]
        rotation['position'] += 1
        info = self.families[family]
        return info['indices'][bisect.bisect_left(info['cdf'], self._rng.random())]

    def __iter__(self):
        # 中途恢复时继续当前epoch；完整消费后下次迭代才开始新epoch。
        if self._epoch == 0 or self._position == self.num_samples:
            self._epoch += 1
            self._position = 0
            if self._epoch == 1:
                self._coverage_order = [self.families[f]['original'] for f in self.keys]
                self._rng.shuffle(self._coverage_order)
        epoch = self._epoch
        while self._position < self.num_samples and self._epoch == epoch:
            if epoch == 1 and self._position < len(self._coverage_order):
                index = self._coverage_order[self._position]
            else:
                index = self._weighted_draw()
            self._position += 1
            yield index

    def state_dict(self):
        return copy.deepcopy({'version': 1, 'signature': self._signature,
                'rng_state': self._rng.getstate(), 'epoch': self._epoch,
                'position': self._position, 'coverage_order': self._coverage_order,
                'rotation': self._rotation})

    def load_state_dict(self, state):
        if state.get('version') != 1 or state.get('signature') != self._signature:
            raise ValueError('采样检查点与当前rows/weights/scene_ids/epoch长度不兼容')
        if not (0 <= state['position'] <= self.num_samples) or state['epoch'] < 0:
            raise ValueError('采样检查点位置无效')
        self._rng.setstate(_tuples(state['rng_state']))
        self._epoch, self._position = state['epoch'], state['position']
        self._coverage_order = copy.deepcopy(state['coverage_order'])
        self._rotation = copy.deepcopy(state['rotation'])


def _descriptor(row):
    """只有target原图需要像素；proxy不比较像素，始终单独成组。"""
    result = {'timestamp': None, 'capture_time': None, 'reason': '', 'rgb': None, 'dhash': None}
    if row['environment'] == 'proxy':
        result['reason'] = 'proxy_family_singleton'
        return result
    if not _session(row) or not row.get('metadata'):
        result['reason'] = 'missing_session_or_capture_metadata'
        return result
    try:
        metadata = json.loads(Path(row['metadata']).read_text(encoding='utf-8-sig'))
        context = metadata.get('camera_context', {})
        recorded = context.get('session_id')
        if recorded and str(recorded) != _session(row):
            result['reason'] = 'metadata_session_conflict'
            return result
        capture = context.get('captured_at_host_approx')
        if not capture:
            result['reason'] = 'missing_capture_time'
            return result
        moment = datetime.fromisoformat(str(capture).replace('Z', '+00:00'))
        if moment.tzinfo is None:
            result['reason'] = 'ambiguous_capture_timezone'
            return result
        result.update(timestamp=moment.timestamp(), capture_time=capture)
        with Image.open(row['source']) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            # 只保留uint8缩略图，避免全量train描述子占用数GB内存。
            result['rgb'] = np.asarray(rgb.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.uint8)
            gray = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
        bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
        result['dhash'] = int.from_bytes(np.packbits(bits).tobytes(), 'big')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(timestamp=None, reason='unusable_capture_or_image:' + type(exc).__name__)
    return result


def build_scene_index(rows, *, max_gap_seconds=2., max_rgb_mae=2./255., max_dhash=3, workers=4):
    """相邻时间门限+固定组代表相似度；无法确认时保守独立成组。"""
    rows = list(rows)
    _train_only(rows)
    if not rows or max_gap_seconds < 0 or max_rgb_mae < 0 or not 0 <= max_dhash <= 64:
        raise ValueError('训练原图/分组阈值无效')
    originals, aliases = {}, defaultdict(list)
    for row in rows:
        if row['kind'] != 'original':
            raise ValueError('分组仅接受train original，不接受ROI或合成图片')
        family = str(row['family_id'])
        if not family:
            raise ValueError('family_id不可为空')
        if family in originals and (_branch(originals[family]), _session(originals[family])) != (_branch(row), _session(row)):
            raise ValueError('同family出现标签/分支/session冲突')
        originals.setdefault(family, row)
        aliases[family].append({'source': row['source'], 'metadata': row.get('metadata', '')})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        descriptors = dict(zip(originals, pool.map(_descriptor, originals.values())))
    buckets = defaultdict(list)
    for family, row in originals.items():
        buckets[(_branch(row), _session(row))].append(family)
    groups = []
    for (branch, session), families in sorted(buckets.items()):
        families.sort(key=lambda f: (descriptors[f]['timestamp'] is None,
                      descriptors[f]['timestamp'] or 0, f))
        current, previous_time = None, None
        for family in families:
            descriptor = descriptors[family]
            timestamp = descriptor['timestamp']
            matched, mae, distance = False, None, None
            if (branch[1] == 'target' and current is not None and timestamp is not None and
                    previous_time is not None and 0 <= timestamp - previous_time <= max_gap_seconds):
                representative = descriptors[current['representative_family']]
                if representative['rgb'] is not None and descriptor['rgb'] is not None:
                    delta = representative['rgb'].astype(np.int16) - descriptor['rgb'].astype(np.int16)
                    mae = float(np.mean(np.abs(delta))) / 255.
                    distance = (representative['dhash'] ^ descriptor['dhash']).bit_count()
                    matched = mae <= max_rgb_mae + 1e-12 and distance <= max_dhash
            if not matched:
                scene_id = 'scene_' + hashlib.sha256(json.dumps([branch, session, family],
                                      ensure_ascii=False).encode()).hexdigest()[:20]
                current = {'scene_id': scene_id, **dict(zip(BRANCH_FIELDS, branch)),
                           'source_session': session, 'representative_family': family,
                           'family_ids': [], 'members': [], 'max_rgb_mae_to_representative': 0.,
                           'max_dhash_to_representative': 0}
                groups.append(current)
            current['family_ids'].append(family)
            current['members'].append({'family_id': family, 'source': originals[family]['source'],
                    'metadata': originals[family].get('metadata', ''),
                    'capture_time': descriptor['capture_time'],
                    'rgb_mae_to_representative': mae if matched else 0.,
                    'dhash_to_representative': distance if matched else 0,
                    'singleton_reason': descriptor['reason'], 'source_aliases': aliases[family]})
            if matched:
                current['max_rgb_mae_to_representative'] = max(current['max_rgb_mae_to_representative'], mae)
                current['max_dhash_to_representative'] = max(current['max_dhash_to_representative'], distance)
            previous_time = timestamp
    family_to_scene = {f: g['scene_id'] for g in groups for f in g['family_ids']}
    settings = {'max_adjacent_capture_gap_seconds': max_gap_seconds,
                'max_full_rgb_mae_normalized': max_rgb_mae, 'max_dhash_hamming': max_dhash,
                'rgb_size': [224, 224], 'dhash_size': [9, 8], 'resize': 'whole_frame_lanczos',
                'compare_to': 'fixed_scene_representative', 'proxy_policy': 'one_scene_per_family',
                'missing_or_conflicting_metadata': 'singleton', 'input_path_key': 'source'}
    index = {'version': 1, 'description': '保守的近似摆放组，不代表物理独立食材个体',
             'settings': settings, 'family_to_scene': family_to_scene, 'scenes': groups}
    sizes = Counter(len(g['family_ids']) for g in groups)
    by_label = {}
    for label in sorted({r['label'] for r in rows}):
        subset = [g for g in groups if g['label'] == label]
        by_label[label] = {'original_rows': sum(r['label'] == label for r in rows),
                'families': sum(len(g['family_ids']) for g in subset), 'scenes': len(subset),
                'target_scenes': sum(g['environment'] == 'target' for g in subset),
                'proxy_scenes': sum(g['environment'] == 'proxy' for g in subset),
                'largest_scene_families': max((len(g['family_ids']) for g in subset), default=0)}
    audit = {'train_original_rows': len(rows), 'unique_families': len(originals),
             'duplicate_family_rows': len(rows)-len(originals), 'scene_count': len(groups),
             'multi_family_scenes': sum(n for size, n in sizes.items() if size > 1),
             'singleton_scenes': sizes[1], 'largest_scene_families': max(sizes),
             'scene_size_histogram': dict(sorted(sizes.items())), 'per_label': by_label,
             'target_families': sum(r['environment'] == 'target' for r in originals.values()),
             'proxy_families': sum(r['environment'] == 'proxy' for r in originals.values()),
             'target_scenes': sum(g['environment'] == 'target' for g in groups),
             'proxy_scenes': sum(g['environment'] == 'proxy' for g in groups),
             'singleton_reason_counts': dict(Counter(d['reason'] for d in descriptors.values() if d['reason'])),
             'settings': settings, 'only_training_originals_read': True,
             'validation_test_images_or_predictions_read': False,
             'physical_individual_independence_proven': False,
             'limitations': '仅时间相邻且全图近似的保守摆放组；不同scene仍可能是同一物理食材，无法据此声称个体泛化。',
             'largest_scenes': [{'scene_id': g['scene_id'], 'label': g['label'],
                    'environment': g['environment'], 'era': g['era'], 'sampling_group': g['sampling_group'],
                    'families': len(g['family_ids']),
                    'max_rgb_mae_to_representative': g['max_rgb_mae_to_representative'],
                    'max_dhash_to_representative': g['max_dhash_to_representative'],
                    'first_capture': g['members'][0]['capture_time'],
                    'last_capture': g['members'][-1]['capture_time']}
                  for g in sorted(groups, key=lambda g: -len(g['family_ids']))[:12]]}
    return index, audit


def write_contacts(index, output_dir, *, count=8, max_members=24):
    """给最大近似摆放组生成审核接触表；只读取索引内已核验的train来源。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else ImageFont.load_default()
    results = []
    largest = sorted(index['scenes'], key=lambda g: -len(g['family_ids']))[:count]
    for number, scene in enumerate(largest, 1):
        members = scene['members']
        choices = np.linspace(0, len(members)-1, min(max_members, len(members)), dtype=int)
        selected = [members[i] for i in choices]
        cols, cell_w, cell_h, top = 4, 300, 220, 86
        sheet = Image.new('RGB', (cols * cell_w, top + math.ceil(len(selected)/cols) * cell_h), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"近似摆放组 {number}: {scene['label']} | {len(members)} families", fill='black', font=font)
        draw.text((10, 33), scene['scene_id'] + ' | ' + scene['era'], fill='black', font=font)
        draw.text((10, 58), '只读训练原图；第一格为固定代表。组间不保证物理个体独立。', fill='black', font=font)
        for cell, member in enumerate(selected):
            left, upper = (cell % cols) * cell_w, top + (cell // cols) * cell_h
            with Image.open(member['source']) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'), (292, 164))
            sheet.paste(thumb, (left + (cell_w-thumb.width)//2, upper))
            name = Path(member['source']).name
            draw.text((left+4, upper+165), name[:39], fill='black', font=small)
            time = (member['capture_time'] or 'no time')[11:26]
            draw.text((left+4, upper+184), f"{time}  MAE255={255*member['rgb_mae_to_representative']:.2f} dh={member['dhash_to_representative']}", fill='black', font=small)
            draw.text((left+4, upper+202), member['family_id'][:24], fill='black', font=small)
        path = output_dir / f'scene_contact_{number:02d}.jpg'
        sheet.save(path, quality=94)
        results.append({'path': str(path), 'scene_id': scene['scene_id'],
                        'scene_families': len(members), 'shown_families': len(selected)})
    return results


def main():
    parser = argparse.ArgumentParser(description='仅基于train原图构建近似摆放组')
    parser.add_argument('--training-base', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = json.loads(args.training_base.read_text(encoding='utf-8'))
    index, audit = build_scene_index(rows, workers=args.workers)
    audit['training_base'] = str(args.training_base.resolve())
    audit['training_base_sha256'] = hashlib.sha256(args.training_base.read_bytes()).hexdigest()
    audit['contact_sheets'] = write_contacts(index, args.output_dir)
    for filename, data in [('scene_index.json', index), ('scene_audit.json', audit)]:
        (args.output_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['train_original_rows', 'unique_families', 'scene_count',
                    'target_scenes', 'proxy_scenes', 'multi_family_scenes', 'largest_scene_families']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
