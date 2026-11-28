"""生成训练视图并按来源家族分配权重，派生图不改变数据集合。"""
import math
import sys
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from prepare_data import RUN, ROOT, load, dump, sha, source_id

sys.path.insert(0, str(RUN / 'source_snapshot'))
from oven_food.augment import prepare_image
from oven_food.manifest import ManifestRecord, write_manifest, validate_manifest
from joint_support import hierarchical_weights


def store_view(image, row, kind, tag):
    """只处理训练原图/已核验合成；归一化使用固定ImageNet常数。"""
    import hashlib
    assert row['split'] == 'train'
    p = RUN / 'inputs/train' / row['label'] / (row['family_id'] + '_' + tag + '.png')
    prepared = prepare_image(image, image_size=224, resize_mode='letterbox')
    prepared.save(p)
    return {**row, 'kind': kind, 'actual_input': str(p), 'prepared_sha256': sha(p),
            'prepared_pixel_sha256': hashlib.sha256(prepared.tobytes()).hexdigest()}


def records(rows):
    return [ManifestRecord(image_path=r['actual_input'], food_label=r['label'],
            environment=r['environment'], session_id=r['source_group'],
            camera_position='fixed_oven_camera', lighting_condition='fixed',
            steam_level='none', occlusion_level='none', split=r['split']) for r in rows]


def main():
    train = load(RUN / 'training_base.json')
    val, test = load(RUN / 'validation_base.json'), load(RUN / 'test_base.json')
    by_source = {r['family_id']: r for r in train}
    annotations = load(RUN / 'segmentation_inputs.json')
    annotation_sources = {r['source_sha256']: dict(r) for r in annotations['sources']}
    imported = load(RUN / 'boxes320_matches.json')
    assert sha(RUN / 'boxes320_matches.json') == '296c3c1966d18a3bdd8ee894119a0bd0e1828c52935c7e80a49ea9bcb9a859f8'
    assert sum(r['training_eligible'] for r in imported) == 266
    for r in imported:
        if not r['training_eligible']:
            continue
        dest = annotation_sources[r['source_sha256']]
        assert dest['label'] == r['label'] and r['split'] == 'train'
        dest['box_xyxy'] = r['effective_box_xyxy']
        dest['box_evidence'] = {'kind': 'final_320_visual_review', 'id': r['id'],
                'manifest': str(RUN / 'boxes320_matches.json'),
                'review': r['final_visual_status'], 'reason': r['box_adjustment_reason']}
    dump(RUN / 'effective_annotation_sources.json', list(annotation_sources.values()))

    def roi(a):
        digest = a['source_sha256']
        if digest not in by_source:
            raise ValueError('ROI来源不属于训练集')
        r = by_source[digest]
        assert a['split'] == 'train' and a['label'] == r['label']
        assert sha(a['source_path']) == digest
        with Image.open(a['source_path']) as original:
            assert list(original.size) == a['image_size']
            l, t, right, bottom = a['box_xyxy']
            assert 0 <= l < right <= original.width and 0 <= t < bottom <= original.height
            # 保留10%周边，容忍用户已接受的小贴边与少量背景；不新增颜色扰动。
            px, py = max(8, math.ceil((right-l)*.10)), max(8, math.ceil((bottom-t)*.10))
            crop = [max(0, l-px), max(0, t-py), min(original.width, right+px), min(original.height, bottom+py)]
            result = store_view(original.crop(crop), r, 'annotation_roi', 'roi')
        result.update(parent_source_sha256=digest, crop_xyxy=crop,
                      annotation_box_xyxy=a['box_xyxy'], annotation_evidence=a['box_evidence'])
        return result
    with ThreadPoolExecutor(max_workers=4) as executor:
        rois = list(executor.map(roi, annotation_sources.values()))

    synthetic = []
    for s in annotations['synthetic_samples']:
        r = by_source[s['source_sha256']]
        assert s['split'] == s['source_split'] == 'train' and s['label'] == r['label']
        assert sha(s['output']) == s['output_sha256']
        with Image.open(s['output']) as image:
            result = store_view(image, r, 'reviewed_food_geometry', 'synth_' + s['id'])
        result.update(parent_source_sha256=r['family_id'], synthetic_id=s['id'],
                      synthetic_source_path=s['output'], synthetic_source_sha256=s['output_sha256'],
                      synthetic_review_evidence=s['review_evidence'])
        synthetic.append(result)
    assert len(synthetic) == 94
    combined = train + rois + synthetic
    held_out = {r['prepared_pixel_sha256'] for r in val + test}
    assert not ({r['prepared_pixel_sha256'] for r in combined} & held_out)
    assert not ({r['family_id'] for r in combined} & {r['family_id'] for r in val + test})
    for r in combined:
        assert r['split'] == 'train' and r['family_id'] in by_source
    validate_manifest(records(combined + val), check_files=True)
    validate_manifest(records(test), check_files=True)
    write_manifest(RUN / 'train_validation.csv', records(combined + val))
    write_manifest(RUN / 'test_only.csv', records(test))
    dump(RUN / 'training_integrated.json', combined)
    # 先在原图池分配类别/域/拍摄组质量，再拆给该来源的原图与派生图。
    raw_weights = hierarchical_weights(train, target_share=.75, max_expected_draws=4.)
    family_mass = defaultdict(float)
    for r, w in zip(train, raw_weights):
        family_mass[r['family_id']] += w
    families = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(combined):
        families[r['family_id']][r['kind']].append(i)
    weights = [0.] * len(combined)
    for family, groups in families.items():
        has_roi, has_synth = 'annotation_roi' in groups, 'reviewed_food_geometry' in groups
        shares = {'annotation_roi': .25 if has_roi else 0., 'reviewed_food_geometry': .10 if has_synth else 0.}
        shares['original'] = 1-sum(shares.values())
        for kind, indices in groups.items():
            for i in indices:
                weights[i] = family_mass[family] * shares[kind] / len(indices)
        assert math.isclose(sum(weights[i] for indices in groups.values() for i in indices), family_mass[family], abs_tol=1e-14)
    assert math.isclose(sum(weights), 1., abs_tol=1e-10) and min(weights) > 0
    dump(RUN / 'sampling_weights.json', weights)
    config = load(ROOT / 'runs/joint_optimized_20261118_202521/train_config.json')
    config.update(experiment_name='integrated_original_mask_box_20261126',
                  manifest=str(RUN / 'train_validation.csv'), output_dir=str(RUN / 'training'),
                  checkpoint=str(RUN / 'official_initialization.pt'))
    dump(RUN / 'train_config.json', config)
    audit = {'source_training_rows': len(train), 'unique_source_families': len(families),
             'validation_originals': len(val), 'test_originals': len(test),
             'integrated_train_views': len(combined), 'view_counts': dict(Counter(r['kind'] for r in combined)),
             'roi_per_class': dict(Counter(r['label'] for r in rois)),
             'new_320_accepted': 266, 'new_320_excluded': 54,
             'class_names': load(RUN / 'official_class_names.json'),
             'training_pixel_or_parent_overlap_with_eval': 0,
             'weighted_epoch_family_mass_preserved': True,
             'full_coverage_epochs': [1,5,9,13,17],
             'cropped_eval_images': 0, 'new_unreviewed_food_synthesis': 0,
             'historical_pending_materials_authorized_by_current_request': True}
    dump(RUN / 'prepared_data_audit.json', audit)
    print(audit, flush=True)


if __name__ == '__main__':
    import torch
    initial = torch.load(RUN / 'official_initialization.pt', map_location='cpu', weights_only=False)
    assert initial['epoch'] == 0
    dump(RUN / 'official_class_names.json', initial['class_names'])
    main()
