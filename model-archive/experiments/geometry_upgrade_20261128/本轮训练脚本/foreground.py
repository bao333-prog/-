"""Offline, same-frame food rotation using only existing verified complete masks.

Dependencies are loaded only when generating: .venv has OpenCV; the existing
segmentation dependency folder provides SciPy. No installation is performed.
Source images and masks are read-only, and all generated samples await review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OLD = ROOT / 'runs/integrated_food_20261126_115154/segmentation_inputs.json'
EFFECTIVE = ROOT / 'outputs/整合训练模型_20261126_115154/数据与隔离记录/effective_annotation_sources.json'
LATEST = ROOT / 'runs/asparagus_half_20261126_172248'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_is_eligible(record, training_rows, heldout_shas):
    """A complete verified mask must belong to a current original train photo."""
    source_sha = record.get('source_sha256')
    return bool(source_sha and source_sha not in heldout_shas
                and record.get('split') == 'train'
                and record.get('mask_usage') == 'full_food_only'
                and record.get('mask_status') == 'verified'
                and any(r.get('raw_sha256') == source_sha
                        and r.get('split') == 'train'
                        and r.get('kind', 'original') == 'original'
                        for r in training_rows))


def rotate_foreground(rgb, mask, angle=90, *, halo=6, reflection_radius=None):
    """Return an image and exact allowed-edit masks, rotating about the bbox center.

    RGB and alpha are resampled together with premultiplied-alpha interpolation
    by Pillow. There is no color, exposure, illumination or scale adjustment.
    The background is filled exclusively from this photo using the previous
    approved repair pipeline's harmonic interpolation. All pixels outside the
    erased region and pasted foreground are copied exactly from the input.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB must be uint8 HxWx3')
    if mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError('invalid foreground mask')
    if not np.isfinite(angle) or abs(angle) % 360 == 0:
        raise ValueError('angle must make a nonzero rotation')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('border-touching mask could clip food')
    if halo < 0 or halo > 32:
        raise ValueError('halo must be between 0 and 32 pixels')
    sys.path.insert(0, str(ROOT / 'runs/segmentation_20261122_203914/dependencies'))
    from scipy import ndimage as ndi
    sys.path.insert(0, str(ROOT / 'runs/masked_samples_20261123_135614'))
    from background_fill import erase_food_background

    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    # Project every foreground pixel, rather than trusting a clipped canvas.
    radians = math.radians(angle)
    dx, dy = xs + .5 - center[0], ys + .5 - center[1]
    rx = center[0] + math.cos(radians) * dx + math.sin(radians) * dy
    ry = center[1] - math.sin(radians) * dx + math.cos(radians) * dy
    if min(rx.min(), ry.min()) < 2 or rx.max() > rgb.shape[1] - 2 or ry.max() > rgb.shape[0] - 2:
        raise ValueError('rotation would clip food outside image bounds')

    distance = ndi.distance_transform_edt(~mask)
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    warm = (r > 28) & (r > g * 1.08) & (r > b * 1.17) & (r - b > 8)
    radius = max(35, min(mask.shape) * .075) if reflection_radius is None else reflection_radius
    reflection = warm & (distance < radius)
    expanded = ndi.binary_dilation(mask, iterations=max(1, int(halo))) if halo else mask.copy()
    erase_input = expanded | reflection
    background, erased = erase_food_background(rgb, erase_input)
    # Harmonic fill supplies the tray's smooth illumination. On foil, restore
    # texture from a clean translated patch in this exact training photograph.
    # This avoids the visibly flat blobs found in the first representative QA.
    import cv2
    border = ndi.binary_dilation(erased, iterations=12) & ~erased
    # Filtering the food-erased image prevents warm food pixels bleeding into
    # the background texture statistics or being copied as a reflection.
    low = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 8)
    high = background.astype(np.float32) - low
    texture_offset = None
    if border.any() and float(high[border].std()) > 3:
        ey, ex = np.where(erased)
        by, bx = np.where(border)
        # The sampled ring ranks candidates; all source-region pixels are
        # checked against the dilated food/reflection exclusion below.
        sample = max(1, len(by) // 2000)
        by, bx = by[::sample], bx[::sample]
        target_border = low[by, bx]
        target_texture_std = float(high[by, bx].std())
        excluded_source = erased
        step = max(10, min(rgb.shape[:2]) // 32)
        y0, y1, x0, x1 = ey.min(), ey.max() + 1, ex.min(), ex.max() + 1
        footprint = erased[y0:y1, x0:x1]
        best = None
        for dy in range(-int(y0), rgb.shape[0] - int(y1) + 1, step):
            for dx in range(-int(x0), rgb.shape[1] - int(x1) + 1, step):
                if abs(dx) + abs(dy) < step:
                    continue
                if excluded_source[y0+dy:y1+dy, x0+dx:x1+dx][footprint].any():
                    continue
                valid = (by + dy >= 0) & (by + dy < rgb.shape[0]) & (bx + dx >= 0) & (bx + dx < rgb.shape[1])
                if valid.sum() < len(by) * .6:
                    continue
                delta = low[by[valid] + dy, bx[valid] + dx] - target_border[valid]
                patch_texture_std = float(high[y0+dy:y1+dy, x0+dx:x1+dx][footprint].std())
                score = (float(np.mean(np.square(delta)))
                         + max(0, target_texture_std - patch_texture_std) ** 2 * 8
                         + (dx * dx + dy * dy) * .0005)
                if best is None or score < best[0]:
                    best = (score, dx, dy)
        if best is None:
            raise ValueError('no clean same-frame texture patch available')
        _, dx, dy = best
        texture = high[ey + dy, ex + dx]
        blend = np.minimum(ndi.distance_transform_edt(erased)[erased] / 5, 1)[:, None]
        background[erased] = np.clip(np.rint(background[erased] + texture * blend), 0, 255).astype(np.uint8)
        texture_offset = [int(dx), int(dy)]
    rgba = Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255)))
    rotated = rgba.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, center=center)
    rotated_mask = np.array(rotated.getchannel('A')) > 0
    canvas = Image.fromarray(background).convert('RGBA')
    canvas.alpha_composite(rotated)
    image = np.array(canvas.convert('RGB'))
    allowed = erased | rotated_mask
    if not np.array_equal(image[~allowed], rgb[~allowed]):
        raise ValueError('pixels changed outside background repair and foreground')
    return {'image': image, 'background': background, 'erase_mask': erased,
            'rotated_mask': rotated_mask, 'center': center, 'source_box': box,
            'changed_pixels': int(np.any(image != rgb, axis=2).sum()),
            'outside_allowed_changed_pixels': 0, 'halo_pixels': int(halo),
            'reflection_radius': float(radius), 'texture_source_offset': texture_offset}


def load_candidates():
    old = json.loads(OLD.read_text(encoding='utf-8'))
    effective = json.loads(EFFECTIVE.read_text(encoding='utf-8'))
    effective_by_sha = {r['source_sha256']: r for r in effective}
    train = json.loads((LATEST / 'training_base.json').read_text(encoding='utf-8'))
    # The held-out manifests are used solely for SHA exclusion; no held-out
    # image or mask is read, decoded or used as a background.
    heldout_rows = sum((json.loads((LATEST / name).read_text(encoding='utf-8'))
                       for name in ('validation_base.json', 'test_base.json')), [])
    heldout = {r[k] for r in heldout_rows
               for k in ('raw_sha256', 'input_sha256', 'prepared_sha256', 'pixel_sha256')
               if r.get(k)}
    originals = {r['raw_sha256']: r for r in train if r.get('kind') == 'original'}
    result = []
    for r in old['sources']:
        e = effective_by_sha.get(r['source_sha256'])
        if not e or not source_is_eligible(r, train, heldout) or not source_is_eligible(e, train, heldout):
            continue
        if e.get('mask_path') != r.get('mask_path') or e.get('current_mask_sha256') != r.get('current_mask_sha256'):
            continue
        source = Path(r['source_path'])
        mask = Path(r['mask_path'])
        if not source.is_file() or not mask.is_file():
            continue
        if sha(source) != r['source_sha256'] or sha(mask) != r['current_mask_sha256']:
            continue
        latest = originals[r['source_sha256']]
        if any(latest.get(k) in heldout for k in ('raw_sha256', 'input_sha256', 'prepared_sha256', 'pixel_sha256') if latest.get(k)):
            continue
        text = '|'.join(str(r.get(k, '')) for k in ('sampling_group', 'physical_sample_id', 'raw_source_path')).lower()
        height = next((h for h in ('high', 'medium', 'low') if h in text), 'other')
        if height == 'other' and 'mid' in text:
            height = 'medium'
        foil = 'without' if any(t in text for t in ('no-', 'no_', 'without', 'nohave', 'nofoil', 'not-have')) else 'with' if any(t in text for t in ('have', 'with_foil')) else 'unknown'
        result.append({**r, 'height': height, 'foil': foil, 'latest': latest})
    return result, train, heldout


def select_candidates(candidates, per_primary=12, per_other=6):
    """Round-robin era/foil/height strata; avoid adjacent same-source frames."""
    labels = ['香肠', '可颂', '鸡翅', '薯条', '芦笋', '西兰花', '三文鱼']
    selected = []
    for label in labels:
        target = per_primary if label in labels[:3] else per_other
        pool = [r for r in candidates if r['label'] == label]
        count = 0
        eras = sorted({r['era'] for r in pool})
        for era_index, era in enumerate(eras):
            quota = target // max(1, len(eras)) + (era_index < target % max(1, len(eras)))
            available = sorted((r for r in pool if r['era'] == era), key=lambda r: r['source_sha256'])
            for slot in range(quota):
                height = ('high', 'medium', 'low')[slot % 3]
                foil = ('with', 'without')[(slot + era_index) % 2]
                choice = next((r for r in available if r['height'] == height and r['foil'] == foil), None)
                if choice is None:
                    choice = next((r for r in available if r['height'] == height), available[0] if available else None)
                if choice is None:
                    continue
                available.remove(choice)
                choice = dict(choice)
                choice['candidate_angle'] = [90, -60, 180, 60, -90, 30][count % 6]
                selected.append(choice)
                count += 1
    return selected


def write_contact_sheets(rows, prefix='foreground', batch_size=6):
    preview = RUN / 'preview'
    preview.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    paths = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        sheet = Image.new('RGB', (1800, 285 * len(batch)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(batch):
            title = f"{row['id']} {row['label']} {row['era']} {row['foil']} {row['height']} {row['angle']}° | 原图 / 旋转候选 / 掩码与擦除范围"
            draw.text((8, i * 285 + 3), title, fill='black', font=font)
            source = Image.open(row['source_path']).convert('RGB')
            out = Image.open(row['output']).convert('RGB')
            overlay = np.array(source).copy()
            erased = np.array(Image.open(row['evidence']['erase_mask_path'])) > 0
            mask = np.array(Image.open(row['mask_path'])) > 0
            overlay[erased] = (.55 * overlay[erased] + .45 * np.array((0, 150, 255))).astype(np.uint8)
            overlay[mask] = (.5 * overlay[mask] + .5 * np.array((255, 70, 0))).astype(np.uint8)
            for j, im in enumerate((source, out, Image.fromarray(overlay))):
                im.thumbnail((592, 248))
                sheet.paste(im, (j * 600, i * 285 + 32))
        dest = preview / f'{prefix}_{offset // batch_size + 1:02d}.jpg'
        sheet.save(dest, quality=95)
        paths.append(str(dest))
    return paths


def generate_samples(records=None, *, output_dir=None, angles=None, halo=6, manifest_path=None):
    """Optional records/angles allow callers to build a small review-only preview."""
    candidates, _, _ = load_candidates()
    canonical_by_sha = {r['source_sha256']: r for r in candidates}
    if records is None:
        records = select_candidates(candidates)
    output_dir = Path(output_dir) if output_dir else RUN / 'foreground_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for index, record in enumerate(records):
        angle = angles[index % len(angles)] if angles else record.get('candidate_angle', 90)
        source = Path(record['source_path'])
        mask_path = Path(record['mask_path'])
        try:
            canonical = canonical_by_sha.get(record.get('source_sha256'))
            fields = ('source_path', 'mask_path', 'current_mask_sha256', 'label',
                      'split', 'mask_usage', 'mask_status')
            if canonical is None or any(record.get(k) != canonical.get(k) for k in fields):
                raise ValueError('source or mask is not an exact verified member of current train pool')
            if record.get('mask_usage') != 'full_food_only' or record.get('mask_status') != 'verified' or record.get('split') != 'train':
                raise ValueError('source does not carry verified complete train mask')
            source_hash, mask_hash = sha(source), sha(mask_path)
            if source_hash != record['source_sha256'] or mask_hash != record['current_mask_sha256']:
                raise ValueError('source or mask SHA mismatch')
            rgb = np.array(Image.open(source).convert('RGB'))
            mask = np.array(Image.open(mask_path).convert('L')) > 0
            result = rotate_foreground(rgb, mask, angle, halo=halo)
            name = f'fg_{index + 1:03d}_{source_hash[:12]}'
            dest = output_dir / f'{name}.png'
            erase_dest = output_dir / f'{name}_erase.png'
            Image.fromarray(result['image']).save(dest)
            Image.fromarray(result['erase_mask'].astype(np.uint8) * 255).save(erase_dest)
            if sha(source) != source_hash or sha(mask_path) != mask_hash:
                raise ValueError('source or mask file changed')
            row = {'id': name, 'source_sha256': source_hash, 'label': record['label'],
                   'source_path': str(source), 'mask_path': str(mask_path), 'mask_sha256': mask_hash,
                   'output': str(dest), 'output_sha256': sha(dest), 'angle': angle,
                   'split': 'train', 'review_status': 'pending', 'user_review_status': 'pending',
                   'assistant_visual_status': 'pending', 'training_eligible': False,
                   'era': record['era'], 'height': record.get('height'), 'foil': record.get('foil'),
                   'source_group': record.get('source_group'), 'sampling_group': record.get('sampling_group'),
                   'source_session': record.get('source_session'), 'physical_sample_id': record.get('physical_sample_id'),
                   'kind': 'foreground_rotation',
                   'evidence': {'annotation_source': str(OLD), 'effective_annotation_source': str(EFFECTIVE),
                                'mask_status': record['mask_status'], 'mask_usage': record['mask_usage'],
                                'mask_id': record.get('mask_id'), 'prior_mask_evidence': record.get('mask_evidence'),
                                'current_train_membership': str(LATEST / 'training_base.json'),
                                'source_file_unchanged': True, 'mask_file_unchanged': True,
                                'output_image_size': [rgb.shape[1], rgb.shape[0]],
                                'center': result['center'], 'source_box': result['source_box'],
                                'erase_mask_path': str(erase_dest), 'erase_mask_sha256': sha(erase_dest),
                                'erased_pixels': int(result['erase_mask'].sum()),
                                'changed_pixels': result['changed_pixels'],
                                'outside_allowed_changed_pixels': result['outside_allowed_changed_pixels'],
                                'halo_pixels': result['halo_pixels'], 'reflection_radius': result['reflection_radius'],
                                'texture_source_offset': result['texture_source_offset'],
                                'background_method': 'same-train-frame reflection exclusion, harmonic illumination and clean translated tray texture',
                                'food_photometric_transform': 'none', 'scale': 1.0}}
            rows.append(row)
            print(f"GENERATED {name} {record['label']} {angle}", flush=True)
        except (ValueError, OSError) as error:
            skipped.append({'source_sha256': record.get('source_sha256'), 'label': record.get('label'), 'reason': str(error)})
            print(f"SKIPPED {record.get('source_sha256', '')[:12]} {error}", flush=True)
    manifest = {'schema': 'foreground-rotation-candidates-v1', 'sources': rows, 'skipped': skipped,
                'policy': {'review_status': 'pending', 'training_eligible': False, 'original_files_modified': False,
                           'heldout_usage': 'manifest hashes for exclusion only', 'fixed_background': True,
                           'color_or_lighting_change': False},
                'contact_sheets': write_contact_sheets(rows, output_dir.name)}
    manifest_path = Path(manifest_path) if manifest_path else RUN / 'foreground_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    candidates, _, _ = load_candidates()
    selected = select_candidates(candidates)
    (RUN / 'foreground_candidates.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.preview:
        selected = [next(r for r in selected if r['label'] == label) for label in ('香肠', '可颂', '鸡翅', '三文鱼')]
        generate_samples(selected, output_dir=RUN / 'foreground_representatives', manifest_path=RUN / 'foreground_preview_manifest.json')
    else:
        generate_samples(selected)
