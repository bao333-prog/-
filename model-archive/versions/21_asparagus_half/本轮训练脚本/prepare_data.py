"""冻结并整合训练素材；只创建新缓存和清单，绝不修改原始照片。"""
import json
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CLEAN = ROOT / 'outputs/数据清理与分割修正_20261123_202414/清理后清单'
sys.path.insert(0, str(ROOT / 'src'))
from PIL import Image


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_id(row):
    # 网图仅保存了历史准备输入，不能拿空raw_sha256折叠全部网图。
    return row.get('raw_sha256') or row['input_sha256']


def cache(row):
    # 延后导入，避免训练进程仅导入通用路径工具时提前加载共享源码。
    from oven_food.augment import prepare_image
    source = Path(row['source'])
    digest = source_id(row)
    if sha(source) != digest:
        raise ValueError('原图指纹不符：' + str(source))
    target = RUN / 'inputs' / row['split'] / row['label'] / (digest + '_full.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = prepare_image(image, image_size=224, resize_mode='letterbox')
        pixels = hashlib.sha256(image.tobytes()).hexdigest()
        image.save(target)
    return {**row, 'family_id': digest, 'kind': 'original',
            'previous_actual_input': row['actual_input'], 'actual_input': str(target),
            'prepared_sha256': sha(target), 'prepared_pixel_sha256': pixels}


def original_data():
    collections = {}
    for role in ['training', 'validation', 'test']:
        rows = load(CLEAN / (role + '_inventory.json'))
        # 同字节原图的两个文件名不能由两个线程同时写同一缓存。
        unique = {(source_id(r), r['label']): r for r in rows}
        with ThreadPoolExecutor(max_workers=4) as pool:
            cached = list(pool.map(cache, unique.values()))
        lookup = {(source_id(r), r['label']): r for r in cached}
        prepared = []
        for row in rows:
            frozen = lookup[source_id(row), row['label']]
            prepared.append({**row, **{k: frozen[k] for k in
                ['family_id', 'kind', 'actual_input', 'prepared_sha256', 'prepared_pixel_sha256']},
                'previous_actual_input': row['actual_input']})
        dump(RUN / (role + '_base.json'), prepared)
        collections[role] = prepared
    for a, b in [('training', 'validation'), ('training', 'test'), ('validation', 'test')]:
        for field in ['family_id', 'prepared_pixel_sha256']:
            if {r[field] for r in collections[a]} & {r[field] for r in collections[b]}:
                raise ValueError(f'{a}/{b}的{field}出现交叉')
        sessions = lambda rs: {r['source_session'] for r in rs if r['environment'] == 'target'}
        if sessions(collections[a]) & sessions(collections[b]):
            raise ValueError(f'{a}/{b}已知采集会话交叉')
    snapshot = RUN / 'source_snapshot/oven_food'
    if not snapshot.exists():
        shutil.copytree(ROOT / 'src/oven_food', snapshot,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for file in ['joint_support.py', 'validation_monitor.py']:
        shutil.copy2(ROOT / 'runs/joint_optimized_20261118_202521' / file, RUN / file)
    init = ROOT / 'runs/masked_samples_20261123_135614/official_initialization.pt'
    assert sha(init) == '5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    shutil.copy2(init, RUN / 'official_initialization.pt')
    dump(RUN / 'original_data_audit.json', {
        'counts': {k: len(v) for k, v in collections.items()},
        'known_file_pixel_session_cross_split_overlap': 0,
        'physical_individual_ids_complete': False,
        'initialization_sha256': sha(RUN / 'official_initialization.pt'),
        'initialization_epoch': 0, 'inherited_task_trained_weights': False,
    })
    print('原图缓存及分组核对完成', {k: len(v) for k, v in collections.items()}, flush=True)


if __name__ == '__main__':
    original_data()
