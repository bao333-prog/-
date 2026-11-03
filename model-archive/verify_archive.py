"""校验归档指纹；可选在独立CPU进程中加载每版模型，不读取评测集。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def archive_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing or unsafe archive file: ' + relative)
    return path


def verify(load_models=False):
    checksums = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for relative, expected in checksums.items():
        if digest(archive_path(relative)) != expected:
            raise ValueError('SHA256 mismatch: ' + relative)
    catalog = json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))
    models = catalog['models']
    assert len(models) == catalog['trained_version_count'] == 8
    by_id = {entry['id']: entry for entry in models}
    assert len(by_id) == 8
    for entry in models:
        assert entry['trained'] is True
        assert digest(archive_path(entry['checkpoint'])) == entry['checkpoint_sha256']
        assert len(entry['class_names']) == entry['class_count']
        assert len(set(entry['class_names'])) == entry['class_count']
        parent = entry['parent_version']
        if parent:
            old_names = by_id[parent]['class_names']
            assert entry['class_names'][:len(old_names)] == old_names
    for entry in catalog['auxiliary']:
        assert entry['trained'] is False
        assert digest(archive_path(entry['checkpoint'])) == entry['checkpoint_sha256']
    provenance = json.loads((ROOT / 'provenance.json').read_text(encoding='utf-8'))
    for original in provenance:
        assert digest(archive_path(original['path'])) == original['sha256'], original['path']
    result = {'status': 'passed', 'files_hashed': len(checksums), 'original_delivery_members': len(provenance),
              'trained_versions': len(models), 'untrained_auxiliary': len(catalog['auxiliary']),
              'test_set_scored': False, 'model_checks': []}
    if load_models:
        # 每版单独进程和源码路径，防止不同版本的同名模块互相污染。
        worker = '''
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]); entry=json.loads(sys.argv[2])
sys.path.insert(0,str(root/entry['directory']/'src'))
from oven_food.model import build_model
torch.set_num_threads(4)
cp=torch.load(root/entry['checkpoint'],map_location='cpu',weights_only=True)
assert cp['class_names']==entry['class_names']
model=build_model(cp['model_name'],num_classes=len(cp['class_names']),pretrained=False)
model.load_state_dict(cp['model_state'],strict=True); model.eval()
side=cp['image_size']
with torch.inference_mode():
    outputs=model(torch.linspace(-1,1,3*side*side).reshape(1,3,side,side))
    probabilities=outputs.softmax(1)
assert list(outputs.shape)==[1,len(cp['class_names'])]
assert torch.isfinite(outputs).all() and torch.isfinite(probabilities).all()
assert abs(float(probabilities.sum())-1)<1e-5
print(json.dumps({'id':entry['id'],'status':'passed','classes':len(cp['class_names']),
 'epoch':cp['epoch'],'parameters':sum(p.numel() for p in model.parameters()),
 'input':'synthetic_tensor','accuracy_measured':False}))
'''
        for entry in models:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
            env.pop('PYTHONPATH', None)
            process = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', worker, str(ROOT),
                                      json.dumps(entry, ensure_ascii=False)], cwd=ROOT, env=env,
                                     capture_output=True, text=True, encoding='utf-8', timeout=120)
            if process.returncode:
                raise RuntimeError(entry['id'] + ': ' + process.stderr)
            result['model_checks'].append(json.loads(process.stdout))
        # 辅助文件格式与训练检查点不同，仅检查其声明与状态字典，不当识别模型。
        import torch
        for entry in catalog['auxiliary']:
            cp = torch.load(archive_path(entry['checkpoint']), map_location='cpu', weights_only=True)
            assert cp['trained'] is False and cp['class_names'] == entry['class_names'] and cp['state_dict']
        result['untrained_metadata_checked'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-models', action='store_true')
    parser.add_argument('--output', type=Path, help='可选本地检查记录；建议保存到归档目录外')
    args = parser.parse_args()
    result = verify(args.load_models)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)
