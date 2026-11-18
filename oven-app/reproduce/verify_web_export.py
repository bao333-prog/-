"""核对网页元数据和真实图片上的 PyTorch/ONNX 数值一致性，不评测准确率。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from predict import image_tensor, load_classifier
from oven_food.model import checkpoint_resize_mode


def verify(model_path, web_dir, image_paths):
    torch.set_num_threads(4)
    model, names, checkpoint = load_classifier(model_path)
    metadata = json.loads((web_dir / 'model.json').read_text(encoding='utf-8'))
    source_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256((web_dir / 'model.onnx').read_bytes()).hexdigest()
    assert metadata['classes'] == names, 'class order differs'
    assert metadata['size'] == checkpoint['image_size'], 'input size differs'
    assert metadata['resize_mode'] == checkpoint_resize_mode(checkpoint), 'resize differs'
    assert metadata['color'] == 'RGB' and metadata['layout'] == 'NCHW'
    assert metadata['output_type'] == 'logits'
    assert metadata['sha256'] == onnx_sha, 'ONNX hash differs'
    if 'source_sha256' in metadata:
        assert metadata['source_sha256'] == source_sha, 'source checkpoint differs'
    for key, default in [('mean', [.485, .456, .406]), ('std', [.229, .224, .225])]:
        np.testing.assert_array_equal(np.asarray(metadata[key], dtype=np.float32),
                                      np.asarray(checkpoint.get('preprocessing', {}).get(key, default), dtype=np.float32))
    session = ort.InferenceSession(str(web_dir / 'model.onnx'), providers=['CPUExecutionProvider'])
    rows = []
    for path in image_paths:
        with Image.open(path) as image:
            tensor = image_tensor(image, checkpoint)
        with torch.inference_mode():
            expected = model(tensor).numpy()
        actual = session.run([metadata['output_name']], {metadata['input_name']: tensor.numpy()})[0]
        assert actual.shape == expected.shape == (1, len(names))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4, err_msg=path.name)
        index = int(expected.argmax())
        rows.append({'image': path.name, 'pytorch_top1': names[index],
                     'onnx_top1': names[int(actual.argmax())],
                     'pytorch_confidence': float(torch.from_numpy(expected).softmax(1)[0, index]),
                     'max_logit_error': float(np.abs(actual - expected).max())})
    assert rows, 'no images supplied'
    assert all(row['pytorch_top1'] == row['onnx_top1'] for row in rows), 'top-1 differs'
    return {'version': metadata['version'], 'source_sha256': source_sha, 'onnx_sha256': onnx_sha,
            'metadata_matches_checkpoint': True, 'image_count': len(rows), 'top1_matches': len(rows),
            'max_logit_error': max(row['max_logit_error'] for row in rows),
            'note': '数值一致性抽查，含重复样例；无独立真值标签，不能据此推断准确率。', 'cases': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--web-dir', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='图片文件或图片目录')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = []
    for item in args.images:
        paths.extend(sorted(p for p in item.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')) if item.is_dir() else [item])
    report = verify(args.model, args.web_dir, paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=True))
