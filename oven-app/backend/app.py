"""本地同源网页与推理接口；图片仅在内存中处理。"""
import io
import os
import threading
import warnings
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
import torch
from predict import load_classifier, image_tensor

ROOT = Path(__file__).resolve().parents[1]

def create_app(model_path=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 11 * 1024 * 1024
    torch.set_num_threads(4)
    selected_path = Path(model_path or os.environ.get('OVEN_MODEL', ROOT / 'models/model_C_v2.pt'))
    model, names, checkpoint = load_classifier(selected_path)
    model_version = selected_path.stem
    lock = threading.Lock()

    @app.get('/')
    def index():
        return send_from_directory(ROOT / 'frontend', 'index.html')

    @app.get('/inference.js')
    def script():
        return send_from_directory(ROOT / 'frontend', 'inference.js')

    @app.get('/api/classes')
    def classes():
        return jsonify(classes=names, model=model_version, version=model_version, name='本机食材分类模型')

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='图片不能超过 10 MB'), 413

    @app.post('/api/predict')
    def predict():
        upload = request.files.get('image')
        if upload is None:
            return jsonify(error='请选择图片'), 400
        content = upload.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            return jsonify(error='图片不能超过 10 MB'), 413
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as picture:
                    if picture.format not in {'JPEG', 'PNG', 'WEBP'}:
                        return jsonify(error='仅支持 JPG、PNG、WebP'), 400
                    if picture.width * picture.height > 20_000_000:
                        return jsonify(error='图片像素总数不能超过两千万'), 400
                    tensor = image_tensor(picture, checkpoint)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return jsonify(error='图片损坏或尺寸不受支持'), 400
        try:
            with lock, torch.inference_mode():
                started = time.perf_counter()
                probabilities = model(tensor).softmax(1)[0]
                inference_ms = (time.perf_counter() - started) * 1000
            if not torch.isfinite(probabilities).all():
                raise ValueError('non-finite prediction')
            scores, indices = probabilities.topk(min(3, len(names)))
            candidates = [{'name': names[i], 'confidence': float(p)} for i, p in zip(indices.tolist(), scores.tolist())]
            return jsonify(candidates=candidates, requires_confirmation=True, model=model_version,
                           model_version=model_version, timing={'inference_ms': inference_ms},
                           uncertain=candidates[0]['confidence'] < 0.8,
                           note='预测置信度不是准确率；模型不能可靠判断未知食材，请人工确认。')
        except Exception:
            app.logger.exception('Inference failed')
            return jsonify(error='预测失败，请重试或手动确认'), 500
    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=int(os.environ.get('OVEN_PORT', '8000')), debug=False)
