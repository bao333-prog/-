"""独立单图预测：类别顺序、输入尺寸与补边方式都来自已冻结权重。"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ['南瓜', '吐司', '牛排', '玉米', '西兰花', '鸡翅', '芦笋',
               '培根', '三文鱼', '整鸡', '土豆', '香肠', '薯条', '可颂']


def require_device(device, torch):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('请求的GPU不可用，已停止；不会自动改用CPU')
    if str(device) != 'cpu' and not str(device).startswith('cuda'):
        raise ValueError('device仅支持cpu或cuda设备')
    return device


def validate_payload(payload):
    if payload.get('class_names') != CLASS_NAMES:
        raise ValueError('权重必须严格包含官方14类及固定顺序')
    if payload.get('image_size') not in (224, 320):
        raise ValueError('仅支持已审核的224或320输入')
    if payload.get('preprocessing', {}).get('resize_mode') != 'letterbox':
        raise ValueError('权重必须明确记录letterbox预处理')
    if payload.get('model_name', 'mobilenet_v3_small') != 'mobilenet_v3_small':
        raise ValueError('不支持的模型结构')


def validate_arm_payload(payload, arm, selection):
    """只检查已保存元数据，不运行推理；选模、最终评分和交付共用此契约。"""
    validate_payload(payload)
    expected_size = {'A224':224, 'B320':320, 'C320':320}.get(arm)
    if expected_size is None or payload['image_size'] != expected_size:
        raise ValueError('分支与checkpoint输入分辨率不一致: '+str(arm))
    if selection.get('arm',arm) != arm:
        raise ValueError('selection分支不一致')
    if (selection.get('variant') not in ('raw','ema','ema_bn') or
            payload.get('variant') != selection['variant']):
        raise ValueError('checkpoint变体与选定raw/EMA/BN不一致')
    if (type(payload.get('epoch')) is not int or payload['epoch'] < 1 or
            payload['epoch'] != selection.get('epoch')):
        raise ValueError('checkpoint与选定epoch不一致')
    if payload.get('selection_key') != selection.get('key'):
        raise ValueError('checkpoint与选定验证键不一致')
    if payload.get('selection_stats') != selection.get('stats'):
        raise ValueError('checkpoint与选定逐类验证统计不一致')
    if set(selection.get('stats',{}).get('per_class',{})) != set(CLASS_NAMES):
        raise ValueError('选模统计必须保留官方14类，缺样本类计数为0')
    loss = payload.get('validation_loss')
    if not isinstance(loss,(int,float)) or not math.isfinite(loss) or not math.isclose(loss,selection.get('loss',-1),rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('checkpoint与选定validation_loss不一致')
    config = payload.get('config',{})
    if config.get('image_size',expected_size) != expected_size or config.get('resize_mode','letterbox') != 'letterbox':
        raise ValueError('checkpoint内部config与分支预处理不一致')


def source_modules(code_root=None):
    root = Path(code_root) if code_root else HERE / ('code' if (HERE/'code').is_dir() else 'source_snapshot')
    root = root.resolve()
    if not (root/'oven_food/model.py').is_file():
        raise FileNotFoundError('缺少随包模型代码: ' + str(root))
    sys.path.insert(0, str(root))
    import oven_food
    if not Path(oven_food.__file__).resolve().is_relative_to(root):
        raise RuntimeError('发现其他位置已导入的oven_food，拒绝混用源码；请在新进程运行')
    from oven_food.model import build_model
    from oven_food.augment import prepare_image
    return build_model, prepare_image


def load_predictor(checkpoint, *, device='cpu', code_root=None):
    import torch
    require_device(device, torch)
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    validate_payload(payload)
    build_model, _ = source_modules(code_root)
    model = build_model(payload.get('model_name', 'mobilenet_v3_small'), num_classes=14, pretrained=False)
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(device).eval()
    return model, payload


def prepare_tensor(image_path, payload, *, code_root=None):
    import numpy as np
    import torch
    from PIL import Image
    validate_payload(payload)
    _, prepare_image = source_modules(code_root)
    with Image.open(image_path) as image:
        prepared = prepare_image(image, image_size=payload['image_size'], resize_mode='letterbox')
    metadata = payload['preprocessing']
    mean = np.asarray(metadata.get('mean', [.485,.456,.406]), dtype=np.float32)
    std = np.asarray(metadata.get('std', [.229,.224,.225]), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('无效的归一化参数')
    pixels = (np.asarray(prepared, dtype=np.float32)/255. - mean)/std
    return torch.from_numpy(pixels.transpose(2,0,1).copy())


def predict_image(checkpoint, image_path, *, device='cpu', code_root=None):
    import torch
    model, payload = load_predictor(checkpoint, device=device, code_root=code_root)
    image = prepare_tensor(image_path, payload, code_root=code_root).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(image)
        if logits.shape != (1,14) or not torch.isfinite(logits).all():
            raise ValueError('预测输出无效')
        probabilities = logits.softmax(1)[0].cpu().tolist()
    index = max(range(14), key=probabilities.__getitem__)
    return dict(label=CLASS_NAMES[index], index=index, confidence=probabilities[index],
                probabilities=dict(zip(CLASS_NAMES, probabilities)), image_size=payload['image_size'],
                resize_mode='letterbox', device=device, variant=payload.get('variant', 'unknown'))


def main():
    parser = argparse.ArgumentParser(description='官方14类食材单图预测')
    parser.add_argument('image_path', nargs='?', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=HERE/'model.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.image is not None and args.image_path is not None:
        parser.error('图片路径只能用位置参数或--image之一')
    image = args.image or args.image_path
    if image is None:
        parser.error('请提供图片路径')
    import torch
    torch.set_num_threads(4)
    print(json.dumps(predict_image(args.checkpoint, image, device=args.device), ensure_ascii=False))


if __name__ == '__main__':
    main()
