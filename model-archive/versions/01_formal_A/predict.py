"""单图预测示例：python predict.py --model models/model_A.pt --image 食物.jpg。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from oven_food.model import build_model, checkpoint_class_names

def load_classifier(model_path):
    """先重建网络，再读取训练参数；类别顺序必须以权重文件为准。"""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    names = checkpoint_class_names(checkpoint)
    if checkpoint.get("preprocessing", {}).get("resize") != "aspect_ratio_pad":
        raise ValueError("此示例仅支持本次正式A的补边预处理权重")
    model = build_model(checkpoint["model_name"], num_classes=len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()  # 预测时关闭Dropout，BatchNorm使用训练时累积的统计量。
    return model, names, checkpoint

def image_tensor(image_path, preprocessing):
    """原图只读：校正方向、透明合成白底、等比例缩放补边、归一化。"""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        side = preprocessing["size"]
        image = ImageOps.pad(image, (side, side), method=Image.Resampling.BILINEAR, color="white")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(preprocessing["mean"], dtype=np.float32)
    std = np.array(preprocessing["std"], dtype=np.float32)
    # 卷积输入排列：[批量, 通道, 高, 宽]。
    return torch.from_numpy(((pixels-mean)/std).transpose(2,0,1).copy()).unsqueeze(0)

def predict_image(model, names, preprocessing, image_path):
    tensor = image_tensor(image_path, preprocessing)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    # 概率仅表示模型在已知六类之间的相对分数，不是校准后的可靠性保证。
    return {"image": str(image_path), "predicted_class": names[index],
            "top1_probability": float(probabilities[index]),
            "probabilities": {name: float(p) for name,p in zip(names,probabilities.tolist())}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="正式六类模型A单图预测")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model,names,checkpoint = load_classifier(Path(args.model))
    result = predict_image(model,names,checkpoint["preprocessing"],Path(args.image))
    print(json.dumps(result,ensure_ascii=False,indent=2))
