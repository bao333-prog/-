"""模型交付包的单图预测示例；输入未画框的完整相机图片。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
sys.path.insert(0, str(HERE / 'source_snapshot'))
from oven_food.model import build_model
from oven_food.augment import prepare_image
from oven_food.dataset import IMAGENET_MEAN, IMAGENET_STD


def main():
    parser = argparse.ArgumentParser(description='完整原图食材分类，不需要人工画框')
    parser.add_argument('image')
    parser.add_argument('--checkpoint', default=str(HERE / 'model.pt'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    names = checkpoint['class_names']
    model = build_model(checkpoint['model_name'], num_classes=len(names), pretrained=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(args.device).eval()
    with Image.open(args.image) as image:
        prepared = prepare_image(image, image_size=checkpoint['image_size'], resize_mode='letterbox')
    pixels = (np.asarray(prepared, dtype=np.float32)/255.-IMAGENET_MEAN)/IMAGENET_STD
    tensor = torch.from_numpy(pixels.transpose(2,0,1).copy())[None].to(args.device)
    with torch.inference_mode():
        scores, indices = model(tensor).softmax(1)[0].topk(3)
    print(json.dumps({'image':args.image,'top3':[{'label':names[i],'confidence':p}
                    for i,p in zip(indices.cpu().tolist(),scores.cpu().tolist())]},ensure_ascii=False))


if __name__ == '__main__':main()
