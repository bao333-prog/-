"""原图预测：与训练/评测共用prepare_image，避免裁剪或透明背景处理不一致。"""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
import torch
from oven_food.augment import prepare_image
from oven_food.dataset import IMAGENET_MEAN, IMAGENET_STD
from oven_food.model import build_model, checkpoint_class_names, checkpoint_resize_mode


def load_classifier(path):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    names=checkpoint_class_names(checkpoint)
    model=build_model(checkpoint["model_name"],num_classes=len(names),pretrained=False)
    model.load_state_dict(checkpoint["model_state"],strict=True)
    return model.eval(),names,checkpoint


def image_tensor(image,checkpoint):
    # 从权重恢复输入约定；既支持旧A，也支持新候选，不依赖调用者手工预处理。
    image=prepare_image(image,image_size=checkpoint["image_size"],resize_mode=checkpoint_resize_mode(checkpoint))
    metadata=checkpoint.get("preprocessing",{})
    mean=np.array(metadata.get("mean",IMAGENET_MEAN),dtype=np.float32)
    std=np.array(metadata.get("std",IMAGENET_STD),dtype=np.float32)
    pixels=(np.asarray(image,dtype=np.float32)/255-mean)/std
    return torch.from_numpy(pixels.transpose(2,0,1).copy()).unsqueeze(0)


def predict_image(model,names,checkpoint,path):
    with Image.open(path) as image:
        tensor=image_tensor(image,checkpoint)
    with torch.inference_mode():
        probabilities=model(tensor).softmax(1)[0]
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("non-finite prediction")
    index=int(probabilities.argmax())
    return {"image":str(path),"predicted_class":names[index],"top1_probability":float(probabilities[index]),
            "probabilities":{name:float(p) for name,p in zip(names,probabilities.tolist())}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="六类基础模型原图预测")
    parser.add_argument("--model",required=True);parser.add_argument("--image",required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    model,names,checkpoint=load_classifier(Path(args.model))
    print(json.dumps(predict_image(model,names,checkpoint,Path(args.image)),ensure_ascii=False,indent=2))
