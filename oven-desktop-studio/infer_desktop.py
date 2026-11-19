import json
from hashlib import sha256
from pathlib import Path
import sys
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from oven_food.augment import prepare_image

class Classifier:
    def __init__(self):
        root = Path(__file__).resolve().parent
        model_root = Path(os.environ.get("OVENDESK_MODEL_DIR", str(root)))
        model = model_root / "food.onnx"
        self.meta = json.loads((model_root / "food.json").read_text(encoding="utf-8-sig"))
        names = self.meta["names"]
        if not isinstance(names, list) or len(names) < 2 or any(
            not isinstance(name, str) or not name.strip() for name in names
        ) or len(set(names)) != len(names):
            raise ValueError("类别名称必须是至少两个不重复的非空名称")
        if not isinstance(self.meta["size"], int) or not 16 <= self.meta["size"] <= 2048:
            raise ValueError("输入尺寸必须在 16 到 2048 之间")
        if self.meta["resize_mode"] not in ("letterbox", "center_crop"):
            raise ValueError("不支持该图片处理方式")
        self.model_hash = sha256(model.read_bytes()).hexdigest()
        self.preprocessing_hash = sha256(
            (root / "oven_food" / "augment.py").read_bytes()
            + Path(__file__).read_bytes()
        ).hexdigest()
        self.mean = np.asarray(self.meta["mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["std"], dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("mean/std 必须是三个有效数值，std 必须大于零")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.runtime = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )
        inputs = self.runtime.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("当前支持单张 RGB 图片、float32 NCHW 输入的分类模型")
        self.input_name = inputs[0].name
        self.version = self.meta.get("version", model_root.name)

    def classify(self, source, diagnostic=None):
        if source is None:
            # A repeatable pipeline check without assuming a food label.
            image = Image.new("RGB", (320, 180), "#6080a0")
            prepared = self.prepare(image)
        else:
            with Image.open(source) as image:
                prepared = self.prepare(image)
        pixels = np.asarray(prepared, dtype=np.float32) / 255
        normalized = (pixels - self.mean) / self.std
        tensor = normalized.transpose(2, 0, 1)[None].copy()
        if diagnostic:
            prepared.save(diagnostic, format="PNG")
        output = self.runtime.run(None, {self.input_name: tensor})[0]
        if output.shape != (1, len(self.meta["names"])):
            raise ValueError("模型输出形状与类别配置不匹配")
        logits = output[0]
        if len(logits) != len(self.meta["names"]) or not np.isfinite(logits).all():
            raise ValueError("分类输出无效")
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        candidates = [
            {"name": self.meta["names"][index], "score": float(scores[index])}
            for index in scores.argsort()[-3:][::-1]
        ]
        result = {
            "candidates": candidates,
            "modelSha256": self.model_hash,
            "preprocessingSha256": self.preprocessing_hash,
            "tensorShape": list(tensor.shape),
            "classes": self.meta["names"],
            "inputSize": list(prepared.size),
            "resizeMode": self.meta["resize_mode"],
            "modelVersion": self.version,
            "metadataSha256": sha256(json.dumps(self.meta, sort_keys=True).encode()).hexdigest(),
            "tensorSha256": sha256(tensor.tobytes()).hexdigest(),
            "sourceSha256": sha256(Path(source).read_bytes()).hexdigest() if source else None,
        }
        return result

    def prepare(self, image):
        return prepare_image(
            image, image_size=self.meta["size"], resize_mode=self.meta["resize_mode"]
        )


def main():
    classifier = Classifier()
    if len(sys.argv) == 4 and sys.argv[1] == "--json":
        result = classifier.classify(sys.argv[2])
        Path(sys.argv[3]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if sys.argv[1:] == ["--worker"]:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                reply = classifier.classify(request["source"], request.get("diagnostic"))
            except Exception as exc:
                reply = {"error": str(exc) or type(exc).__name__}
            print(json.dumps(reply), flush=True)
        return

    for candidate in classifier.classify(sys.argv[1])["candidates"]:
        print(f"{candidate['name']}\t{candidate['score']:.6f}")


if __name__ == "__main__":
    main()
