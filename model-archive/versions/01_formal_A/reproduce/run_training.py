"""执行一次固定配置的正式A训练；不会续训试跑模型，也不会读取测试集。"""
from pathlib import Path
import hashlib
import json
import time
import traceback

import torch
import torchvision
from oven_food.config import load_experiment_config
import oven_food.train as training

OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)

# 在本次运行中检查损失/梯度；发现NaN或无穷大时在更新参数前停止。
# 只包裹现有训练入口，不改变项目其他实验的行为。
BaseLoss = torch.nn.CrossEntropyLoss
class CheckedLoss(BaseLoss):
    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("损失出现NaN或无穷大，停止本次训练")
        return loss
torch.nn.CrossEntropyLoss = CheckedLoss
original_build = training.build_model
def checked_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    def check_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("梯度出现NaN或无穷大，停止本次训练")
        return gradient
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_hook(check_gradient)
    return model
training.build_model = checked_build

started = time.perf_counter()
result = {"status": "running", "stage": "formal_A", "torch": torch.__version__,
          "torchvision": torchvision.__version__, "device": "cpu", "threads": 4}
try:
    config = load_experiment_config(OUT / "train_config.json")
    assert config.checkpoint is None and config.pretrained and config.epochs == 20
    checkpoint_path = training.train_experiment(config)
    # 保存输入约定，避免别人只拿权重却用错图片预处理或类别顺序。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["stage"] = "formal_A_web_v1"
    checkpoint["data_version"] = OUT.name
    checkpoint["manifest_sha256"] = hashlib.sha256((OUT / "manifest.csv").read_bytes()).hexdigest()
    checkpoint["preprocessing"] = {
        "exif_transpose": True, "alpha_background": "white", "resize": "aspect_ratio_pad",
        "size": 224, "resample": "bilinear", "padding_color": "white",
        "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]}
    torch.save(checkpoint, checkpoint_path)
    result.update(status="completed", checkpoint=str(checkpoint_path.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status="failed", error=repr(exc))
    traceback.print_exc()
    raise
finally:
    inventory = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    result["original_image_hashes_unchanged"] = all(
        hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
        for row in inventory)
    result["training_seconds"] = time.perf_counter() - started
    (OUT / "execution.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)
