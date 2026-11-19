import sys, torch
pkg, cfg_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, pkg)
from oven_food.config import load_experiment_config
from oven_food.manifest import read_manifest, validate_manifest
from oven_food.dataset import FoodImageDataset
from oven_food.model import build_model

cfg = load_experiment_config(cfg_path)
records = validate_manifest(read_manifest(cfg.manifest), data_root=cfg.data_root, check_files=True)
classes = sorted({r.food_label for r in records})
train = [r for r in records if r.split == "train"][:8]
ds = FoodImageDataset(train, data_root=cfg.data_root, class_names=classes,
                      augmentation_profile=cfg.augmentation, image_size=224,
                      seed=42, resize_mode="letterbox")
dl = torch.utils.data.DataLoader(ds, batch_size=8)
ckpt = torch.load(cfg.init_backbone_from, map_location="cpu", weights_only=False)
state = {k: v for k, v in ckpt["model_state"].items() if not k.startswith("classifier.")}
model = build_model("mobilenet_v3_small", num_classes=len(classes), pretrained=False)
missing, unexpected = model.load_state_dict(state, strict=False)
print("classes", len(classes), classes)
print("missing(non-classifier)", [k for k in missing if not k.startswith("classifier.")])
print("unexpected", list(unexpected))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(dev)
x, y = next(iter(dl)); x, y = x.to(dev), y.to(dev)
out = model(x); loss = torch.nn.functional.cross_entropy(out, y); loss.backward()
print("device", dev, "logits", tuple(out.shape), "loss", round(float(loss), 4))
print("SMOKE OK")
