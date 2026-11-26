import sys, os, json, csv, collections
sys.path.insert(0, "experiments/B_recipe/src")
import numpy as np, torch
from PIL import Image
from oven_food.model import build_model, checkpoint_class_names, checkpoint_resize_mode
from oven_food.augment import prepare_image
from oven_food.manifest import read_manifest, validate_manifest

ROOT = os.getcwd()
OUT = "experiments/eval"
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

records = validate_manifest(read_manifest("experiments/dataset/manifest.csv"),
                            data_root=ROOT, check_files=True)
test = [r for r in records if r.split == "test"]

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = checkpoint_class_names(ck)
    m = build_model(ck.get("model_name", "mobilenet_v3_small"),
                    num_classes=len(names), pretrained=False)
    m.load_state_dict(ck["model_state"]); m.eval(); m.to(DEV)
    return ck, names, m

@torch.inference_mode()
def predict(model, rec, image_size, resize_mode):
    with Image.open(rec.image_path) as im:
        img = prepare_image(im, image_size=image_size, resize_mode=resize_mode)
    x = np.asarray(img, np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy()).unsqueeze(0).to(DEV)
    prob = torch.softmax(model(x), 1)[0]
    top = int(prob.argmax())
    return top, float(prob[top]), prob.cpu().numpy()

def prf(matrix, names):
    per = {}
    for i, name in enumerate(names):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(names))) - tp
        fn = sum(matrix[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[name] = {"precision": p, "recall": r, "f1": f, "support": sum(matrix[i])}
    macro = {k: sum(v[k] for v in per.values()) / len(per) for k in ("precision", "recall", "f1")}
    return per, macro

def metrics(trues, preds, names):
    n = len(names)
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(trues, preds): mat[t][p] += 1
    correct = sum(mat[i][i] for i in range(n))
    per, macro = prf(mat, names)
    return {
        "num_samples": len(trues),
        "overall_accuracy": correct / len(trues) if trues else None,
        "per_class": per,
        "macro": macro,
        "class_names": names,
        "confusion_matrix": mat,
    }

MODELS = [
    ("A_target12", "experiments/runs/A_target12_gentle/best.pt", "full"),
    ("B_target12", "experiments/runs/B_target12_gentle_plus/best.pt", "full"),
    ("B_plus_lr3e4", "experiments/runs/B_target12_gentle_plus_lr3e4/best.pt", "full"),
    ("A_v2_zeroshot", "六类模型A候选v2_改进代码与权重/models/A_candidate_v2.pt", "subset"),
    ("B_zeroshot", "十类B_新增增强候选_代码与权重/models/model_B_augmentation_candidate.pt", "subset"),
]

summary = {}
rows = []
for tag, path, mode in MODELS:
    ck, names, model = load(path)
    image_size = ck.get("image_size", 224)
    resize_mode = checkpoint_resize_mode(ck)
    name_to_idx = {n: i for i, n in enumerate(names)}
    selected = test if mode == "full" else [r for r in test if r.food_label in name_to_idx]
    trues, preds = [], []
    conf_all = []
    for rec in selected:
        top, conf, prob = predict(model, rec, image_size, resize_mode)
        pred_label = names[top]
        trues.append(name_to_idx[rec.food_label]); preds.append(top)
        conf_all.append(conf)
        rows.append({
            "model": tag, "file": rec.image_path, "true_label": rec.food_label,
            "pred_label": pred_label, "confidence": round(conf, 6),
            "correct": int(pred_label == rec.food_label),
            "in_model_classes": int(rec.food_label in name_to_idx),
            "sample_prefix": rec.session_id,
        })
    met = metrics(trues, preds, names)
    met.update({
        "checkpoint": path, "mode": mode,
        "best_epoch": ck.get("epoch"),
        "checkpoint_val_accuracy": ck.get("validation_accuracy"),
        "mean_confidence": float(np.mean(conf_all)) if conf_all else None,
        "image_size": image_size, "resize_mode": resize_mode,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    })
    summary[tag] = met
    print(f"{tag}: {met['num_samples']} 张, acc={met['overall_accuracy']:.4f}, "
          f"macro_f1={met['macro']['f1']:.4f}, mean_conf={met['mean_confidence']:.4f}")

# 原模型在全 test 上的预测分布（只看它们把新类预测成什么）
for tag, path, mode in MODELS:
    if mode != "subset": continue
    ck, names, model = load(path)
    c = collections.Counter()
    for rec in test:
        top, conf, prob = predict(model, rec, ck.get("image_size", 224), checkpoint_resize_mode(ck))
        c[(rec.food_label, names[top])] += 1
    summary[tag]["full_test_prediction_distribution"] = {
        f"{t}->{p}": n for (t, p), n in sorted(c.items())
    }

with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
with open(f"{OUT}/predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for tag, met in summary.items():
    with open(f"{OUT}/confusion_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred", *met["class_names"]])
        for name, row in zip(met["class_names"], met["confusion_matrix"]):
            w.writerow([name, *row])
print("已写", OUT)
