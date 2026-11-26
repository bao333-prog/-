import os, json, glob, csv, re, itertools, random, collections
from pathlib import Path

ROOT = Path("按食材分类")
OUT = Path("experiments/dataset")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)

# 类名 -> 英文前缀（用于报告）
EN = {"三文鱼":"salmon","可颂":"croissant","吐司":"toast","土豆":"potato","培根":"bacon",
      "牛排":"beef","玉米":"corn","芦笋":"asparagus","薯条":"chips","西兰花":"broccoli",
      "香肠":"sausage","鸡翅":"wings"}

def fire_of(p):
    p = p.lower()
    if "high" in p: return "high"
    if "mid" in p: return "mid"
    if "low" in p: return "low"
    return "unknown"

def steam_of(p):
    p = p.lower()
    if "without_foil" in p or re.search(r'(^|[_-])no([_-]|$)', p): return "none"
    if "with_foil" in p or re.search(r'(^|[_-])have([_-]|$)', p): return "medium"
    return "none"

records = []
for cls_dir in sorted(ROOT.iterdir()):
    if not cls_dir.is_dir(): continue
    cls = cls_dir.name
    for img in sorted(cls_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        jp = img.with_suffix(".json")
        meta = {}
        if jp.exists():
            meta = json.load(open(jp, encoding="utf-8-sig"))
        cc = meta.get("camera_context") or {}
        prefix = meta.get("sample_prefix") or re.sub(r'[_-]?\d+$', '', img.stem)
        records.append({
            "image_path": str(img),
            "food_label": cls,
            "environment": "target",
            "session_id": prefix,          # 物理样本组，代码要求一组只含一类、只在一个 split
            "camera_position": cc.get("placement") or "center",
            "lighting_condition": fire_of(prefix),
            "steam_level": steam_of(prefix),
            "occlusion_level": "none",
            "split": "",
            # 额外溯源列，训练代码会忽略
            "sample_prefix": prefix,
            "capture_session": cc.get("session_id") or "",
            "label_status": meta.get("label_status") or "missing_json",
            "human_label_confirmed": str(cc.get("human_label_confirmed", "")),
            "candidate_class_key_unreviewed": cc.get("candidate_class_key_unreviewed") or "",
        })

# 按类/组聚合
by_class = collections.defaultdict(lambda: collections.defaultdict(list))
for r in records:
    by_class[r["food_label"]][r["sample_prefix"]].append(r)

def best_subset(sizes, target, min_count, forbid=()):
    """枚举组子集，使总图片数最接近 target，且至少 min_count 张、至少一组。"""
    keys = list(sizes)
    best = None
    n = len(keys)
    for mask in range(1, 1 << n):
        chosen = [keys[i] for i in range(n) if mask >> i & 1]
        if any(k in forbid for k in chosen): continue
        total = sum(sizes[k] for k in chosen)
        if total < min_count: continue
        score = (abs(total - target), -len(chosen), total)
        if best is None or score < best[0]:
            best = (score, chosen, total)
    return best[1], best[2]

summary = {}
for cls, groups in sorted(by_class.items()):
    total_n = sum(len(v) for v in groups.values())
    target_test = round(total_n * 0.15)
    target_val = round(total_n * 0.15)
    sizes = {k: len(v) for k, v in groups.items()}
    test_groups, test_n = best_subset(sizes, target_test, min(10, target_test))
    val_groups, val_n = best_subset(sizes, target_val, min(10, target_val), forbid=set(test_groups))
    train_groups = [k for k in sizes if k not in test_groups and k not in val_groups]
    train_n = sum(sizes[k] for k in train_groups)

    for k in test_groups:
        for r in groups[k]: r["split"] = "test"
    for k in val_groups:
        for r in groups[k]: r["split"] = "val"
    for k in train_groups:
        for r in groups[k]: r["split"] = "train"

    summary[cls] = {
        "total": total_n, "groups": len(sizes),
        "train": {"n": train_n, "groups": train_groups},
        "val": {"n": val_n, "groups": val_groups},
        "test": {"n": test_n, "groups": test_groups},
        "ratio": f"{train_n/total_n:.0%}/{val_n/total_n:.0%}/{test_n/total_n:.0%}",
    }

# 写 CSV
fields = ["image_path","food_label","environment","session_id","camera_position",
          "lighting_condition","steam_level","occlusion_level","split",
          "sample_prefix","capture_session","label_status","human_label_confirmed",
          "candidate_class_key_unreviewed"]
with open(OUT/"manifest.csv","w",encoding="utf-8-sig",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(records)

json.dump(summary, open(OUT/"split_summary.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

# 控制台摘要
print("总图片", len(records), "类", len(summary))
print(f"{'类':<8}{'总数':>5}{'组':>3} | {'train':>14} {'val':>12} {'test':>12} | 比例")
for cls, s in summary.items():
    print(f"{cls:<8}{s['total']:>5}{s['groups']:>3} | "
          f"{s['train']['n']:>4}张/{len(s['train']['groups'])}组 "
          f"{s['val']['n']:>4}张/{len(s['val']['groups'])}组 "
          f"{s['test']['n']:>4}张/{len(s['test']['groups'])}组 | {s['ratio']}")
tot = collections.Counter(r["split"] for r in records)
print("\n总计:", dict(tot), f"({tot['train']+tot['val']+tot['test']} 张)")
print("已写", OUT/"manifest.csv")
