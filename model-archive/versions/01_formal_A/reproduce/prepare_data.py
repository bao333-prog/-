"""为本次正式A固定数据版本：原图只读，筛选、分组、划分均保留记录。"""
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse, unquote
import hashlib
import itertools
import json
import random
import re

import numpy as np
from PIL import Image, ImageOps
from oven_food.manifest import ManifestRecord, validate_manifest, write_manifest

OUT = Path(__file__).resolve().parent
ROWS = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
CLASSES = ["牛排", "鸡翅", "南瓜", "吐司", "玉米", "西兰花"]
# 编号对应inventory内的class/index，不是原始文件名中的数字。
EXCLUDED = {
    "鸡翅_001": "部位疑似鸡腿，不能可靠确认为翅根，暂排除",
    "鸡翅_010": "外观为长骨鸡腿，标签疑点",
    "鸡翅_012": "外观为长骨鸡腿，标签疑点",
    "鸡翅_015": "部位疑似鸡腿，不能可靠确认为翅根，暂排除",
    "鸡翅_019": "外观为长骨鸡腿，标签疑点",
    "鸡翅_022": "有大面积文字标识，避免模型学习文字捷径",
    "鸡翅_034": "明显过曝，主体细节难辨",
    "鸡翅_052": "带明显商标文字，暂排除",
    "南瓜_131": "半瓜切面深褐色，熟制状态疑点",
    "南瓜_149": "来源记录标为Illustration，真实性未核实，暂排除",
    "玉米_029": "颗粒皱缩，食品状态疑点；与小样本复核标准一致",
    "西兰花_106": "来源记录明确标为插画资产，暂排除",
}
COOKED = re.compile(r"\b(?:boiled|cooked|roasted|grilled|steamed|precooked|pre-cooked)\b", re.I)
for row in ROWS:
    title = row["description"].split("\n")[0]
    if row["class"] == "玉米" and COOKED.search(title):
        EXCLUDED[row["id"]] = "来源记录标题指向熟制或预煮玉米，暂排除：" + title

# 用并查集合并同域来源和近似图片：宁可保守合组，也不让已发现的同源跨集合。
parent = {row["id"]: row["id"] for row in ROWS}
def find(key):
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key

edges = []
def union(left, right, reason):
    if find(left) != find(right):
        parent[find(right)] = find(left)
    edges.append({"left": left, "right": right, "reason": reason})

def rgb_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")

def canonical_image(image):
    """仅用于近重复筛查：按与角落背景的差异估计主体框，不修改训练原图。"""
    small = ImageOps.contain(image, (256, 256))
    pixels = np.asarray(small).astype(np.int16)
    corners = np.array([pixels[0, 0], pixels[-1, 0], pixels[0, -1], pixels[-1, -1]])
    background = np.median(corners, axis=0)
    mask = np.max(np.abs(pixels - background), axis=2) > 35
    ys, xs = np.nonzero(mask)
    if len(xs):
        small = small.crop((max(0, int(xs.min())-2), max(0, int(ys.min())-2),
                            min(small.width, int(xs.max())+3), min(small.height, int(ys.max())+3)))
    return ImageOps.pad(small, (32, 32), color="white")

# 64位感知哈希只产生相似性证据；低阈值和像素差联合使用，不能证明独立来源。
cosine = np.cos(np.pi / 32 * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None])
def phash(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low = cosine @ gray @ cosine.T
    bits = low.ravel() > np.median(low.ravel()[1:])
    return sum(int(bit) << i for i, bit in enumerate(bits))

seen = {}
features = {}
for row in ROWS:
    image = rgb_image(row["path"])
    canonical = canonical_image(image)
    pixels = np.asarray(canonical, dtype=np.float32) / 255
    features[row["id"]] = (phash(canonical), phash(ImageOps.mirror(canonical)), pixels)
    for name, value in [("file_sha256", row["sha256"]), ("original_sha256", row["original_sha256"])]:
        if not value:
            continue
        key = (row["class"], name, value)
        if key in seen:
            union(seen[key], row["id"], name)
        seen[key] = row["id"]
    page = row["page_url"]
    if page.startswith("http"):
        # 同一个网站的商品多角度图片和转载有潜在关系，按域名保守归组。
        key = (row["class"], "source_domain", urlparse(page).hostname)
    elif page:
        # 旧数据来源没有拍摄批次，整个旧来源先视为一组。
        key = (row["class"], "legacy_source", page)
    else:
        continue
    if key in seen:
        union(seen[key], row["id"], str(key[1:]))
    seen[key] = row["id"]

similar_pairs = []
for label in CLASSES:
    rows = [r for r in ROWS if r["class"] == label]
    for left, right in itertools.combinations(rows, 2):
        a, _, p = features[left["id"]]
        b, bm, q = features[right["id"]]
        distance = min((a ^ b).bit_count(), (a ^ bm).bit_count())
        if distance <= 7:
            mae = float(min(np.abs(p-q).mean(), np.abs(p-q[:, ::-1]).mean()))
            if distance <= 4 or mae < 0.075:
                union(left["id"], right["id"], f"visual_similarity phash={distance},mae={mae:.4f}")
                similar_pairs.append({"left": left["id"], "right": right["id"], "phash_distance": distance, "pixel_mae": mae})

# 逐张预览发现的保守视觉组，特别是没有来源元数据的鸡翅。
manual = {"鸡翅": [[17,83], [38,78,82], [44,62,68], [46,89]],
          "南瓜": [[9,38], [52,53], [157,159], [18,156]],
          "西兰花": [list(range(170,186))]}
for label, groups in manual.items():
    for ids in groups:
        for idx in ids[1:]:
            union(f"{label}_{ids[0]:03d}", f"{label}_{idx:03d}", "manual_conservative_visual_group")

grouped = defaultdict(list)
for row in ROWS:
    if row["id"] not in EXCLUDED:
        grouped[find(row["id"])].append(row)

def prior_test(row):
    path = unquote(row["image_url"]).replace("\\", "/").lower()
    return "/test/" in path

rng = random.Random(42)
assignments = {}
for label in CLASSES:
    groups = [g for g in grouped.values() if g[0]["class"] == label]
    total = sum(map(len, groups))
    forced = [g for g in groups if any(prior_test(r) for r in g)]
    for group in forced:
        assignments[find(group[0]["id"])] = "test"
    remaining = [g for g in groups if g not in forced]
    rng.shuffle(remaining)
    # 以整组做子集和选择，先测试再验证，尽量接近15%，其余训练。
    def choose(pool, target):
        possible = {0: []}
        for i,g in enumerate(pool):
            for n, ids in list(possible.items()):
                possible.setdefault(n+len(g), ids+[i])
        options = [(n,ids) for n,ids in possible.items() if n > 0 and len(ids) < len(pool)]
        if not options:
            raise ValueError(f"{label}同源组不足，无法独立划分")
        return min(options, key=lambda item: abs(item[0]-target))[1]
    already_test = sum(map(len, forced))
    if already_test < round(total*.15):
        ids = choose(remaining, round(total*.15)-already_test)
        for i in ids:
            assignments[find(remaining[i][0]["id"])] = "test"
        remaining = [g for i,g in enumerate(remaining) if i not in ids]
    ids = choose(remaining, round(total*.15))
    for i,g in enumerate(remaining):
        assignments[find(g[0]["id"])] = "val" if i in ids else "train"

records = []
review = []
for row in ROWS:
    if row["id"] in EXCLUDED:
        review.append({**row, "included": False, "reason": EXCLUDED[row["id"]]})
        continue
    group = find(row["id"])
    split = assignments[group]
    destination = OUT / "prepared_images" / row["class"] / (row["id"] + ".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.pad(rgb_image(row["path"]), (224,224), method=Image.Resampling.BILINEAR, color="white")
    image.save(destination)
    records.append(ManifestRecord(str(destination), row["class"], "proxy", group,
                                  "unknown_web", "unknown_web", "none", "none", split))
    review.append({**row, "included": True, "split": split, "group": group, "prepared_path": str(destination)})

validate_manifest(records, check_files=True)
write_manifest(OUT / "manifest.csv", records)
summary = {label: {s: sum(r.food_label==label and r.split==s for r in records)
                   for s in ["train","val","test"]} for label in CLASSES}
assert all(all(n>0 for n in counts.values()) for counts in summary.values())
group_splits = defaultdict(set)
for r in records:
    group_splits[r.session_id].add(r.split)
assert all(len(s)==1 for s in group_splits.values())
file_splits = defaultdict(set)
for r in review:
    if r["included"]:
        file_splits[r["sha256"]].add(r["split"])
assert all(len(s)==1 for s in file_splits.values())

for name,value in [("review_and_split.json",review), ("grouping_edges.json",edges),
                   ("similar_pairs.json",similar_pairs), ("split_summary.json",summary)]:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
config = dict(mode="train", experiment_name="formal_A_six_class_web_v1",
              manifest=str(OUT/"manifest.csv"), data_root=str(OUT), output_dir=str(OUT/"training"),
              train_environments=["proxy"], validation_environments=["proxy"],
              augmentation="baseline", model_name="mobilenet_v3_small", image_size=224,
              batch_size=16,num_workers=0,epochs=20,learning_rate=0.0003,weight_decay=0.0001,
              seed=42,device="cpu",pretrained=True,freeze_backbone=False)
(OUT/"train_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
with (OUT/"progress.md").open("a",encoding="utf-8") as f:
    f.write("\n全部1090张已解码并查看联系表，疑点排除原因逐张记录。按来源域名、旧来源整体、低阈值感知哈希及人工疑似同源合组。\n")
    f.write("旧来源test归组后只进test；缺少来源的鸡翅不能保证来源独立，结果必须声明限制。\n")
    f.write(json.dumps(summary,ensure_ascii=False)+"\n")
print(json.dumps({"included":len(records),"excluded":len(EXCLUDED),"groups":len(grouped),
                  "similar_pairs":len(similar_pairs),"summary":summary},ensure_ascii=False,indent=2))
