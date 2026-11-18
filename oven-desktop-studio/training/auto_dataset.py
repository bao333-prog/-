"""Create reproducible class-balanced splits without altering original photos."""
import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path


def prepare(source, output, validation=None):
    for train_name, val_name in (("train", "val"), ("train", "validation"), ("采集数据集", "验证数据集")):
        if (source / train_name).is_dir() and (source / val_name).is_dir():
            validation = source / val_name
            source = source / train_name
            break
    items = []
    seen = {}
    roots = [(source, None)] if validation is None else [(source, "train"), (validation, "val")]
    fixed = {}
    for root, usage in roots:
      extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
      flat = any(p.is_file() and p.suffix.lower() in extensions for p in root.iterdir())
      for category in ([root] if flat else sorted(root.iterdir())):
        if not category.is_dir():
            continue
        for photo in sorted(category.rglob("*")):
            if photo.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            digest = hashlib.sha256(photo.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest] != (category.name, usage):
                    raise ValueError("相同照片被标为不同食材：" + str(photo))
                continue
            seen[digest] = (category.name, usage)
            relative = photo.relative_to(category)
            batch = (usage or "auto") + "/" + category.name + "/" + (relative.parts[0] if len(relative.parts) > 1 else digest)
            grouped = len(relative.parts) > 1
            sidecar = photo.with_suffix(".json")
            if sidecar.is_file():
                info = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                if info.get("schema") == "het-oven-capture/2":
                    if info.get("image_file") != photo.name or info.get("sha256") != digest:
                        raise ValueError("照片与同名 JSON 的文件名或校验值不一致：" + str(photo))
                    session = info["camera_context"]["session_id"]
                    batch = (usage or "auto") + "/" + category.name + "/" + session
                    grouped = True
            if usage:
                fixed[batch] = usage
            items.append((photo, category.name, digest, batch, grouped))
    if not items:
        raise ValueError("未找到照片，请在总文件夹内按食材名称建立子文件夹")
    splits = dict(fixed)
    rng = random.Random(42)
    for label in sorted({item[1] for item in items}) if validation is None else []:
        groups = sorted({item[3] for item in items if item[1] == label})
        rng.shuffle(groups)
        count = max(1, round(len(groups) * 0.2)) if len(groups) >= 2 else 0
        for index, group in enumerate(groups):
            splits[group] = "val" if index < count else "train"
    if "val" not in splits.values():
        raise ValueError("只有一个拍摄批次，暂时无法留出验证照片。请用“采集验证照片”另拍一批，或选择已有的独立验证集。")
    if "train" not in splits.values():
        raise ValueError("未找到训练照片，请选择按食材名称分文件夹的训练数据集")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with zipfile.ZipFile(output / "samples.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for photo, label, digest, batch, grouped in items:
            name = "images/" + digest + photo.suffix.lower()
            archive.write(photo, name)
            rows.append(dict(image=name, label=label, batch=batch, imageSha256=digest))
        archive.writestr("labels.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    (output / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    train = sum(splits[item[3]] == "train" for item in items)
    warning = "" if all(item[4] for item in items) else "\n部分照片没有拍摄批次，已按图片划分；连拍或相似图片仍可能跨集合，验证成绩需谨慎使用。"
    message = f"整理完成，去重后共 {len(items)} 张。训练 {train} 张，验证 {len(items)-train} 张。" + warning
    (output / "summary.txt").write_text(message, encoding="utf-8")
    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.source, args.output, args.validation))
    except (ValueError, OSError) as error:
        print(str(error), file=__import__("sys").stderr)
        raise SystemExit(1)


