"""Prepare replay and independent acceptance data for desktop training."""
import csv
import hashlib
import json
import os

import zipfile
from collections import Counter
from pathlib import Path

from auto_dataset import prepare as legacy_prepare

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare(source, output, validation=None, base=None):
    if base is None:
        return legacy_prepare(source, output, validation)
    names = json.loads((Path(__file__).resolve().parents[1] / "food.json").read_bytes())["names"]
    rows = []
    seen = {}
    sessions = {}
    def add(path, label, split, session):
        if label not in names:
            raise ValueError("未知食材：" + label)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != (split, label):
                raise ValueError("照片跨集合或标签冲突：" + str(path))
            return
        seen[digest] = (split, label)
        side = path.with_suffix(".json")
        if side.exists():
            meta = json.loads(side.read_bytes())
            if meta.get("schema") == "het-oven-capture/2":
                if meta.get("sha256") != digest or meta.get("image_file") != path.name:
                    raise ValueError("照片标注校验失败：" + str(path))
                session = meta["camera_context"]["session_id"]
        if session and sessions.setdefault(session, split) != split:
            raise ValueError("同一拍摄批次跨集合：" + session)
        rows.append(dict(path=str(path.resolve()), label=label, split=split, sha=digest, batch=session or digest))

    def scan(root, split):
        if not root.is_dir():
            raise ValueError("缺少数据目录：" + str(root))
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in EXT:
                continue
            parts = path.relative_to(root).parts
            label = root.name if root.name in names else parts[0]
            group = str(path.parent.resolve()) if path.parent.name != label else ""
            add(path, label, split, group)

    manifest = base / "manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                if r["split"] in ("train", "val", "test"):
                    add(base / r["image_path"], r["food_label"], r["split"], r.get("session_id"))
    else:
        for split in ("train", "val", "test"):
            scan(base / split, split)
    for split in ("train", "val", "test"):
        missing = set(names) - {r["label"] for r in rows if r["split"] == split}
        if missing:
            raise ValueError(split + " 底库缺少：" + "、".join(sorted(missing)))
    if source.resolve() != base.resolve():
        scan(source, "train")
    if validation:
        scan(validation, "val")
    output.mkdir(parents=True, exist_ok=False)
    splits = {}
    labels = []
    with zipfile.ZipFile(output / "samples.zip", "w") as archive:
        for r in rows:
            if r["split"] == "test":
                continue
            name = "images/" + r["sha"] + Path(r["path"]).suffix
            archive.write(r["path"], name)
            splits[r["batch"]] = r["split"]
            labels.append(dict(image=name, label=r["label"], batch=r["batch"]))
        archive.writestr("labels.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in labels))
    (output / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (output / "heldout.json").write_text(json.dumps([r for r in rows if r["split"]=="test"], ensure_ascii=False), encoding="utf-8")
    counts = Counter((r["label"], r["split"]) for r in rows)
    return "食材：训练 / 验证 / 独立测试\n" + "\n".join(n+"："+ " / ".join(str(counts[n,s]) for s in ("train","val","test")) for n in names)


def evaluate(index, current, candidate):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from infer_desktop import Classifier
    predictions = []
    rows = json.loads(index.read_bytes())
    for folder in (current, candidate):
        os.environ["OVENDESK_MODEL_DIR"] = str(folder)
        model = Classifier()
        results = []
        for r in rows:
            if hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha"]:
                raise ValueError("测试图片已改变：" + r["path"])
            results.append(model.classify(r["path"])["candidates"][0]["name"])
        predictions.append(results)
    report = []
    for r, old, new in zip(rows, *predictions):
        report.append(dict(image=r["path"], label=r["label"], before=old, after=new))
    lines = ["独立测试：原版正确 / 新版正确 / 总张数"]
    for label in sorted({r["label"] for r in report}):
        group = [r for r in report if r["label"] == label]
        lines.append(f"{label}：{sum(r['before']==label for r in group)} / {sum(r['after']==label for r in group)} / {len(group)}")
    regressions = [r for r in report if r["before"] == r["label"] and r["after"] != r["label"]]
    lines.append(f"旧版正确、新版错误：{len(regressions)} 张")
    lines.extend(f"{r['image']}：{r['label']} → {r['after']}" for r in regressions)
    (candidate / "test-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate / "test-summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)

