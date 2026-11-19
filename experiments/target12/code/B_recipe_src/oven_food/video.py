from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .manifest import ManifestRecord


def average_hash(image: Image.Image, *, hash_size: int = 8) -> int:
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    grayscale = image.convert("L").resize(
        (hash_size, hash_size), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(grayscale, dtype=np.float32)
    bits = pixels > pixels.mean()
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def is_near_duplicate(
    candidate_hash: int,
    existing_hashes: Iterable[int],
    *,
    threshold: int = 4,
) -> bool:
    if not 0 <= threshold <= 64:
        raise ValueError("duplicate threshold must be between 0 and 64")
    return any(
        hamming_distance(candidate_hash, existing_hash) <= threshold
        for existing_hash in existing_hashes
    )


def extract_video_frames(
    video_path: str | Path,
    *,
    output_dir: str | Path,
    data_root: str | Path,
    food_label: str,
    environment: str,
    session_id: str,
    camera_position: str,
    lighting_condition: str,
    steam_level: str,
    occlusion_level: str,
    interval_seconds: float = 1.0,
    duplicate_threshold: int = 4,
) -> list[ManifestRecord]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            "Video extraction requires OpenCV. Install with: pip install -e .[video]"
        ) from exc

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if not 0 <= duplicate_threshold <= 64:
        raise ValueError("duplicate threshold must be between 0 and 64")

    video = Path(video_path)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        capture.release()
        raise ValueError(f"video reports an invalid frame rate: {fps}")

    root = Path(data_root).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    frame_step = max(1, round(fps * interval_seconds))
    frame_index = 0
    hashes: list[int] = []
    records: list[ManifestRecord] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb, mode="RGB")
            image_hash = average_hash(image)
            if is_near_duplicate(image_hash, hashes, threshold=duplicate_threshold):
                frame_index += 1
                continue

            filename = f"{session_id}_{frame_index:08d}.jpg"
            saved_path = destination / filename
            image.save(saved_path, format="JPEG", quality=95)
            hashes.append(image_hash)
            try:
                manifest_path = saved_path.resolve().relative_to(root).as_posix()
            except ValueError:
                manifest_path = str(saved_path.resolve())
            records.append(
                ManifestRecord(
                    image_path=manifest_path,
                    food_label=food_label,
                    environment=environment,
                    session_id=session_id,
                    camera_position=camera_position,
                    lighting_condition=lighting_condition,
                    steam_level=steam_level,
                    occlusion_level=occlusion_level,
                )
            )
            frame_index += 1
    finally:
        capture.release()
    if not records:
        raise ValueError("no frames were extracted after interval and duplicate filtering")
    return records

