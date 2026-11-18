"""Convert a training predictor's saved results to the desktop comparison format.

The source must be the exact original photo passed to that predictor.
Predictions JSON is a list of {"name": "...", "score": 0.9} entries.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Expected a nonempty list of name/score objects")
    for item in candidates:
        if not isinstance(item["name"], str) or not 0 <= item["score"] <= 1:
            raise ValueError("Invalid name or score")
    result = {
        "sourceSha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "modelSha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "modelVersion": args.weights.name,
        "candidates": sorted(candidates, key=lambda item: item["score"], reverse=True),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
