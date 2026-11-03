from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_experiment_config
from .evaluate import evaluate_experiment
from .manifest import read_manifest, validate_manifest, write_manifest
from .report import build_comparison, select_best_proxy_checkpoint, write_comparison
from .split import split_by_session
from .train import train_experiment
from .video import extract_video_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oven-food",
        description="目标烤箱代理环境食材识别训练工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="校验数据清单")
    validate.add_argument("manifest")
    validate.add_argument("--data-root", default=".")
    validate.add_argument("--skip-files", action="store_true", help="不检查图片文件是否存在")

    split = subparsers.add_parser("split-manifest", help="按拍摄批次划分数据")
    split.add_argument("manifest")
    split.add_argument("output")
    split.add_argument("--data-root", default=".")
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--val-ratio", type=float, default=0.15)
    split.add_argument("--test-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--skip-files", action="store_true")

    train = subparsers.add_parser("train", help="训练代理或目标微调模型")
    train.add_argument("config")

    evaluate = subparsers.add_parser("evaluate", help="在固定测试集上评测检查点")
    evaluate.add_argument("config")

    extract = subparsers.add_parser("extract-frames", help="从实时视频抽帧并去除近重复图片")
    extract.add_argument("video")
    extract.add_argument("output_dir")
    extract.add_argument("manifest_output")
    extract.add_argument("--data-root", default=".")
    extract.add_argument("--food-label", required=True)
    extract.add_argument("--environment", choices=("proxy", "target"), required=True)
    extract.add_argument("--session-id", required=True)
    extract.add_argument("--camera-position", required=True)
    extract.add_argument("--lighting-condition", required=True)
    extract.add_argument("--steam-level", choices=("none", "low", "medium", "high"), default="none")
    extract.add_argument(
        "--occlusion-level", choices=("none", "low", "medium", "high"), default="none"
    )
    extract.add_argument("--interval-seconds", type=float, default=1.0)
    extract.add_argument("--duplicate-threshold", type=int, default=4)

    compare = subparsers.add_parser("compare", help="汇总多组 metrics.json")
    compare.add_argument("output")
    compare.add_argument("metrics", nargs="+")

    select = subparsers.add_parser(
        "select-proxy-checkpoint", help="按代理验证集准确率选择最佳检查点"
    )
    select.add_argument("output")
    select.add_argument("summaries", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        records = validate_manifest(
            read_manifest(args.manifest),
            data_root=args.data_root,
            check_files=not args.skip_files,
        )
        print(f"Valid manifest: {len(records)} records")
        return 0
    if args.command == "split-manifest":
        records = validate_manifest(
            read_manifest(args.manifest),
            data_root=args.data_root,
            check_files=not args.skip_files,
        )
        split_records = split_by_session(
            records,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        write_manifest(args.output, split_records)
        print(f"Wrote {len(split_records)} records to {args.output}")
        return 0
    if args.command == "train":
        checkpoint = train_experiment(load_experiment_config(args.config))
        print(f"Best checkpoint: {checkpoint}")
        return 0
    if args.command == "evaluate":
        metrics = evaluate_experiment(load_experiment_config(args.config))
        print(f"Metrics: {metrics}")
        return 0
    if args.command == "extract-frames":
        records = extract_video_frames(
            args.video,
            output_dir=args.output_dir,
            data_root=args.data_root,
            food_label=args.food_label,
            environment=args.environment,
            session_id=args.session_id,
            camera_position=args.camera_position,
            lighting_condition=args.lighting_condition,
            steam_level=args.steam_level,
            occlusion_level=args.occlusion_level,
            interval_seconds=args.interval_seconds,
            duplicate_threshold=args.duplicate_threshold,
        )
        validate_manifest(records, data_root=args.data_root, check_files=True)
        write_manifest(args.manifest_output, records)
        print(f"Extracted {len(records)} unique frames to {args.manifest_output}")
        return 0
    if args.command == "compare":
        rows = build_comparison(args.metrics)
        write_comparison(args.output, rows)
        print(f"Wrote {len(rows)} experiment rows to {args.output}")
        return 0
    if args.command == "select-proxy-checkpoint":
        selection = select_best_proxy_checkpoint(args.summaries, args.output)
        print(
            "Selected "
            f"{selection['selected_experiment']}: {selection['selected_checkpoint']}"
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")

