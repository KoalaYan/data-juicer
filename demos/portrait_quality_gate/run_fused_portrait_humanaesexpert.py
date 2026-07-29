#!/usr/bin/env python3
"""Run one-download portrait triage and HumanAesExpert scoring on Ray."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from data_juicer.utils.cache_quota import FileCacheQuota  # noqa: E402


DEFAULT_MODEL_CACHE = Path(
    "/mnt/afs/yanpeishen/model_cache/huggingface"
)
DEFAULT_YOLO_MODEL = Path(
    "/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n.pt"
)
DEFAULT_YOLO_POSE_MODEL = Path(
    "/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n-pose.pt"
)


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            f"path must be absolute, got: {value}"
        )
    return path


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def gpu_fraction(value: str) -> float:
    result = float(value)
    if not 0 < result <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return result


def safe_cache_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"unsafe cache root: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download each image once, run portrait triage, immediately "
            "release ineligible cache files, and score eligible images on a "
            "persistent HumanAesExpert GPU actor pool."
        )
    )
    parser.add_argument("--input", required=True, type=absolute_path)
    parser.add_argument("--output", required=True, type=absolute_path)
    parser.add_argument("--cache-root", required=True, type=absolute_path)
    parser.add_argument("--model", default="KlingTeam/HumanAesExpert-8B")
    parser.add_argument(
        "--model-cache",
        type=absolute_path,
        default=DEFAULT_MODEL_CACHE,
    )
    parser.add_argument(
        "--yolo-model",
        type=absolute_path,
        default=DEFAULT_YOLO_MODEL,
    )
    parser.add_argument(
        "--yolo-pose-model",
        type=absolute_path,
        default=DEFAULT_YOLO_POSE_MODEL,
    )
    parser.add_argument("--max-cache-files", type=positive_int, default=2048)
    parser.add_argument(
        "--max-cache-bytes",
        type=positive_int,
        default=200 * 1024**3,
        help="Hard cache byte cap; default: 200 GiB.",
    )
    parser.add_argument("--download-workers", type=positive_int, default=8)
    parser.add_argument("--download-concurrency", type=positive_int, default=1)
    parser.add_argument("--quality-workers", type=positive_int, default=4)
    parser.add_argument(
        "--quality-gpus-per-worker",
        type=gpu_fraction,
        default=0.25,
        help=(
            "Ray GPU reservation per lightweight YOLO actor. Defaults to "
            "four actors sharing one H100 in total."
        ),
    )
    parser.add_argument("--quality-batch-size", type=positive_int, default=64)
    parser.add_argument("--router-workers", type=positive_int, default=16)
    parser.add_argument("--router-batch-size", type=positive_int, default=128)
    parser.add_argument(
        "--score-workers",
        type=positive_int,
        default=7,
        help=(
            "Persistent HumanAesExpert actors. The 8-H100 default reserves "
            "seven cards for scoring and one card for lightweight triage."
        ),
    )
    parser.add_argument(
        "--score-batch-size",
        type=positive_int,
        default=16,
        help=(
            "Ray scheduling batch, not a native multi-image model batch. "
            "Smaller values improve load balancing when hit rates vary."
        ),
    )
    parser.add_argument("--input-size", type=positive_int, default=448)
    parser.add_argument("--max-num", type=positive_int, default=12)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--persistent-actor-pool",
        action="store_true",
        help=(
            "Route scoring to an already-started detached Ray actor pool "
            "instead of loading a model in each Ray Data actor."
        ),
    )
    parser.add_argument(
        "--persistent-actor-namespace",
        default="portrait-quality-gate",
    )
    parser.add_argument(
        "--persistent-actor-prefix",
        default="humanaesexpert",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access when weights are not cached.",
    )
    args = parser.parse_args()

    if "AOSS_CONF" not in os.environ:
        parser.error("AOSS_CONF must point to the private AOSS config")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_cache = args.model_cache.expanduser().resolve()
    cache_root = safe_cache_root(args.cache_root)
    yolo_model = args.yolo_model.resolve()
    yolo_pose_model = args.yolo_pose_model.resolve()
    if input_path == output_path:
        parser.error("--input and --output must differ")
    for model_path in (yolo_model, yolo_pose_model):
        if not model_path.is_file():
            parser.error(f"YOLO model does not exist: {model_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    FileCacheQuota(
        str(cache_root),
        max_files=args.max_cache_files,
        max_bytes=args.max_cache_bytes,
    ).prepare_for_new_run()

    config = {
        "project_name": "portrait-humanaesexpert-fused",
        "executor_type": "ray",
        "ray_address": args.ray_address,
        "dataset_path": str(input_path),
        "export_path": str(output_path),
        "text_keys": "text",
        "image_key": "images",
        "image_bytes_key": "image_bytes",
        "skip_op_error": False,
        "keep_stats_in_res_ds": True,
        "use_cache": False,
        "process": [
            {
                "s3_download_file_mapper": {
                    "download_field": "images",
                    "save_dir": str(cache_root),
                    "source_field": "source_images",
                    "resume_download": True,
                    "preserve_s3_paths": True,
                    "s3_backend": "aoss",
                    "aoss_config_env": "AOSS_CONF",
                    "auto_op_parallelism": False,
                    "num_proc": args.download_workers,
                    "batch_size": 1,
                    "max_concurrent": args.download_concurrency,
                    "max_cache_files": args.max_cache_files,
                    "max_cache_bytes": args.max_cache_bytes,
                    "fail_on_download_error": True,
                }
            },
            {
                "image_portrait_quality_mapper": {
                    "auto_op_parallelism": False,
                    "num_proc": args.quality_workers,
                    "num_gpus": args.quality_gpus_per_worker,
                    "batch_size": args.quality_batch_size,
                    "inference_batch_size": args.quality_batch_size,
                    "delete_local_cache_after_processing": False,
                    "detect_people": True,
                    "detect_faces_enabled": True,
                    "detect_pose": True,
                    "require_human": True,
                    "yolo_model_path": str(yolo_model),
                    "yolo_pose_model_path": str(yolo_pose_model),
                    "max_analysis_side": 1024,
                    "min_sharpness_score": 0.0,
                }
            },
            {
                "image_portrait_cache_router_mapper": {
                    "auto_op_parallelism": False,
                    "num_proc": args.router_workers,
                    "batch_size": args.router_batch_size,
                    "keep_human_statuses": [
                        "portrait_clear",
                        "human_present",
                    ],
                    "local_cache_root": str(cache_root),
                    "source_image_key": "source_images",
                }
            },
            {
                "image_humanaesexpert_mapper": {
                    "auto_op_parallelism": False,
                    "num_proc": args.score_workers,
                    "num_gpus": 0 if args.persistent_actor_pool else 1,
                    "batch_size": args.score_batch_size,
                    "model_name_or_path": args.model,
                    "model_cache_dir": str(model_cache),
                    "input_size": args.input_size,
                    "max_num": args.max_num,
                    "local_files_only": not args.allow_model_download,
                    "delete_local_cache_after_processing": True,
                    "local_cache_root": str(cache_root),
                    "source_image_key": "source_images",
                    "skip_ineligible": True,
                    "persistent_actor_pool": args.persistent_actor_pool,
                    "persistent_actor_namespace": (
                        args.persistent_actor_namespace
                    ),
                    "persistent_actor_prefix": args.persistent_actor_prefix,
                    "persistent_actor_pool_size": args.score_workers,
                }
            },
        ],
    }

    environment = os.environ.copy()
    environment.setdefault("RAY_USE_MULTIPROCESSING_CPU_COUNT", "1")
    environment.setdefault("DATA_JUICER_MODELS_CACHE", str(model_cache))
    environment.setdefault("DATA_JUICER_LAZY_OP_IMPORT", "1")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{REPOSITORY}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(REPOSITORY)
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="portrait-fused-",
        encoding="utf-8",
    ) as config_file:
        yaml.safe_dump(config, config_file, allow_unicode=True, sort_keys=False)
        config_file.flush()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.process_data",
                "--config",
                config_file.name,
            ],
            cwd=REPOSITORY,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
