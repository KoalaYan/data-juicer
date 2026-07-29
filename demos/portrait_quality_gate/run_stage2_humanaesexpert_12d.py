#!/usr/bin/env python3
"""Filter portrait classes and run HumanAesExpert Expert Head 12D scoring."""

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


def safe_cache_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"unsafe cache root: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=absolute_path,
        help="Stage-1 output containing portrait_quality metadata.",
    )
    parser.add_argument("--output", required=True, type=absolute_path)
    parser.add_argument("--cache-root", required=True, type=absolute_path)
    parser.add_argument("--model", default="KlingTeam/HumanAesExpert-8B")
    parser.add_argument(
        "--model-cache",
        type=absolute_path,
        default=DEFAULT_MODEL_CACHE,
    )
    parser.add_argument("--max-cache-files", type=positive_int, default=256)
    parser.add_argument(
        "--max-cache-bytes",
        type=positive_int,
        default=100 * 1024**3,
        help="Hard cache byte cap; default: 100 GiB.",
    )
    parser.add_argument("--download-workers", type=positive_int, default=4)
    parser.add_argument(
        "--download-batch-size",
        type=int,
        choices=[1],
        default=1,
        help=(
            "Must remain 1 with a hard quota so a partially downloaded Ray "
            "batch cannot hold all capacity while waiting for itself."
        ),
    )
    parser.add_argument("--download-concurrency", type=positive_int, default=1)
    parser.add_argument("--score-batch-size", type=positive_int, default=4)
    parser.add_argument("--score-workers", type=positive_int, default=1)
    parser.add_argument("--input-size", type=positive_int, default=448)
    parser.add_argument("--max-num", type=positive_int, default=12)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access when weights are not cached.",
    )
    parser.add_argument(
        "--exclude-hard-rejects",
        action="store_true",
        help=(
            "Additionally exclude hard-quality reject rows. By default the "
            "selection follows only the two requested human-status classes."
        ),
    )
    args = parser.parse_args()

    if "AOSS_CONF" not in os.environ:
        parser.error("AOSS_CONF must point to the private AOSS config")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_cache = args.model_cache.expanduser().resolve()
    cache_root = safe_cache_root(args.cache_root)
    if input_path == output_path:
        parser.error("--input and --output must differ")
    cache_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    FileCacheQuota(
        str(cache_root),
        max_files=args.max_cache_files,
        max_bytes=args.max_cache_bytes,
    ).prepare_for_new_run()

    hard_statuses = (
        ["pass", "uncertain"]
        if args.exclude_hard_rejects
        else ["pass", "uncertain", "reject"]
    )
    config = {
        "project_name": "portrait-humanaesexpert-stage2",
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
                "image_portrait_quality_filter": {
                    "keep_statuses": hard_statuses,
                    "keep_human_statuses": [
                        "portrait_clear",
                        "human_present",
                    ],
                    "any_or_all": "all",
                    "keep_missing": False,
                }
            },
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
                    "batch_size": args.download_batch_size,
                    "max_concurrent": args.download_concurrency,
                    "max_cache_files": args.max_cache_files,
                    "max_cache_bytes": args.max_cache_bytes,
                    "fail_on_download_error": True,
                }
            },
            {
                "image_humanaesexpert_mapper": {
                    "num_gpus": 1,
                    "auto_op_parallelism": False,
                    "num_proc": args.score_workers,
                    "batch_size": args.score_batch_size,
                    "model_name_or_path": args.model,
                    "model_cache_dir": str(model_cache),
                    "input_size": args.input_size,
                    "max_num": args.max_num,
                    "local_files_only": not args.allow_model_download,
                    "delete_local_cache_after_processing": True,
                    "local_cache_root": str(cache_root),
                    "source_image_key": "source_images",
                }
            },
        ],
    }

    environment = os.environ.copy()
    environment.setdefault("RAY_USE_MULTIPROCESSING_CPU_COUNT", "1")
    environment.setdefault("DATA_JUICER_MODELS_CACHE", str(model_cache))
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{REPOSITORY}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(REPOSITORY)
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="portrait-stage2-",
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
