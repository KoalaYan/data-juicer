#!/usr/bin/env python3
"""Start, inspect, or stop detached HumanAesExpert Ray actors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

HAE_DEPENDENCIES = {
    "accelerate": "0.33.0",
    "einops": "0.8.2",
    "opencv-contrib-python-headless": "4.11.0.86",
    "sentencepiece": "0.2.0",
    "timm": "0.6.7",
    "transformers": "4.44.2",
}


def validate_dependencies(dependencies=None):
    dependencies = dependencies or HAE_DEPENDENCIES
    installed = {}
    failures = []
    for distribution, required_version in dependencies.items():
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{distribution}=={required_version} (missing)")
            continue
        installed[distribution] = installed_version
        if installed_version != required_version:
            failures.append(
                f"{distribution}=={required_version} "
                f"(found {installed_version})"
            )
    if failures:
        requirements = REPOSITORY / (
            "demos/portrait_quality_gate/"
            "requirements-humanaesexpert.txt"
        )
        raise RuntimeError(
            "HumanAesExpert dependency preflight failed:\n- "
            + "\n- ".join(failures)
            + "\nInstall the pinned environment with:\n"
            + f"{sys.executable} -m pip install -r {requirements}"
        )
    return installed


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["check", "start", "status", "stop"],
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--namespace", default="portrait-quality-gate")
    parser.add_argument("--prefix", default="humanaesexpert")
    parser.add_argument("--size", type=positive_int, default=7)
    parser.add_argument("--model", default="KlingTeam/HumanAesExpert-8B")
    parser.add_argument("--model-cache", default="")
    parser.add_argument("--input-size", type=positive_int, default=448)
    parser.add_argument("--max-num", type=positive_int, default=12)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DATA_JUICER_LAZY_OP_IMPORT", "1")
    if args.action in {"check", "start"}:
        dependency_versions = validate_dependencies()
        if args.action == "check":
            print(
                json.dumps(
                    {"status": "ok", "dependencies": dependency_versions},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

    import ray

    from data_juicer.utils.humanaesexpert_ray_pool import (
        get_pool_handles,
        start_pool,
        stop_pool,
    )

    ray.init(address=args.ray_address, namespace=args.namespace)
    if args.action == "start":
        scorer_config = {
            "model_name_or_path": args.model,
            "model_cache_dir": args.model_cache,
            "input_size": args.input_size,
            "max_num": args.max_num,
            "local_files_only": not args.allow_model_download,
        }
        result = start_pool(
            ray,
            args.namespace,
            args.prefix,
            args.size,
            scorer_config,
        )
    elif args.action == "status":
        handles = get_pool_handles(
            ray,
            args.namespace,
            args.prefix,
            args.size,
        )
        result = ray.get([handle.identity.remote() for handle in handles])
    else:
        result = {
            "stopped": stop_pool(
                ray,
                args.namespace,
                args.prefix,
                args.size,
            )
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
