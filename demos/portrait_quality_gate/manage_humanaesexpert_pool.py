#!/usr/bin/env python3
"""Start, inspect, or stop detached HumanAesExpert Ray actors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "stop"])
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
