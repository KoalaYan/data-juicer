#!/usr/bin/env python3
"""Show progress for the portrait logical/micro-shard runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            f"path must be absolute, got: {value}"
        )
    return path


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def cache_usage(cache_root: Path) -> tuple[int, int]:
    files = 0
    byte_count = 0
    if not cache_root.is_dir():
        return files, byte_count
    controls = {
        ".data_juicer_cache_quota.json",
        ".data_juicer_cache_quota.lock",
    }
    for path in cache_root.rglob("*"):
        if path.is_file() and path.name not in controls:
            files += 1
            byte_count += path.stat().st_size
    return files, byte_count


def snapshot(
    output_root: Path,
    work_root: Path,
    cache_root: Path | None,
    logical_shard_index: int | None,
) -> str:
    manifest_path = work_root / "INPUT_MANIFEST"
    manifest = load_json(manifest_path)
    all_shards = manifest.get("shards") or []
    if not all_shards:
        return f"INPUT_MANIFEST is not ready: {manifest_path}"
    shards = all_shards
    if logical_shard_index is not None:
        shards = [
            shard
            for shard in all_shards
            if shard.get("logical_index") == logical_shard_index
        ]
    completed = 0
    completed_input_rows = 0
    completed_output_rows = 0
    next_task = None
    logical_indices = sorted(
        {shard["logical_index"] for shard in shards}
    )
    completed_logical = 0
    for logical_index in logical_indices:
        if (
            output_root
            / f"shard-{logical_index:06d}"
            / "SUCCESS"
        ).is_file():
            completed_logical += 1
    for shard in shards:
        marker_path = (
            output_root
            / f"shard-{shard['logical_index']:06d}"
            / f"micro-{shard['micro_index']:04d}"
            / "SUCCESS"
        )
        marker = load_json(marker_path)
        if (
            marker.get("micro_shard_index") == shard["index"]
            and marker.get("input_sha256") == shard["sha256"]
        ):
            completed += 1
            completed_input_rows += int(marker.get("input_rows", 0))
            completed_output_rows += int(marker.get("output_rows", 0))
        elif next_task is None:
            next_task = (
                f"shard-{shard['logical_index']:06d}/"
                f"micro-{shard['micro_index']:04d}"
            )

    attempts_root = output_root / ".attempts"
    attempts = (
        sorted(path.name for path in attempts_root.iterdir())
        if attempts_root.is_dir()
        else []
    )
    failures_root = work_root / "failures"
    failures = (
        sorted(path.name for path in failures_root.iterdir())
        if failures_root.is_dir()
        else []
    )
    expected_rows = sum(int(shard["rows"]) for shard in shards)
    percent = 100.0 * completed / len(shards) if shards else 0.0
    lines = [
        f"time: {datetime.now().isoformat(timespec='seconds')}",
        (
            f"logical shards: {completed_logical}/{len(logical_indices)}; "
            f"micro shards: {completed}/{len(shards)} ({percent:.1f}%)"
        ),
        (
            f"completed rows: input={completed_input_rows}/{expected_rows}, "
            f"output={completed_output_rows}"
        ),
        f"next incomplete: {next_task or 'none'}",
        f"active attempts: {len(attempts)}"
        + (f" ({', '.join(attempts)})" if attempts else ""),
        f"failure records: {len(failures)}"
        + (f" ({', '.join(failures[-3:])})" if failures else ""),
    ]
    window_root = work_root / "window_checkpoints"
    if window_root.is_dir():
        window_manifests = list(
            window_root.rglob("WINDOW_MANIFEST")
        )
        if logical_shard_index is not None:
            logical_name = f"shard-{logical_shard_index:06d}"
            window_manifests = [
                path
                for path in window_manifests
                if logical_name in path.parts
            ]
        total_windows = 0
        completed_windows = 0
        next_window = None
        for manifest_path in sorted(window_manifests):
            manifest = load_json(manifest_path)
            for window in manifest.get("windows") or []:
                total_windows += 1
                window_dir = (
                    manifest_path.parent
                    / f"window-{int(window['index']):04d}"
                )
                if (window_dir / "WINDOW_SUCCESS").is_file():
                    completed_windows += 1
                elif next_window is None:
                    next_window = str(
                        window_dir.relative_to(window_root)
                    )
        if total_windows:
            lines.append(
                f"execution windows: {completed_windows}/{total_windows}; "
                f"next={next_window or 'none'}"
            )
    if cache_root is not None:
        cache_files, cache_bytes = cache_usage(cache_root)
        lines.append(
            f"cache payload: files={cache_files}, bytes={cache_bytes}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=absolute_path)
    parser.add_argument("--work-root", required=True, type=absolute_path)
    parser.add_argument("--cache-root", type=absolute_path)
    parser.add_argument(
        "--logical-shard-index",
        type=non_negative_int,
    )
    parser.add_argument("--watch-seconds", type=non_negative_int, default=0)
    args = parser.parse_args()

    while True:
        print(
            snapshot(
                args.output_root.resolve(),
                args.work_root.resolve(),
                args.cache_root.resolve() if args.cache_root else None,
                args.logical_shard_index,
            ),
            flush=True,
        )
        if args.watch_seconds == 0:
            return
        print("", flush=True)
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
