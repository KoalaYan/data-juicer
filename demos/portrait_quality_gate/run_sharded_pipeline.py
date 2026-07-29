#!/usr/bin/env python3
"""Run portrait pipelines with task-level shard checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


REPOSITORY = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPTS = {
    "stage1": (
        REPOSITORY
        / "demos/portrait_quality_gate/run_stage1_portrait_quality_all.py"
    ),
    "stage2": (
        REPOSITORY
        / "demos/portrait_quality_gate/run_stage2_humanaesexpert_12d.py"
    ),
    "fused": (
        REPOSITORY
        / "demos/portrait_quality_gate/"
        "run_fused_portrait_humanaesexpert.py"
    ),
}
DATA_SUFFIXES = {".json", ".jsonl"}
CONTROL_FILE_NAMES = {
    "INPUT_MANIFEST",
    "SUCCESS",
    "SHARD_INFO",
}
PORTRAIT_STATUSES = {"portrait_clear", "human_present"}
ALL_HARD_STATUSES = {"pass", "uncertain", "reject"}
NON_REJECT_HARD_STATUSES = {"pass", "uncertain"}


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


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(parent)]) == str(parent)
    except ValueError:
        return False


def require_safe_child(path: Path, parent: Path) -> None:
    path = path.resolve()
    parent = parent.resolve()
    if path == parent or not is_relative_to(path, parent):
        raise ValueError(f"refusing to remove unsafe path: {path}")


def safe_rmtree(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    require_safe_child(path, parent)
    if path.is_symlink():
        raise ValueError(f"refusing to remove symlink: {path}")
    shutil.rmtree(path)


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def discover_source_files(input_path: Path) -> List[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in DATA_SUFFIXES:
            raise ValueError(
                f"input must be JSON/JSONL, got: {input_path}"
            )
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    files = []
    for candidate in input_path.rglob("*"):
        if (
            candidate.is_file()
            and candidate.suffix.lower() in DATA_SUFFIXES
            and candidate.name not in CONTROL_FILE_NAMES
            and not any(part.startswith(".") for part in candidate.parts)
        ):
            files.append(candidate.resolve())
    files.sort(key=str)
    if not files:
        raise ValueError(f"no JSON/JSONL data files found under {input_path}")
    return files


def source_fingerprint(files: Sequence[Path]) -> Tuple[str, List[dict]]:
    entries = []
    for path in files:
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), entries


def iter_nonempty_lines(files: Iterable[Path]) -> Iterator[bytes]:
    for path in files:
        with path.open("rb") as source:
            for line in source:
                normalized = line.strip()
                if normalized:
                    yield normalized + b"\n"


def build_input_shards(
    source_files: Sequence[Path],
    source_entries: List[dict],
    fingerprint: str,
    shard_size: int,
    work_root: Path,
    rebuild: bool,
) -> Dict[str, Any]:
    manifest_path = work_root / "INPUT_MANIFEST"
    input_shards_root = work_root / "input_shards"
    if manifest_path.is_file() and not rebuild:
        manifest = load_json(manifest_path)
        expected = {
            "version": 1,
            "source_fingerprint": fingerprint,
            "shard_size": shard_size,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    "existing input shards do not match the current source "
                    f"or shard size ({key}); use --rebuild-input-shards or "
                    "choose another --work-root"
                )
        for shard in manifest.get("shards", []):
            path = Path(shard["path"])
            if (
                not path.is_file()
                or path.stat().st_size != shard.get("bytes")
                or path.stat().st_mtime_ns != shard.get("mtime_ns")
            ):
                raise RuntimeError(
                    f"input shard is missing or changed: {path}; use "
                    "--rebuild-input-shards"
                )
        return manifest

    if input_shards_root.exists():
        safe_rmtree(input_shards_root, work_root)
    if manifest_path.exists():
        manifest_path.unlink()
    input_shards_root.mkdir(parents=True, exist_ok=True)

    shards = []
    target = None
    target_tmp = None
    hasher = None
    rows = 0
    byte_count = 0
    total_rows = 0

    def finish_shard() -> None:
        nonlocal target, target_tmp, hasher, rows, byte_count
        if target is None or target_tmp is None or hasher is None:
            return
        target.flush()
        os.fsync(target.fileno())
        target.close()
        index = len(shards)
        final_path = input_shards_root / f"shard-{index:06d}.jsonl"
        os.replace(target_tmp, final_path)
        final_stat = final_path.stat()
        shards.append(
            {
                "index": index,
                "path": str(final_path),
                "rows": rows,
                "bytes": byte_count,
                "mtime_ns": final_stat.st_mtime_ns,
                "sha256": hasher.hexdigest(),
            }
        )
        target = None
        target_tmp = None
        hasher = None
        rows = 0
        byte_count = 0

    try:
        for line in iter_nonempty_lines(source_files):
            if target is None:
                index = len(shards)
                target_tmp = (
                    input_shards_root
                    / f".shard-{index:06d}.jsonl.tmp.{os.getpid()}"
                )
                target = target_tmp.open("wb")
                hasher = hashlib.sha256()
            target.write(line)
            hasher.update(line)
            rows += 1
            byte_count += len(line)
            total_rows += 1
            if rows >= shard_size:
                finish_shard()
        finish_shard()
    except BaseException:
        if target is not None:
            target.close()
        if target_tmp is not None:
            target_tmp.unlink(missing_ok=True)
        raise

    if not shards:
        raise ValueError("input contains no non-empty JSONL records")
    manifest = {
        "version": 1,
        "source_fingerprint": fingerprint,
        "source_files": source_entries,
        "shard_size": shard_size,
        "total_rows": total_rows,
        "shards": shards,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def stage2_record_is_selected(
    record: Dict[str, Any],
    exclude_hard_rejects: bool,
) -> bool:
    meta = record.get("__dj__meta__") or record.get("meta") or {}
    quality_records = meta.get("portrait_quality")
    if not quality_records:
        return False
    hard_statuses = (
        NON_REJECT_HARD_STATUSES
        if exclude_hard_rejects
        else ALL_HARD_STATUSES
    )
    return all(
        quality.get("status") in hard_statuses
        and quality.get("human_status") in PORTRAIT_STATUSES
        for quality in quality_records
    )


def expected_output_rows(
    mode: str,
    input_shard: Path,
    input_rows: int,
    pipeline_args: Sequence[str],
) -> int:
    if mode != "stage2":
        return input_rows
    exclude_hard_rejects = "--exclude-hard-rejects" in pipeline_args
    selected = 0
    observed = 0
    with input_shard.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            observed += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid input JSON at {input_shard}:{line_number}"
                ) from error
            if stage2_record_is_selected(record, exclude_hard_rejects):
                selected += 1
    if observed != input_rows:
        raise RuntimeError(
            f"input shard row count changed: expected={input_rows}, "
            f"observed={observed}, path={input_shard}"
        )
    return selected


def discover_output_parts(raw_output: Path) -> List[Path]:
    if raw_output.is_file():
        return [raw_output]
    if not raw_output.is_dir():
        raise RuntimeError(f"pipeline output does not exist: {raw_output}")
    parts = sorted(
        path
        for path in raw_output.rglob("*")
        if path.is_file() and path.suffix.lower() in DATA_SUFFIXES
    )
    if not parts:
        raise RuntimeError(f"pipeline output has no JSON parts: {raw_output}")
    return parts


def normalize_output(
    raw_output: Path,
    normalized_output: Path,
) -> Tuple[int, str]:
    parts = discover_output_parts(raw_output)
    temporary = normalized_output.with_name(
        f".{normalized_output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    output_rows = 0
    hasher = hashlib.sha256()
    try:
        with temporary.open("wb") as target:
            for part in parts:
                with part.open("rb") as source:
                    for line_number, line in enumerate(source, start=1):
                        normalized = line.strip()
                        if not normalized:
                            continue
                        try:
                            json.loads(normalized)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid output JSON at {part}:{line_number}"
                            ) from error
                        normalized += b"\n"
                        target.write(normalized)
                        hasher.update(normalized)
                        output_rows += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, normalized_output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_rows, hasher.hexdigest()


def cache_usage(cache_root: Path) -> Dict[str, int]:
    state_path = cache_root / ".data_juicer_cache_quota.json"
    references = 0
    if state_path.is_file():
        state = load_json(state_path)
        references = sum(
            int(entry.get("references", 0))
            for entry in state.get("entries", {}).values()
        )

    files = 0
    byte_count = 0
    if cache_root.is_dir():
        for path in cache_root.rglob("*"):
            if (
                path.is_file()
                and path.name
                not in {
                    ".data_juicer_cache_quota.json",
                    ".data_juicer_cache_quota.lock",
                }
            ):
                files += 1
                byte_count += path.stat().st_size
    return {
        "files": files,
        "bytes": byte_count,
        "references": references,
    }


def wait_for_zero_cache(
    cache_root: Path,
    timeout_seconds: int,
) -> Dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        usage = cache_usage(cache_root)
        if not any(usage.values()):
            return usage
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"cache did not return to zero: root={cache_root}, "
                f"usage={usage}"
            )
        time.sleep(1)


def valid_success_marker(
    shard_dir: Path,
    mode: str,
    shard: Dict[str, Any],
) -> bool:
    marker_path = shard_dir / "SUCCESS"
    data_path = shard_dir / "data.jsonl"
    if not marker_path.is_file() or not data_path.is_file():
        return False
    try:
        marker = load_json(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        marker.get("version") == 1
        and marker.get("mode") == mode
        and marker.get("shard_index") == shard["index"]
        and marker.get("input_rows") == shard["rows"]
        and marker.get("input_sha256") == shard["sha256"]
    )


def write_failure(
    failures_root: Path,
    shard_index: int,
    error: BaseException,
) -> None:
    atomic_write_json(
        failures_root / f"shard-{shard_index:06d}",
        {
            "version": 1,
            "shard_index": shard_index,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def validate_cli_ownership(argv: Sequence[str]) -> None:
    for reserved in ("--output",):
        if any(
            token == reserved or token.startswith(f"{reserved}=")
            for token in argv
        ):
            raise ValueError(
                f"{reserved} is controlled by the shard runner"
            )
    for required in ("--input", "--output-root", "--cache-root"):
        occurrences = sum(
            token == required or token.startswith(f"{required}=")
            for token in argv
        )
        if occurrences != 1:
            raise ValueError(
                f"{required} must be provided exactly once"
            )
    mode_occurrences = sum(
        token == "--mode" or token.startswith("--mode=")
        for token in argv
    )
    if mode_occurrences != 1:
        raise ValueError("--mode must be provided exactly once")


def main() -> None:
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Split JSONL input into deterministic task shards and run one "
            "portrait pipeline subprocess per shard."
        )
    )
    parser.add_argument("--mode", required=True, choices=sorted(PIPELINE_SCRIPTS))
    parser.add_argument("--pipeline-script", type=absolute_path)
    parser.add_argument("--input", required=True, type=absolute_path)
    parser.add_argument("--output-root", required=True, type=absolute_path)
    parser.add_argument("--cache-root", required=True, type=absolute_path)
    parser.add_argument("--work-root", type=absolute_path)
    parser.add_argument("--shard-size", type=positive_int, default=100_000)
    parser.add_argument("--shard-index", type=non_negative_int)
    parser.add_argument(
        "--cache-zero-timeout",
        type=non_negative_int,
        default=60,
    )
    parser.add_argument("--rebuild-input-shards", action="store_true")
    args, pipeline_args = parser.parse_known_args()
    validate_cli_ownership(raw_argv)

    python = Path(sys.executable).resolve()
    pipeline_script = (
        args.pipeline_script.resolve()
        if args.pipeline_script
        else PIPELINE_SCRIPTS[args.mode].resolve()
    )
    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()
    work_root = (
        args.work_root.resolve()
        if args.work_root
        else output_root.parent / f".{output_root.name}.work"
    )
    if not python.is_absolute() or not os.access(python, os.X_OK):
        parser.error(f"Python interpreter is not executable: {python}")
    if not pipeline_script.is_file():
        parser.error(f"pipeline script does not exist: {pipeline_script}")
    generated_roots = (output_root, cache_root, work_root)
    unsafe_roots = {Path("/"), Path.home().resolve()}
    if any(path in unsafe_roots for path in generated_roots):
        parser.error("output, cache, and work roots must not be / or HOME")
    if input_path in set(generated_roots):
        parser.error("input, output, cache, and work paths must differ")
    for index, root in enumerate(generated_roots):
        for other in generated_roots[index + 1 :]:
            if is_relative_to(root, other) or is_relative_to(other, root):
                parser.error(
                    "output, cache, and work roots must not overlap: "
                    f"{root}, {other}"
                )
    if input_path.is_dir():
        for generated_root in generated_roots:
            if is_relative_to(generated_root, input_path):
                parser.error(
                    f"generated path must not be inside input: {generated_root}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    attempts_root = output_root / ".attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    failures_root = work_root / "failures"
    failures_root.mkdir(parents=True, exist_ok=True)

    source_files = discover_source_files(input_path)
    fingerprint, source_entries = source_fingerprint(source_files)
    manifest = build_input_shards(
        source_files,
        source_entries,
        fingerprint,
        args.shard_size,
        work_root,
        args.rebuild_input_shards,
    )
    if args.rebuild_input_shards:
        for stale_output in output_root.glob("shard-*"):
            if stale_output.is_dir():
                safe_rmtree(stale_output, output_root)
    shards = manifest["shards"]
    if args.shard_index is not None:
        shards = [
            shard
            for shard in shards
            if shard["index"] == args.shard_index
        ]
        if not shards:
            parser.error(
                f"--shard-index {args.shard_index} is outside "
                f"0..{len(manifest['shards']) - 1}"
            )

    completed = 0
    skipped = 0
    for shard in shards:
        index = shard["index"]
        shard_name = f"shard-{index:06d}"
        final_dir = output_root / shard_name
        failure_path = failures_root / shard_name
        if valid_success_marker(final_dir, args.mode, shard):
            print(f"[skip] {shard_name}: valid SUCCESS")
            skipped += 1
            continue
        if final_dir.exists():
            safe_rmtree(final_dir, output_root)

        for stale_attempt in attempts_root.glob(f"{shard_name}.*"):
            safe_rmtree(stale_attempt, attempts_root)
        attempt_dir = (
            attempts_root
            / f"{shard_name}.{os.getpid()}.{uuid.uuid4().hex}"
        )
        attempt_dir.mkdir()
        raw_output = attempt_dir / "ray_output.jsonl"
        normalized_output = attempt_dir / "data.jsonl"
        shard_cache = cache_root / shard_name
        shard_cache.mkdir(parents=True, exist_ok=True)
        expected_rows = expected_output_rows(
            args.mode,
            Path(shard["path"]),
            shard["rows"],
            pipeline_args,
        )
        command = [
            str(python),
            str(pipeline_script),
            "--input",
            shard["path"],
            "--output",
            str(raw_output),
            "--cache-root",
            str(shard_cache),
            *pipeline_args,
        ]
        print(
            f"[run] {shard_name}: input_rows={shard['rows']} "
            f"expected_output_rows={expected_rows}"
        )
        try:
            subprocess.run(command, cwd=REPOSITORY, check=True)
            output_rows, output_sha256 = normalize_output(
                raw_output,
                normalized_output,
            )
            if output_rows != expected_rows:
                raise RuntimeError(
                    f"output row mismatch for {shard_name}: "
                    f"expected={expected_rows}, observed={output_rows}"
                )
            usage = wait_for_zero_cache(
                shard_cache,
                args.cache_zero_timeout,
            )
            if raw_output.is_dir():
                safe_rmtree(raw_output, attempt_dir)
            elif raw_output.exists():
                raw_output.unlink()
            if shard_cache.exists():
                safe_rmtree(shard_cache, cache_root)
            os.replace(attempt_dir, final_dir)
            marker = {
                "version": 1,
                "mode": args.mode,
                "shard_index": index,
                "input_path": shard["path"],
                "input_rows": shard["rows"],
                "input_sha256": shard["sha256"],
                "expected_output_rows": expected_rows,
                "output_rows": output_rows,
                "output_sha256": output_sha256,
                "cache": usage,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(final_dir / "SUCCESS", marker)
            failure_path.unlink(missing_ok=True)
            completed += 1
            print(f"[done] {shard_name}: {output_rows} rows")
        except BaseException as error:
            if attempt_dir.exists():
                safe_rmtree(attempt_dir, attempts_root)
            if (
                final_dir.exists()
                and not (final_dir / "SUCCESS").is_file()
            ):
                safe_rmtree(final_dir, output_root)
            write_failure(failures_root, index, error)
            print(
                f"[failed] {shard_name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            raise

    try:
        attempts_root.rmdir()
    except OSError:
        pass
    print(
        f"shard run complete: completed={completed}, skipped={skipped}, "
        f"selected={len(shards)}, total={len(manifest['shards'])}"
    )


if __name__ == "__main__":
    main()
