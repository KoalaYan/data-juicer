#!/usr/bin/env python3
"""Run a bounded single-node portrait pipeline without Ray.

The driver keeps one lightweight portrait-quality worker and several
HumanAesExpert workers resident on dedicated GPUs. AOSS downloads run in a
thread pool and overlap both GPU stages. Local images are deleted immediately
after they leave their final consumer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_MODEL_CACHE = Path(
    "/mnt/afs/yanpeishen/model_cache/huggingface"
)
DEFAULT_YOLO_MODEL = Path(
    "/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n.pt"
)
DEFAULT_YOLO_POSE_MODEL = Path(
    "/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n-pose.pt"
)
STOP_MESSAGE = ("stop", -1, None, None)
PORTRAIT_STATUSES = {"portrait_clear", "human_present"}


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


def non_negative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def require_safe_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"unsafe generated root: {resolved}")
    return resolved


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(parent)]) == str(parent)
    except ValueError:
        return False


def safe_rmtree(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_parent = parent.resolve()
    if (
        resolved == resolved_parent
        or not is_relative_to(resolved, resolved_parent)
        or path.is_symlink()
    ):
        raise ValueError(f"refusing unsafe directory removal: {path}")
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
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def jsonl_summary(path: Path) -> Tuple[int, str]:
    rows = 0
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for line in source:
            normalized = line.strip()
            if not normalized:
                continue
            json.loads(normalized)
            normalized += b"\n"
            hasher.update(normalized)
            rows += 1
    return rows, hasher.hexdigest()


def read_jsonl(path: Path, expected_rows: int) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_number}"
                )
            records.append(record)
    if len(records) != expected_rows:
        raise RuntimeError(
            f"input row mismatch: expected={expected_rows} "
            f"observed={len(records)} path={path}"
        )
    return records


def queue_size(work_queue: Any) -> int:
    try:
        return int(work_queue.qsize())
    except (AttributeError, NotImplementedError):
        return -1


def visible_gpu_tokens(required: int) -> List[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = (
        [token.strip() for token in configured.split(",") if token.strip()]
        if configured
        else [str(index) for index in range(required)]
    )
    if len(tokens) < required:
        raise RuntimeError(
            f"need {required} visible GPUs, found {len(tokens)}: {tokens}"
        )
    return tokens[:required]


def configure_gpu_worker(gpu_token: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token
    os.environ["DATA_JUICER_LAZY_OP_IMPORT"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def configure_cpu_worker(cpu_threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DATA_JUICER_LAZY_OP_IMPORT"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)


def resolve_worker_devices(
    score_workers: int,
    quality_device: str,
) -> Tuple[str, List[str]]:
    quality_uses_cuda = quality_device == "cuda"
    tokens = visible_gpu_tokens(score_workers + int(quality_uses_cuda))
    if quality_uses_cuda:
        return tokens[0], tokens[1:]
    return "cpu", tokens


def safe_delete_cached_paths(
    record: Dict[str, Any],
    cache_root: str,
) -> None:
    from data_juicer.utils.cache_quota import (
        delete_file_and_release_cache_quota,
    )

    paths = record.get("images") or []
    if not isinstance(paths, list):
        paths = [paths]
    resolved_root = os.path.realpath(os.path.abspath(cache_root))
    for path in paths:
        if not isinstance(path, str) or path.startswith("s3://"):
            continue
        resolved_path = os.path.realpath(os.path.abspath(path))
        try:
            inside = (
                os.path.commonpath([resolved_root, resolved_path])
                == resolved_root
            )
        except ValueError:
            inside = False
        if inside and os.path.isfile(resolved_path) and not os.path.islink(
            resolved_path
        ):
            delete_file_and_release_cache_quota(
                resolved_root,
                resolved_path,
            )


def error_payload(stage: str, error: BaseException) -> Dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _minimal_batch(records: Sequence[Dict[str, Any]]) -> Dict[str, list]:
    from data_juicer.utils.constant import Fields

    return {
        "images": [copy.deepcopy(record.get("images") or []) for record in records],
        "source_images": [
            copy.deepcopy(record.get("source_images") or [])
            for record in records
        ],
        Fields.meta: [
            copy.deepcopy(record.get(Fields.meta) or {})
            for record in records
        ],
    }


def _merge_minimal_batch(
    records: Sequence[Dict[str, Any]],
    batch: Dict[str, list],
) -> None:
    from data_juicer.utils.constant import Fields

    for index, record in enumerate(records):
        record["images"] = batch["images"][index]
        record["source_images"] = batch["source_images"][index]
        record[Fields.meta] = batch[Fields.meta][index]


def quality_worker_main(
    device_token: str,
    quality_device: str,
    cpu_threads: int,
    input_queue: Any,
    score_queue: Any,
    result_queue: Any,
    ready_queue: Any,
    cache_root: str,
    yolo_model: str,
    yolo_pose_model: str,
    batch_size: int,
    batch_wait_seconds: float,
) -> None:
    use_cuda = quality_device == "cuda"
    if use_cuda:
        configure_gpu_worker(device_token)
    else:
        configure_cpu_worker(cpu_threads)
    try:
        import torch

        from data_juicer.ops.mapper.image_portrait_cache_router_mapper import (
            ImagePortraitCacheRouterMapper,
        )
        from data_juicer.ops.mapper.image_portrait_quality_mapper import (
            ImagePortraitQualityMapper,
        )
        from data_juicer.utils.constant import Fields, MetaKeys
        from data_juicer.utils.model_utils import get_model
        from data_juicer.utils.process_utils import setup_worker_threads

        if use_cuda:
            torch.cuda.set_device(0)
        else:
            # get_model() normally constrains multiprocessing workers to one
            # thread. Configure this dedicated CPU inference worker first so
            # the two YOLO models can use the explicitly assigned CPU budget.
            setup_worker_threads(num_threads=cpu_threads)
            # setup_worker_threads() may already have been called by an
            # imported Data-Juicer module. set_num_threads() is safe to call
            # again and makes the dedicated worker's CPU budget authoritative.
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(min(cpu_threads, 4))
            except RuntimeError:
                pass
            print(
                "[quality-runtime] "
                f"device=cpu "
                f"torch_threads={torch.get_num_threads()} "
                f"torch_interop_threads={torch.get_num_interop_threads()}",
                flush=True,
            )
        quality = ImagePortraitQualityMapper(
            yolo_model_path=yolo_model,
            yolo_pose_model_path=yolo_pose_model,
            detect_people=True,
            detect_faces_enabled=True,
            detect_pose=True,
            require_human=True,
            inference_batch_size=batch_size,
            max_analysis_side=1024,
            min_sharpness_score=0.0,
            accelerator=quality_device,
            auto_op_parallelism=False,
        )
        get_model(quality.person_model_key, rank=0, use_cuda=use_cuda)
        get_model(quality.pose_model_key, rank=0, use_cuda=use_cuda)
        router = ImagePortraitCacheRouterMapper(
            keep_human_statuses=sorted(PORTRAIT_STATUSES),
            local_cache_root=cache_root,
            source_image_key="source_images",
            auto_op_parallelism=False,
        )
        ready_queue.put(("ready", "quality", device_token, None))
    except BaseException as error:
        ready_queue.put(
            (
                "error",
                "quality",
                device_token,
                error_payload("startup", error),
            )
        )
        return

    stop_after_batch = False
    while True:
        try:
            first = input_queue.get()
        except (EOFError, OSError):
            return
        if first[0] == "stop":
            return
        items = [first]
        while len(items) < batch_size:
            try:
                item = input_queue.get(timeout=batch_wait_seconds)
            except queue.Empty:
                break
            if item[0] == "stop":
                stop_after_batch = True
                break
            items.append(item)

        records = [item[2] for item in items]
        started = time.monotonic()
        try:
            batch = _minimal_batch(records)
            quality.process_batched(batch, rank=0)
            router.process_batched(batch)
            _merge_minimal_batch(records, batch)
            elapsed = time.monotonic() - started
            for item, record in zip(items, records):
                _, sequence, _, metrics = item
                metrics["quality_seconds"] = elapsed / max(1, len(items))
                quality_records = (
                    (record.get(Fields.meta) or {}).get(
                        MetaKeys.portrait_quality
                    )
                    or []
                )
                if any(
                    bool(result.get("humanaesexpert_eligible"))
                    for result in quality_records
                ):
                    score_queue.put(("record", sequence, record, metrics))
                else:
                    record.setdefault(Fields.meta, {})[
                        MetaKeys.humanaesexpert_expert_scores
                    ] = [None] * len(quality_records)
                    result_queue.put(("ok", sequence, record, metrics))
        except BaseException as error:
            payload = error_payload("quality", error)
            for item, record in zip(items, records):
                safe_delete_cached_paths(record, cache_root)
                result_queue.put(
                    ("error", item[1], payload, item[3])
                )
        if stop_after_batch:
            return


def score_worker_main(
    worker_index: int,
    gpu_token: str,
    input_queue: Any,
    result_queue: Any,
    ready_queue: Any,
    cache_root: str,
    model_name: str,
    model_cache: str,
    input_size: int,
    max_num: int,
    use_flash_attn: bool,
    allow_model_download: bool,
) -> None:
    configure_gpu_worker(gpu_token)
    try:
        import torch

        from data_juicer.ops.mapper.image_humanaesexpert_mapper import (
            ImageHumanAesExpertMapper,
        )

        torch.cuda.set_device(0)
        scorer = ImageHumanAesExpertMapper(
            model_name_or_path=model_name,
            model_cache_dir=model_cache,
            input_size=input_size,
            max_num=max_num,
            local_files_only=not allow_model_download,
            delete_local_cache_after_processing=True,
            local_cache_root=cache_root,
            source_image_key="source_images",
            skip_ineligible=True,
            use_flash_attn=use_flash_attn,
            accelerator="cuda",
            auto_op_parallelism=False,
        )
        scorer._load_model()
        ready_queue.put(
            ("ready", f"score-{worker_index}", gpu_token, None)
        )
    except BaseException as error:
        ready_queue.put(
            (
                "error",
                f"score-{worker_index}",
                gpu_token,
                error_payload("startup", error),
            )
        )
        return

    while True:
        try:
            kind, sequence, record, metrics = input_queue.get()
        except (EOFError, OSError):
            return
        if kind == "stop":
            return
        started = time.monotonic()
        try:
            scorer.process_single(record, rank=0)
            metrics["score_seconds"] = time.monotonic() - started
            metrics["score_worker"] = worker_index
            result_queue.put(("ok", sequence, record, metrics))
        except BaseException as error:
            safe_delete_cached_paths(record, cache_root)
            result_queue.put(
                (
                    "error",
                    sequence,
                    error_payload("score", error),
                    metrics,
                )
            )


class DirectDownloader:
    """Thread-safe AOSS downloader backed by the existing Data-Juicer mapper."""

    def __init__(
        self,
        cache_root: Path,
        max_cache_files: int,
        max_cache_bytes: int,
        attempts: int,
        retry_initial_delay: float,
        retry_max_delay: float,
        retry_jitter: float,
    ):
        os.environ["DATA_JUICER_LAZY_OP_IMPORT"] = "1"
        from data_juicer.ops.mapper.s3_download_file_mapper import (
            S3DownloadFileMapper,
        )

        self.cache_root = cache_root.resolve()
        self.mapper = S3DownloadFileMapper(
            download_field="images",
            save_dir=str(self.cache_root),
            resume_download=True,
            preserve_s3_paths=True,
            source_field="source_images",
            s3_backend="aoss",
            aoss_config_env="AOSS_CONF",
            aoss_stream_to_file=True,
            max_cache_files=max_cache_files,
            max_cache_bytes=max_cache_bytes,
            fail_on_download_error=True,
            aoss_max_attempts=attempts,
            aoss_retry_initial_delay=retry_initial_delay,
            aoss_retry_max_delay=retry_max_delay,
            aoss_retry_jitter=retry_jitter,
            auto_op_parallelism=False,
        )

    def _download_uri(self, uri: str) -> Tuple[str, int]:
        if not isinstance(uri, str) or not uri.startswith("s3://"):
            raise ValueError(f"direct pipeline requires an s3:// URI: {uri}")
        local_path = self.mapper._get_local_save_path(
            uri,
            str(self.cache_root),
        )
        if os.path.isfile(local_path):
            if self.mapper._cache_quota is not None:
                self.mapper._cache_quota.retain(local_path)
            return local_path, os.path.getsize(local_path)
        status, response, _, saved_path = self.mapper._download_from_s3(
            uri,
            local_path,
            False,
        )
        if status != "success" or not saved_path:
            raise RuntimeError(response or f"failed to download {uri}")
        return saved_path, os.path.getsize(saved_path)

    def __call__(
        self,
        sequence: int,
        record: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        started = time.monotonic()
        source_paths = record.get("images") or []
        if not isinstance(source_paths, list):
            source_paths = [source_paths]
        if not source_paths:
            raise ValueError(f"record {sequence} has no images")
        local_paths = []
        downloaded_bytes = 0
        updated = copy.deepcopy(record)
        try:
            for uri in source_paths:
                local_path, size = self._download_uri(uri)
                local_paths.append(local_path)
                downloaded_bytes += size
            updated["source_images"] = copy.deepcopy(source_paths)
            updated["images"] = local_paths
            return (
                sequence,
                updated,
                {
                    "download_bytes": downloaded_bytes,
                    "download_seconds": time.monotonic() - started,
                },
            )
        except BaseException:
            updated["images"] = local_paths
            safe_delete_cached_paths(updated, str(self.cache_root))
            raise


def download_producer(
    records: Sequence[Dict[str, Any]],
    download_one: Callable[
        [int, Dict[str, Any]],
        Tuple[int, Dict[str, Any], Dict[str, Any]],
    ],
    workers: int,
    max_pending: int,
    quality_queue: Any,
    result_queue: Any,
    state: Dict[str, Any],
    stop_event: threading.Event,
    cleanup_record: Callable[[Dict[str, Any]], None],
) -> None:
    futures: Dict[Future, int] = {}
    next_sequence = 0
    finalized: set[int] = set()

    def dispatch_quality(item: tuple) -> bool:
        while not stop_event.is_set():
            try:
                quality_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    try:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="aoss-download",
        ) as executor:
            while futures or next_sequence < len(records):
                if stop_event.is_set() and not futures:
                    break
                while (
                    next_sequence < len(records)
                    and len(futures) < max_pending
                    and not stop_event.is_set()
                ):
                    future = executor.submit(
                        download_one,
                        next_sequence,
                        records[next_sequence],
                    )
                    futures[future] = next_sequence
                    next_sequence += 1
                completed, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    sequence = futures.pop(future)
                    try:
                        result_sequence, record, metrics = future.result()
                        if stop_event.is_set():
                            cleanup_record(record)
                        else:
                            dispatched = dispatch_quality(
                                (
                                    "record",
                                    result_sequence,
                                    record,
                                    metrics,
                                ),
                            )
                            if not dispatched:
                                cleanup_record(record)
                    except BaseException as error:
                        if not stop_event.is_set():
                            result_queue.put(
                                (
                                    "error",
                                    sequence,
                                    error_payload("download", error),
                                    {},
                                )
                            )
                    finalized.add(sequence)
                    state["dispatched"] = int(
                        state.get("dispatched", 0)
                    ) + 1
                if stop_event.is_set():
                    for future in futures:
                        future.cancel()
                    break
    except BaseException as error:
        state["fatal"] = error_payload("download-producer", error)
        if not stop_event.is_set():
            for sequence in sorted(set(range(len(records))) - finalized):
                result_queue.put(
                    (
                        "error",
                        sequence,
                        state["fatal"],
                        {},
                    )
                )
    finally:
        state["done"] = True


def valid_micro_success(
    micro_dir: Path,
    shard: Dict[str, Any],
) -> bool:
    marker_path = micro_dir / "SUCCESS"
    data_path = micro_dir / "data.jsonl"
    if not marker_path.is_file() or not data_path.is_file():
        return False
    try:
        marker = load_json(marker_path)
        output_rows, output_sha256 = jsonl_summary(data_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        marker.get("version") == 1
        and marker.get("mode") == "direct-fused"
        and marker.get("micro_shard_index") == shard["index"]
        and marker.get("logical_shard_index") == shard["logical_index"]
        and marker.get("input_rows") == shard["rows"]
        and marker.get("input_sha256") == shard["sha256"]
        and marker.get("output_rows") == output_rows
        and marker.get("output_sha256") == output_sha256
        and output_rows == shard["rows"]
        and marker.get("cache", {}).get("files") == 0
        and marker.get("cache", {}).get("bytes") == 0
        and marker.get("cache", {}).get("references") == 0
        and marker.get("cache", {}).get("partial_files") == 0
    )


def write_logical_success(
    output_root: Path,
    logical_index: int,
    logical_shards: Sequence[Dict[str, Any]],
) -> bool:
    logical_dir = output_root / f"shard-{logical_index:06d}"
    markers = []
    for shard in logical_shards:
        micro_dir = logical_dir / f"micro-{shard['micro_index']:04d}"
        if not valid_micro_success(micro_dir, shard):
            (logical_dir / "SUCCESS").unlink(missing_ok=True)
            return False
        markers.append(load_json(micro_dir / "SUCCESS"))
    atomic_write_json(
        logical_dir / "SUCCESS",
        {
            "version": 1,
            "mode": "direct-fused",
            "logical_shard_index": logical_index,
            "micro_shards": len(logical_shards),
            "input_rows": sum(item["input_rows"] for item in markers),
            "output_rows": sum(item["output_rows"] for item in markers),
            "scored_rows": sum(item["scored_rows"] for item in markers),
            "quality_only_rows": sum(
                item["quality_only_rows"] for item in markers
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return True


def wait_for_workers(
    ready_queue: Any,
    processes: Sequence[mp.Process],
    expected: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    ready = 0
    while ready < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out waiting for GPU workers: ready={ready}/{expected}"
            )
        try:
            status, name, gpu_token, details = ready_queue.get(
                timeout=min(5.0, remaining)
            )
        except queue.Empty:
            dead = [
                f"{process.name}:{process.exitcode}"
                for process in processes
                if process.exitcode is not None
            ]
            if dead:
                raise RuntimeError(
                    "GPU worker exited during startup: " + ", ".join(dead)
                )
            continue
        if status != "ready":
            raise RuntimeError(
                f"{name} failed on GPU {gpu_token}: "
                f"{json.dumps(details, ensure_ascii=False)}"
            )
        ready += 1
        print(
            f"[worker-ready] name={name} gpu={gpu_token} "
            f"ready={ready}/{expected}",
            flush=True,
        )


def ensure_workers_alive(processes: Sequence[mp.Process]) -> None:
    dead = [
        f"{process.name}:{process.exitcode}"
        for process in processes
        if process.exitcode is not None
    ]
    if dead:
        raise RuntimeError(
            "GPU worker exited unexpectedly: " + ", ".join(dead)
        )


def cache_snapshot(cache_root: Path) -> Dict[str, int]:
    from data_juicer.utils.cache_quota import FileCacheQuota

    quota = FileCacheQuota(str(cache_root), max_files=1)
    snapshot = quota.snapshot()
    partial_files = sum(
        1
        for path in cache_root.rglob("*")
        if path.is_file() and ".part." in path.name
    )
    return {
        "files": int(snapshot["files"]),
        "bytes": int(snapshot["bytes"]),
        "references": int(snapshot["references"]),
        "partial_files": partial_files,
    }


def process_micro_shard(
    shard: Dict[str, Any],
    output_path: Path,
    records: Sequence[Dict[str, Any]],
    downloader: DirectDownloader,
    download_workers: int,
    download_prefetch: int,
    quality_queue: Any,
    score_queue: Any,
    result_queue: Any,
    worker_processes: Sequence[mp.Process],
    cache_root: Path,
    progress_interval: float,
    stall_timeout: float,
) -> Dict[str, Any]:
    producer_state: Dict[str, Any] = {
        "done": False,
        "dispatched": 0,
        "fatal": None,
    }
    producer_stop = threading.Event()
    producer = threading.Thread(
        target=download_producer,
        kwargs={
            "records": records,
            "download_one": downloader,
            "workers": download_workers,
            "max_pending": download_prefetch,
            "quality_queue": quality_queue,
            "result_queue": result_queue,
            "state": producer_state,
            "stop_event": producer_stop,
            "cleanup_record": lambda record: safe_delete_cached_paths(
                record,
                str(cache_root),
            ),
        },
        name=f"download-producer-{shard['index']}",
        daemon=True,
    )
    producer.start()

    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    received = 0
    next_sequence = 0
    pending: Dict[int, Tuple[str, Any, Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    hasher = hashlib.sha256()
    downloaded_bytes = 0
    scored = 0
    quality_only = 0
    started = time.monotonic()
    last_result = started
    last_report = started

    try:
        with temporary.open("wb") as target:
            while received < len(records):
                try:
                    status, sequence, payload, metrics = result_queue.get(
                        timeout=5.0
                    )
                except queue.Empty:
                    ensure_workers_alive(worker_processes)
                    if time.monotonic() - last_result > stall_timeout:
                        raise TimeoutError(
                            "pipeline made no record-level progress for "
                            f"{stall_timeout:.0f}s"
                        )
                    continue
                if sequence < 0 or sequence >= len(records):
                    raise RuntimeError(
                        f"invalid result sequence: {sequence}"
                    )
                if sequence in pending or sequence < next_sequence:
                    raise RuntimeError(
                        f"duplicate result sequence: {sequence}"
                    )
                pending[sequence] = (status, payload, metrics)
                received += 1
                last_result = time.monotonic()
                downloaded_bytes += int(metrics.get("download_bytes", 0))
                if "score_seconds" in metrics:
                    scored += 1
                else:
                    quality_only += 1

                while next_sequence in pending:
                    item_status, item_payload, _ = pending.pop(
                        next_sequence
                    )
                    if item_status == "ok":
                        encoded = (
                            json.dumps(
                                item_payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                        target.write(encoded)
                        hasher.update(encoded)
                    else:
                        failures.append(
                            {
                                "sequence": next_sequence,
                                **item_payload,
                            }
                        )
                    next_sequence += 1

                now = time.monotonic()
                if (
                    received == len(records)
                    or now - last_report >= progress_interval
                ):
                    elapsed = max(1e-6, now - started)
                    cache = cache_snapshot(cache_root)
                    print(
                        "[progress] "
                        f"micro={shard['index']} "
                        f"completed={received}/{len(records)} "
                        f"rate={received / elapsed:.2f}_images/s "
                        f"download={downloaded_bytes / elapsed / 1024**2:.2f}_MiB/s "
                        f"scored={scored} quality_only={quality_only} "
                        f"queues=download:{queue_size(quality_queue)},"
                        f"score:{queue_size(score_queue)},"
                        f"result:{queue_size(result_queue)} "
                        f"cache_files={cache['files']} "
                        f"cache_gib={cache['bytes'] / 1024**3:.2f}",
                        flush=True,
                    )
                    last_report = now

            target.flush()
            os.fsync(target.fileno())
        producer.join(timeout=30.0)
        if producer.is_alive():
            raise RuntimeError("download producer did not stop")
        if producer_state.get("fatal"):
            failures.append(
                {
                    "sequence": -1,
                    **producer_state["fatal"],
                }
            )
        if failures:
            preview = json.dumps(
                failures[:3],
                ensure_ascii=False,
            )
            raise RuntimeError(
                f"micro-shard has {len(failures)} failed records: {preview}"
            )
        if next_sequence != len(records):
            raise RuntimeError(
                f"ordered writer stopped at {next_sequence}/{len(records)}"
            )
        os.replace(temporary, output_path)
    except BaseException:
        producer_stop.set()
        producer.join(timeout=60.0)
        temporary.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - started
    return {
        "output_rows": len(records),
        "output_sha256": hasher.hexdigest(),
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(len(records) / max(elapsed, 1e-6), 4),
        "downloaded_bytes": downloaded_bytes,
        "download_mib_per_second": round(
            downloaded_bytes / max(elapsed, 1e-6) / 1024**2,
            4,
        ),
        "scored_rows": scored,
        "quality_only_rows": quality_only,
    }


def prepare_cache(cache_root: Path) -> Dict[str, int]:
    from data_juicer.utils.cache_quota import FileCacheQuota

    quota = FileCacheQuota(str(cache_root), max_files=1)
    previous = quota.prepare_for_new_run()
    return {
        "files": int(previous["files"]),
        "bytes": int(previous["bytes"]),
    }


def cleanup_cache(cache_root: Path) -> None:
    for path in sorted(
        cache_root.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_symlink():
            continue
        if path.is_file():
            if path.name in {
                ".data_juicer_cache_quota.json",
                ".data_juicer_cache_quota.lock",
            }:
                continue
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    prepare_cache(cache_root)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded AOSS -> portrait quality -> HumanAesExpert pipeline "
            "for one fixed 8-GPU node. Ray is not used."
        )
    )
    parser.add_argument("--input", required=True, type=absolute_path)
    parser.add_argument("--output-root", required=True, type=absolute_path)
    parser.add_argument("--work-root", required=True, type=absolute_path)
    parser.add_argument("--cache-root", required=True, type=absolute_path)
    parser.add_argument(
        "--input-adapter",
        choices=["none", "purchased-selection"],
        default="none",
    )
    parser.add_argument("--logical-shard-size", type=positive_int, default=100_000)
    parser.add_argument("--micro-shard-size", type=positive_int, default=10_000)
    parser.add_argument(
        "--logical-shard-index",
        type=non_negative_int,
    )
    parser.add_argument("--rebuild-input-shards", action="store_true")
    parser.add_argument("--download-workers", type=positive_int, default=24)
    parser.add_argument("--download-prefetch", type=positive_int, default=96)
    parser.add_argument("--download-queue-size", type=positive_int, default=128)
    parser.add_argument("--score-queue-size", type=positive_int, default=128)
    parser.add_argument("--result-queue-size", type=positive_int, default=512)
    parser.add_argument("--quality-batch-size", type=positive_int, default=64)
    parser.add_argument(
        "--quality-device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--quality-cpu-threads",
        type=positive_int,
        default=16,
        help="Torch/OMP threads used when --quality-device=cpu.",
    )
    parser.add_argument(
        "--quality-batch-wait",
        type=non_negative_float,
        default=0.05,
        help="Maximum wait between records while filling one quality batch.",
    )
    parser.add_argument("--score-workers", type=positive_int, default=7)
    parser.add_argument("--max-cache-files", type=positive_int, default=512)
    parser.add_argument(
        "--max-cache-bytes",
        type=positive_int,
        default=50 * 1024**3,
    )
    parser.add_argument("--aoss-download-attempts", type=positive_int, default=5)
    parser.add_argument(
        "--aoss-retry-initial-delay",
        type=non_negative_float,
        default=1.5,
    )
    parser.add_argument(
        "--aoss-retry-max-delay",
        type=non_negative_float,
        default=12.0,
    )
    parser.add_argument(
        "--aoss-retry-jitter",
        type=non_negative_float,
        default=1.0,
    )
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
    parser.add_argument("--input-size", type=positive_int, default=448)
    parser.add_argument("--max-num", type=positive_int, default=12)
    parser.add_argument("--disable-flash-attn", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--worker-start-timeout",
        type=positive_int,
        default=900,
    )
    parser.add_argument(
        "--stall-timeout",
        type=positive_int,
        default=1800,
    )
    parser.add_argument(
        "--progress-interval",
        type=non_negative_float,
        default=10.0,
    )
    return parser


def main() -> None:
    def interrupt_handler(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)
    args = build_argument_parser().parse_args()
    if "AOSS_CONF" not in os.environ:
        raise RuntimeError(
            "AOSS_CONF must point to the private AOSS config file"
        )
    if not Path(os.environ["AOSS_CONF"]).is_absolute():
        raise RuntimeError("AOSS_CONF must be an absolute path")
    if args.logical_shard_size % args.micro_shard_size != 0:
        raise ValueError(
            "--logical-shard-size must be an exact multiple of "
            "--micro-shard-size"
        )
    if args.download_prefetch < args.download_workers:
        raise ValueError(
            "--download-prefetch must be >= --download-workers"
        )
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    required_gpus = args.score_workers + int(args.quality_device == "cuda")
    if required_gpus > 8:
        raise ValueError(
            "this single-node layout supports at most 8 GPU workers in total"
        )

    input_path = args.input.resolve()
    output_root = require_safe_root(args.output_root)
    work_root = require_safe_root(args.work_root)
    cache_root = require_safe_root(args.cache_root)
    model_cache = args.model_cache.resolve()
    for generated in (output_root, work_root, cache_root):
        if generated == input_path:
            raise ValueError("input and generated roots must differ")
    for index, root in enumerate((output_root, work_root, cache_root)):
        for other in (output_root, work_root, cache_root)[index + 1 :]:
            if is_relative_to(root, other) or is_relative_to(other, root):
                raise ValueError(
                    f"generated roots must not overlap: {root}, {other}"
                )
    for model_path in (args.yolo_model, args.yolo_pose_model):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    if not model_cache.is_dir():
        raise FileNotFoundError(model_cache)

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    attempts_root = output_root / ".attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    failures_root = work_root / "failures"
    failures_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SCRIPT_DIRECTORY))
    from run_sharded_pipeline import (
        build_input_shards,
        discover_source_files,
        source_fingerprint,
    )

    source_files = discover_source_files(input_path)
    fingerprint, source_entries = source_fingerprint(source_files)
    manifest = build_input_shards(
        source_files,
        source_entries,
        fingerprint,
        args.logical_shard_size,
        args.micro_shard_size,
        args.input_adapter,
        work_root,
        args.rebuild_input_shards,
    )
    all_shards = manifest["shards"]
    shards = all_shards
    if args.logical_shard_index is not None:
        shards = [
            shard
            for shard in shards
            if shard["logical_index"] == args.logical_shard_index
        ]
        if not shards:
            raise ValueError(
                f"logical shard {args.logical_shard_index} does not exist"
            )

    pending = []
    skipped = 0
    for shard in shards:
        final_dir = (
            output_root
            / f"shard-{shard['logical_index']:06d}"
            / f"micro-{shard['micro_index']:04d}"
        )
        if valid_micro_success(final_dir, shard):
            print(
                f"[skip] logical={shard['logical_index']} "
                f"micro={shard['micro_index']}: valid SUCCESS",
                flush=True,
            )
            skipped += 1
        else:
            pending.append(shard)
    if not pending:
        for logical_index in sorted(
            {shard["logical_index"] for shard in shards}
        ):
            write_logical_success(
                output_root,
                logical_index,
                [
                    shard
                    for shard in all_shards
                    if shard["logical_index"] == logical_index
                ],
            )
        print(f"[done] all selected micro-shards complete; skipped={skipped}")
        return

    previous_cache = prepare_cache(cache_root)
    if previous_cache["files"] or previous_cache["bytes"]:
        print(
            "[cache-recovery] deleting stale cache from interrupted run: "
            f"files={previous_cache['files']} "
            f"bytes={previous_cache['bytes']}",
            flush=True,
        )
        cleanup_cache(cache_root)

    quality_token, score_gpu_tokens = resolve_worker_devices(
        args.score_workers,
        args.quality_device,
    )
    context = mp.get_context("spawn")
    quality_queue = context.Queue(maxsize=args.download_queue_size)
    score_queue = context.Queue(maxsize=args.score_queue_size)
    result_queue = context.Queue(maxsize=args.result_queue_size)
    ready_queue = context.Queue(maxsize=args.score_workers + 1)
    processes: List[mp.Process] = []

    quality_process = context.Process(
        target=quality_worker_main,
        kwargs={
            "device_token": quality_token,
            "quality_device": args.quality_device,
            "cpu_threads": args.quality_cpu_threads,
            "input_queue": quality_queue,
            "score_queue": score_queue,
            "result_queue": result_queue,
            "ready_queue": ready_queue,
            "cache_root": str(cache_root),
            "yolo_model": str(args.yolo_model.resolve()),
            "yolo_pose_model": str(args.yolo_pose_model.resolve()),
            "batch_size": args.quality_batch_size,
            "batch_wait_seconds": args.quality_batch_wait,
        },
        name=f"portrait-quality-{args.quality_device}",
    )
    processes.append(quality_process)
    for worker_index in range(args.score_workers):
        processes.append(
            context.Process(
                target=score_worker_main,
                kwargs={
                    "worker_index": worker_index,
                    "gpu_token": score_gpu_tokens[worker_index],
                    "input_queue": score_queue,
                    "result_queue": result_queue,
                    "ready_queue": ready_queue,
                    "cache_root": str(cache_root),
                    "model_name": args.model,
                    "model_cache": str(model_cache),
                    "input_size": args.input_size,
                    "max_num": args.max_num,
                    "use_flash_attn": not args.disable_flash_attn,
                    "allow_model_download": args.allow_model_download,
                },
                name=f"humanaesexpert-gpu-{worker_index}",
            )
        )

    started_workers = False
    try:
        for process in processes:
            process.start()
        started_workers = True
        wait_for_workers(
            ready_queue,
            processes,
            len(processes),
            args.worker_start_timeout,
        )
        downloader = DirectDownloader(
            cache_root,
            args.max_cache_files,
            args.max_cache_bytes,
            args.aoss_download_attempts,
            args.aoss_retry_initial_delay,
            args.aoss_retry_max_delay,
            args.aoss_retry_jitter,
        )
        completed = 0
        for shard in pending:
            ensure_workers_alive(processes)
            logical_dir = (
                output_root
                / f"shard-{shard['logical_index']:06d}"
            )
            final_dir = logical_dir / f"micro-{shard['micro_index']:04d}"
            logical_dir.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                safe_rmtree(final_dir, logical_dir)
            attempt_dir = (
                attempts_root
                / (
                    f"shard-{shard['logical_index']:06d}-"
                    f"micro-{shard['micro_index']:04d}."
                    f"{os.getpid()}.{uuid.uuid4().hex}"
                )
            )
            attempt_dir.mkdir()
            failure_path = failures_root / (
                f"shard-{shard['logical_index']:06d}-"
                f"micro-{shard['micro_index']:04d}.json"
            )
            print(
                f"[micro-start] logical={shard['logical_index']} "
                f"micro={shard['micro_index']} rows={shard['rows']}",
                flush=True,
            )
            try:
                records = read_jsonl(
                    Path(shard["path"]),
                    shard["rows"],
                )
                summary = process_micro_shard(
                    shard=shard,
                    output_path=attempt_dir / "data.jsonl",
                    records=records,
                    downloader=downloader,
                    download_workers=args.download_workers,
                    download_prefetch=args.download_prefetch,
                    quality_queue=quality_queue,
                    score_queue=score_queue,
                    result_queue=result_queue,
                    worker_processes=processes,
                    cache_root=cache_root,
                    progress_interval=args.progress_interval,
                    stall_timeout=args.stall_timeout,
                )
                cache = cache_snapshot(cache_root)
                if (
                    cache["files"]
                    or cache["bytes"]
                    or cache["references"]
                    or cache["partial_files"]
                ):
                    raise RuntimeError(
                        "cache is not empty after micro-shard: "
                        f"{json.dumps(cache, sort_keys=True)}"
                    )
                marker = {
                    "version": 1,
                    "mode": "direct-fused",
                    "logical_shard_index": shard["logical_index"],
                    "micro_shard_index": shard["index"],
                    "micro_index_within_logical_shard": shard[
                        "micro_index"
                    ],
                    "input_path": shard["path"],
                    "input_rows": shard["rows"],
                    "input_sha256": shard["sha256"],
                    **summary,
                    "cache": cache,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write_json(attempt_dir / "SUCCESS", marker)
                os.replace(attempt_dir, final_dir)
                failure_path.unlink(missing_ok=True)
                completed += 1
                atomic_write_json(
                    output_root / "PROGRESS.json",
                    {
                        "status": "running",
                        "completed_micro_shards": completed,
                        "skipped_micro_shards": skipped,
                        "pending_micro_shards": len(pending) - completed,
                        "last_completed": {
                            "logical_shard_index": shard["logical_index"],
                            "micro_index": shard["micro_index"],
                            **summary,
                        },
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    f"[micro-done] logical={shard['logical_index']} "
                    f"micro={shard['micro_index']} "
                    f"rate={summary['images_per_second']}_images/s",
                    flush=True,
                )
            except BaseException as error:
                if attempt_dir.exists():
                    safe_rmtree(attempt_dir, attempts_root)
                cleanup_cache(cache_root)
                atomic_write_json(
                    failure_path,
                    {
                        "version": 1,
                        "mode": "direct-fused",
                        "logical_shard_index": shard["logical_index"],
                        "micro_shard_index": shard["index"],
                        "micro_index_within_logical_shard": shard[
                            "micro_index"
                        ],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "failed_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )
                atomic_write_json(
                    output_root / "PROGRESS.json",
                    {
                        "status": "failed",
                        "completed_micro_shards": completed,
                        "skipped_micro_shards": skipped,
                        "failed_logical_shard_index": shard[
                            "logical_index"
                        ],
                        "failed_micro_index": shard["micro_index"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "updated_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )
                raise

        atomic_write_json(
            output_root / "PROGRESS.json",
            {
                "status": "complete",
                "completed_micro_shards": completed,
                "skipped_micro_shards": skipped,
                "pending_micro_shards": 0,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"[done] completed={completed} skipped={skipped}",
            flush=True,
        )
        selected_logical_indices = sorted(
            {shard["logical_index"] for shard in shards}
        )
        for logical_index in selected_logical_indices:
            logical_shards = [
                shard
                for shard in all_shards
                if shard["logical_index"] == logical_index
            ]
            write_logical_success(
                output_root,
                logical_index,
                logical_shards,
            )
    finally:
        if started_workers:
            try:
                quality_queue.put(STOP_MESSAGE, timeout=1.0)
            except queue.Full:
                pass
            for _ in range(args.score_workers):
                try:
                    score_queue.put(STOP_MESSAGE, timeout=1.0)
                except queue.Full:
                    break
            for process in processes:
                process.join(timeout=30.0)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=10.0)


if __name__ == "__main__":
    main()
