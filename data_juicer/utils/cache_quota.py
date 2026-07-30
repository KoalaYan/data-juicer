"""Cross-process quota accounting for ephemeral file caches.

The quota state lives inside the cache root and is protected with ``flock``.
It is intended for producer/consumer pipelines where one operator downloads
files and a downstream operator deletes them after successful processing.
"""

from __future__ import annotations

import errno
import json
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import fcntl


_STATE_NAME = ".data_juicer_cache_quota.json"
_LOCK_NAME = ".data_juicer_cache_quota.lock"
_CONTROL_NAMES = {_STATE_NAME, _LOCK_NAME}
_PARTIAL_FILE_PATTERN = re.compile(r"\.part\.\d+\.\d+$")


class FileCacheQuota:
    """Reserve bounded file-count and byte capacity across worker processes."""

    def __init__(
        self,
        cache_root: str,
        max_files: int = 0,
        max_bytes: int = 0,
        wait_timeout: float = 1800.0,
        poll_interval: float = 1.0,
        stale_reservation_seconds: float = 600.0,
    ):
        if max_files < 0 or max_bytes < 0:
            raise ValueError("cache quota limits must be non-negative")
        if max_files == 0 and max_bytes == 0:
            raise ValueError("at least one cache quota limit must be positive")
        if wait_timeout <= 0 or poll_interval <= 0:
            raise ValueError("cache quota wait settings must be positive")
        self.cache_root = os.path.realpath(os.path.abspath(cache_root))
        self.max_files = int(max_files)
        self.max_bytes = int(max_bytes)
        self.wait_timeout = float(wait_timeout)
        self.poll_interval = float(poll_interval)
        self.stale_reservation_seconds = float(stale_reservation_seconds)
        os.makedirs(self.cache_root, exist_ok=True)
        self.state_path = os.path.join(self.cache_root, _STATE_NAME)
        self.lock_path = os.path.join(self.cache_root, _LOCK_NAME)

    def _safe_path(self, path: str) -> str:
        resolved = os.path.realpath(os.path.abspath(path))
        try:
            inside = os.path.commonpath([self.cache_root, resolved]) == self.cache_root
        except ValueError:
            inside = False
        if not inside or resolved == self.cache_root:
            raise ValueError(f"cache quota path is outside cache root: {path}")
        return resolved

    @contextmanager
    def _locked(self):
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            deadline = time.monotonic() + self.wait_timeout
            delay = 0.01
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "timed out acquiring file cache quota lock: "
                            f"{self.lock_path}"
                        ) from error
                    sleep_for = min(
                        remaining,
                        delay + random.uniform(0.0, delay),
                    )
                    time.sleep(sleep_for)
                    delay = min(0.25, delay * 2)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _scan_entries(self) -> Dict[str, dict]:
        entries = {}
        for directory, _, filenames in os.walk(self.cache_root):
            for filename in filenames:
                if directory == self.cache_root and filename in _CONTROL_NAMES:
                    continue
                path = os.path.realpath(os.path.join(directory, filename))
                if not os.path.isfile(path) or os.path.islink(path):
                    continue
                entries[path] = {
                    "bytes": os.path.getsize(path),
                    "references": 0,
                    "state": "ready",
                    "updated_at": time.time(),
                }
        return entries

    def _load_state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as state_file:
                state = json.load(state_file)
            if state.get("version") != 1 or not isinstance(state.get("entries"), dict):
                raise ValueError("unsupported cache quota state")
            return state
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {"version": 1, "entries": self._scan_entries()}

    def _write_state(self, state: dict) -> None:
        temporary = (
            f"{self.state_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        with open(temporary, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, ensure_ascii=False, separators=(",", ":"))
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary, self.state_path)

    def _reconcile(self, state: dict) -> None:
        now = time.time()
        reconciled = {}
        for path, entry in state["entries"].items():
            if os.path.isfile(path) and not os.path.islink(path):
                reconciled[path] = {
                    "bytes": os.path.getsize(path),
                    "references": int(entry.get("references", 0)),
                    "state": "ready",
                    "updated_at": now,
                }
                continue
            updated_at = float(entry.get("updated_at", 0.0))
            if (
                entry.get("state") == "reserved"
                and now - updated_at < self.stale_reservation_seconds
            ):
                reconciled[path] = entry
        state["entries"] = reconciled

    def _has_capacity(self, entries: Dict[str, dict], size: int) -> bool:
        file_count = len(entries)
        byte_count = sum(int(entry.get("bytes", 0)) for entry in entries.values())
        if self.max_files and file_count + 1 > self.max_files:
            return False
        if self.max_bytes and byte_count + size > self.max_bytes:
            return False
        return True

    def _evict_unreferenced(
        self,
        entries: Dict[str, dict],
        size: int,
    ) -> None:
        candidates = sorted(
            (
                (path, entry)
                for path, entry in entries.items()
                if int(entry.get("references", 0)) <= 0
                and entry.get("state") == "ready"
            ),
            key=lambda item: float(item[1].get("updated_at", 0.0)),
        )
        for path, _ in candidates:
            if self._has_capacity(entries, size):
                break
            if os.path.isfile(path) and not os.path.islink(path):
                os.remove(path)
            entries.pop(path, None)

    def acquire(self, path: str, size: int) -> bool:
        """Reserve capacity for ``path``.

        Returns ``True`` when the caller owns a new reservation and should
        write the file. Returns ``False`` when another worker has already made
        the same file available.
        """
        if size < 0:
            raise ValueError("cache reservation size must be non-negative")
        if self.max_bytes and size > self.max_bytes:
            raise ValueError(
                f"single file size {size} exceeds cache byte limit {self.max_bytes}"
            )
        path = self._safe_path(path)
        deadline = time.monotonic() + self.wait_timeout
        while True:
            with self._locked():
                state = self._load_state()
                self._reconcile(state)
                entry = state["entries"].get(path)
                if os.path.isfile(path):
                    state["entries"][path] = {
                        "bytes": os.path.getsize(path),
                        "references": int(
                            (entry or {}).get("references", 0)
                        )
                        + 1,
                        "state": "ready",
                        "updated_at": time.time(),
                    }
                    self._write_state(state)
                    return False
                self._evict_unreferenced(state["entries"], size)
                if entry is None and self._has_capacity(state["entries"], size):
                    state["entries"][path] = {
                        "bytes": int(size),
                        "references": 1,
                        "state": "reserved",
                        "updated_at": time.time(),
                    }
                    self._write_state(state)
                    return True
                self._write_state(state)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for file cache quota capacity: "
                    f"root={self.cache_root} max_files={self.max_files} "
                    f"max_bytes={self.max_bytes}"
                )
            time.sleep(self.poll_interval)

    def mark_ready(self, path: str) -> None:
        path = self._safe_path(path)
        with self._locked():
            state = self._load_state()
            self._reconcile(state)
            if os.path.isfile(path):
                state["entries"][path] = {
                    "bytes": os.path.getsize(path),
                    "references": int(
                        state["entries"].get(path, {}).get("references", 1)
                    ),
                    "state": "ready",
                    "updated_at": time.time(),
                }
            self._write_state(state)

    def retain(self, path: str) -> None:
        """Register another downstream consumer for an existing cache file."""
        path = self._safe_path(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with self._locked():
            state = self._load_state()
            self._reconcile(state)
            entry = state["entries"].get(path, {})
            state["entries"][path] = {
                "bytes": os.path.getsize(path),
                "references": int(entry.get("references", 0)) + 1,
                "state": "ready",
                "updated_at": time.time(),
            }
            self._write_state(state)

    def release(self, path: str) -> None:
        path = self._safe_path(path)
        with self._locked():
            state = self._load_state()
            state["entries"].pop(path, None)
            self._reconcile(state)
            self._write_state(state)

    def delete_after_consume(self, path: str) -> bool:
        """Release one reference and delete the file after the last consumer."""
        path = self._safe_path(path)
        with self._locked():
            state = self._load_state()
            self._reconcile(state)
            entry = state["entries"].get(path)
            if entry is not None and int(entry.get("references", 1)) > 1:
                entry["references"] = int(entry["references"]) - 1
                entry["updated_at"] = time.time()
                self._write_state(state)
                return False
            deleted = False
            if os.path.isfile(path) and not os.path.islink(path):
                os.remove(path)
                deleted = True
            state["entries"].pop(path, None)
            self._write_state(state)
            return deleted

    def snapshot(self) -> dict:
        """Return reconciled quota usage for diagnostics and tests."""
        with self._locked():
            state = self._load_state()
            self._reconcile(state)
            self._write_state(state)
            entries = state["entries"]
            return {
                "files": len(entries),
                "bytes": sum(
                    int(entry.get("bytes", 0))
                    for entry in entries.values()
                ),
                "references": sum(
                    int(entry.get("references", 0))
                    for entry in entries.values()
                ),
            }

    def prepare_for_new_run(self) -> dict:
        """Reset stale consumers for an exclusively owned cache directory.

        Complete files remain available for ``resume_download`` but start with
        zero references and can be evicted under quota pressure. Private
        temporary files left by an interrupted downloader are removed.
        Call this once in the driver before starting any workers.
        """
        with self._locked():
            state = self._load_state()
            for directory, _, filenames in os.walk(self.cache_root):
                for filename in filenames:
                    if not _PARTIAL_FILE_PATTERN.search(filename):
                        continue
                    path = os.path.join(directory, filename)
                    if os.path.isfile(path) and not os.path.islink(path):
                        os.remove(path)
            self._reconcile(state)
            for entry in state["entries"].values():
                entry["references"] = 0
                entry["updated_at"] = time.time()
            self._write_state(state)
            return {
                "files": len(state["entries"]),
                "bytes": sum(
                    int(entry.get("bytes", 0))
                    for entry in state["entries"].values()
                ),
            }


def release_file_cache_quota(cache_root: str, path: str) -> None:
    """Release ``path`` from an existing quota state, if one is active."""
    state_path = os.path.join(
        os.path.realpath(os.path.abspath(cache_root)),
        _STATE_NAME,
    )
    if not os.path.isfile(state_path):
        return
    quota = FileCacheQuota(cache_root, max_files=1)
    quota.release(path)


def delete_file_and_release_cache_quota(cache_root: str, path: str) -> bool:
    """Delete a cache file, respecting duplicate-reference accounting."""
    state_path = os.path.join(
        os.path.realpath(os.path.abspath(cache_root)),
        _STATE_NAME,
    )
    if not os.path.isfile(state_path):
        if os.path.isfile(path) and not os.path.islink(path):
            os.remove(path)
            return True
        return False
    quota = FileCacheQuota(cache_root, max_files=1)
    return quota.delete_after_consume(path)
