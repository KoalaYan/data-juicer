#!/usr/bin/env python3
"""Build a viewer-compatible conv.json from portrait quality mapper output."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DJ_META_KEY = "__dj__meta__"
QUALITY_KEY = "portrait_quality"


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            f"path must be absolute, got: {value}"
        )
    return path


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def list_value(record: Dict[str, Any], key: str) -> List[Any]:
    value = record.get(key)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def split_root(uri: str, configured_root: str = "") -> Tuple[str, str]:
    root = configured_root
    if root and uri.startswith(root):
        return root, uri[len(root) :].lstrip("/")
    if uri.startswith("s3://"):
        bucket, _, key = uri[5:].partition("/")
        root = f"s3://{bucket}/"
        return root, key
    return "", uri


def original_caption(record: Dict[str, Any]) -> str:
    for item in record.get("conversations") or []:
        if item.get("from") == "human" and isinstance(item.get("value"), str):
            return item["value"]
    return ""


def format_annotation_text(
    record: Dict[str, Any],
    quality: Dict[str, Any],
    image_uri: str,
    local_image: str,
) -> str:
    return (
        f"ID: {record.get('id', '')}\n"
        f"Image URI: {image_uri}\n"
        f"Local image: {local_image}\n"
        f"Hard-quality status: {quality.get('status')}\n"
        f"Human status: {quality.get('human_status')}\n"
        f"Reject reasons: {quality.get('reject_reasons', [])}\n"
        f"Warning reasons: {quality.get('warning_reasons', [])}\n"
        f"Person count: {quality.get('person_count')}\n"
        f"Face count: {quality.get('face_count')}\n"
        f"Sharpness: {quality.get('sharpness_score')}\n"
        f"Global exposure: {quality.get('global_exposure')}\n"
        f"Subject exposure: {quality.get('subject_exposure')}\n"
        f"Background exposure: {quality.get('background_exposure')}\n"
        f"Source metadata: {record.get('source_meta', record.get('__dj__source_file__', ''))}\n"
        f"Source offset: {record.get('source_offset', record.get('offset', ''))}\n"
        f"Source caption: {original_caption(record)}"
    )


def build_rows(
    records: List[Dict[str, Any]],
    image_key: str,
    source_image_key: str,
    configured_root: str,
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        meta = record.get(DJ_META_KEY) or record.get("meta") or {}
        quality_records = meta.get(QUALITY_KEY) or []
        local_images = list_value(record, image_key)
        source_images = list_value(record, source_image_key) or local_images

        for image_index, quality in enumerate(quality_records):
            local_image = local_images[image_index] if image_index < len(local_images) else ""
            if str(local_image).startswith("s3://"):
                local_image = ""
            image_uri = source_images[image_index] if image_index < len(source_images) else local_image
            root_hint = configured_root or record.get("image_root") or record.get("root") or ""
            root, relative_image = split_root(str(image_uri), str(root_hint))
            status = quality.get("status", "unknown")
            human_status = quality.get("human_status", "unknown")
            sample_id = record.get("id") or f"sample-{record_index:06d}"
            row = {
                "id": f"{sample_id}::{image_index}",
                "image": relative_image,
                "image_uri": image_uri,
                "image_root": root,
                "relative_image": relative_image,
                "local_image": local_image,
                "source_meta": record.get("source_meta", record.get("__dj__source_file__")),
                "source_offset": record.get("source_offset", record.get("offset")),
                "hard_quality_status": status,
                "human_status": human_status,
                "reject_reasons": quality.get("reject_reasons", []),
                "warning_reasons": quality.get("warning_reasons", []),
                "portrait_quality": quality,
                "width": quality.get("width", record.get("width")),
                "height": quality.get("height", record.get("height")),
                "conversations": [
                    {
                        "from": "human",
                        "value": format_annotation_text(record, quality, str(image_uri), str(local_image)),
                    },
                    {"from": "gpt", "value": "<image>"},
                ],
            }
            groups[(status, human_status, root)].append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=absolute_path,
        help="Scored JSONL exported by Data-Juicer",
    )
    parser.add_argument("--output-dir", required=True, type=absolute_path)
    parser.add_argument("--image-key", default="images", help="Local/cache image field")
    parser.add_argument("--source-image-key", default="source_images", help="Original URI field")
    parser.add_argument("--root", default="", help="Optional viewer root override")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = read_jsonl(args.input, args.limit)
    groups = build_rows(records, args.image_key, args.source_image_key, args.root)
    manifest = {}
    summary = {
        "input_records": len(records),
        "images": 0,
        "by_status": {},
        "by_human_status": {},
    }
    status_order = {"reject": 0, "uncertain": 1, "pass": 2, "unknown": 3}
    human_status_order = {
        "no_human": 0,
        "human_uncertain": 1,
        "human_present": 2,
        "portrait_clear": 3,
        "unknown": 4,
    }

    for group_index, ((status, human_status, root), rows) in enumerate(
        sorted(
            groups.items(),
            key=lambda item: (
                status_order.get(item[0][0], 99),
                human_status_order.get(item[0][1], 99),
                item[0][2],
            ),
        )
    ):
        rows.sort(key=lambda row: (row["reject_reasons"], row["warning_reasons"], row["id"]))
        annotation_path = (
            args.output_dir
            / "viewer_annotations"
            / f"{status}_{human_status}_{group_index:02d}.jsonl"
        )
        count = atomic_jsonl(annotation_path, rows)
        manifest[f"portrait_hard_quality_{status}_{human_status}_{group_index:02d}"] = {
            "root": root,
            "annotation": str(annotation_path.resolve()),
            "length": count,
            "repeat_time": 1,
        }
        summary["images"] += count
        summary["by_status"][status] = summary["by_status"].get(status, 0) + count
        summary["by_human_status"][human_status] = (
            summary["by_human_status"].get(human_status, 0) + count
        )

    atomic_json(args.output_dir / "conv.json", manifest)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    print((args.output_dir / "conv.json").resolve())


if __name__ == "__main__":
    main()
