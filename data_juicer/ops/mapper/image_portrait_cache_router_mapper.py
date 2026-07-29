from __future__ import annotations

import copy
import os
from typing import List, Union

from loguru import logger

from data_juicer.utils.cache_quota import (
    delete_file_and_release_cache_quota,
)
from data_juicer.utils.constant import Fields, MetaKeys

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

OP_NAME = "image_portrait_cache_router_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ImagePortraitCacheRouterMapper(Mapper):
    """Release cached images that do not need expensive portrait scoring.

    Eligible images keep their local paths for the downstream GPU scorer.
    Ineligible images are deleted immediately and restored to their remote URI.
    Row cardinality and image order are preserved.
    """

    _batched_op = True

    def __init__(
        self,
        quality_key: str = MetaKeys.portrait_quality,
        keep_human_statuses: Union[str, List[str]] = (
            "portrait_clear",
            "human_present",
        ),
        local_cache_root: str = "",
        source_image_key: str = "source_images",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(keep_human_statuses, str):
            keep_human_statuses = [keep_human_statuses]
        if not keep_human_statuses:
            raise ValueError("keep_human_statuses must not be empty")
        if not local_cache_root:
            raise ValueError("local_cache_root is required")
        self.quality_key = quality_key
        self.keep_human_statuses = set(keep_human_statuses)
        self.local_cache_root = self._validate_cache_root(local_cache_root)
        self.source_image_key = source_image_key

    @staticmethod
    def _validate_cache_root(cache_root: str) -> str:
        root = os.path.realpath(os.path.abspath(cache_root))
        forbidden = {
            os.path.realpath(os.path.abspath(os.sep)),
            os.path.realpath(os.path.expanduser("~")),
        }
        if root in forbidden:
            raise ValueError(f"Unsafe local_cache_root: {cache_root}")
        return root

    def _delete_cached_path(self, path: str) -> bool:
        if not isinstance(path, str) or path.startswith("s3://"):
            return False
        absolute_path = os.path.abspath(path)
        resolved_path = os.path.realpath(absolute_path)
        try:
            inside = os.path.commonpath(
                [self.local_cache_root, resolved_path]
            ) == self.local_cache_root
        except ValueError:
            inside = False
        if not inside or os.path.islink(absolute_path):
            logger.warning(
                f"Refusing to delete path outside cache root or symlink: {path}"
            )
            return False
        if not os.path.isfile(absolute_path):
            return False
        return delete_file_and_release_cache_quota(
            self.local_cache_root,
            resolved_path,
        )

    def process_single(self, sample):
        meta = sample.get(Fields.meta) or {}
        records = meta.get(self.quality_key) or []
        local_paths = sample.get(self.image_key) or []
        source_paths = sample.get(self.source_image_key) or []
        if not isinstance(local_paths, list):
            local_paths = [local_paths]
        if not isinstance(source_paths, list):
            source_paths = [source_paths]
        if len(local_paths) != len(source_paths):
            raise ValueError(
                "Source image paths and local cached image paths must have "
                "the same length"
            )
        if len(records) != len(local_paths):
            raise ValueError(
                "Portrait-quality records and image paths must have the same "
                "length"
            )

        routed_paths = copy.deepcopy(local_paths)
        for index, (path, source, record) in enumerate(
            zip(local_paths, source_paths, records)
        ):
            eligible = record.get("human_status") in self.keep_human_statuses
            record["humanaesexpert_eligible"] = eligible
            if not eligible:
                record["local_cache_deleted"] = self._delete_cached_path(path)
                routed_paths[index] = source
        sample[self.image_key] = routed_paths
        return sample

    def process_batched(self, samples):
        if not samples:
            return samples
        sample_count = len(next(iter(samples.values())))
        for sample_index in range(sample_count):
            sample = {
                key: values[sample_index]
                for key, values in samples.items()
            }
            sample = self.process_single(sample)
            for key, value in sample.items():
                if key not in samples:
                    samples[key] = [None] * sample_count
                samples[key][sample_index] = value
        return samples
