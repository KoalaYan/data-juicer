from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image, ImageFile

from data_juicer.utils.cache_quota import (
    delete_file_and_release_cache_quota,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")
sentencepiece = LazyLoader("sentencepiece")

OP_NAME = "image_humanaesexpert_mapper"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

EXPERT_DIMENSIONS: Tuple[Tuple[str, str, str], ...] = (
    ("facial_brightness", "Facial Brightness", "面部亮度"),
    ("facial_feature_clarity", "Facial Feature Clarity", "五官清晰度"),
    ("facial_skin_tone", "Facial Skin Tone", "面部肤色"),
    ("facial_structure", "Facial Structure", "面部结构"),
    ("facial_contour_clarity", "Facial Contour Clarity", "面部轮廓清晰度"),
    ("facial_aesthetic_score", "Facial Aesthetic Score", "面部综合美学"),
    ("outfit", "Outfit", "服装"),
    ("body_shape", "Body Shape", "体型"),
    ("looks", "Looks", "外貌"),
    ("environment", "Environment", "环境"),
    (
        "general_appearance_aesthetic_score",
        "General Appearance Aesthetic Score",
        "整体外观美学",
    ),
    (
        "comprehensive_aesthetic_score",
        "Comprehensive Aesthetic Score",
        "综合美学总分",
    ),
)


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_diff:
            best_diff = difference
            best_ratio = ratio
        elif difference == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    max_num: int,
    image_size: int,
) -> List[Image.Image]:
    width, height = image.size
    target_ratios = sorted(
        {
            (columns, rows)
            for count in range(1, max_num + 1)
            for columns in range(1, count + 1)
            for rows in range(1, count + 1)
            if 1 <= columns * rows <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = _closest_aspect_ratio(
        width / height,
        target_ratios,
        width,
        height,
        image_size,
    )
    resized = image.resize(
        (image_size * columns, image_size * rows),
        resample=Image.Resampling.BICUBIC,
    )
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(
            resized.crop(
                (
                    left,
                    top,
                    left + image_size,
                    top + image_size,
                )
            )
        )
    if len(tiles) != 1:
        tiles.append(
            image.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        )
    return tiles


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ImageHumanAesExpertMapper(Mapper):
    """Attach official HumanAesExpert-8B Expert Head 12D scores.

    The operator calls the model's trusted remote-code
    ``expert_score(tokenizer, pixel_values)`` method and verifies all returned
    values against the official English-key mapping. One score record is
    written per image to ``meta.humanaesexpert_expert_scores``.
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        model_name_or_path: str = "KlingTeam/HumanAesExpert-8B",
        model_cache_dir: str = "",
        output_key: str = MetaKeys.humanaesexpert_expert_scores,
        input_size: int = 448,
        max_num: int = 12,
        local_files_only: bool = True,
        delete_local_cache_after_processing: bool = False,
        local_cache_root: str = "",
        source_image_key: str = "source_images",
        eligibility_key: str = "humanaesexpert_eligible",
        skip_ineligible: bool = False,
        required_transformers_version: str = "4.44.2",
        allow_unsupported_transformers: bool = False,
        required_sentencepiece_version: str = "0.2.0",
        allow_unsupported_sentencepiece: bool = False,
        persistent_actor_pool: bool = False,
        persistent_actor_namespace: str = "portrait-quality-gate",
        persistent_actor_prefix: str = "humanaesexpert",
        persistent_actor_pool_size: int = 7,
        *args,
        **kwargs,
    ):
        if persistent_actor_pool:
            kwargs.setdefault("memory", "1GB")
            kwargs.setdefault("accelerator", "cpu")
            kwargs.setdefault("ray_execution_mode", "actor")
        else:
            kwargs.setdefault("memory", "24GB")
        super().__init__(*args, **kwargs)
        if input_size < 64:
            raise ValueError("input_size must be at least 64")
        if max_num < 1:
            raise ValueError("max_num must be positive")
        if delete_local_cache_after_processing and not local_cache_root:
            raise ValueError(
                "local_cache_root is required when "
                "delete_local_cache_after_processing=True"
            )
        self.model_name_or_path = model_name_or_path
        self.model_cache_dir = model_cache_dir or os.environ.get(
            "DATA_JUICER_MODELS_CACHE",
            "",
        )
        self.output_key = output_key
        self.input_size = int(input_size)
        self.max_num = int(max_num)
        self.local_files_only = bool(local_files_only)
        self.delete_local_cache_after_processing = (
            delete_local_cache_after_processing
        )
        self.local_cache_root = (
            self._validate_cache_root(local_cache_root)
            if delete_local_cache_after_processing
            else ""
        )
        self.source_image_key = source_image_key
        self.eligibility_key = eligibility_key
        self.skip_ineligible = bool(skip_ineligible)
        self.required_transformers_version = required_transformers_version
        self.allow_unsupported_transformers = bool(
            allow_unsupported_transformers
        )
        self.required_sentencepiece_version = (
            required_sentencepiece_version
        )
        self.allow_unsupported_sentencepiece = bool(
            allow_unsupported_sentencepiece
        )
        if persistent_actor_pool_size < 1:
            raise ValueError("persistent_actor_pool_size must be positive")
        self.persistent_actor_pool = bool(persistent_actor_pool)
        self.persistent_actor_namespace = persistent_actor_namespace
        self.persistent_actor_prefix = persistent_actor_prefix
        self.persistent_actor_pool_size = int(persistent_actor_pool_size)
        self._model = None
        self._tokenizer = None
        self._persistent_handles = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        state["_persistent_handles"] = None
        return state

    def _get_persistent_handles(self):
        if self._persistent_handles is None:
            import ray

            from data_juicer.utils.humanaesexpert_ray_pool import (
                get_pool_handles,
            )

            self._persistent_handles = get_pool_handles(
                ray,
                self.persistent_actor_namespace,
                self.persistent_actor_prefix,
                self.persistent_actor_pool_size,
            )
        return self._persistent_handles

    def _score_paths(self, paths: Sequence[str]) -> List[Dict]:
        if not self.persistent_actor_pool:
            return [self._score_image(path) for path in paths]
        import ray

        handles = self._get_persistent_handles()
        references = [
            handles[index % len(handles)].score_image.remote(path)
            for index, path in enumerate(paths)
        ]
        return ray.get(references)

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

    def _load_model(self):
        if self._model is not None:
            return self._model, self._tokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("HumanAesExpert-8B Expert Head requires CUDA")
        installed_version = transformers.__version__.split("+", 1)[0]
        if (
            self.required_transformers_version
            and installed_version != self.required_transformers_version
            and not self.allow_unsupported_transformers
        ):
            raise RuntimeError(
                "HumanAesExpert requires transformers=="
                f"{self.required_transformers_version}; found "
                f"{installed_version}. Use a dedicated pipeline environment "
                "or explicitly set allow_unsupported_transformers=True after "
                "validating the model."
            )
        installed_sentencepiece = sentencepiece.__version__.split("+", 1)[0]
        if (
            self.required_sentencepiece_version
            and installed_sentencepiece
            != self.required_sentencepiece_version
            and not self.allow_unsupported_sentencepiece
        ):
            raise RuntimeError(
                "HumanAesExpert tokenizer requires sentencepiece=="
                f"{self.required_sentencepiece_version}; found "
                f"{installed_sentencepiece}. A newer release can fail with "
                "'piece must not include null character'."
            )
        cache_dir = self.model_cache_dir or None
        self._model = (
            transformers.AutoModel.from_pretrained(
                self.model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            .eval()
            .cuda()
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=self.local_files_only,
        )
        return self._model, self._tokenizer

    def _load_image_tensor(self, path: str):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensors = []
        for tile in _dynamic_preprocess(
            image,
            max_num=self.max_num,
            image_size=self.input_size,
        ):
            values = np.asarray(tile, dtype=np.float32) / 255.0
            values = (values - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(
                torch.from_numpy(
                    np.ascontiguousarray(values.transpose(2, 0, 1))
                )
            )
        return torch.stack(tensors)

    def _score_image(self, path: str) -> Dict:
        model, tokenizer = self._load_model()
        pixel_values = self._load_image_tensor(path).to(
            dtype=torch.float16,
            device="cuda",
        )
        with torch.inference_mode():
            score_tensor, official_mapping = model.expert_score(
                tokenizer,
                pixel_values,
            )
        values = score_tensor.detach().float().cpu().reshape(-1).tolist()
        record = self._build_score_record(
            values,
            official_mapping,
            int(pixel_values.shape[0]),
        )
        record["transformers_version"] = transformers.__version__
        record["sentencepiece_version"] = sentencepiece.__version__
        return record

    def _build_score_record(
        self,
        values,
        official_mapping,
        tile_count: int,
    ) -> Dict:
        if len(values) != len(EXPERT_DIMENSIONS):
            raise RuntimeError(
                f"expected {len(EXPERT_DIMENSIONS)} Expert Head scores, "
                f"got {len(values)}"
            )
        dimensions = {}
        for index, (key, english, _) in enumerate(EXPERT_DIMENSIONS):
            value = float(values[index])
            if not math.isfinite(value):
                raise ValueError(f"non-finite Expert Head score: {key}={value}")
            official = float(official_mapping[english])
            if abs(official - value) > 1e-6:
                raise RuntimeError(
                    f"official Expert Head mapping mismatch: {english}"
                )
            dimensions[key] = value
        return {
            "version": 1,
            "model": self.model_name_or_path,
            "head": "expert_head_12d",
            "method": "official expert_score() 12-dimensional Expert Head",
            "score": dimensions["comprehensive_aesthetic_score"],
            "dimensions": dimensions,
            "input_size": self.input_size,
            "max_num": self.max_num,
            "tile_count": int(tile_count),
            "torch_dtype": "float16",
        }

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

    def process_single(self, sample, rank=None):
        if Fields.meta not in sample or sample[Fields.meta] is None:
            sample[Fields.meta] = {}
        if self.output_key in sample[Fields.meta]:
            return sample
        local_paths = sample.get(self.image_key) or []
        if not isinstance(local_paths, list):
            local_paths = [local_paths]
        source_paths = sample.get(self.source_image_key) or []
        if not isinstance(source_paths, list):
            source_paths = [source_paths]
        if self.delete_local_cache_after_processing and len(source_paths) != len(
            local_paths
        ):
            raise ValueError(
                "Source image paths and local cached image paths must have "
                "the same length before cleanup"
            )
        quality_records = (
            sample[Fields.meta].get(MetaKeys.portrait_quality) or []
        )
        if self.skip_ineligible and len(quality_records) != len(local_paths):
            raise ValueError(
                "Portrait-quality records and image paths must have the same "
                "length when skip_ineligible=True"
            )
        scores = []
        for image_index, local_path in enumerate(local_paths):
            eligible = (
                not self.skip_ineligible
                or bool(
                    quality_records[image_index].get(self.eligibility_key)
                )
            )
            if not eligible:
                scores.append(None)
                continue
            score = self._score_paths([local_path])[0]
            if self.delete_local_cache_after_processing:
                score["local_cache_deleted"] = self._delete_cached_path(
                    local_path
                )
            scores.append(score)
        sample[Fields.meta][self.output_key] = scores
        if self.delete_local_cache_after_processing:
            sample[self.image_key] = copy.deepcopy(
                sample[self.source_image_key]
            )
        return sample

    def process_batched(self, samples, rank=None):
        if not samples:
            return samples
        sample_count = len(next(iter(samples.values())))
        if Fields.meta not in samples:
            samples[Fields.meta] = [{} for _ in range(sample_count)]
        else:
            samples[Fields.meta] = [
                meta if meta is not None else {}
                for meta in samples[Fields.meta]
            ]
        if self.persistent_actor_pool:
            return self._process_batched_with_persistent_pool(
                samples,
                sample_count,
            )
        for sample_index in range(sample_count):
            sample = {
                key: values[sample_index]
                for key, values in samples.items()
            }
            sample = self.process_single(sample, rank=rank)
            for key, value in sample.items():
                if key not in samples:
                    samples[key] = [None] * sample_count
                samples[key][sample_index] = value
        return samples

    def _process_batched_with_persistent_pool(self, samples, sample_count):
        pending = []
        prepared = []
        for sample_index in range(sample_count):
            meta = samples[Fields.meta][sample_index]
            if self.output_key in meta:
                prepared.append(None)
                continue
            local_paths = samples.get(self.image_key, [None] * sample_count)[
                sample_index
            ] or []
            if not isinstance(local_paths, list):
                local_paths = [local_paths]
            source_paths = samples.get(
                self.source_image_key,
                [None] * sample_count,
            )[sample_index] or []
            if not isinstance(source_paths, list):
                source_paths = [source_paths]
            if self.delete_local_cache_after_processing and len(
                source_paths
            ) != len(local_paths):
                raise ValueError(
                    "Source image paths and local cached image paths must "
                    "have the same length before cleanup"
                )
            quality_records = meta.get(MetaKeys.portrait_quality) or []
            if self.skip_ineligible and len(quality_records) != len(
                local_paths
            ):
                raise ValueError(
                    "Portrait-quality records and image paths must have the "
                    "same length when skip_ineligible=True"
                )
            scores = [None] * len(local_paths)
            prepared.append((local_paths, source_paths, scores))
            for image_index, local_path in enumerate(local_paths):
                eligible = (
                    not self.skip_ineligible
                    or bool(
                        quality_records[image_index].get(
                            self.eligibility_key
                        )
                    )
                )
                if eligible:
                    pending.append(
                        (sample_index, image_index, local_path)
                    )

        score_records = self._score_paths(
            [item[2] for item in pending]
        )
        for (sample_index, image_index, local_path), score in zip(
            pending,
            score_records,
        ):
            if self.delete_local_cache_after_processing:
                score["local_cache_deleted"] = self._delete_cached_path(
                    local_path
                )
            prepared[sample_index][2][image_index] = score

        for sample_index, item in enumerate(prepared):
            if item is None:
                continue
            _, source_paths, scores = item
            samples[Fields.meta][sample_index][self.output_key] = scores
            if self.delete_local_cache_after_processing:
                samples[self.image_key][sample_index] = copy.deepcopy(
                    source_paths
                )
        return samples
