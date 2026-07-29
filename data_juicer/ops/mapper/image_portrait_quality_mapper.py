from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    detect_faces,
    load_data_with_context,
    load_image,
)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

cv2 = LazyLoader("cv2", "opencv-contrib-python")

OP_NAME = "image_portrait_quality_mapper"


def _resize_for_analysis(image, max_side: int):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, 1.0, 1.0
    scale = max_side / float(longest)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=3,
    )
    return resized, resized.width / width, resized.height / height


def _luminance(rgb: np.ndarray) -> np.ndarray:
    rgb_float = rgb.astype(np.float32)
    return 0.2126 * rgb_float[..., 0] + 0.7152 * rgb_float[..., 1] + 0.0722 * rgb_float[..., 2]


def _exposure_metrics(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "mean": -1.0,
            "std": -1.0,
            "p01": -1.0,
            "p50": -1.0,
            "p99": -1.0,
            "highlight_clip_ratio": -1.0,
            "shadow_clip_ratio": -1.0,
            "dynamic_range": -1.0,
        }
    p01, p50, p99 = np.percentile(values, [1, 50, 99])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "highlight_clip_ratio": float(np.mean(values >= 250.0)),
        "shadow_clip_ratio": float(np.mean(values <= 5.0)),
        "dynamic_range": float(p99 - p01),
    }


def _sharpness_score(luma: np.ndarray) -> float:
    """Resolution-normalized gradient energy used as a conservative blur signal."""
    if min(luma.shape[:2]) < 3:
        return 0.0
    dx = np.diff(luma, axis=1)
    dy = np.diff(luma, axis=0)
    return float((np.mean(dx * dx) + np.mean(dy * dy)) / 2.0)


def _clip_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_metrics(
    box: Sequence[float],
    width: int,
    height: int,
    edge_margin_ratio: float,
) -> Dict:
    clipped = _clip_box(box, width, height)
    if clipped is None:
        return {
            "area_ratio": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "width_height_ratio": 0.0,
            "touches_edges": [],
        }
    x1, y1, x2, y2 = clipped
    box_width = x2 - x1
    box_height = y2 - y1
    x_margin = max(1, round(width * edge_margin_ratio))
    y_margin = max(1, round(height * edge_margin_ratio))
    touches_edges = []
    if x1 <= x_margin:
        touches_edges.append("left")
    if y1 <= y_margin:
        touches_edges.append("top")
    if x2 >= width - x_margin:
        touches_edges.append("right")
    if y2 >= height - y_margin:
        touches_edges.append("bottom")
    return {
        "area_ratio": float(box_width * box_height / max(1, width * height)),
        "width_ratio": float(box_width / max(1, width)),
        "height_ratio": float(box_height / max(1, height)),
        "width_height_ratio": float(box_width / max(1, box_height)),
        "touches_edges": touches_edges,
    }


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImagePortraitQualityMapper(Mapper):
    """Attach conservative hard-quality and portrait-presence signals to images.

    This lightweight mapper is intended for the first stage of portrait raw-data
    curation. It computes global, subject and background exposure metrics,
    gradient-based sharpness, and human-presence signals from a small YOLO person
    detector, an optional YOLO pose model, and an OpenCV frontal-face detector.
    Results are written to ``meta.portrait_quality`` as one record per input
    image.

    Human presence is classified independently as ``portrait_clear``,
    ``human_present``, ``human_uncertain`` or ``no_human``. The mapper also emits
    an overall ``pass``, ``uncertain`` or ``reject`` hard-quality status. Only
    high-confidence failures become ``reject``. Partial/cropped or low-confidence
    human evidence remains ``uncertain`` for a downstream VLM or human review.
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        output_key: str = MetaKeys.portrait_quality,
        delete_local_cache_after_processing: bool = False,
        local_cache_root: str = "",
        source_image_key: str = "source_images",
        detect_people: bool = True,
        detect_faces_enabled: bool = True,
        detect_pose: bool = False,
        require_human: bool = True,
        yolo_model_path: str = "yolo11n.pt",
        yolo_pose_model_path: str = "yolo11n-pose.pt",
        yolo_image_size: int = 640,
        inference_batch_size: int = 16,
        person_confidence: float = 0.35,
        strong_person_confidence: float = 0.55,
        pose_confidence: float = 0.35,
        pose_keypoint_confidence: float = 0.35,
        min_pose_keypoints: int = 4,
        face_classifier: str = "",
        face_scale_factor: float = 1.1,
        face_min_neighbors: int = 3,
        clear_face_area_ratio_min: float = 0.01,
        clear_face_sharpness_min: float = 20.0,
        edge_margin_ratio: float = 0.01,
        partial_person_width_height_ratio_min: float = 1.5,
        max_analysis_side: int = 1024,
        black_p99_max: float = 8.0,
        white_p01_min: float = 247.0,
        global_highlight_clip_min: float = 0.65,
        global_overexposure_dynamic_range_max: float = 40.0,
        subject_highlight_clip_min: float = 0.30,
        subject_overexposure_std_max: float = 15.0,
        background_highlight_clip_min: float = 0.70,
        normal_subject_highlight_clip_max: float = 0.15,
        min_sharpness_score: float = 0.0,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("memory", "1200MB")
        super().__init__(*args, **kwargs)
        if require_human and not (detect_people or detect_faces_enabled or detect_pose):
            raise ValueError("require_human=True needs at least one enabled human detector")
        if max_analysis_side < 64:
            raise ValueError("max_analysis_side must be at least 64")
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        if not 0 <= edge_margin_ratio < 0.5:
            raise ValueError("edge_margin_ratio must be in [0, 0.5)")
        if delete_local_cache_after_processing and not local_cache_root:
            raise ValueError(
                "local_cache_root is required when "
                "delete_local_cache_after_processing=True"
            )

        self.output_key = output_key
        self.delete_local_cache_after_processing = delete_local_cache_after_processing
        self.local_cache_root = (
            self._validate_cache_root(local_cache_root)
            if delete_local_cache_after_processing
            else ""
        )
        self.source_image_key = source_image_key
        self.detect_people = detect_people
        self.detect_faces_enabled = detect_faces_enabled
        self.detect_pose = detect_pose
        self.require_human = require_human
        self.yolo_image_size = yolo_image_size
        self.inference_batch_size = inference_batch_size
        self.person_confidence = person_confidence
        self.strong_person_confidence = strong_person_confidence
        self.pose_confidence = pose_confidence
        self.pose_keypoint_confidence = pose_keypoint_confidence
        self.min_pose_keypoints = min_pose_keypoints
        self.face_scale_factor = face_scale_factor
        self.face_min_neighbors = face_min_neighbors
        self.clear_face_area_ratio_min = clear_face_area_ratio_min
        self.clear_face_sharpness_min = clear_face_sharpness_min
        self.edge_margin_ratio = edge_margin_ratio
        self.partial_person_width_height_ratio_min = partial_person_width_height_ratio_min
        self.max_analysis_side = max_analysis_side

        self.black_p99_max = black_p99_max
        self.white_p01_min = white_p01_min
        self.global_highlight_clip_min = global_highlight_clip_min
        self.global_overexposure_dynamic_range_max = global_overexposure_dynamic_range_max
        self.subject_highlight_clip_min = subject_highlight_clip_min
        self.subject_overexposure_std_max = subject_overexposure_std_max
        self.background_highlight_clip_min = background_highlight_clip_min
        self.normal_subject_highlight_clip_max = normal_subject_highlight_clip_max
        self.min_sharpness_score = min_sharpness_score

        self.person_model_key = None
        if self.detect_people:
            self.person_model_key = prepare_model(model_type="yolo", model_path=yolo_model_path)

        self.pose_model_key = None
        if self.detect_pose:
            self.pose_model_key = prepare_model(model_type="yolo", model_path=yolo_pose_model_path)

        self.face_model_key = None
        if self.detect_faces_enabled:
            if not face_classifier:
                face_classifier = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
            self.face_model_key = prepare_model(model_type="opencv_classifier", model_path=face_classifier)

    @staticmethod
    def _validate_cache_root(cache_root: str) -> str:
        root = os.path.realpath(os.path.abspath(cache_root))
        forbidden_roots = {
            os.path.realpath(os.path.abspath(os.sep)),
            os.path.realpath(os.path.expanduser("~")),
        }
        if root in forbidden_roots:
            raise ValueError(f"Unsafe local_cache_root: {cache_root}")
        return root

    def _delete_cached_paths(self, paths) -> List[bool]:
        deleted = []
        for path in paths:
            if not isinstance(path, str) or path.startswith("s3://"):
                deleted.append(False)
                continue
            absolute_path = os.path.abspath(path)
            resolved_path = os.path.realpath(absolute_path)
            try:
                inside_cache = os.path.commonpath(
                    [self.local_cache_root, resolved_path]
                ) == self.local_cache_root
            except ValueError:
                inside_cache = False
            if not inside_cache or os.path.islink(absolute_path):
                logger.warning(
                    f"Refusing to delete path outside cache root or symlink: {path}"
                )
                deleted.append(False)
                continue
            try:
                if os.path.isfile(absolute_path):
                    os.remove(absolute_path)
                    deleted.append(True)
                else:
                    deleted.append(False)
            except OSError as e:
                logger.warning(f"Failed to delete portrait cache file {path}: {e}")
                deleted.append(False)
        return deleted

    def _cleanup_single_sample_cache(self, sample, local_paths):
        if not self.delete_local_cache_after_processing:
            return
        if self.source_image_key not in sample:
            raise ValueError(
                f"{self.source_image_key!r} is required to restore remote image "
                "URIs before deleting local cache files"
            )
        source_paths = sample[self.source_image_key]
        if not isinstance(source_paths, list):
            source_paths = [source_paths]
        if len(source_paths) != len(local_paths):
            raise ValueError(
                "Source image paths and local cached image paths must have "
                "the same length before cleanup"
            )
        deleted = self._delete_cached_paths(local_paths)
        quality_records = (sample.get(Fields.meta) or {}).get(self.output_key) or []
        for quality, was_deleted in zip(quality_records, deleted):
            quality["local_cache_deleted"] = was_deleted
        sample[self.image_key] = copy.deepcopy(sample[self.source_image_key])

    def _detect_people(self, image, rank=None) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        model = get_model(self.person_model_key, rank=rank, use_cuda=self.use_cuda())
        prediction = model(
            image,
            imgsz=self.yolo_image_size,
            conf=self.person_confidence,
            classes=[0],
            verbose=False,
        )[0]
        boxes = prediction.boxes.xyxy.detach().cpu().numpy().tolist()
        confidences = prediction.boxes.conf.detach().cpu().numpy().tolist()
        return boxes, confidences

    def _detect_people_batch(self, images, rank=None):
        model = get_model(self.person_model_key, rank=rank, use_cuda=self.use_cuda())
        predictions = model(
            images,
            imgsz=self.yolo_image_size,
            conf=self.person_confidence,
            classes=[0],
            verbose=False,
        )
        return [
            (
                prediction.boxes.xyxy.detach().cpu().numpy().tolist(),
                prediction.boxes.conf.detach().cpu().numpy().tolist(),
            )
            for prediction in predictions
        ]

    @staticmethod
    def _pose_prediction_to_records(prediction) -> List[Dict]:
        if prediction.keypoints is None:
            return []
        boxes = prediction.boxes.xyxy.detach().cpu().numpy().tolist()
        confidences = prediction.boxes.conf.detach().cpu().numpy().tolist()
        keypoints = prediction.keypoints.xy.detach().cpu().numpy().tolist()
        if prediction.keypoints.conf is None:
            keypoint_confidences = [
                [1.0 if x != 0.0 or y != 0.0 else 0.0 for x, y in points]
                for points in keypoints
            ]
        else:
            keypoint_confidences = prediction.keypoints.conf.detach().cpu().numpy().tolist()
        return [
            {
                "box_xyxy": box,
                "confidence": confidence,
                "keypoints_xy": points,
                "keypoint_confidences": point_confidences,
            }
            for box, confidence, points, point_confidences in zip(
                boxes,
                confidences,
                keypoints,
                keypoint_confidences,
            )
        ]

    def _detect_poses(self, image, rank=None) -> List[Dict]:
        model = get_model(self.pose_model_key, rank=rank, use_cuda=self.use_cuda())
        prediction = model(
            image,
            imgsz=self.yolo_image_size,
            conf=self.pose_confidence,
            verbose=False,
        )[0]
        return self._pose_prediction_to_records(prediction)

    def _detect_poses_batch(self, images, rank=None):
        model = get_model(self.pose_model_key, rank=rank, use_cuda=self.use_cuda())
        predictions = model(
            images,
            imgsz=self.yolo_image_size,
            conf=self.pose_confidence,
            verbose=False,
        )
        return [self._pose_prediction_to_records(prediction) for prediction in predictions]

    def _detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        model = get_model(self.face_model_key)
        detections = detect_faces(
            image,
            model,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=None,
            maxSize=None,
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    def _detect_faces_at_analysis_resolution(self, image) -> List[Tuple[int, int, int, int]]:
        resized, scale_x, scale_y = _resize_for_analysis(image, self.max_analysis_side)
        detections = self._detect_faces(resized)
        if scale_x == 1.0 and scale_y == 1.0:
            return detections
        return [
            (
                int(round(x / scale_x)),
                int(round(y / scale_y)),
                int(round(w / scale_x)),
                int(round(h / scale_y)),
            )
            for x, y, w, h in detections
        ]

    def _analyze_image(self, image, rank=None, detections=None) -> Dict:
        original_width, original_height = image.size
        if detections is None:
            person_boxes: List[Tuple[float, float, float, float]] = []
            person_confidences: List[float] = []
            face_boxes: List[Tuple[int, int, int, int]] = []
            poses: List[Dict] = []
            detection_errors: List[str] = []

            if self.detect_people:
                try:
                    person_boxes, person_confidences = self._detect_people(image, rank=rank)
                except Exception as e:
                    detection_errors.append(f"person_detector:{type(e).__name__}")
                    logger.warning(f"Portrait person detection failed: {e}")
            if self.detect_faces_enabled:
                try:
                    face_boxes = self._detect_faces_at_analysis_resolution(image)
                except Exception as e:
                    detection_errors.append(f"face_detector:{type(e).__name__}")
                    logger.warning(f"Portrait face detection failed: {e}")
            if self.detect_pose:
                try:
                    poses = self._detect_poses(image, rank=rank)
                except Exception as e:
                    detection_errors.append(f"pose_detector:{type(e).__name__}")
                    logger.warning(f"Portrait pose detection failed: {e}")
        else:
            person_boxes = detections["person_boxes"]
            person_confidences = detections["person_confidences"]
            face_boxes = detections["face_boxes"]
            poses = detections["poses"]
            detection_errors = detections["detection_errors"]

        resized, scale_x, scale_y = _resize_for_analysis(image, self.max_analysis_side)
        rgb = np.asarray(resized.convert("RGB"), dtype=np.uint8)
        luma = _luminance(rgb)
        global_metrics = _exposure_metrics(luma)

        scaled_person_boxes = [
            _clip_box(
                (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y),
                resized.width,
                resized.height,
            )
            for box in person_boxes
        ]
        scaled_face_boxes = [
            _clip_box(
                (x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y),
                resized.width,
                resized.height,
            )
            for x, y, w, h in face_boxes
        ]
        all_subject_boxes = [box for box in scaled_person_boxes + scaled_face_boxes if box is not None]

        person_metrics = []
        for box, confidence in zip(person_boxes, person_confidences):
            metrics = _box_metrics(
                box,
                original_width,
                original_height,
                self.edge_margin_ratio,
            )
            suspicious_partial = (
                "top" in metrics["touches_edges"]
                or metrics["width_height_ratio"] >= self.partial_person_width_height_ratio_min
            )
            person_metrics.append(
                {
                    **metrics,
                    "confidence": float(confidence),
                    "suspicious_partial": suspicious_partial,
                }
            )

        face_quality = []
        for original_box, scaled_box in zip(face_boxes, scaled_face_boxes):
            if scaled_box is None:
                continue
            x, y, w, h = original_box
            metrics = _box_metrics(
                (x, y, x + w, y + h),
                original_width,
                original_height,
                self.edge_margin_ratio,
            )
            x1, y1, x2, y2 = scaled_box
            face_sharpness = _sharpness_score(luma[y1:y2, x1:x2])
            clear = (
                metrics["area_ratio"] >= self.clear_face_area_ratio_min
                and face_sharpness >= self.clear_face_sharpness_min
                and not metrics["touches_edges"]
            )
            face_quality.append(
                {
                    **metrics,
                    "sharpness_score": face_sharpness,
                    "clear": clear,
                }
            )

        pose_quality = []
        for pose in poses:
            point_confidences = np.asarray(pose["keypoint_confidences"], dtype=np.float32)
            valid = point_confidences >= self.pose_keypoint_confidence
            valid_count = int(np.sum(valid))
            head_count = int(np.sum(valid[:5]))
            torso_count = int(np.sum(valid[[5, 6, 11, 12]])) if valid.size >= 13 else 0
            metrics = _box_metrics(
                pose["box_xyxy"],
                original_width,
                original_height,
                self.edge_margin_ratio,
            )
            suspicious_partial = (
                "top" in metrics["touches_edges"]
                or metrics["width_height_ratio"] >= self.partial_person_width_height_ratio_min
            )
            confident_human = (
                pose["confidence"] >= self.pose_confidence
                and valid_count >= self.min_pose_keypoints
                and (head_count >= 2 or torso_count >= 2)
            )
            pose_quality.append(
                {
                    **metrics,
                    "confidence": float(pose["confidence"]),
                    "valid_keypoint_count": valid_count,
                    "head_keypoint_count": head_count,
                    "torso_keypoint_count": torso_count,
                    "confident_human": confident_human,
                    "suspicious_partial": suspicious_partial,
                    "keypoints_xy": pose["keypoints_xy"],
                    "keypoint_confidences": pose["keypoint_confidences"],
                }
            )

        subject_mask = np.zeros(luma.shape, dtype=bool)
        for x1, y1, x2, y2 in all_subject_boxes:
            subject_mask[y1:y2, x1:x2] = True
        subject_metrics = _exposure_metrics(luma[subject_mask])
        background_metrics = _exposure_metrics(luma[~subject_mask]) if subject_mask.any() else _exposure_metrics([])

        reject_reasons: List[str] = []
        warning_reasons: List[str] = []
        if global_metrics["p99"] <= self.black_p99_max:
            reject_reasons.append("near_black")
        if global_metrics["p01"] >= self.white_p01_min:
            reject_reasons.append("near_white")
        if (
            global_metrics["highlight_clip_ratio"] >= self.global_highlight_clip_min
            and global_metrics["dynamic_range"] <= self.global_overexposure_dynamic_range_max
        ):
            reject_reasons.append("severe_global_overexposure")
        if (
            subject_metrics["highlight_clip_ratio"] >= self.subject_highlight_clip_min
            and subject_metrics["std"] <= self.subject_overexposure_std_max
        ):
            reject_reasons.append("severe_subject_overexposure")

        sharpness = _sharpness_score(luma)
        if self.min_sharpness_score > 0 and sharpness < self.min_sharpness_score:
            reject_reasons.append("severe_blur")

        human_evidence = bool(person_boxes or face_boxes or poses)
        enabled_detector_count = int(self.detect_people) + int(self.detect_faces_enabled) + int(self.detect_pose)
        detection_complete = len(detection_errors) == 0 and enabled_detector_count > 0

        clear_face_found = any(face["clear"] for face in face_quality)
        strong_complete_person = any(
            person["confidence"] >= self.strong_person_confidence and not person["suspicious_partial"]
            for person in person_metrics
        )
        strong_complete_pose = any(
            pose["confident_human"] and not pose["suspicious_partial"] for pose in pose_quality
        )
        suspicious_partial_human = any(
            item["suspicious_partial"] for item in person_metrics + pose_quality
        )

        if clear_face_found:
            human_status = "portrait_clear"
        elif face_quality or strong_complete_person or strong_complete_pose:
            human_status = "human_present"
        elif human_evidence or not detection_complete:
            human_status = "human_uncertain"
        else:
            human_status = "no_human"

        if self.require_human:
            if human_status == "no_human":
                reject_reasons.append("no_human")
            elif human_status == "human_uncertain":
                warning_reasons.append("human_presence_uncertain")
                if suspicious_partial_human:
                    warning_reasons.append("partial_human_or_bad_crop")
                if person_confidences and max(person_confidences) < self.strong_person_confidence:
                    warning_reasons.append("low_confidence_human_detection")

        if (
            background_metrics["highlight_clip_ratio"] >= self.background_highlight_clip_min
            and subject_metrics["highlight_clip_ratio"] >= 0
            and subject_metrics["highlight_clip_ratio"] <= self.normal_subject_highlight_clip_max
        ):
            warning_reasons.append("background_overexposed")
        if detection_errors:
            warning_reasons.append("detector_error")

        if reject_reasons:
            status = "reject"
        elif warning_reasons:
            status = "uncertain"
        else:
            status = "pass"

        return {
            "version": 2,
            "status": status,
            "human_status": human_status,
            "reject_reasons": reject_reasons,
            "warning_reasons": warning_reasons,
            "width": original_width,
            "height": original_height,
            "sharpness_score": sharpness,
            "person_count": len(person_boxes),
            "face_count": len(face_boxes),
            "max_person_confidence": float(max(person_confidences, default=0.0)),
            "human_found": human_status != "no_human",
            "detection_complete": detection_complete,
            "detection_errors": detection_errors,
            "global_exposure": global_metrics,
            "subject_exposure": subject_metrics,
            "background_exposure": background_metrics,
            "person_boxes_xyxy": [[float(value) for value in box] for box in person_boxes],
            "face_boxes_xywh": [[int(value) for value in box] for box in face_boxes],
            "person_quality": person_metrics,
            "face_quality": face_quality,
            "pose_quality": pose_quality,
        }

    def _run_batched_detectors(self, images, rank=None):
        batch_size = len(images)
        people = [([], []) for _ in range(batch_size)]
        faces = [[] for _ in range(batch_size)]
        poses = [[] for _ in range(batch_size)]
        errors = [[] for _ in range(batch_size)]

        if self.detect_people:
            try:
                people = self._detect_people_batch(images, rank=rank)
            except Exception as batch_error:
                logger.warning(f"Portrait batched person detection failed; retrying per image: {batch_error}")
                for idx, image in enumerate(images):
                    try:
                        people[idx] = self._detect_people(image, rank=rank)
                    except Exception as e:
                        errors[idx].append(f"person_detector:{type(e).__name__}")
                        logger.warning(f"Portrait person detection failed: {e}")

        if self.detect_faces_enabled:
            for idx, image in enumerate(images):
                try:
                    faces[idx] = self._detect_faces_at_analysis_resolution(image)
                except Exception as e:
                    errors[idx].append(f"face_detector:{type(e).__name__}")
                    logger.warning(f"Portrait face detection failed: {e}")

        if self.detect_pose:
            try:
                poses = self._detect_poses_batch(images, rank=rank)
            except Exception as batch_error:
                logger.warning(f"Portrait batched pose detection failed; retrying per image: {batch_error}")
                for idx, image in enumerate(images):
                    try:
                        poses[idx] = self._detect_poses(image, rank=rank)
                    except Exception as e:
                        errors[idx].append(f"pose_detector:{type(e).__name__}")
                        logger.warning(f"Portrait pose detection failed: {e}")

        return [
            {
                "person_boxes": person_boxes,
                "person_confidences": person_confidences,
                "face_boxes": image_faces,
                "poses": image_poses,
                "detection_errors": image_errors,
            }
            for (person_boxes, person_confidences), image_faces, image_poses, image_errors in zip(
                people,
                faces,
                poses,
                errors,
            )
        ]

    def process_single(self, sample, rank=None, context=False):
        if Fields.meta not in sample or sample[Fields.meta] is None:
            sample[Fields.meta] = {}
        if self.output_key in sample[Fields.meta]:
            return sample

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.meta][self.output_key] = []
            return sample

        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample,
            context,
            loaded_image_keys,
            load_image,
            mm_bytes_key=self.image_bytes_key,
        )
        sample[Fields.meta][self.output_key] = [
            self._analyze_image(images[key], rank=rank) for key in loaded_image_keys
        ]
        self._cleanup_single_sample_cache(sample, loaded_image_keys)
        return sample

    def process_batched(self, samples, rank=None, context=False):
        """Score all images in a sample batch with true YOLO batch inference."""
        if not samples:
            return samples
        num_samples = len(next(iter(samples.values())))
        if Fields.meta not in samples:
            samples[Fields.meta] = [{} for _ in range(num_samples)]
        else:
            samples[Fields.meta] = [
                meta if meta is not None else {} for meta in samples[Fields.meta]
            ]

        flat_images = []
        image_owners = []
        for sample_idx in range(num_samples):
            meta = samples[Fields.meta][sample_idx]
            if self.output_key in meta:
                continue
            sample_images = (
                samples[self.image_key][sample_idx]
                if self.image_key in samples and samples[self.image_key][sample_idx]
                else []
            )
            if not sample_images:
                meta[self.output_key] = []
                continue
            samples, loaded_images = load_data_with_context(
                samples,
                context,
                sample_images,
                load_image,
                mm_bytes_key=self.image_bytes_key,
                sample_idx=sample_idx,
            )
            for image_idx, image_key in enumerate(sample_images):
                flat_images.append(loaded_images[image_key])
                image_owners.append((sample_idx, image_idx))

        quality_by_sample = {}
        for start in range(0, len(flat_images), self.inference_batch_size):
            end = start + self.inference_batch_size
            image_chunk = flat_images[start:end]
            detection_chunk = self._run_batched_detectors(image_chunk, rank=rank)
            for image, detections, (sample_idx, image_idx) in zip(
                image_chunk,
                detection_chunk,
                image_owners[start:end],
            ):
                quality_by_sample.setdefault(sample_idx, []).append(
                    (
                        image_idx,
                        self._analyze_image(
                            image,
                            rank=rank,
                            detections=detections,
                        ),
                    )
                )

        for sample_idx, indexed_quality in quality_by_sample.items():
            indexed_quality.sort(key=lambda item: item[0])
            samples[Fields.meta][sample_idx][self.output_key] = [
                quality for _, quality in indexed_quality
            ]

        if self.delete_local_cache_after_processing:
            if self.source_image_key not in samples:
                raise ValueError(
                    f"{self.source_image_key!r} is required to restore remote "
                    "image URIs before deleting local cache files"
                )
            for sample_idx in range(num_samples):
                local_paths = (
                    samples[self.image_key][sample_idx]
                    if self.image_key in samples
                    else []
                )
                if not isinstance(local_paths, list):
                    local_paths = [local_paths]
                source_paths = samples[self.source_image_key][sample_idx]
                if not isinstance(source_paths, list):
                    source_paths = [source_paths]
                if len(source_paths) != len(local_paths):
                    raise ValueError(
                        "Source image paths and local cached image paths must "
                        "have the same length before cleanup"
                    )
                deleted = self._delete_cached_paths(local_paths)
                quality_records = (
                    samples[Fields.meta][sample_idx].get(self.output_key) or []
                )
                for quality, was_deleted in zip(quality_records, deleted):
                    quality["local_cache_deleted"] = was_deleted
                samples[self.image_key][sample_idx] = copy.deepcopy(
                    samples[self.source_image_key][sample_idx]
                )
        return samples
