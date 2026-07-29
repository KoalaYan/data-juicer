import unittest

import numpy as np
from PIL import Image

from data_juicer.ops.mapper.image_portrait_quality_mapper import (
    ImagePortraitQualityMapper,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImagePortraitQualityMapperTest(DataJuicerTestCaseBase):

    def _rule_only_op(self, **kwargs):
        return ImagePortraitQualityMapper(
            detect_people=False,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=False,
            **kwargs,
        )

    def test_near_black_is_rejected(self):
        result = self._rule_only_op()._analyze_image(Image.new("RGB", (64, 64), (0, 0, 0)))
        self.assertEqual(result["status"], "reject")
        self.assertIn("near_black", result["reject_reasons"])

    def test_normal_image_passes_rule_only_gate(self):
        image = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1))[:, :, None].repeat(3, axis=2) * 3)
        result = self._rule_only_op()._analyze_image(image)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["reject_reasons"])

    def test_no_human_rejected_when_detection_completed(self):
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([], [])
        result = op._analyze_image(Image.new("RGB", (64, 64), (128, 128, 128)))
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["human_status"], "no_human")
        self.assertIn("no_human", result["reject_reasons"])

    def test_background_only_overexposure_is_uncertain(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[20:80, 30:70] = 128
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([(30, 20, 70, 80)], [0.95])
        result = op._analyze_image(Image.fromarray(image))
        self.assertEqual(result["status"], "uncertain")
        self.assertNotIn("severe_subject_overexposure", result["reject_reasons"])
        self.assertIn("background_overexposed", result["warning_reasons"])

    def test_clear_large_sharp_face_is_portrait_clear(self):
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        checkerboard = (np.indices((40, 40)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image[20:60, 30:70] = checkerboard[:, :, None]
        op = ImagePortraitQualityMapper(
            detect_people=False,
            detect_faces_enabled=True,
            detect_pose=False,
            require_human=True,
            clear_face_area_ratio_min=0.01,
            clear_face_sharpness_min=20.0,
        )
        op._detect_faces = lambda image: [(30, 20, 40, 40)]
        result = op._analyze_image(Image.fromarray(image))
        self.assertEqual(result["human_status"], "portrait_clear")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["face_quality"][0]["clear"])

    def test_strong_complete_person_is_human_present(self):
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([(30, 10, 70, 95)], [0.9])
        result = op._analyze_image(Image.new("RGB", (100, 100), (128, 128, 128)))
        self.assertEqual(result["human_status"], "human_present")
        self.assertEqual(result["status"], "pass")

    def test_low_confidence_partial_person_is_human_uncertain(self):
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([(0, 45, 100, 100)], [0.43])
        result = op._analyze_image(Image.new("RGB", (100, 100), (128, 128, 128)))
        self.assertEqual(result["human_status"], "human_uncertain")
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("partial_human_or_bad_crop", result["warning_reasons"])
        self.assertIn("low_confidence_human_detection", result["warning_reasons"])

    def test_top_cropped_people_are_human_uncertain(self):
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            detect_pose=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: (
            [(20, 0, 45, 80), (55, 0, 85, 70)],
            [0.91, 0.80],
        )
        result = op._analyze_image(Image.new("RGB", (100, 100), (128, 128, 128)))
        self.assertEqual(result["human_status"], "human_uncertain")
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("partial_human_or_bad_crop", result["warning_reasons"])


if __name__ == "__main__":
    unittest.main()
