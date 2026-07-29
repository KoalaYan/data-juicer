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
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([], [])
        result = op._analyze_image(Image.new("RGB", (64, 64), (128, 128, 128)))
        self.assertEqual(result["status"], "reject")
        self.assertIn("no_human", result["reject_reasons"])

    def test_background_only_overexposure_is_uncertain(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[20:80, 30:70] = 128
        op = ImagePortraitQualityMapper(
            detect_people=True,
            detect_faces_enabled=False,
            require_human=True,
        )
        op._detect_people = lambda image, rank=None: ([(30, 20, 70, 80)], [0.95])
        result = op._analyze_image(Image.fromarray(image))
        self.assertEqual(result["status"], "uncertain")
        self.assertNotIn("severe_subject_overexposure", result["reject_reasons"])
        self.assertIn("background_overexposed", result["warning_reasons"])


if __name__ == "__main__":
    unittest.main()
