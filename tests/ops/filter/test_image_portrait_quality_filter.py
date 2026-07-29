import unittest

from data_juicer.ops.filter.image_portrait_quality_filter import (
    ImagePortraitQualityFilter,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImagePortraitQualityFilterTest(DataJuicerTestCaseBase):

    def test_default_keeps_pass_and_uncertain(self):
        op = ImagePortraitQualityFilter()
        for status in ("pass", "uncertain"):
            sample = {
                Fields.meta: {
                    MetaKeys.portrait_quality: [{"status": status}],
                }
            }
            self.assertTrue(op.process_single(sample))

    def test_default_rejects_reject_status(self):
        op = ImagePortraitQualityFilter()
        sample = {
            Fields.meta: {
                MetaKeys.portrait_quality: [{"status": "reject"}],
            }
        }
        self.assertFalse(op.process_single(sample))

    def test_all_strategy_for_multiple_images(self):
        op = ImagePortraitQualityFilter(any_or_all="all")
        sample = {
            Fields.meta: {
                MetaKeys.portrait_quality: [
                    {"status": "pass"},
                    {"status": "reject"},
                ],
            }
        }
        self.assertFalse(op.process_single(sample))


if __name__ == "__main__":
    unittest.main()
