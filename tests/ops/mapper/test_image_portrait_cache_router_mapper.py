import os
import tempfile

from data_juicer.ops.mapper.image_portrait_cache_router_mapper import (
    ImagePortraitCacheRouterMapper,
)
from data_juicer.utils.cache_quota import FileCacheQuota
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImagePortraitCacheRouterMapperTest(DataJuicerTestCaseBase):

    def test_releases_ineligible_and_preserves_eligible_cache(self):
        with tempfile.TemporaryDirectory() as cache_root:
            clear_path = os.path.join(cache_root, "clear.jpg")
            empty_path = os.path.join(cache_root, "empty.jpg")
            quota = FileCacheQuota(cache_root, max_files=2)
            for path in (clear_path, empty_path):
                self.assertTrue(quota.acquire(path, 1))
                with open(path, "wb") as output:
                    output.write(b"x")
                quota.mark_ready(path)

            source_paths = [
                "s3://bucket/clear.jpg",
                "s3://bucket/empty.jpg",
            ]
            records = [
                {"human_status": "portrait_clear"},
                {"human_status": "no_human"},
            ]
            op = ImagePortraitCacheRouterMapper(
                local_cache_root=cache_root,
            )
            result = op.process_single(
                {
                    "images": [clear_path, empty_path],
                    "source_images": source_paths,
                    Fields.meta: {MetaKeys.portrait_quality: records},
                }
            )

            self.assertEqual(
                result["images"],
                [clear_path, source_paths[1]],
            )
            self.assertTrue(os.path.exists(clear_path))
            self.assertFalse(os.path.exists(empty_path))
            self.assertTrue(records[0]["humanaesexpert_eligible"])
            self.assertFalse(records[1]["humanaesexpert_eligible"])
            self.assertTrue(records[1]["local_cache_deleted"])
            self.assertEqual(quota.snapshot()["files"], 1)
