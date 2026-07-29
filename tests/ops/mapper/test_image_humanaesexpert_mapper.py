import io
import os
import tempfile
import unittest

from PIL import Image

from data_juicer.ops.mapper.image_humanaesexpert_mapper import (
    EXPERT_DIMENSIONS,
    ImageHumanAesExpertMapper,
    _dynamic_preprocess,
)
from data_juicer.utils.cache_quota import FileCacheQuota
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageHumanAesExpertMapperTest(DataJuicerTestCaseBase):

    def test_dynamic_preprocess_respects_tile_limit(self):
        tiles = _dynamic_preprocess(
            Image.new("RGB", (1600, 400)),
            max_num=4,
            image_size=64,
        )
        self.assertLessEqual(len(tiles), 5)
        self.assertTrue(all(tile.size == (64, 64) for tile in tiles))

    def test_build_score_record_checks_official_mapping(self):
        op = ImageHumanAesExpertMapper()
        values = [float(index) / 10 for index in range(12)]
        mapping = {
            english: values[index]
            for index, (_, english, _) in enumerate(EXPERT_DIMENSIONS)
        }
        score = op._build_score_record(values, mapping, tile_count=5)
        self.assertEqual(score["tile_count"], 5)
        self.assertEqual(
            score["score"],
            score["dimensions"]["comprehensive_aesthetic_score"],
        )

        mapping["Facial Brightness"] += 1
        with self.assertRaises(RuntimeError):
            op._build_score_record(values, mapping, tile_count=5)

    def test_process_batched_writes_scores_deletes_cache_and_restores_uri(self):
        with tempfile.TemporaryDirectory() as cache_root:
            local_path = os.path.join(cache_root, "bucket", "image.jpg")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            image_bytes = io.BytesIO()
            Image.new("RGB", (64, 64), (128, 128, 128)).save(
                image_bytes,
                format="JPEG",
            )
            content = image_bytes.getvalue()
            quota = FileCacheQuota(cache_root, max_files=1)
            self.assertTrue(quota.acquire(local_path, len(content)))
            with open(local_path, "wb") as output:
                output.write(content)
            quota.mark_ready(local_path)

            op = ImageHumanAesExpertMapper(
                delete_local_cache_after_processing=True,
                local_cache_root=cache_root,
                source_image_key="source_images",
            )
            op._score_image = lambda path: {
                "score": 0.8,
                "dimensions": {
                    key: 0.8 for key, _, _ in EXPERT_DIMENSIONS
                },
            }
            source = [["s3://bucket/image.jpg"]]
            result = op.process_batched(
                {
                    "text": ["portrait"],
                    "images": [[local_path]],
                    "source_images": source,
                    Fields.meta: [{}],
                }
            )

            self.assertEqual(result["images"], source)
            self.assertFalse(os.path.exists(local_path))
            score = result[Fields.meta][0][
                MetaKeys.humanaesexpert_expert_scores
            ][0]
            self.assertTrue(score["local_cache_deleted"])
            self.assertEqual(score["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
