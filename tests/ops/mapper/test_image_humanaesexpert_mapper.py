import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import data_juicer.ops.mapper.image_humanaesexpert_mapper as hae_module
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

    def test_model_load_rejects_unsupported_transformers(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        )
        fake_transformers = SimpleNamespace(__version__="4.57.1")
        op = ImageHumanAesExpertMapper()
        with (
            patch.object(hae_module, "torch", fake_torch),
            patch.object(
                hae_module,
                "transformers",
                fake_transformers,
            ),
            self.assertRaisesRegex(RuntimeError, "4.44.2"),
        ):
            op._load_model()

    def test_model_load_rejects_unsupported_sentencepiece(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True)
        )
        fake_transformers = SimpleNamespace(__version__="4.44.2")
        fake_sentencepiece = SimpleNamespace(__version__="0.2.2")
        op = ImageHumanAesExpertMapper()
        with (
            patch.object(hae_module, "torch", fake_torch),
            patch.object(
                hae_module,
                "transformers",
                fake_transformers,
            ),
            patch.object(
                hae_module,
                "sentencepiece",
                fake_sentencepiece,
            ),
            self.assertRaisesRegex(RuntimeError, "sentencepiece==0.2.0"),
        ):
            op._load_model()

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

    def test_process_batched_skips_ineligible_without_loading_model(self):
        source = [["s3://bucket/no-human.jpg"]]
        op = ImageHumanAesExpertMapper(
            skip_ineligible=True,
            delete_local_cache_after_processing=True,
            local_cache_root=tempfile.gettempdir(),
            source_image_key="source_images",
        )
        op._score_image = lambda path: self.fail(
            "ineligible image must not be scored"
        )
        result = op.process_batched(
            {
                "text": ["empty"],
                "images": source.copy(),
                "source_images": source,
                Fields.meta: [
                    {
                        MetaKeys.portrait_quality: [
                            {
                                "human_status": "no_human",
                                "humanaesexpert_eligible": False,
                            }
                        ]
                    }
                ],
            }
        )
        self.assertEqual(result["images"], source)
        self.assertEqual(
            result[Fields.meta][0][
                MetaKeys.humanaesexpert_expert_scores
            ],
            [None],
        )

    def test_persistent_pool_scores_one_batch_and_preserves_alignment(self):
        source = [
            ["s3://bucket/eligible.jpg", "s3://bucket/ineligible.jpg"],
            ["s3://bucket/second.jpg"],
        ]
        op = ImageHumanAesExpertMapper(
            persistent_actor_pool=True,
            persistent_actor_pool_size=7,
            skip_ineligible=True,
        )
        observed = []

        def fake_score_paths(paths):
            observed.append(list(paths))
            return [{"score": float(index)} for index, _ in enumerate(paths)]

        op._score_paths = fake_score_paths
        result = op.process_batched(
            {
                "text": ["first", "second"],
                "images": source,
                Fields.meta: [
                    {
                        MetaKeys.portrait_quality: [
                            {"humanaesexpert_eligible": True},
                            {"humanaesexpert_eligible": False},
                        ]
                    },
                    {
                        MetaKeys.portrait_quality: [
                            {"humanaesexpert_eligible": True}
                        ]
                    },
                ],
            }
        )

        self.assertEqual(
            observed,
            [[source[0][0], source[1][0]]],
        )
        first_scores = result[Fields.meta][0][
            MetaKeys.humanaesexpert_expert_scores
        ]
        second_scores = result[Fields.meta][1][
            MetaKeys.humanaesexpert_expert_scores
        ]
        self.assertEqual(first_scores, [{"score": 0.0}, None])
        self.assertEqual(second_scores, [{"score": 1.0}])
        self.assertEqual(op.accelerator, "cpu")
        self.assertEqual(op.ray_execution_mode, "actor")
        self.assertEqual(op.memory, 1.0)

    def test_flash_attention_is_enabled_by_default(self):
        op = ImageHumanAesExpertMapper()
        self.assertTrue(op.use_flash_attn)

        eager_op = ImageHumanAesExpertMapper(use_flash_attn=False)
        self.assertFalse(eager_op.use_flash_attn)


if __name__ == "__main__":
    unittest.main()
