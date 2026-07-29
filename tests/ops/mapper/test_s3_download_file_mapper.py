import os
import dill
import tempfile
import unittest
from unittest.mock import patch

from data_juicer.ops.mapper.s3_download_file_mapper import S3DownloadFileMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class _FakeAOSSClient:
    def get(self, url):
        return f"content:{url}".encode()


class _FailingAOSSClient:
    def get(self, url):
        raise TimeoutError(f"timed out: {url}")


class S3DownloadFileMapperTest(DataJuicerTestCaseBase):

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_aoss_backend_downloads_to_memory(self):
        op = S3DownloadFileMapper(
            download_field="images",
            save_field="image_bytes",
            s3_backend="aoss",
        )
        op._thread_local.aoss_client = _FakeAOSSClient()
        status, error, content, save_path = op._download_from_s3(
            "s3://infographics/example.jpg",
            return_content=True,
        )
        self.assertEqual(status, "success")
        self.assertIsNone(error)
        self.assertEqual(content, b"content:s3://infographics/example.jpg")
        self.assertIsNone(save_path)

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_aoss_operator_is_serializable(self):
        op = S3DownloadFileMapper(
            download_field="images",
            save_field="image_bytes",
            s3_backend="aoss",
        )
        restored = dill.loads(dill.dumps(op))
        self.assertEqual(restored.s3_backend, "aoss")
        self.assertIsNotNone(restored._thread_local)

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_aoss_preserves_uri_and_bucket_key_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = S3DownloadFileMapper(
                download_field="images",
                save_dir=tmpdir,
                source_field="source_images",
                preserve_s3_paths=True,
                s3_backend="aoss",
                resume_download=True,
            )
            op._create_aoss_client = lambda: _FakeAOSSClient()
            samples = {"images": [["s3://infographics/a/b/example.jpg"]]}
            output = op.process_batched(samples)
            expected_path = os.path.join(tmpdir, "infographics", "a", "b", "example.jpg")
            self.assertEqual(output["source_images"], [["s3://infographics/a/b/example.jpg"]])
            self.assertEqual(output["images"], [[expected_path]])
            self.assertTrue(os.path.isfile(expected_path))

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_resume_download_reuses_existing_file_without_aoss_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = S3DownloadFileMapper(
                download_field="images",
                save_dir=tmpdir,
                preserve_s3_paths=True,
                s3_backend="aoss",
                resume_download=True,
            )
            uri = "s3://infographics/a/b/existing.jpg"
            cache_path = op._get_local_save_path(uri, tmpdir)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as target:
                target.write(b"cached-image")

            def fail_if_called():
                raise AssertionError("AOSS should not be called on a cache hit")

            op._create_aoss_client = fail_if_called
            output = op.process_batched({"images": [[uri]]})
            self.assertEqual(output["images"], [[cache_path]])
            with open(cache_path, "rb") as source:
                self.assertEqual(source.read(), b"cached-image")

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_aoss_download_registers_hard_cache_quota(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = S3DownloadFileMapper(
                download_field="images",
                save_dir=tmpdir,
                preserve_s3_paths=True,
                s3_backend="aoss",
                resume_download=True,
                max_cache_files=1,
                max_cache_bytes=1024,
            )
            op._create_aoss_client = lambda: _FakeAOSSClient()
            uri = "s3://infographics/a.jpg"
            output = op.process_batched({"images": [[uri]]})
            cache_path = output["images"][0][0]
            self.assertTrue(os.path.isfile(cache_path))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        tmpdir,
                        ".data_juicer_cache_quota.json",
                    )
                )
            )

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    def test_fail_on_download_error_raises_at_download_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = S3DownloadFileMapper(
                download_field="images",
                save_dir=tmpdir,
                preserve_s3_paths=True,
                s3_backend="aoss",
                fail_on_download_error=True,
            )
            op._create_aoss_client = lambda: _FailingAOSSClient()
            with self.assertRaisesRegex(
                RuntimeError,
                "one or more downloads failed",
            ):
                op.process_batched(
                    {"images": [["s3://infographics/timeout.jpg"]]}
                )

    @patch.dict(os.environ, {}, clear=True)
    def test_aoss_backend_requires_environment_variable(self):
        with self.assertRaisesRegex(ValueError, "AOSS_CONF"):
            S3DownloadFileMapper(
                download_field="images",
                save_field="image_bytes",
                s3_backend="aoss",
            )


if __name__ == '__main__':
    unittest.main()
