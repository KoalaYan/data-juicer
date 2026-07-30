import os
import dill
import tempfile
import unittest
from unittest.mock import patch

from data_juicer.ops.mapper.s3_download_file_mapper import S3DownloadFileMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class _FakeAOSSClient:
    def __init__(self):
        self.get_calls = 0
        self.download_file_calls = 0

    def get(self, url):
        self.get_calls += 1
        return f"content:{url}".encode()

    def download_file(self, url, path):
        self.download_file_calls += 1
        with open(path, "wb") as target:
            target.write(f"streamed:{url}".encode())


class _FailingAOSSClient:
    def get(self, url):
        raise TimeoutError(f"timed out: {url}")


class _FlakyAOSSClient:
    def __init__(self):
        self.attempts = 0

    def get(self, url):
        self.attempts += 1
        if self.attempts < 3:
            raise BlockingIOError(11, "Resource temporarily unavailable")
        return f"content:{url}".encode()


class _MissingAOSSClient:
    def __init__(self):
        self.attempts = 0

    def get(self, url):
        self.attempts += 1
        return None


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
    def test_aoss_can_stream_directly_to_atomic_cache_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = _FakeAOSSClient()
            op = S3DownloadFileMapper(
                download_field="images",
                save_dir=tmpdir,
                preserve_s3_paths=True,
                s3_backend="aoss",
                aoss_stream_to_file=True,
                resume_download=True,
                max_cache_files=1,
                max_cache_bytes=1024,
            )
            op._create_aoss_client = lambda: client
            uri = "s3://infographics/a/b/streamed.jpg"
            output = op.process_batched({"images": [[uri]]})
            cache_path = output["images"][0][0]
            with open(cache_path, "rb") as source:
                self.assertEqual(
                    source.read(),
                    b"streamed:s3://infographics/a/b/streamed.jpg",
                )
            self.assertEqual(client.download_file_calls, 1)
            self.assertEqual(client.get_calls, 0)
            self.assertFalse(
                any(".part." in name for name in os.listdir(tmpdir))
            )

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
                aoss_max_attempts=1,
            )
            op._create_aoss_client = lambda: _FailingAOSSClient()
            with self.assertRaisesRegex(
                RuntimeError,
                "one or more downloads failed",
            ):
                op.process_batched(
                    {"images": [["s3://infographics/timeout.jpg"]]}
                )

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    @patch(
        "data_juicer.ops.mapper.s3_download_file_mapper.random.uniform",
        return_value=0.25,
    )
    @patch("data_juicer.ops.mapper.s3_download_file_mapper.time.sleep")
    def test_aoss_retryable_error_uses_backoff_and_recovers(
        self,
        sleep,
        _uniform,
    ):
        op = S3DownloadFileMapper(
            download_field="images",
            save_field="image_bytes",
            s3_backend="aoss",
            aoss_max_attempts=3,
            aoss_retry_initial_delay=1.5,
            aoss_retry_max_delay=12.0,
            aoss_retry_jitter=1.0,
        )
        client = _FlakyAOSSClient()
        op._thread_local.aoss_client = client
        status, error, content, _ = op._download_from_s3(
            "s3://infographics/transient.jpg",
            return_content=True,
        )
        self.assertEqual(status, "success")
        self.assertIsNone(error)
        self.assertEqual(
            content,
            b"content:s3://infographics/transient.jpg",
        )
        self.assertEqual(client.attempts, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.75, 3.25],
        )

    @patch.dict(os.environ, {"AOSS_CONF": "/private/runtime/aoss.conf"})
    @patch("data_juicer.ops.mapper.s3_download_file_mapper.time.sleep")
    def test_aoss_missing_object_is_not_retried(self, sleep):
        op = S3DownloadFileMapper(
            download_field="images",
            save_field="image_bytes",
            s3_backend="aoss",
            aoss_max_attempts=5,
        )
        client = _MissingAOSSClient()
        op._thread_local.aoss_client = client
        status, error, _, _ = op._download_from_s3(
            "s3://infographics/missing.jpg",
            return_content=True,
        )
        self.assertEqual(status, "failed")
        self.assertIn("AOSS returned no data", error)
        self.assertEqual(client.attempts, 1)
        sleep.assert_not_called()

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
