import os
import tempfile
import unittest
from unittest.mock import patch

from data_juicer.utils.cache_quota import FileCacheQuota


class FileCacheQuotaTest(unittest.TestCase):

    @patch(
        "data_juicer.utils.cache_quota.random.uniform",
        return_value=0.005,
    )
    @patch("data_juicer.utils.cache_quota.time.sleep")
    @patch("data_juicer.utils.cache_quota.fcntl.flock")
    def test_retries_transient_afs_lock_contention(
        self,
        flock,
        sleep,
        _uniform,
    ):
        flock.side_effect = [
            BlockingIOError(11, "Resource temporarily unavailable"),
            BlockingIOError(11, "Resource temporarily unavailable"),
            None,
            None,
        ]
        with tempfile.TemporaryDirectory() as root:
            quota = FileCacheQuota(root, max_files=1)
            self.assertEqual(quota.snapshot()["files"], 0)
        self.assertEqual(flock.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.015, 0.025],
        )

    def test_enforces_byte_limit_and_releases_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first.jpg")
            second = os.path.join(root, "second.jpg")
            quota = FileCacheQuota(
                root,
                max_files=2,
                max_bytes=10,
                wait_timeout=0.05,
                poll_interval=0.01,
            )
            self.assertTrue(quota.acquire(first, 6))
            with open(first, "wb") as output:
                output.write(b"123456")
            quota.mark_ready(first)

            with self.assertRaises(TimeoutError):
                quota.acquire(second, 5)

            os.remove(first)
            quota.release(first)
            self.assertTrue(quota.acquire(second, 5))

    def test_duplicate_path_is_not_reserved_twice(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "same.jpg")
            quota = FileCacheQuota(root, max_files=1)
            self.assertTrue(quota.acquire(path, 3))
            with open(path, "wb") as output:
                output.write(b"abc")
            quota.mark_ready(path)
            self.assertFalse(quota.acquire(path, 3))
            self.assertFalse(quota.delete_after_consume(path))
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(quota.delete_after_consume(path))
            self.assertFalse(os.path.exists(path))

    def test_new_run_reuses_or_evicts_zero_reference_files(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first.jpg")
            second = os.path.join(root, "second.jpg")
            partial = f"{first}.part.123.456"
            quota = FileCacheQuota(root, max_files=1)
            self.assertTrue(quota.acquire(first, 3))
            with open(first, "wb") as output:
                output.write(b"abc")
            quota.mark_ready(first)
            with open(partial, "wb") as output:
                output.write(b"partial")

            prepared = quota.prepare_for_new_run()
            self.assertEqual(prepared["files"], 1)
            self.assertFalse(os.path.exists(partial))
            self.assertFalse(quota.acquire(first, 3))
            self.assertTrue(quota.delete_after_consume(first))

            self.assertTrue(quota.acquire(first, 3))
            with open(first, "wb") as output:
                output.write(b"abc")
            quota.mark_ready(first)
            quota.prepare_for_new_run()
            self.assertTrue(quota.acquire(second, 3))
            self.assertFalse(os.path.exists(first))

    def test_initial_scan_ignores_active_partial_downloads(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "image.jpg")
            partial = f"{target}.part.123.456"
            with open(partial, "wb") as output:
                output.write(b"partial")
            quota = FileCacheQuota(
                root,
                max_files=1,
                max_bytes=1024,
            )
            self.assertTrue(quota.acquire(target, 16))
            self.assertTrue(os.path.isfile(partial))
            snapshot = quota.snapshot()
            self.assertEqual(snapshot["files"], 1)
            self.assertEqual(snapshot["bytes"], 16)


if __name__ == "__main__":
    unittest.main()
