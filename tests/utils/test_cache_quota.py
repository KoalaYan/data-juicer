import os
import tempfile
import unittest

from data_juicer.utils.cache_quota import FileCacheQuota


class FileCacheQuotaTest(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
