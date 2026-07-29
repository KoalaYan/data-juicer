import asyncio
import copy
import hashlib
import os
import os.path as osp
import threading
from typing import List, Union

from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.cache_quota import FileCacheQuota
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.s3_utils import get_aws_credentials

boto3 = LazyLoader("boto3")
botocore_exceptions = LazyLoader("botocore.exceptions")

OP_NAME = "s3_download_file_mapper"


@OPERATORS.register_module(OP_NAME)
class S3DownloadFileMapper(Mapper):
    """Mapper to download files from S3 to local files or load them into memory.

    This operator downloads files from S3 URLs (s3://...) or handles local files. It supports:
    - Downloading multiple files concurrently
    - Saving files to a specified directory or loading content into memory
    - Resume download functionality
    - S3 authentication with access keys
    - Custom S3 endpoints (for S3-compatible services like MinIO)

    The operator processes nested lists of URLs/paths, maintaining the original structure
    in the output."""

    _batched_op = True

    def __init__(
        self,
        download_field: str = None,
        save_dir: str = None,
        save_field: str = None,
        resume_download: bool = False,
        timeout: int = 30,
        max_concurrent: int = 10,
        # S3 credentials
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        aws_region: str = None,
        endpoint_url: str = None,
        s3_backend: str = "auto",
        aoss_config_env: str = "AOSS_CONF",
        preserve_s3_paths: bool = False,
        source_field: str = None,
        max_cache_files: int = 0,
        max_cache_bytes: int = 0,
        cache_quota_wait_timeout: float = 1800.0,
        cache_quota_poll_interval: float = 1.0,
        fail_on_download_error: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param download_field: The field name to get the URL/path to download.
        :param save_dir: The directory to save downloaded files.
        :param save_field: The field name to save the downloaded file content.
        :param resume_download: Whether to resume download. If True, skip the sample if it exists.
        :param timeout: (Deprecated) Kept for backward compatibility, not used for S3 downloads.
        :param max_concurrent: Maximum concurrent downloads.
        :param aws_access_key_id: AWS access key ID for S3.
        :param aws_secret_access_key: AWS secret access key for S3.
        :param aws_session_token: AWS session token for S3 (optional).
        :param aws_region: AWS region for S3.
        :param endpoint_url: Custom S3 endpoint URL (for S3-compatible services).
        :param s3_backend: S3 client backend. ``auto`` selects the internal
            AOSS client when ``aoss_config_env`` is set, otherwise boto3.
            Supported values are ``auto``, ``boto3`` and ``aoss``.
        :param aoss_config_env: Name of the environment variable that points
            to the private AOSS client config. The config path and credentials
            are never stored in the operator config or output samples.
        :param preserve_s3_paths: When saving to ``save_dir``, preserve the
            bucket/key hierarchy instead of using only the basename. This
            avoids filename collisions in large datasets.
        :param source_field: Optional field used to preserve the original
            remote URLs before ``download_field`` is replaced by local paths.
        :param max_cache_files: Hard upper bound on downloaded files waiting
            in ``save_dir``. Zero disables the file-count bound.
        :param max_cache_bytes: Hard upper bound on downloaded bytes waiting
            in ``save_dir``. Zero disables the byte bound.
        :param cache_quota_wait_timeout: Maximum time a downloader waits for a
            downstream consumer to delete files and return cache capacity.
        :param cache_quota_poll_interval: Cache-capacity polling interval.
        :param fail_on_download_error: Raise immediately after a download
            batch contains failed objects instead of passing their remote URI
            to downstream operators.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.download_field = download_field
        self.save_dir = save_dir
        self.save_field = save_field
        self.resume_download = resume_download

        if not (self.save_dir or self.save_field):
            logger.warning(
                "Both `save_dir` and `save_field` are not specified. Use the default `image_bytes` key to "
                "save the downloaded contents."
            )
            self.save_field = self.image_bytes_key

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.timeout = timeout
        self.max_concurrent = max_concurrent
        if s3_backend not in {"auto", "boto3", "aoss"}:
            raise ValueError("s3_backend must be one of ['auto', 'boto3', 'aoss']")
        self.aoss_config_env = aoss_config_env
        self.aoss_config_path = os.environ.get(aoss_config_env)
        if s3_backend == "auto":
            self.s3_backend = "aoss" if self.aoss_config_path else "boto3"
        else:
            self.s3_backend = s3_backend
        if self.s3_backend == "aoss" and not self.aoss_config_path:
            raise ValueError(
                f"AOSS backend requires environment variable {self.aoss_config_env!r} "
                "to point to the private client config"
            )
        self.preserve_s3_paths = preserve_s3_paths
        self.source_field = source_field
        self._thread_local = threading.local()
        if max_cache_files < 0 or max_cache_bytes < 0:
            raise ValueError("max_cache_files and max_cache_bytes must be non-negative")
        if (max_cache_files or max_cache_bytes) and not self.save_dir:
            raise ValueError("cache quota limits require save_dir")
        self.max_cache_files = int(max_cache_files)
        self.max_cache_bytes = int(max_cache_bytes)
        self.cache_quota_wait_timeout = float(cache_quota_wait_timeout)
        self.cache_quota_poll_interval = float(cache_quota_poll_interval)
        self.fail_on_download_error = fail_on_download_error
        self._cache_quota = self._create_cache_quota()

        # Prepare config dict for get_aws_credentials
        ds_config = {}
        if aws_access_key_id:
            ds_config["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            ds_config["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            ds_config["aws_session_token"] = aws_session_token
        if aws_region:
            ds_config["aws_region"] = aws_region
        if endpoint_url:
            ds_config["endpoint_url"] = endpoint_url

        # Get credentials with priority: environment variables > operator parameters
        (
            resolved_access_key_id,
            resolved_secret_access_key,
            resolved_session_token,
            resolved_region,
        ) = get_aws_credentials(ds_config)

        # Store S3 configuration (don't create client here to avoid serialization issues)
        self.s3_config = None
        self._s3_client = None
        if self.s3_backend == "aoss":
            logger.info(f"Using AOSS backend configured via environment variable {self.aoss_config_env}")
        elif resolved_access_key_id and resolved_secret_access_key:
            self.s3_config = {
                "aws_access_key_id": resolved_access_key_id,
                "aws_secret_access_key": resolved_secret_access_key,
            }
            if resolved_session_token:
                self.s3_config["aws_session_token"] = resolved_session_token
            if resolved_region:
                self.s3_config["region_name"] = resolved_region
            if endpoint_url:
                self.s3_config["endpoint_url"] = endpoint_url
            logger.info(f"S3 configuration stored with endpoint: {endpoint_url or 'default'}")
        else:
            logger.info("No S3 credentials provided. S3 URLs will not be supported.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3_client"] = None
        state["_thread_local"] = None
        state["_cache_quota"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._s3_client = None
        self._thread_local = threading.local()
        self._cache_quota = self._create_cache_quota()

    def _create_cache_quota(self):
        if not (self.max_cache_files or self.max_cache_bytes):
            return None
        return FileCacheQuota(
            self.save_dir,
            max_files=self.max_cache_files,
            max_bytes=self.max_cache_bytes,
            wait_timeout=self.cache_quota_wait_timeout,
            poll_interval=self.cache_quota_poll_interval,
        )

    @property
    def s3_client(self):
        """Lazy initialization of S3 client to avoid serialization issues with Ray."""
        if self._s3_client is None and self.s3_config is not None:
            self._s3_client = boto3.client("s3", **self.s3_config)
            logger.debug("S3 client initialized (lazy)")
        return self._s3_client

    def _create_aoss_client(self):
        try:
            from aoss_client.client import Client
        except ImportError as e:
            raise ImportError(
                "AOSS backend requires the internal 'aoss_client' package in the runtime environment"
            ) from e
        return Client(self.aoss_config_path)

    @property
    def aoss_client(self):
        """Return one AOSS client per worker thread.

        The internal client is not assumed to be thread-safe. The private
        config path comes exclusively from the configured environment
        variable and is never logged or serialized into output samples.
        """
        if not hasattr(self._thread_local, "aoss_client"):
            self._thread_local.aoss_client = self._create_aoss_client()
        return self._thread_local.aoss_client

    def _is_s3_url(self, url: str) -> bool:
        """Check if the URL is an S3 URL."""
        return url.startswith("s3://")

    def _parse_s3_url(self, s3_url: str):
        """Parse S3 URL into bucket and key.

        Example: s3://bucket-name/path/to/file.mp4 -> ('bucket-name', 'path/to/file.mp4')
        """
        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {s3_url}")

        parts = s3_url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        return bucket, key

    def _get_local_save_path(self, s3_url: str, save_dir: str) -> str:
        bucket, key = self._parse_s3_url(s3_url)
        if not bucket or bucket in {".", ".."} or "/" in bucket or "\\" in bucket:
            raise ValueError(f"Unsafe S3 bucket: {bucket}")
        if self.preserve_s3_paths:
            normalized_key = osp.normpath(key).lstrip("/\\")
            if normalized_key == ".." or normalized_key.startswith(("../", "..\\")):
                raise ValueError(f"Unsafe S3 key: {key}")
            target = osp.abspath(osp.join(save_dir, bucket, normalized_key))
            cache_root = osp.abspath(save_dir)
            if osp.commonpath([cache_root, target]) != cache_root:
                raise ValueError(f"Unsafe S3 cache target for key: {key}")
            return target

        filename = osp.basename(key)
        if not filename:
            filename = hashlib.sha256(s3_url.encode("utf-8")).hexdigest()
        return osp.join(save_dir, filename)

    @staticmethod
    def _temporary_save_path(save_path: str) -> str:
        return (
            f"{save_path}.part.{os.getpid()}."
            f"{threading.get_ident()}"
        )

    def _download_from_aoss(self, s3_url: str, save_path: str = None, return_content: bool = False):
        reservation_owned = False
        temporary_path = None
        try:
            content = self.aoss_client.get(s3_url)
            if hasattr(content, "read"):
                content = content.read()
            if content is None:
                raise FileNotFoundError(f"AOSS returned no data for {s3_url}")
            if not isinstance(content, bytes):
                content = bytes(content)

            if save_path:
                if self._cache_quota is not None:
                    reservation_owned = self._cache_quota.acquire(
                        save_path,
                        len(content),
                    )
                    if not reservation_owned:
                        return (
                            "success",
                            None,
                            content if return_content else None,
                            save_path,
                        )
                save_parent = osp.dirname(save_path)
                if save_parent:
                    os.makedirs(save_parent, exist_ok=True)
                temporary_path = self._temporary_save_path(save_path)
                with open(temporary_path, "wb") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary_path, save_path)
                temporary_path = None
                if self._cache_quota is not None:
                    self._cache_quota.mark_ready(save_path)

            return "success", None, content if return_content else None, save_path
        except Exception as e:
            if temporary_path and os.path.isfile(temporary_path):
                os.remove(temporary_path)
            if reservation_owned and self._cache_quota is not None and save_path:
                self._cache_quota.release(save_path)
            error_msg = f"AOSS download error for {s3_url}: {e}"
            logger.error(error_msg)
            return "failed", error_msg, None, None

    def _download_from_s3(self, s3_url: str, save_path: str = None, return_content: bool = False):
        """Download a file from S3.

        :param s3_url: S3 URL (s3://bucket/key)
        :param save_path: Local path to save the file
        :param return_content: Whether to return file content as bytes
        :return: (status, response, content, save_path)
        """
        if self.s3_backend == "aoss":
            return self._download_from_aoss(s3_url, save_path, return_content)

        if not self.s3_client:
            raise ValueError("S3 client not initialized. Please provide AWS credentials.")

        reservation_owned = False
        temporary_path = None
        try:
            bucket, key = self._parse_s3_url(s3_url)

            if save_path:
                # Ensure parent directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

                if self._cache_quota is not None:
                    object_size = int(
                        self.s3_client.head_object(Bucket=bucket, Key=key)[
                            "ContentLength"
                        ]
                    )
                    reservation_owned = self._cache_quota.acquire(
                        save_path,
                        object_size,
                    )
                    if not reservation_owned:
                        content = None
                        if return_content:
                            with open(save_path, "rb") as f:
                                content = f.read()
                        return "success", None, content, save_path

                # Download to a private temporary file and atomically publish
                # it. Duplicate consumers never observe a partial image.
                temporary_path = self._temporary_save_path(save_path)
                self.s3_client.download_file(bucket, key, temporary_path)
                os.replace(temporary_path, save_path)
                temporary_path = None
                if self._cache_quota is not None:
                    self._cache_quota.mark_ready(save_path)
                logger.debug(f"Downloaded S3 file: {s3_url} -> {save_path}")

                # Read content if needed
                content = None
                if return_content:
                    with open(save_path, "rb") as f:
                        content = f.read()

                return "success", None, content, save_path

            elif return_content:
                # Download to memory
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read()
                logger.debug(f"Downloaded S3 file to memory: {s3_url}")

                return "success", None, content, None

            else:
                return "success", None, None, None

        except botocore_exceptions.ClientError as e:
            if temporary_path and os.path.isfile(temporary_path):
                os.remove(temporary_path)
            if reservation_owned and self._cache_quota is not None and save_path:
                self._cache_quota.release(save_path)
            error_msg = f"S3 download failed: {e}"
            logger.error(error_msg)
            return "failed", error_msg, None, None
        except Exception as e:
            if temporary_path and os.path.isfile(temporary_path):
                os.remove(temporary_path)
            if reservation_owned and self._cache_quota is not None and save_path:
                self._cache_quota.release(save_path)
            error_msg = f"S3 download error: {e}"
            logger.error(error_msg)
            return "failed", error_msg, None, None

    async def download_files_async(self, urls, return_contents, save_dir=None, **kwargs):
        """Download files asynchronously from S3."""

        async def _download_file(
            semaphore: asyncio.Semaphore,
            idx: int,
            url: str,
            save_dir=None,
            return_content=False,
            **kwargs,
        ) -> dict:
            async with semaphore:
                try:
                    status, response, content, save_path = "success", None, None, None

                    # Handle S3 URLs (synchronous operation in async context)
                    if self._is_s3_url(url):
                        if save_dir:
                            save_path = self._get_local_save_path(url, save_dir)

                            # Check if file exists and resume is enabled
                            if os.path.exists(save_path) and self.resume_download:
                                if self._cache_quota is not None:
                                    self._cache_quota.retain(save_path)
                                if return_content:
                                    with open(save_path, "rb") as f:
                                        content = f.read()
                                return idx, save_path, status, response, content

                        # Download from S3 (run in executor to avoid blocking)
                        loop = asyncio.get_event_loop()
                        status, response, content, save_path = await loop.run_in_executor(
                            None, self._download_from_s3, url, save_path, return_content
                        )
                        return idx, save_path, status, response, content

                    # Check for HTTP/HTTPS URLs - not supported
                    if url.startswith("http://") or url.startswith("https://"):
                        raise ValueError(
                            f"HTTP/HTTPS URLs are not supported. This mapper only supports S3 URLs (s3://...) and local files. Got: {url}"
                        )

                    # Handle local files
                    if return_content:
                        with open(url, "rb") as f:
                            content = f.read()
                    if save_dir:
                        save_path = url

                    return idx, save_path, status, response, content

                except Exception as e:
                    status = "failed"
                    response = str(e)
                    save_path = None
                    content = None

                return idx, save_path, status, response, content

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            _download_file(semaphore, idx, url, save_dir, return_contents[idx], **kwargs)
            for idx, url in enumerate(urls)
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        return results

    def _flat_urls(self, nested_urls):
        """Flatten nested URLs while preserving structure information."""
        flat_urls = []
        structure_info = []  # save as original index, sub index

        for idx, urls in enumerate(nested_urls):
            if isinstance(urls, list):
                for sub_idx, url in enumerate(urls):
                    flat_urls.append(url)
                    structure_info.append((idx, sub_idx))
            else:
                flat_urls.append(urls)
                structure_info.append((idx, -1))  # -1 means single str element

        return flat_urls, structure_info

    def _create_path_struct(self, nested_urls, keep_failed_url=True) -> List[Union[str, List[str]]]:
        """Create path structure for output."""
        if keep_failed_url:
            reconstructed = copy.deepcopy(nested_urls)
        else:
            reconstructed = []
            for item in nested_urls:
                if isinstance(item, list):
                    reconstructed.append([None] * len(item))
                else:
                    reconstructed.append(None)

        return reconstructed

    def _create_save_field_struct(self, nested_urls, save_field_contents=None) -> List[Union[bytes, List[bytes]]]:
        """Create save field structure for output."""
        if save_field_contents is None:
            save_field_contents = []
            for item in nested_urls:
                if isinstance(item, list):
                    save_field_contents.append([None] * len(item))
                else:
                    save_field_contents.append(None)
        else:
            # check whether the save_field_contents format is correct and correct it automatically
            for i, item in enumerate(nested_urls):
                if isinstance(item, list):
                    if not save_field_contents[i] or len(save_field_contents[i]) != len(item):
                        save_field_contents[i] = [None] * len(item)

        return save_field_contents

    async def download_nested_urls(
        self, nested_urls: List[Union[str, List[str]]], save_dir=None, save_field_contents=None
    ):
        """Download nested URLs with structure preservation."""
        flat_urls, structure_info = self._flat_urls(nested_urls)

        if save_field_contents is None:
            # not save contents, set return_contents to False
            return_contents = [False] * len(flat_urls)
        else:
            # if original content None, set bool value to True to get content else False to skip reload it
            return_contents = []
            for item in save_field_contents:
                if isinstance(item, list):
                    return_contents.extend([not c for c in item])
                else:
                    return_contents.append(not item)

        download_results = await self.download_files_async(
            flat_urls,
            return_contents,
            save_dir,
        )

        if self.save_dir:
            reconstructed_path = self._create_path_struct(nested_urls)
        else:
            reconstructed_path = None

        failed_info = ""
        for i, (idx, save_path, status, response, content) in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            if status != "success":
                save_path = flat_urls[i]
                failed_info += "\n" + str(response)

            if save_field_contents is not None:
                if return_contents[i]:
                    if sub_idx == -1:
                        save_field_contents[orig_idx] = content
                    else:
                        save_field_contents[orig_idx][sub_idx] = content

            if self.save_dir:
                if sub_idx == -1:
                    reconstructed_path[orig_idx] = save_path
                else:
                    reconstructed_path[orig_idx][sub_idx] = save_path

        return save_field_contents, reconstructed_path, failed_info

    def process_batched(self, samples):
        """Process a batch of samples."""
        if self.download_field not in samples or not samples[self.download_field]:
            return samples

        batch_nested_urls = samples[self.download_field]
        if self.source_field:
            if self.source_field in samples and not self.resume_download:
                raise ValueError(
                    f"{self.source_field} is already in samples. "
                    "Choose another source_field or set resume_download=True"
                )
            if self.source_field not in samples:
                samples[self.source_field] = copy.deepcopy(batch_nested_urls)

        if self.save_field:
            if not self.resume_download:
                if self.save_field in samples:
                    raise ValueError(
                        f"{self.save_field} is already in samples. "
                        f"If you want to resume download, please set `resume_download=True`"
                    )
                save_field_contents = self._create_save_field_struct(batch_nested_urls)
            else:
                if self.save_field not in samples:
                    save_field_contents = self._create_save_field_struct(batch_nested_urls)
                else:
                    save_field_contents = self._create_save_field_struct(batch_nested_urls, samples[self.save_field])
        else:
            save_field_contents = None

        save_field_contents, reconstructed_path, failed_info = asyncio.run(
            self.download_nested_urls(
                batch_nested_urls, save_dir=self.save_dir, save_field_contents=save_field_contents
            )
        )

        if self.save_dir:
            samples[self.download_field] = reconstructed_path

        if self.save_field:
            samples[self.save_field] = save_field_contents

        if len(failed_info):
            logger.error(f"Failed files:\n{failed_info}")
            if self.fail_on_download_error:
                raise RuntimeError(
                    "one or more downloads failed:" + failed_info
                )

        return samples
