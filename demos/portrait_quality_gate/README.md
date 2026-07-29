# Portrait hard-quality gate

This demo performs conservative first-stage filtering for portrait raw data:

- AOSS-backed `s3://...` materialization with source URI preservation;
- near-black/near-white and severe exposure detection;
- four-level portrait presence from YOLO person/pose plus frontal-face fallback;
- background-only overexposure routed to `uncertain`;
- viewer-compatible `conv.json` output for small-scale review.

## Runtime configuration

The private AOSS config must only be supplied through the environment:

```bash
export AOSS_CONF="/path/to/private/aoss.conf"
```

Do not add the config file, its contents, or credentials to YAML or Git.
The runtime environment must already provide the internal `aoss_client`
package. Install Data-Juicer with its vision dependencies for YOLO and OpenCV.

## Input schema

The input JSONL should contain an `images` list of local paths or S3 URIs. The
generic Data-Juicer formatter also requires a non-empty `text` field before it
runs any operator. The portrait gate never consumes this field: copy the
source caption into `text`, or use a fixed placeholder such as
`"portrait_quality_gate"` for image-only raw data.

Useful provenance fields such as `id`, `image_root`, `source_meta`,
`source_offset`, and `conversations` are passed through and included in the
review output when available.

## Score before filtering

Edit only the input/output/cache paths in `score.yaml`, then run:

```bash
dj-process --config demos/portrait_quality_gate/score.yaml
```

The AOSS mapper copies original remote paths to `source_images`, replaces
`images` with collision-safe local cache paths, and stores no AOSS credential
information in output records.

## Batch execution

`image_portrait_quality_mapper` implements native `process_batched()`:

- images across samples are flattened while retaining sample/image indices;
- YOLO person and pose models each receive image lists instead of single images;
- `inference_batch_size` bounds each model micro-batch;
- OpenCV face detection and deterministic metrics remain per-image CPU work;
- results are restored to the original nested image order.

`batch_size` controls the Data-Juicer dataset batch, while
`inference_batch_size` controls the largest GPU call. For a single GPU, start
with both set to 8 and tune upward based on VRAM. Ray execution can be enabled
with `executor_type: ray` when the runtime has Ray configured.
The demo declares `num_gpus: 1`; remove that resource request and set
`accelerator: cpu` only when intentionally using the slower CPU fallback.

## Image and dataset caches

There are three independent caches:

1. The AOSS materialization cache is `s3_download_file_mapper.save_dir`.
   With `preserve_s3_paths: true`, an object is stored as
   `<save_dir>/<bucket>/<key>`. `source_images` retains the original URI and
   `images` is replaced with the local path.
2. With `resume_download: true`, an existing local path is reused without
   another AOSS request. This is an existence-based cache; it does not currently
   compare an object ETag or checksum.
3. Data-Juicer's `use_cache`/`ds_cache_dir` caches transformed dataset states
   by input and operator fingerprint. Model weights use
   `DATA_JUICER_MODELS_CACHE` separately.

The demo uses `/tmp/data_juicer_portrait_cache` only for smoke tests. `/tmp` is
node-local and may be cleaned after reboot or by system policy. Production
recipes should change `save_dir` to a persistent AFS path outside the Git
repository. Do not store AOSS credentials in that directory or in YAML.

## Streaming cleanup for very large datasets

Do not use the default executor with a download-then-score recipe for hundreds
of millions of images. The default executor materializes one operator over the
dataset before starting the next operator, so local storage can fill before
scoring begins.

Use `score_streaming_cleanup.yaml` for production-scale runs. It:

- uses Ray Data block streaming;
- bounds the AOSS stage with explicit batch size and concurrency;
- requires a shared AFS cache visible to download and GPU workers;
- deletes each local image only after its person, pose, face, exposure, and
  sharpness results have been produced;
- refuses to delete paths outside the explicit `local_cache_root`, refuses
  symlinks, and never recursively deletes directories;
- restores `images` to the original `source_images` S3 URIs before export;
- writes `local_cache_deleted` into each image's quality record.

The recipe starts a single-machine Ray runtime with `ray_address: local`.
Change it to `auto` only when connecting to a Ray cluster that was started
separately. In containers where Ray reports zero available CPUs, set
`RAY_USE_MULTIPROCESSING_CPU_COUNT=1` for the process. When running directly
from a checkout that is not installed in editable mode, put that checkout on
`PYTHONPATH` so both the driver and Ray workers load the same operators:

```bash
PYTHONPATH="$PWD" \
RAY_USE_MULTIPROCESSING_CPU_COUNT=1 \
dj-process --config demos/portrait_quality_gate/score_streaming_cleanup.yaml
```

The cleanup root and S3 download `save_dir` must resolve to exactly the same
directory. `/tmp` must not be used with a multi-node Ray cluster because it is
node-local. If a task fails before cleanup, only in-flight batch files may
remain; the next run reuses them through `resume_download`.

For visualization, run a separate small sample with cleanup disabled. A
full-scale output intentionally contains no durable local image path; selected
review images should be downloaded again into a dedicated viewer cache.

Each image receives a `__dj__meta__.portrait_quality` record with:

- `status`: `pass`, `uncertain`, or `reject`;
- `human_status`: `portrait_clear`, `human_present`, `human_uncertain`,
  or `no_human`;
- path-independent exposure and sharpness metrics;
- face area and face-region sharpness;
- person/face counts, detector confidence, crop-edge diagnostics, and pose
  keypoint completeness;
- explicit `reject_reasons` and `warning_reasons`.

The human-presence levels mean:

- `portrait_clear`: a sufficiently large, sharp, non-edge-clipped face;
- `human_present`: strong complete person/pose evidence, including small,
  occluded, or back-facing faces;
- `human_uncertain`: low-confidence or suspiciously partial/cropped evidence;
- `no_human`: no face, person, or pose evidence after all enabled detectors
  completed.

Only high-confidence failures are rejected. Partial bodies, suspicious crop,
background-only overexposure, and detector failures are marked `uncertain`.

## Build a small review manifest

```bash
python demos/portrait_quality_gate/build_conv_manifest.py \
  --input ./outputs/portrait-hard-quality/scored.jsonl \
  --output-dir ./outputs/portrait-hard-quality/viewer \
  --limit 500
```

The output contains:

```text
viewer/conv.json
viewer/summary.json
viewer/viewer_annotations/*.jsonl
```

Each annotation retains the original S3 URI, image root, relative path, local
cache path, source metadata, source offset, all quality signals, and a readable
review conversation.

## Apply the gate

After reviewing all `reject` samples and calibrating thresholds, run:

```bash
dj-process --config demos/portrait_quality_gate/filter.yaml
```

The default filter keeps `pass` and `uncertain`; it removes only `reject`.
