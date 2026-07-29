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
export AOSS_CONF="/mnt/afs/private/path/to/aoss.conf"
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

## Two-stage full-dataset scoring

The tested cluster interpreter is
`/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python`.
All dataset, output, cache, model-cache, and detector-model arguments must be
absolute paths. The launchers below also use the absolute repository path
`/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer`; they deliberately
do not hard-code the private `AOSS_CONF` path.

`run_stage1_portrait_quality_all.py` annotates every input row and performs no
filtering. Its Ray JSON output is therefore 1:1 with each input shard, while
`images` is restored to the original S3 URI and
`__dj__meta__.portrait_quality` contains the hard-quality and four-level human
presence results.

```bash
export AOSS_CONF="/mnt/afs/private/path/to/aoss.conf"

/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_stage1_cluster.sh \
  --input /mnt/afs/yanpeishen/datasets/raw.jsonl \
  --output-root /mnt/afs/yanpeishen/results/stage1_portrait_quality \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-stage1 \
  --work-root /mnt/afs/yanpeishen/work/portrait-stage1 \
  --shard-size 100000 \
  --max-cache-files 1024 \
  --max-cache-bytes 214748364800
```

`run_stage2_humanaesexpert_12d.py` reads that output, keeps
`portrait_clear` and `human_present` samples, downloads those images again,
and runs the official HumanAesExpert-8B Expert Head. Add
`--exclude-hard-rejects` only when the second stage should also discard the
first-stage hard-quality `reject` rows:

```bash
/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_stage2_cluster_8h100.sh \
  --input /mnt/afs/yanpeishen/results/stage1_portrait_quality \
  --output-root /mnt/afs/yanpeishen/results/stage2_humanaesexpert_12d \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-stage2 \
  --work-root /mnt/afs/yanpeishen/work/portrait-stage2 \
  --shard-size 100000 \
  --max-cache-files 256 \
  --max-cache-bytes 107374182400
```

The Expert Head record is stored in
`__dj__meta__.humanaesexpert_expert_scores`, with the official 12 dimensions:
facial brightness, feature clarity, skin tone, structure, contour clarity,
facial aesthetics, outfit, body shape, looks, environment, general appearance
aesthetics, and comprehensive aesthetics. The aggregate `score` is the
`comprehensive_aesthetic_score`. Skin tone, body shape, and looks are retained
as model diagnostics and must not be used directly as automatic deletion
criteria.

Both pipelines use a cross-process cache quota. Download workers reserve file
and byte capacity before writing; after scoring, the consumer deletes the
local file and returns the reservation. When either limit is reached, new
downloads wait while the GPU continues consuming completed items. Download
batch size is fixed to one so a partially produced Ray batch cannot occupy the
entire quota and deadlock itself. Duplicate S3 paths are reference-counted and
are deleted after their final in-flight consumer.

## Task-level shard checkpoints

The three cluster launchers call `run_sharded_pipeline.py`. It streams the
source JSONL (or a directory containing Ray JSON parts) into deterministic
input shards. The default is 100,000 records per shard; use
`--shard-size 1000000` for one-million-record tasks. A completed split is
described by the atomic `INPUT_MANIFEST` under `--work-root` and is reused on
restart. The source file list, sizes, mtimes, shard size, row counts, byte
counts, and per-shard SHA-256 values prevent stale input shards from being
silently reused.

Each task uses:

```text
<output-root>/
  shard-000000/
    data.jsonl
    SUCCESS
  shard-000001/
    data.jsonl
    SUCCESS
```

Ray first writes into
`<output-root>/.attempts/shard-N.<pid>.<uuid>/ray_output.jsonl/`. Its JSON
parts are validated and normalized into one bounded `data.jsonl` for that
task. There is deliberately no final cross-shard merge.

`SUCCESS` is created atomically only after:

- every non-empty output line parses as JSON;
- stage 1 and fused output rows equal input rows;
- stage 2 output rows equal the exact count implied by its portrait filter;
- the shard-local image cache has zero payload files, bytes, and references.

A restart skips a shard only when its `SUCCESS`, `data.jsonl`, mode, input row
count, and input SHA-256 agree. An incomplete final directory and any stale
attempt directory are removed before that shard is rerun. Failed-attempt
diagnostics are kept under `<work-root>/failures`, while incomplete output is
removed. The failed shard's cache remains available for safe resumable
downloads and is removed after a successful retry.

To rerun or inspect only shard 42:

```bash
/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_fused_cluster_8h100.sh \
  --input /mnt/afs/yanpeishen/datasets/raw.jsonl \
  --output-root /mnt/afs/yanpeishen/results/portrait_quality_and_expert12d \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-fused \
  --work-root /mnt/afs/yanpeishen/work/portrait-fused \
  --shard-size 100000 \
  --shard-index 42
```

If the source or shard size intentionally changes, pass
`--rebuild-input-shards`. This discards the old input split and completed
output shard directories before rebuilding, so it should not be used during a
normal resume.

## One-download fused mode on 8 H100s

Run this pipeline in a dedicated environment with
`transformers==4.44.2`, `accelerate==0.33.0`, and
`sentencepiece==0.2.0`, as required by the tested HumanAesExpert model stack.
SentencePiece 0.2.2 fails to load this tokenizer vocabulary. The
repository's general all-operator dependency set currently pins a newer
Transformers release, so do not install that full extra into this environment.
A practical cluster setup is to clone the existing Data-Juicer environment,
pin `transformers==4.44.2` and `accelerate==0.33.0`, and verify that Ray,
PyArrow, Ultralytics, the internal AOSS client, Torch, and Torchvision remain
importable on every node. The operator fails early on a version mismatch
instead of producing unverified scores.

For the full run, the recommended high-throughput mode is a bounded fused
pipeline:

```bash
/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_fused_cluster_8h100.sh \
  --input /mnt/afs/yanpeishen/datasets/raw.jsonl \
  --output-root /mnt/afs/yanpeishen/results/portrait_quality_and_expert12d \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-fused \
  --work-root /mnt/afs/yanpeishen/work/portrait-fused \
  --shard-size 100000 \
  --max-cache-files 2048 \
  --max-cache-bytes 214748364800
```

Each fused output shard remains 1:1 with its raw input shard. Every row has
`portrait_quality`; images classified as `portrait_clear` or `human_present`
also have a 12D Expert Head record. Non-eligible positions contain `null` in
the aligned `humanaesexpert_expert_scores` list.

The default 8-H100 split uses four lightweight YOLO actors sharing one GPU and
seven persistent HumanAesExpert actors using one GPU each. The cache router
deletes `human_uncertain` and `no_human` images before they enter the GPU
scoring stage. Eligible files are deleted immediately after scoring. The
cross-process file/byte quota applies to both groups and blocks producers when
either limit is reached.

Use a dedicated cache root per running job. At startup, the scripts reset stale
consumer references from an interrupted prior run in that directory, remove
private partial-download files, and keep complete files as resumable
zero-reference entries. Those entries are reused when encountered again and
are evicted first if new downloads need quota capacity.

HumanAesExpert's official `expert_score()` accepts one image at a time; its
dynamic tiles form the model's internal vision batch. Therefore
`--score-batch-size` is a Ray scheduling batch, not a native multi-image model
batch. A moderate value such as 8–16 lets Ray dynamically distribute many
small tasks across the seven actors, which avoids persistent imbalance when
different source groups have different portrait hit rates.

The original two-stage scripts remain useful when a durable stage-1 checkpoint,
independent reruns, or threshold audits are more important than avoiding a
second S3 download.

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
/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python \
  /mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/build_conv_manifest.py \
  --input /mnt/afs/yanpeishen/results/portrait-hard-quality/scored.jsonl \
  --output-dir /mnt/afs/yanpeishen/results/portrait-hard-quality/viewer \
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
