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
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
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
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
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

The stage-specific and non-windowed pipelines support a cross-process cache
quota. AOSS downloads use `Client.download_file()` to stream into private
temporary files and atomically publish complete files.

The recommended fused cluster launcher instead uses bounded 500-row execution
windows. The window size itself is the hard payload-file bound, so this mode
does not create or update the per-image AFS quota JSON. Ray streams download,
portrait triage, routing, and HumanAesExpert scoring within each window.
Published images are deleted by the router or scorer, and at most one window
plus the bounded in-flight temporary downloads occupies storage.

## Task-level shard checkpoints

The three cluster launchers call `run_sharded_pipeline.py`. It streams the
source JSONL (or a directory containing Ray JSON parts) into deterministic
10,000-record micro-shards grouped under 100,000-record logical shards. Both
sizes are configurable, and the logical size must be an exact multiple of the
micro size. A completed split is described by the atomic `INPUT_MANIFEST`
under `--work-root` and is reused on restart. The source file list, sizes,
mtimes, both shard sizes, row counts, byte counts, and per-micro-shard SHA-256
values prevent stale inputs from being silently reused.

Each task uses:

```text
<output-root>/
  shard-000000/
    SUCCESS
    micro-0000/
      data.jsonl
      SUCCESS
    micro-0001/
      data.jsonl
      SUCCESS
    ...
```

Ray first writes into
`<output-root>/.attempts/shard-N-micro-M.<pid>.<uuid>/ray_output.jsonl/`.
Its JSON parts are validated and normalized into one bounded `data.jsonl` for
that micro-shard. There is deliberately no final merge across micro-shards or
logical shards.

Each micro-shard `SUCCESS` is created atomically only after:

- every non-empty output line parses as JSON;
- stage 1 and fused output rows equal input rows;
- stage 2 output rows equal the exact count implied by its portrait filter;
- the shard-local image cache has zero payload files, bytes, and references.

A logical-shard `SUCCESS` is created after all of its micro-shards have valid
markers. A restart skips a micro-shard only when its `SUCCESS`, `data.jsonl`,
mode, input row count, and input SHA-256 agree. An incomplete final directory
and any stale attempt directory are removed before that micro-shard is rerun.
Failed-attempt diagnostics are kept under `<work-root>/failures`, while
incomplete output is removed. The failed micro-shard's cache remains available
for safe resumable downloads and is removed after a successful retry.

To rerun or inspect only shard 42:

```bash
/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_fused_cluster_8h100.sh \
  --input /mnt/afs/yanpeishen/datasets/raw.jsonl \
  --output-root /mnt/afs/yanpeishen/results/portrait_quality_and_expert12d \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-fused \
  --work-root /mnt/afs/yanpeishen/work/portrait-fused \
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
  --logical-shard-index 42
```

If the source or shard size intentionally changes, pass
`--rebuild-input-shards`. This discards the old input split and completed
output shard directories before rebuilding, so it should not be used during a
normal resume.

`--input-adapter purchased-selection` converts the existing purchased-data
selection manifest (`image` plus `_sample.image_uri`) into the Data-Juicer
`images` and non-empty `text` fields while preserving all provenance fields.

For the fixed `human_baixing_0515` first-100k fused smoke run, use:

```bash
export AOSS_CONF="/mnt/afs/private/path/to/aoss.conf"

/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_fused_first100k_cluster_8h100.sh
```

This command is a background dispatcher: it starts the real launcher under
`nohup`, writes its PID and state under the run's `logs` directory, prints the
log-follow command, and exits. Re-running it on the same node refuses to start
while the recorded PID is alive. The worker survives an SSH or terminal
disconnect.

The smoke launcher uses eight Ray download workers, batches of eight rows,
and up to four AOSS requests inside each task, for a hard upper bound of 32
in-flight downloads. Unlike the earlier slow path, it does not call
`Client.get()` and then synchronously rewrite and `fsync` the full object.
The installed AOSS client performs up to ten internal attempts without any
delay. The download operator therefore adds five outer attempts for retryable
system and connection failures, with exponential backoff starting at 1.5
seconds, capped at 12 seconds, plus up to one second of random jitter. Missing
objects are not retried. A final download failure is raised in the download
operator so the current micro-shard fails with its S3 URI instead of failing
later during image decoding. Concurrency can be tuned with
`--download-workers`, `--download-batch-size`, and
`--download-concurrency`; retries can be changed with
`--aoss-download-attempts`, `--aoss-retry-initial-delay`,
`--aoss-retry-max-delay`, and `--aoss-retry-jitter`.

Each 10,000-row micro-shard is internally divided into 500-row execution
windows. Every window has an input hash, normalized output, output hash, cache
zero check, and atomic `WINDOW_SUCCESS` under:

```text
<work-root>/window_checkpoints/
  shard-000000/micro-0000/
    window-0000/
      input.jsonl
      data.jsonl
      WINDOW_SUCCESS
```

If a later window fails, a restart validates and skips earlier successful
windows. After all 20 windows finish, their outputs are merged and validated
against the 10,000-row micro-shard before its normal `SUCCESS` is published.
The temporary window checkpoints are then removed. The seven detached
HumanAesExpert actors remain loaded across all window subprocesses.

Monitor structured progress:

```bash
/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python \
  /mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/show_sharded_progress.py \
  --output-root /mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729/output \
  --work-root /mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729/work \
  --cache-root /mnt/afs/yanpeishen/cache/data-juicer/human_baixing_0515_fused_first100k_micro10k_20260729 \
  --logical-shard-index 0 \
  --watch-seconds 10
```

Monitor the full pipeline log:

```bash
/usr/bin/tail -n 200 -F \
  /mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729/logs/pipeline.log
```

Inspect the background launcher state and PID:

```bash
cat /mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729/logs/pipeline.state
cat /mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729/logs/pipeline.pid
```

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

Install the complete pinned HumanAesExpert remote-code dependency set with:

```bash
/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python \
  -m pip install \
  -r /mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/requirements-humanaesexpert.txt
```

The cluster launcher runs `manage_humanaesexpert_pool.py check` before
starting Ray. It reports every missing or mismatched package together, so a
dependency error does not first appear after seven GPU actors start warming.
The pinned environment uses `opencv-contrib-python-headless`; compute nodes
do not need the GUI package's system-level `libGL.so.1`.

HumanAesExpert actors use FlashAttention2 by default for both the InternViT
vision encoder and InternLM2 attention paths. Pass `--disable-flash-attn` to
the fused pipeline and pool manager only for numerical comparison or fallback.
The H100-only source build used for this environment is:

```bash
PATH=/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin:$PATH \
CUDA_HOME=/usr/local/cuda \
MAX_JOBS=8 \
FLASH_ATTN_CUDA_ARCHS=90 \
/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python \
  -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

For the full run, the recommended high-throughput mode is a bounded fused
pipeline:

```bash
/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/run_fused_cluster_8h100.sh \
  --input /mnt/afs/yanpeishen/datasets/raw.jsonl \
  --output-root /mnt/afs/yanpeishen/results/portrait_quality_and_expert12d \
  --cache-root /mnt/afs/yanpeishen/cache/portrait-fused \
  --work-root /mnt/afs/yanpeishen/work/portrait-fused \
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
  --execution-window-size 500 \
  --download-workers 8 \
  --download-batch-size 8 \
  --download-concurrency 4 \
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

`run_fused_cluster_8h100.sh` starts one external Ray cluster and seven named,
detached HumanAesExpert actors before the shard runner. All 10 micro-shard
subprocesses connect to that cluster, so the model replicas are loaded once
per launcher run rather than once per micro-shard. The shell trap stops the
named actors and a Ray cluster that it started on success, failure, or
interruption. It refuses to reuse an existing Ray cluster by default; set
`PORTRAIT_ALLOW_EXISTING_RAY=1` only when the allocated node and seven free
GPUs are exclusively owned by this job.

The fused launcher also sets `DATA_JUICER_LAZY_OP_IMPORT=1`. In this opt-in
mode Data-Juicer registers only the operator modules named in the generated
YAML (`s3_download_file_mapper`, `image_portrait_quality_mapper`,
`image_portrait_cache_router_mapper`, and
`image_humanaesexpert_mapper`). General Data-Juicer commands keep the original
eager all-operator registration behavior when this environment variable is
absent.

Inspect the warm actor pool while the job is running:

```bash
DATA_JUICER_LAZY_OP_IMPORT=1 \
PYTHONPATH=/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer \
/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python \
  /mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer/demos/portrait_quality_gate/manage_humanaesexpert_pool.py \
  status \
  --ray-address auto \
  --namespace portrait-quality-gate \
  --prefix humanaesexpert \
  --size 7
```

Micro-shard validation and `SUCCESS` creation remain in
`run_sharded_pipeline.py`: each 10,000-row result is normalized, row-count
checked, and cache-usage checked before its marker is written. Keeping the Ray
cluster alive does not weaken or coarsen this checkpoint boundary.

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
