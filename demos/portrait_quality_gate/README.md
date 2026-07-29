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
