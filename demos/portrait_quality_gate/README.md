# Portrait hard-quality gate

This demo performs conservative first-stage filtering for portrait raw data:

- AOSS-backed `s3://...` materialization with source URI preservation;
- near-black/near-white and severe exposure detection;
- person presence from YOLO plus frontal-face fallback;
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
- path-independent exposure and sharpness metrics;
- person/face counts and detector confidence;
- explicit `reject_reasons` and `warning_reasons`.

Only high-confidence failures are rejected. Background-only overexposure and
detector failures are marked `uncertain`.

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
