#!/usr/bin/env bash
set -Eeuo pipefail

readonly PYTHON="/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python"
readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly SCRIPT="${REPOSITORY}/demos/portrait_quality_gate/run_fused_portrait_humanaesexpert.py"
readonly MODEL_CACHE="/mnt/afs/yanpeishen/model_cache/huggingface"
readonly YOLO_MODEL="/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n.pt"
readonly YOLO_POSE_MODEL="/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n-pose.pt"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Pipeline script does not exist: ${SCRIPT}" >&2
  exit 1
fi
if [[ -z "${AOSS_CONF:-}" || "${AOSS_CONF}" != /* || ! -f "${AOSS_CONF}" ]]; then
  echo "AOSS_CONF must be an absolute path to an existing private config file." >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT}" \
  --model-cache "${MODEL_CACHE}" \
  --yolo-model "${YOLO_MODEL}" \
  --yolo-pose-model "${YOLO_POSE_MODEL}" \
  --ray-address local \
  --quality-workers 4 \
  --quality-gpus-per-worker 0.25 \
  --score-workers 7 \
  "$@"
