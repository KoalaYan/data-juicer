#!/usr/bin/env bash
set -Eeuo pipefail

readonly PYTHON="/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python"
readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly SCRIPT="${REPOSITORY}/demos/portrait_quality_gate/run_sharded_pipeline.py"
readonly MODEL_CACHE="/mnt/afs/yanpeishen/model_cache/huggingface"

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
  --mode stage2 \
  --model-cache "${MODEL_CACHE}" \
  --ray-address local \
  --score-workers 8 \
  "$@"
