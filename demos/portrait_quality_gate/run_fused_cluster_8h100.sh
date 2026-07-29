#!/usr/bin/env bash
set -Eeuo pipefail

readonly PYTHON="/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python"
readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly SCRIPT="${REPOSITORY}/demos/portrait_quality_gate/run_sharded_pipeline.py"
readonly POOL_MANAGER="${REPOSITORY}/demos/portrait_quality_gate/manage_humanaesexpert_pool.py"
readonly MODEL_CACHE="/mnt/afs/yanpeishen/model_cache/huggingface"
readonly YOLO_MODEL="/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n.pt"
readonly YOLO_POSE_MODEL="/mnt/afs/yanpeishen/.cache/data_juicer/models/yolo11n-pose.pt"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON}" >&2
  exit 1
fi
for required in "${SCRIPT}" "${POOL_MANAGER}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required script does not exist: ${required}" >&2
    exit 1
  fi
done
if [[ -z "${AOSS_CONF:-}" || "${AOSS_CONF}" != /* || ! -f "${AOSS_CONF}" ]]; then
  echo "AOSS_CONF must be an absolute path to an existing private config file." >&2
  exit 1
fi

export DATA_JUICER_LAZY_OP_IMPORT=1
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"

ray_owned=0
cleanup() {
  local exit_code=$?
  trap - EXIT
  "${PYTHON}" "${POOL_MANAGER}" stop \
    --ray-address auto \
    --namespace portrait-quality-gate \
    --prefix humanaesexpert \
    --size 7 >/dev/null 2>&1 || true
  if [[ "${ray_owned}" -eq 1 ]]; then
    "${PYTHON}" -m ray stop --force >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

if "${PYTHON}" -m ray status >/dev/null 2>&1; then
  if [[ "${PORTRAIT_ALLOW_EXISTING_RAY:-0}" != "1" ]]; then
    echo "A Ray cluster is already running. Set PORTRAIT_ALLOW_EXISTING_RAY=1" \
      "only when this job owns its seven free GPUs." >&2
    exit 1
  fi
else
  "${PYTHON}" -m ray start --head --disable-usage-stats
  ray_owned=1
fi

"${PYTHON}" "${POOL_MANAGER}" start \
  --ray-address auto \
  --namespace portrait-quality-gate \
  --prefix humanaesexpert \
  --size 7 \
  --model-cache "${MODEL_CACHE}"

"${PYTHON}" "${SCRIPT}" \
  --mode fused \
  --model-cache "${MODEL_CACHE}" \
  --yolo-model "${YOLO_MODEL}" \
  --yolo-pose-model "${YOLO_POSE_MODEL}" \
  --ray-address auto \
  --quality-workers 4 \
  --quality-gpus-per-worker 0.25 \
  --score-workers 7 \
  --persistent-actor-pool \
  --persistent-actor-namespace portrait-quality-gate \
  --persistent-actor-prefix humanaesexpert \
  "$@"
