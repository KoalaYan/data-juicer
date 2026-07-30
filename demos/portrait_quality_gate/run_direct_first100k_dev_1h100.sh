#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly CLUSTER_LAUNCHER="${REPOSITORY}/demos/portrait_quality_gate/run_direct_first100k_cluster_8h100.sh"

if [[ ! -x "${CLUSTER_LAUNCHER}" ]]; then
  echo "Required launcher is not executable: ${CLUSTER_LAUNCHER}" >&2
  exit 1
fi

# The development host has one H100. GPU quality remains the default because
# the CPU path is substantially slower on this host. Set
# DIRECT_QUALITY_DEVICE=cpu explicitly for further CPU experiments.
visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
dev_gpu_token="${DIRECT_DEV_GPU_TOKEN:-${visible_devices%%,*}}"
quality_device="${DIRECT_QUALITY_DEVICE:-cuda}"
if [[ -z "${dev_gpu_token}" ]]; then
  echo "Unable to resolve the development GPU token." >&2
  exit 1
fi
if [[ "${quality_device}" != "cpu" && "${quality_device}" != "cuda" ]]; then
  echo "DIRECT_QUALITY_DEVICE must be cpu or cuda." >&2
  exit 1
fi

if [[ "${quality_device}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${dev_gpu_token},${dev_gpu_token}"
else
  export CUDA_VISIBLE_DEVICES="${dev_gpu_token}"
fi
export DIRECT_LAUNCHER_ROLE=dev
export DIRECT_SCORE_WORKERS=1
export DIRECT_QUALITY_DEVICE="${quality_device}"
export DIRECT_QUALITY_CPU_THREADS="${DIRECT_QUALITY_CPU_THREADS:-14}"
export DIRECT_DOWNLOAD_WORKERS="${DIRECT_DOWNLOAD_WORKERS:-16}"
export DIRECT_DOWNLOAD_PREFETCH="${DIRECT_DOWNLOAD_PREFETCH:-64}"
export DIRECT_DOWNLOAD_QUEUE_SIZE="${DIRECT_DOWNLOAD_QUEUE_SIZE:-64}"
export DIRECT_SCORE_QUEUE_SIZE="${DIRECT_SCORE_QUEUE_SIZE:-32}"
export DIRECT_QUALITY_BATCH_SIZE="${DIRECT_QUALITY_BATCH_SIZE:-32}"

exec "${CLUSTER_LAUNCHER}" "$@"
