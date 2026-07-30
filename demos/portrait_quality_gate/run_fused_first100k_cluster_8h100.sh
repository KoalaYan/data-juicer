#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly LAUNCHER="${REPOSITORY}/demos/portrait_quality_gate/run_fused_cluster_8h100.sh"
readonly INPUT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/humanaesexpert_8b/human_baixing_0515_first100000_20260728/selection/first_100000_ordered.jsonl"
readonly RUN_ROOT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_fused_first100k_micro10k_20260729"
readonly OUTPUT_ROOT="${RUN_ROOT}/output"
readonly WORK_ROOT="${RUN_ROOT}/work"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly CACHE_ROOT="/mnt/afs/yanpeishen/cache/data-juicer/human_baixing_0515_fused_first100k_micro10k_20260729"
readonly LOG_FILE="${LOG_ROOT}/pipeline.log"
readonly STATE_FILE="${LOG_ROOT}/pipeline.state"

if [[ -z "${AOSS_CONF:-}" || "${AOSS_CONF}" != /* || ! -f "${AOSS_CONF}" ]]; then
  echo "AOSS_CONF must be an absolute path to an existing private config file." >&2
  exit 1
fi
for required in "${LAUNCHER}" "${INPUT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path does not exist: ${required}" >&2
    exit 1
  fi
done

/bin/mkdir -p "${LOG_ROOT}"

on_error() {
  local exit_code=$?
  echo "FAILED time=$(/bin/date --iso-8601=seconds) exit_code=${exit_code}" \
    > "${STATE_FILE}"
  exit "${exit_code}"
}
trap on_error ERR

echo "RUNNING time=$(/bin/date --iso-8601=seconds) pid=$$" \
  > "${STATE_FILE}"
echo "[run] input=${INPUT}"
echo "[run] output_root=${OUTPUT_ROOT}"
echo "[run] work_root=${WORK_ROOT}"
echo "[run] cache_root=${CACHE_ROOT}"
echo "[run] logical_shard=100000 micro_shard=10000 index=0"

"${LAUNCHER}" \
  --input "${INPUT}" \
  --input-adapter purchased-selection \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --work-root "${WORK_ROOT}" \
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
  --logical-shard-index 0 \
  --download-workers 8 \
  --download-concurrency 1 \
  --aoss-download-attempts 5 \
  --aoss-retry-initial-delay 1.5 \
  --aoss-retry-max-delay 12 \
  --aoss-retry-jitter 1 \
  --max-cache-files 2048 \
  --max-cache-bytes 214748364800 \
  "$@" 2>&1 | /usr/bin/tee -a "${LOG_FILE}"

echo "SUCCEEDED time=$(/bin/date --iso-8601=seconds) pid=$$" \
  > "${STATE_FILE}"
