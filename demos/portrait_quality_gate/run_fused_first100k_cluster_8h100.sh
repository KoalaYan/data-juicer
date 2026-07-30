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
readonly PID_FILE="${LOG_ROOT}/pipeline.pid"

foreground=0
if [[ "${1:-}" == "--foreground" ]]; then
  foreground=1
  shift
fi

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

if [[ "${foreground}" -eq 0 ]]; then
  if [[ -s "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] \
      && /bin/kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Pipeline is already running with pid=${existing_pid}" >&2
      echo "Log: ${LOG_FILE}" >&2
      exit 1
    fi
  fi
  /usr/bin/nohup "$0" --foreground "$@" \
    >> "${LOG_FILE}" 2>&1 </dev/null &
  background_pid=$!
  echo "${background_pid}" > "${PID_FILE}"
  echo "[started] pid=${background_pid}"
  echo "[log] ${LOG_FILE}"
  echo "[state] ${STATE_FILE}"
  echo "[tail] /usr/bin/tail -n 200 -F ${LOG_FILE}"
  exit 0
fi

on_error() {
  local exit_code=$?
  echo "FAILED time=$(/bin/date --iso-8601=seconds) exit_code=${exit_code}" \
    > "${STATE_FILE}"
  /bin/rm -f "${PID_FILE}"
  exit "${exit_code}"
}
trap on_error ERR
trap 'echo "INTERRUPTED time=$(/bin/date --iso-8601=seconds) signal=TERM" > "${STATE_FILE}"; /bin/rm -f "${PID_FILE}"; exit 143' TERM
trap 'echo "INTERRUPTED time=$(/bin/date --iso-8601=seconds) signal=INT" > "${STATE_FILE}"; /bin/rm -f "${PID_FILE}"; exit 130' INT

echo "$$" > "${PID_FILE}"
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
  --execution-window-size 500 \
  --logical-shard-index 0 \
  --download-workers 8 \
  --download-batch-size 8 \
  --download-concurrency 4 \
  --aoss-download-attempts 5 \
  --aoss-retry-initial-delay 1.5 \
  --aoss-retry-max-delay 12 \
  --aoss-retry-jitter 1 \
  --max-cache-files 2048 \
  --max-cache-bytes 214748364800 \
  "$@"

echo "SUCCEEDED time=$(/bin/date --iso-8601=seconds) pid=$$" \
  > "${STATE_FILE}"
/bin/rm -f "${PID_FILE}"
