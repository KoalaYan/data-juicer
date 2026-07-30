#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="/mnt/afs/yanpeishen/project/t2i/data-pipeline/data-juicer"
readonly PYTHON="/mnt/afs/yanpeishen/.conda/envs/portrait-hae-datajuicer/bin/python"
readonly PIPELINE="${REPOSITORY}/demos/portrait_quality_gate/run_direct_portrait_humanaesexpert.py"
readonly LOCK_HELPER="${REPOSITORY}/demos/portrait_quality_gate/direct_pipeline_lock.sh"
readonly INPUT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/humanaesexpert_8b/human_baixing_0515_first100000_20260728/selection/first_100000_ordered.jsonl"
readonly RUN_ROOT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly OUTPUT_ROOT="${RUN_ROOT}/output"
readonly WORK_ROOT="${RUN_ROOT}/work"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly CACHE_ROOT="/mnt/afs/yanpeishen/cache/data-juicer/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly MODEL_CACHE="/mnt/afs/yanpeishen/model_cache/huggingface"
readonly LAUNCHER_ROLE="${DIRECT_LAUNCHER_ROLE:-cluster}"
if [[ "${LAUNCHER_ROLE}" == "dev" ]]; then
  readonly LOG_SUFFIX="-dev"
elif [[ "${LAUNCHER_ROLE}" == "cluster" ]]; then
  readonly LOG_SUFFIX=""
else
  echo "DIRECT_LAUNCHER_ROLE must be cluster or dev." >&2
  exit 1
fi
readonly LOG_FILE="${LOG_ROOT}/pipeline${LOG_SUFFIX}.log"
readonly STATE_FILE="${LOG_ROOT}/pipeline${LOG_SUFFIX}.state"
readonly PID_FILE="${LOG_ROOT}/pipeline${LOG_SUFFIX}.pid"
readonly SCRIPT_NAME="${0##*/}"

pipeline_pid_is_active() {
  local pid="$1"
  local process_state
  local process_cmdline

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${pid}/stat" && -r "/proc/${pid}/cmdline" ]] || return 1
  process_state="$(/usr/bin/awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
  [[ -n "${process_state}" && "${process_state}" != "Z" && "${process_state}" != "X" ]] \
    || return 1
  process_cmdline="$(/usr/bin/tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${process_cmdline}" == *"${SCRIPT_NAME}"* ]]
}

foreground=0
if [[ "${1:-}" == "--foreground" ]]; then
  foreground=1
  shift
fi

if [[ -z "${AOSS_CONF:-}" || "${AOSS_CONF}" != /* || ! -f "${AOSS_CONF}" ]]; then
  echo "AOSS_CONF must be an absolute path to an existing private config file." >&2
  exit 1
fi
for required in "${PYTHON}" "${PIPELINE}" "${LOCK_HELPER}" "${INPUT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path does not exist: ${required}" >&2
    exit 1
  fi
done
source "${LOCK_HELPER}"

/bin/mkdir -p "${LOG_ROOT}"

if [[ "${foreground}" -eq 0 ]]; then
  if [[ -s "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if pipeline_pid_is_active "${existing_pid}"; then
      echo "Pipeline is already running with pid=${existing_pid}" >&2
      echo "Log: ${LOG_FILE}" >&2
      exit 1
    fi
    echo "[stale] removing inactive, zombie, or reused pipeline pid=${existing_pid}" >&2
    /bin/rm -f "${PID_FILE}"
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
pipeline_child_pid=""
on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  if [[ -n "${pipeline_child_pid}" ]] \
    && /bin/kill -0 "${pipeline_child_pid}" 2>/dev/null; then
    /bin/kill "-${signal_name}" "${pipeline_child_pid}" 2>/dev/null || true
    wait "${pipeline_child_pid}" 2>/dev/null || true
  fi
  echo "INTERRUPTED time=$(/bin/date --iso-8601=seconds) signal=${signal_name}" \
    > "${STATE_FILE}"
  /bin/rm -f "${PID_FILE}"
  exit "${exit_code}"
}
trap on_error ERR
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap release_direct_pipeline_lock EXIT

acquire_direct_pipeline_lock "${RUN_ROOT}" "${SCRIPT_NAME}"

echo "$$" > "${PID_FILE}"
echo "RUNNING time=$(/bin/date --iso-8601=seconds) pid=$$" \
  > "${STATE_FILE}"

export DATA_JUICER_LAZY_OP_IMPORT=1
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "[run] executor=direct-no-ray"
echo "[run] launcher_role=${LAUNCHER_ROLE}"
echo "[run] python=${PYTHON}"
echo "[run] input=${INPUT}"
echo "[run] output_root=${OUTPUT_ROOT}"
echo "[run] work_root=${WORK_ROOT}"
echo "[run] cache_root=${CACHE_ROOT}"
echo "[run] logical_shard=100000 micro_shard=10000 index=0"
echo "[run] download_workers=${DIRECT_DOWNLOAD_WORKERS:-24}"
echo "[run] score_workers=${DIRECT_SCORE_WORKERS:-7}"
echo "[run] execution_mode=${DIRECT_EXECUTION_MODE:-fused}"
echo "[run] stage_block_size=${DIRECT_STAGE_BLOCK_SIZE:-2000}"
echo "[run] quality_device=${DIRECT_QUALITY_DEVICE:-cuda}"
echo "[run] quality_cpu_threads=${DIRECT_QUALITY_CPU_THREADS:-16}"
echo "[run] visible_gpu_tokens=${CUDA_VISIBLE_DEVICES}"

"${PYTHON}" "${PIPELINE}" \
  --input "${INPUT}" \
  --input-adapter purchased-selection \
  --output-root "${OUTPUT_ROOT}" \
  --work-root "${WORK_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --model-cache "${MODEL_CACHE}" \
  --logical-shard-size 100000 \
  --micro-shard-size 10000 \
  --execution-mode "${DIRECT_EXECUTION_MODE:-fused}" \
  --stage-block-size "${DIRECT_STAGE_BLOCK_SIZE:-2000}" \
  --logical-shard-index 0 \
  --download-workers "${DIRECT_DOWNLOAD_WORKERS:-24}" \
  --download-prefetch "${DIRECT_DOWNLOAD_PREFETCH:-96}" \
  --download-queue-size "${DIRECT_DOWNLOAD_QUEUE_SIZE:-128}" \
  --score-queue-size "${DIRECT_SCORE_QUEUE_SIZE:-128}" \
  --result-queue-size "${DIRECT_RESULT_QUEUE_SIZE:-512}" \
  --quality-batch-size "${DIRECT_QUALITY_BATCH_SIZE:-64}" \
  --quality-batch-wait "${DIRECT_QUALITY_BATCH_WAIT:-0.05}" \
  --quality-device "${DIRECT_QUALITY_DEVICE:-cuda}" \
  --quality-cpu-threads "${DIRECT_QUALITY_CPU_THREADS:-16}" \
  --score-workers "${DIRECT_SCORE_WORKERS:-7}" \
  --max-cache-files "${DIRECT_MAX_CACHE_FILES:-512}" \
  --max-cache-bytes "${DIRECT_MAX_CACHE_BYTES:-53687091200}" \
  --aoss-download-attempts 5 \
  --aoss-retry-initial-delay 1.5 \
  --aoss-retry-max-delay 12 \
  --aoss-retry-jitter 1 \
  "$@" &
pipeline_child_pid=$!
wait "${pipeline_child_pid}"
pipeline_child_pid=""

echo "SUCCEEDED time=$(/bin/date --iso-8601=seconds) pid=$$" \
  > "${STATE_FILE}"
/bin/rm -f "${PID_FILE}"
