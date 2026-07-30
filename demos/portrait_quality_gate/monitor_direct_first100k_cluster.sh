#!/usr/bin/env bash
set -Eeuo pipefail

readonly RUN_ROOT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly CACHE_ROOT="/mnt/afs/yanpeishen/cache/data-juicer/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly LOG_FILE="${RUN_ROOT}/logs/pipeline.log"
readonly STATE_FILE="${RUN_ROOT}/logs/pipeline.state"
readonly PID_FILE="${RUN_ROOT}/logs/pipeline.pid"
readonly PROGRESS_FILE="${RUN_ROOT}/output/PROGRESS.json"

echo "===== state ====="
if [[ -f "${STATE_FILE}" ]]; then
  /bin/cat "${STATE_FILE}"
else
  echo "missing: ${STATE_FILE}"
fi

echo "===== pid ====="
if [[ -f "${PID_FILE}" ]]; then
  pipeline_pid="$(<"${PID_FILE}")"
  echo "${pipeline_pid}"
  /bin/ps -o pid=,ppid=,stat=,etime=,cmd= -p "${pipeline_pid}" || true
else
  echo "no active pid file"
fi

echo "===== progress ====="
if [[ -f "${PROGRESS_FILE}" ]]; then
  /bin/cat "${PROGRESS_FILE}"
else
  echo "no completed micro-shard yet"
fi

echo "===== cache ====="
cache_files="$(
  /usr/bin/find "${CACHE_ROOT}" -type f \
    ! -name '.data_juicer_cache_quota.json' \
    ! -name '.data_juicer_cache_quota.lock' 2>/dev/null \
    | /usr/bin/wc -l
)"
echo "files=${cache_files}"
/usr/bin/du -sh "${CACHE_ROOT}" 2>/dev/null || true

echo "===== recent throughput ====="
if [[ -f "${LOG_FILE}" ]]; then
  /usr/bin/tail -c 200000 "${LOG_FILE}" \
    | /usr/bin/tr -d '\000' \
    | /usr/bin/grep -E \
      '\[progress\]|\[micro-(start|done)\]|\[worker-ready\]|\[done\]|Traceback|FAILED' \
    | /usr/bin/tail -n 50
else
  echo "missing: ${LOG_FILE}"
fi
