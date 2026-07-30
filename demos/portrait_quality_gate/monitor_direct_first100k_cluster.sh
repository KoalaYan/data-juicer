#!/usr/bin/env bash
set -Eeuo pipefail

readonly RUN_ROOT="/mnt/afs/yanpeishen/project/t2i/purchased-data-governance/results/portrait_quality_gate/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly CACHE_ROOT="/mnt/afs/yanpeishen/cache/data-juicer/human_baixing_0515_direct_first100k_micro10k_20260730"
readonly LOG_FILE="${RUN_ROOT}/logs/pipeline.log"
readonly DEV_LOG_FILE="${RUN_ROOT}/logs/pipeline-dev.log"
readonly PROGRESS_FILE="${RUN_ROOT}/output/PROGRESS.json"

for role_suffix in "" "-dev"; do
  if [[ -z "${role_suffix}" ]]; then
    role="cluster"
  else
    role="dev"
  fi
  state_file="${RUN_ROOT}/logs/pipeline${role_suffix}.state"
  pid_file="${RUN_ROOT}/logs/pipeline${role_suffix}.pid"
  echo "===== ${role} state ====="
  if [[ -f "${state_file}" ]]; then
    /bin/cat "${state_file}"
  else
    echo "missing: ${state_file}"
  fi

  echo "===== ${role} pid ====="
  if [[ -f "${pid_file}" ]]; then
    pipeline_pid="$(<"${pid_file}")"
    echo "${pipeline_pid}"
    /bin/ps -o pid=,ppid=,stat=,etime=,cmd= -p "${pipeline_pid}" || true
  else
    echo "no active pid file"
  fi
done

echo "===== shared checkpoint lock ====="
if [[ -f "${RUN_ROOT}/DIRECT_PIPELINE_LOCK/owner" ]]; then
  /bin/cat "${RUN_ROOT}/DIRECT_PIPELINE_LOCK/owner"
else
  echo "unlocked"
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
for log_file in "${LOG_FILE}" "${DEV_LOG_FILE}"; do
  echo "--- ${log_file} ---"
  if [[ -f "${log_file}" ]]; then
    {
      /usr/bin/tail -c 200000 "${log_file}" \
        | /usr/bin/tr -d '\000' \
        | /usr/bin/grep -E \
          '\[progress\]|\[block-(start|progress|done|skip)\]|\[micro-(start|done)\]|\[worker-ready\]|\[done\]|\[lock\]|Traceback|FAILED' \
        | /usr/bin/tail -n 50
    } || true
  else
    echo "missing"
  fi
done
