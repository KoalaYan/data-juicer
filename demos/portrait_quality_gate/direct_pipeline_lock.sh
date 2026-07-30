#!/usr/bin/env bash

# Shared AFS lock for mutually exclusive direct-pipeline launchers.
# The caller must set -u/-e as desired and call release_direct_pipeline_lock
# from an EXIT trap.

direct_lock_owned=0
direct_lock_dir=""

direct_process_is_active() {
  local pid="$1"
  local state

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${pid}/stat" ]] || return 1
  state="$(/usr/bin/awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)"
  [[ -n "${state}" && "${state}" != "Z" && "${state}" != "X" ]]
}

direct_lock_value() {
  local key="$1"
  local owner_file="$2"

  /usr/bin/awk -F= -v wanted="${key}" \
    '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' \
    "${owner_file}" 2>/dev/null || true
}

remove_confirmed_local_stale_lock() {
  local current_host="$1"
  local owner_file="${direct_lock_dir}/owner"
  local owner_host
  local owner_pid

  [[ -f "${owner_file}" ]] || return 1
  owner_host="$(direct_lock_value host "${owner_file}")"
  owner_pid="$(direct_lock_value pid "${owner_file}")"
  [[ "${owner_host}" == "${current_host}" ]] || return 1
  if direct_process_is_active "${owner_pid}"; then
    return 1
  fi

  echo "[lock] removing confirmed stale local lock pid=${owner_pid}" >&2
  /bin/rm -f "${owner_file}"
  /bin/rmdir "${direct_lock_dir}" 2>/dev/null || return 1
}

acquire_direct_pipeline_lock() {
  local run_root="$1"
  local launcher_name="$2"
  local current_host
  local created=0
  local owner_file

  direct_lock_dir="${run_root}/DIRECT_PIPELINE_LOCK"
  owner_file="${direct_lock_dir}/owner"
  current_host="$(/bin/hostname)"
  /bin/mkdir -p "${run_root}"

  if /bin/mkdir "${direct_lock_dir}" 2>/dev/null; then
    created=1
  else
    if remove_confirmed_local_stale_lock "${current_host}" \
      && /bin/mkdir "${direct_lock_dir}" 2>/dev/null; then
      created=1
    fi
  fi
  if [[ "${created}" -ne 1 && -f "${owner_file}" ]]; then
    echo "Another direct pipeline owns the shared AFS checkpoint lock:" >&2
    /bin/cat "${owner_file}" >&2
    echo "Stop that job before switching between development and cluster nodes." >&2
    return 1
  elif [[ "${created}" -ne 1 ]]; then
    echo "Shared lock directory exists without owner metadata: ${direct_lock_dir}" >&2
    echo "Inspect it manually before removing it." >&2
    return 1
  fi

  {
    echo "host=${current_host}"
    echo "pid=$$"
    echo "launcher=${launcher_name}"
    echo "started_at=$(/bin/date --iso-8601=seconds)"
  } > "${owner_file}"
  direct_lock_owned=1
  echo "[lock] acquired ${direct_lock_dir}"
}

release_direct_pipeline_lock() {
  if [[ "${direct_lock_owned:-0}" -ne 1 || -z "${direct_lock_dir:-}" ]]; then
    return
  fi
  /bin/rm -f "${direct_lock_dir}/owner"
  /bin/rmdir "${direct_lock_dir}" 2>/dev/null || true
  direct_lock_owned=0
}
