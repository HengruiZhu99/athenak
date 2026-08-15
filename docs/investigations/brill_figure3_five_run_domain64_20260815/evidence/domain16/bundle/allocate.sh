#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test "${root}" = "${REMOTE_ROOT}"
test "${COLLAPSE_ACCOUNT:-}" = "${EXPECTED_ACCOUNT}"
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${root}/run"
test ! -e "${root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_JOB_NAME}" -o '%A')"
mkdir "${root}/allocation"
set +e
salloc --account="${COLLAPSE_ACCOUNT}" --job-name="${EXPECTED_JOB_NAME}" \
  --qos=interactive --constraint='gpu&hbm40g' --nodes=1 \
  --ntasks="${EXPECTED_RANKS}" --ntasks-per-node="${EXPECTED_RANKS}" \
  --cpus-per-task=8 --gpus-per-node="${EXPECTED_GPUS}" --time=04:00:00 \
  bash "${bundle_dir}/run_allocation.sh" > \
  "${root}/allocation/allocation.stdout" 2> \
  "${root}/allocation/allocation.stderr"
status=$?
set -e
printf '%s\n' "${status}" > "${root}/allocation/allocation.status"
mapfile -t job_ids < <(grep -Eho 'job allocation [0-9]+' \
  "${root}/allocation/allocation.stdout" \
  "${root}/allocation/allocation.stderr" | awk '{print $3}' | sort -u)
if [[ ${#job_ids[@]} -eq 1 && ${job_ids[0]} =~ ^[0-9]+$ ]]; then
  job_id=${job_ids[0]}
  printf '%s\n' "${job_id}" > "${root}/allocation/job-id.txt"
  for _ in $(seq 1 120); do
    sacct -n -P -j "${job_id}" \
      -o JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocNodes,NTasks,ReqCPUS,AllocTRES,NodeList | \
      sed '/^[[:space:]]*$/d' | sort -t '|' -k1,1V > \
      "${root}/allocation/sacct-current.psv"
    parent_state=$(awk -F'|' -v id="${job_id}" '$1==id {print $3}' \
      "${root}/allocation/sacct-current.psv")
    case "${parent_state}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) break ;;
    esac
    sleep 2
  done
  mv "${root}/allocation/sacct-current.psv" \
    "${root}/allocation/sacct-settled.psv"
fi
(cd "${root}" && find run allocation bundle preflight -type f \
  ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 | \
  sort -z | xargs -0 -r sha256sum) > "${root}/SHA256SUMS"
sha256sum "${root}/SHA256SUMS" > "${root}/SHA256SUMS.sha256"
exit "${status}"
