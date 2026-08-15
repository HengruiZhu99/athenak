#!/bin/bash
set -euo pipefail
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
campaign_root=$(cd "${bundle_dir}/.." && pwd)
# shellcheck disable=SC1091
source "${bundle_dir}/contract.env"
bash "${bundle_dir}/require_bound_contract.sh"
test -z "${SLURM_JOB_ID:-}"
test "${COLLAPSE_ACCOUNT:-}" = "${EXPECTED_ACCOUNT}"
test -f "${campaign_root}/PREFLIGHT_COMPLETE"
test ! -e "${campaign_root}/run"
test ! -e "${campaign_root}/build"
test ! -e "${campaign_root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${EXPECTED_JOB_NAME}" -o '%A')"
(cd "${bundle_dir}" && sha256sum -c bundle.sha256 >/dev/null)
mkdir "${campaign_root}/allocation"
set +e
salloc --account="${COLLAPSE_ACCOUNT}" --job-name="${EXPECTED_JOB_NAME}" \
  --qos=interactive --constraint='gpu&hbm40g' --nodes=1 --ntasks=4 \
  --ntasks-per-node=4 --cpus-per-task=8 --gpus-per-node=4 --time=04:00:00 \
  bash "${bundle_dir}/run_allocation.sh" > \
  "${campaign_root}/allocation/allocation.stdout" 2> \
  "${campaign_root}/allocation/allocation.stderr"
allocation_status=$?
set -e
printf '%s\n' "${allocation_status}" > "${campaign_root}/allocation/allocation.status"
mapfile -t job_ids < <(grep -Eho 'job allocation [0-9]+' \
  "${campaign_root}/allocation/allocation.stdout" \
  "${campaign_root}/allocation/allocation.stderr" | awk '{print $3}' | sort -u)
if [[ ${#job_ids[@]} -eq 0 && ${allocation_status} -ne 0 ]]; then
  printf 'no_job_id_preallocation_failure\n' > \
    "${campaign_root}/allocation/job-id.status"
  exit "${allocation_status}"
fi
test "${#job_ids[@]}" -eq 1
job_id=${job_ids[0]}
[[ ${job_id} =~ ^[0-9]+$ ]]
printf '%s\n' "${job_id}" > "${campaign_root}/allocation/job-id.txt"

mode=failure
poll_limit=60
if [[ ${allocation_status} -eq 0 ]]; then mode=success; poll_limit=120; fi
settled=false
for _ in $(seq 1 "${poll_limit}"); do
  sacct -n -P -j "${job_id}" \
    -o JobIDRaw,JobName,Partition,QOS,State,ExitCode,ElapsedRaw,AllocNodes,NTasks,ReqCPUS,AllocTRES,NodeList | \
    sed '/^[[:space:]]*$/d' | sort -t '|' -k1,1V > \
    "${campaign_root}/allocation/sacct-current.psv"
  if awk -v job="${job_id}" -v expected="${EXPECTED_NUMBERED_STEPS}" \
      -v expected_name="${EXPECTED_JOB_NAME}" -v mode="${mode}" \
      -f "${bundle_dir}/validate_accounting.awk" \
      "${campaign_root}/allocation/sacct-current.psv" > \
      "${campaign_root}/allocation/accounting-verdict.txt"; then
    settled=true; break
  fi
  sleep 2
done
test "${settled}" = true
mv "${campaign_root}/allocation/sacct-current.psv" \
  "${campaign_root}/allocation/sacct-settled.psv"
final_status=${allocation_status}
if [[ ${mode} == success ]]; then
  set +e
  test "${allocation_status}" -eq 0 && \
    grep -Fxq HALF_PLANE_KERR_MOVING_PUNCTURE_PASS \
      "${campaign_root}/run/verdict.txt"
  final_status=$?
  set -e
else
  test "${allocation_status}" -ne 0
fi
set +e
bash "${bundle_dir}/finalize_campaign.sh" "${campaign_root}" "${SOURCE_ROOT}" \
  "${EXPECTED_SOURCE_COMMIT}" "${EXPECTED_SOURCE_TREE}"
finalizer_status=$?
set -e
if [[ ${finalizer_status} -ne 0 ]]; then exit 2; fi
exit "${final_status}"
