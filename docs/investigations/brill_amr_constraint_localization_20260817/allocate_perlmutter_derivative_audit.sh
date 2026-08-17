#!/usr/bin/env bash
set -euo pipefail

root=/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-constraint-localization-55f9147b-v4-20260817
account=${COLLAPSE_ACCOUNT:?COLLAPSE_ACCOUNT must be set}
job=cartoon-brill-localize-55f914-v4
test "${account}" = m3328_g
test -f "${root}/PREFLIGHT_COMPLETE"
test ! -e "${root}/build"
test ! -e "${root}/run"
test ! -e "${root}/allocation"
test -z "$(squeue -h -u "${USER}" -n "${job}" -o '%A')"
mkdir "${root}/allocation"
set +e
salloc --account="${account}" --job-name="${job}" --qos=shared_interactive \
  --constraint='gpu&hbm40g' --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=32 --gpus-per-node=1 --time=00:30:00 \
  bash "${root}/run_perlmutter_derivative_audit.sh" \
  > "${root}/allocation/allocation.stdout" \
  2> "${root}/allocation/allocation.stderr"
status=$?
set -e
printf '%s\n' "${status}" > "${root}/allocation/allocation.status"
mapfile -t ids < <(grep -Eho 'job allocation [0-9]+' \
  "${root}/allocation/allocation.stdout" "${root}/allocation/allocation.stderr" | \
  awk '{print $3}' | sort -u)
if [[ ${#ids[@]} -eq 1 && ${ids[0]} =~ ^[0-9]+$ ]]; then
  id=${ids[0]}
  printf '%s\n' "${id}" > "${root}/allocation/job-id.txt"
  for _ in {1..60}; do
    state=$(sacct -n -X -P -j "${id}" -o State | sed '/^[[:space:]]*$/d' | \
      head -1 | cut -d'|' -f1)
    case "${state}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|REVOKED) break ;;
    esac
    sleep 2
  done
  sacct -n -P -j "${id}" -o \
    JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocNodes,NTasks,ReqCPUS,AllocTRES,NodeList | \
    sed '/^[[:space:]]*$/d' | sort -t'|' -k1,1V > \
    "${root}/allocation/sacct-settled.psv"
fi
find "${root}/allocation" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "${root}/allocation/SHA256SUMS"
sha256sum "${root}/allocation/SHA256SUMS" > \
  "${root}/allocation/SHA256SUMS.sha256"
exit "${status}"
