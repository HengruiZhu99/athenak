#!/bin/bash
# Record the exact Aurora rank-to-PVC-tile binding, then execute AthenaK.
set -euo pipefail
: "${RANK_EVIDENCE_ROOT:?}"
rank=${PALS_RANKID:-${PMI_RANK:-unknown}}
local_rank=${PALS_LOCAL_RANKID:-${MPI_LOCALRANKID:-unknown}}
mkdir -p "${RANK_EVIDENCE_ROOT}"
{
  printf 'rank=%s\nlocal_rank=%s\n' "${rank}" "${local_rank}"
  printf 'ZE_AFFINITY_MASK=%s\n' "${ZE_AFFINITY_MASK:-unset}"
  printf 'ONEAPI_DEVICE_SELECTOR=%s\n' "${ONEAPI_DEVICE_SELECTOR:-unset}"
  printf 'Cpus_allowed_list=%s\n' "$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
  sycl-ls
} > "${RANK_EVIDENCE_ROOT}/rank-${rank}.log" 2>&1
exec "$@"
