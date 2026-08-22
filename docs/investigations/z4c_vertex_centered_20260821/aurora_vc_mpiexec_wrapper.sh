#!/usr/bin/env bash
set -euo pipefail

test "$#" -ge 3
test "$1" = -n
ranks=$2
shift 2
test "${ranks}" -ge 1
test "${ranks}" -le 2
unset ZE_AFFINITY_MASK
exec mpiexec -n "${ranks}" -ppn "${ranks}" --depth 8 --cpu-bind depth \
  /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh "$@"
