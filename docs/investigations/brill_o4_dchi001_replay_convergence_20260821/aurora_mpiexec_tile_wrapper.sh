#!/bin/bash
# Adapter for tests that expect an mpiexec-compatible launcher. It inserts the
# Aurora local-rank-to-PVC-tile wrapper while preserving the requested rank count.
set -euo pipefail
test "$#" -ge 3
test "$1" = -n
ranks=$2
shift 2
exec mpiexec -n "${ranks}" -ppn "${ranks}" --depth 8 --cpu-bind depth \
  /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh "$@"
