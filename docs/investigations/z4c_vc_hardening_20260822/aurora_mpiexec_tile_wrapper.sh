#!/bin/bash
set -euo pipefail

test "${1:-}" = "-n"
test -n "${2:-}"
ranks=$2
shift 2

tile_wrapper=/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh
test -x "${tile_wrapper}"
exec /opt/cray/pals/1.8/bin/mpiexec -n "${ranks}" -ppn "${ranks}" \
  --depth 8 --cpu-bind depth "${tile_wrapper}" "$@"
