#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ATHENA_EXE RUN_ROOT"
  exit 2
fi

source_root=$(cd "$(dirname "$0")/.." && pwd)
exec "${source_root}/scripts/run_z4c_cpbc_oblique_pulse.sh" "$@" 64 3
