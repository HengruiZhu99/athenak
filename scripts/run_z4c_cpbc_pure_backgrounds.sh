#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ATHENA_EXE RUN_ROOT"
  exit 2
fi

athena_exe=$1
run_root=$2
source_root=$(cd "$(dirname "$0")/.." && pwd)
input_file="${source_root}/inputs/tests/z4c_characteristic_cpbc_purebg.athinput"
checker="${source_root}/analysis/z4c_characteristic/check_pure_background.py"
mkdir -p "${run_root}"

backgrounds=(minkowski schwarzschild kerr_a09)
for background in "${backgrounds[@]}"; do
  run_dir="${run_root}/${background}"
  if [[ -e ${run_dir} ]]; then
    echo "refusing to reuse ${run_dir}" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  background_args=()
  case "${background}" in
    minkowski)
      background_args=(
        coord/minkowski=true coord/a=0.0
        problem/bh_mass=0.0 problem/bh_spin=0.0
      )
      ;;
    schwarzschild)
      background_args=(
        coord/minkowski=false coord/a=0.0
        problem/bh_mass=1.0 problem/bh_spin=0.0
        problem/excision_damp_rate=50.0
        problem/excision_project_state=true
        problem/excision_freeze_radius=1.0
        problem/excision_ramp_radius=1.4
      )
      ;;
    kerr_a09)
      background_args=(
        coord/minkowski=false coord/a=0.9
        problem/bh_mass=1.0 problem/bh_spin=0.9
        problem/excision_damp_rate=50.0
        problem/excision_project_state=true
        problem/excision_freeze_radius=1.0
        problem/excision_ramp_radius=1.4
      )
      ;;
  esac

  echo "CPBC exact background ${background}"
  "${athena_exe}" -i "${input_file}" -d "${run_dir}" \
    problem/outer_sponge_enabled=false \
    problem/outer_sponge_geometry=radial \
    problem/outer_sponge_start_radius=4.0 \
    problem/outer_sponge_ramp_width=2.0 \
    problem/outer_sponge_damping_time=16.0 \
    "${background_args[@]}" >"${run_dir}/stdout.log" 2>&1

  history_file=$(find "${run_dir}" -name '*.user.hst' \
    ! -name '*.z4c.user.hst' -type f | sort | head -1)
  if [[ -z ${history_file} ]]; then
    echo "missing user history for ${background}" >&2
    exit 1
  fi
  python3 "${checker}" "${history_file}" "${run_dir}/stdout.log" |
    tee "${run_dir}/acceptance.txt"
done
