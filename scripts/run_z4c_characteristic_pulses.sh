#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 ATHENA_EXE RUN_ROOT [RESOLUTION=128] [CENTER=3] [TLIM=12]"
  exit 2
fi

athena_exe=$1
run_root=$2
resolution=${3:-128}
pulse_center=${4:-3.0}
tlim=${5:-12.0}
extrap_order=${CPBC_EXTRAP_ORDER:-2}
half_width=${CPBC_HALF_WIDTH:-8.0}
maximum_interior_ratio=${CPBC_MAX_INTERIOR_RATIO:-0.02}
boundary_rhs=${Z4C_BOUNDARY_RHS:-characteristic_cpbc}
characteristic_bc_source=${Z4C_CHARACTERISTIC_BC_SOURCE:-zero_rate}
control_root=${Z4C_CONTROL_ROOT:-}
check_boundary_diagnostic=${Z4C_CHECK_BOUNDARY_DIAGNOSTIC:-1}
source_root=$(cd "$(dirname "$0")/.." && pwd)
input_file="${source_root}/inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput"
checker="${source_root}/analysis/z4c_characteristic/check_pulse_log.py"
binary_checker="${source_root}/analysis/z4c_characteristic/check_pulse_binary.py"

mkdir -p "${run_root}"

default_families=(
  lapse
  shift_longitudinal
  constraint_scalar_theta
  constraint_scalar_z
  shift_transverse_1
  shift_transverse_2
  constraint_transverse_1
  constraint_transverse_2
  tt_plus
  tt_cross
)
read -r -a axes <<< "${CPBC_AXES:-1 2 3}"
read -r -a sides <<< "${CPBC_SIDES:--1 1}"
if [[ -n ${CPBC_FAMILIES:-} ]]; then
  read -r -a families <<< "${CPBC_FAMILIES}"
else
  families=("${default_families[@]}")
fi

sector_for_family() {
  case "$1" in
    lapse|shift_longitudinal|shift_transverse_1|shift_transverse_2)
      echo gauge
      ;;
    constraint_*)
      echo constraint
      ;;
    tt_*)
      echo radiation
      ;;
    *)
      return 1
      ;;
  esac
}

for axis in "${axes[@]}"; do
  mesh_args=()
  for coordinate in 1 2 3; do
    if [[ ${coordinate} -eq ${axis} ]]; then
      mesh_args+=(
        "mesh/nx${coordinate}=${resolution}"
        "mesh/x${coordinate}min=-${half_width}"
        "mesh/x${coordinate}max=${half_width}"
        "mesh/ix${coordinate}_bc=outflow"
        "mesh/ox${coordinate}_bc=outflow"
        "meshblock/nx${coordinate}=64"
      )
    else
      mesh_args+=(
        "mesh/nx${coordinate}=8"
        "mesh/x${coordinate}min=-0.5"
        "mesh/x${coordinate}max=0.5"
        "mesh/ix${coordinate}_bc=periodic"
        "mesh/ox${coordinate}_bc=periodic"
        "meshblock/nx${coordinate}=8"
      )
    fi
  done
  for side in "${sides[@]}"; do
    for family in "${families[@]}"; do
      run_name="axis${axis}_side${side}_${family}"
      run_dir="${run_root}/${run_name}"
      mkdir -p "${run_dir}"
      echo "${boundary_rhs}/${characteristic_bc_source} pulse ${run_name}"
      "${athena_exe}" -i "${input_file}" -d "${run_dir}" \
        "${mesh_args[@]}" \
        "time/tlim=${tlim}" \
        "problem/characteristic_test_family=${family}" \
        "problem/characteristic_test_axis=${axis}" \
        "problem/characteristic_test_side=${side}" \
        "problem/characteristic_test_center=${pulse_center}" \
        "z4c/extrap_order=${extrap_order}" \
        "z4c/boundary_rhs=${boundary_rhs}" \
        "z4c/characteristic_bc_source=${characteristic_bc_source}" \
        >"${run_dir}/stdout.log" 2>&1
      if [[ ${boundary_rhs} == characteristic_cpbc &&
            ${check_boundary_diagnostic} == 1 ]]; then
        python3 "${checker}" "${run_dir}/stdout.log" \
          "$(sector_for_family "${family}")" \
          >"${run_dir}/reflection.txt"
        cat "${run_dir}/reflection.txt"
      fi
      initial_binary=$(find "${run_dir}" -name '*.bin' -type f | sort | head -1)
      final_binary=$(find "${run_dir}" -name '*.bin' -type f | sort | tail -1)
      if [[ -z ${initial_binary} || -z ${final_binary} ||
            ${initial_binary} == "${final_binary}" ]]; then
        echo "missing distinct initial/final binary output for ${run_name}" >&2
        exit 1
      fi
      binary_checker_args=(
        "${initial_binary}" "${final_binary}" "${family}" "${axis}" "${side}"
        --maximum-ratio "${maximum_interior_ratio}"
      )
      if [[ -n ${control_root} ]]; then
        control_dir="${control_root}/${run_name}"
        control_final=$(find "${control_dir}" -name '*.bin' -type f |
          sort | tail -1)
        if [[ -z ${control_final} ]]; then
          echo "missing control final output for ${run_name}" >&2
          exit 1
        fi
        binary_checker_args+=(--control-final "${control_final}")
      fi
      python3 "${binary_checker}" "${binary_checker_args[@]}" \
        >"${run_dir}/interior_reflection.txt"
      cat "${run_dir}/interior_reflection.txt"
    done
  done
done
