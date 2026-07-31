#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 ATHENA_EXE RUN_ROOT [RESOLUTION=64] [DIMENSIONS=3]"
  exit 2
fi

athena_exe=$1
run_root=$2
resolution=${3:-64}
dimensions=${4:-3}
side=${CPBC_OBLIQUE_SIDE:-1}
family=${CPBC_OBLIQUE_FAMILY:-tt_cross}
boundary_rhs=${Z4C_BOUNDARY_RHS:-characteristic_cpbc}
characteristic_bc_source=${Z4C_CHARACTERISTIC_BC_SOURCE:-zero_rate}
# An L=1 face-normal condition is not exact for a pulse incident obliquely on
# two or three planar faces.  This suite checks finite evolution and repeated
# edge/corner ownership; normal-incidence reflection is gated separately.
maximum_ratio=${CPBC_MAX_INTERIOR_RATIO:-1.0}
control_root=${Z4C_CONTROL_ROOT:-}
repeat_count=${CPBC_OBLIQUE_REPEATS:-2}
boundary_cells=${CPBC_OBLIQUE_BOUNDARY_CELLS:-4}
half_width=${CPBC_OBLIQUE_HALF_WIDTH:-8.0}
center_magnitude=${CPBC_OBLIQUE_CENTER:-6.0}
pulse_width=${CPBC_OBLIQUE_WIDTH:-1.5}
transverse_width=${CPBC_OBLIQUE_TRANSVERSE_WIDTH:-1.5}
tlim=${CPBC_OBLIQUE_TLIM:-14.0}
source_root=$(cd "$(dirname "$0")/.." && pwd)
input_file="${source_root}/inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput"
checker="${source_root}/analysis/z4c_characteristic/check_oblique_pulse_binary.py"

if [[ ${dimensions} != 2 && ${dimensions} != 3 ]]; then
  echo "DIMENSIONS must be 2 (edge) or 3 (corner)" >&2
  exit 2
fi
if [[ ${side} != -1 && ${side} != 1 ]]; then
  echo "CPBC_OBLIQUE_SIDE must be -1 or 1" >&2
  exit 2
fi
if (( resolution < 16 || resolution % 2 != 0 )); then
  echo "RESOLUTION must be even and at least 16" >&2
  exit 2
fi
if (( boundary_cells < 1 || 2 * boundary_cells >= resolution )); then
  echo "CPBC_OBLIQUE_BOUNDARY_CELLS must leave a nonempty interior" >&2
  exit 2
fi
python3 - "${dimensions}" "${resolution}" "${half_width}" \
  "${center_magnitude}" "${pulse_width}" "${transverse_width}" \
  "${tlim}" <<'PY'
import math
import sys

dimensions, resolution = map(int, sys.argv[1:3])
half_width, center, width, transverse_width, tlim = map(
    float, sys.argv[3:])
values = (half_width, center, width, transverse_width, tlim)
if not all(math.isfinite(value) and value > 0.0 for value in values):
    raise SystemExit("oblique geometry values must be finite and positive")
dx = 2.0 * half_width / resolution
boundary_coordinate = math.sqrt(dimensions) * half_width
travel = boundary_coordinate - center
minimum_time = travel + 2.0 * width + 4.0 * dx
if travel <= 2.0 * width:
    raise SystemExit(
        "oblique pulse is not initially separated from the incident boundary")
if min(width, transverse_width) / dx < 2.5:
    raise SystemExit(
        "oblique pulse has fewer than 2.5 coarse cells per Gaussian width")
if tlim <= minimum_time:
    raise SystemExit(
        "oblique tlim={} is too short for a measured return; require > {}"
        .format(tlim, minimum_time))
print(
    "CPBC oblique geometry: dimensions={} dx={:.8e} "
    "boundary_normal_coordinate={:.8e} one_way_time={:.8e} "
    "minimum_tlim={:.8e}".format(
        dimensions, dx, boundary_coordinate, travel, minimum_time))
PY

block_size=$((resolution / 2))
far_resolution=$((3 * resolution / 2))
center=${center_magnitude}
mkdir -p "${run_root}"

small_mesh=()
control_mesh=()
for axis in 1 2 3; do
  small_mesh+=(
    "mesh/nx${axis}=${resolution}"
    "mesh/x${axis}min=-${half_width}"
    "mesh/x${axis}max=${half_width}"
    "mesh/ix${axis}_bc=outflow"
    "mesh/ox${axis}_bc=outflow"
    "meshblock/nx${axis}=${block_size}"
  )
  if (( axis <= dimensions )); then
    if (( side > 0 )); then
      control_min="-${half_width}"
      control_max=$(python3 -c "print(2.0 * float(${half_width}))")
    else
      control_min=$(python3 -c "print(-2.0 * float(${half_width}))")
      control_max="${half_width}"
    fi
    control_mesh+=(
      "mesh/nx${axis}=${far_resolution}"
      "mesh/x${axis}min=${control_min}"
      "mesh/x${axis}max=${control_max}"
    )
  else
    control_mesh+=(
      "mesh/nx${axis}=${resolution}"
      "mesh/x${axis}min=-${half_width}"
      "mesh/x${axis}max=${half_width}"
    )
  fi
  control_mesh+=(
    "mesh/ix${axis}_bc=outflow"
    "mesh/ox${axis}_bc=outflow"
    "meshblock/nx${axis}=${block_size}"
  )
done

common_args=(
  "time/tlim=${tlim}"
  "output3/dt=${tlim}"
  "problem/characteristic_test_family=${family}"
  "problem/characteristic_test_side=${side}"
  "problem/characteristic_test_center=${center}"
  "problem/characteristic_test_width=${pulse_width}"
  "problem/characteristic_test_oblique=true"
  "problem/characteristic_test_oblique_dimensions=${dimensions}"
  "problem/characteristic_test_transverse_width=${transverse_width}"
  "z4c/extrap_order=4"
)

if [[ -z ${control_root} ]]; then
  control_root="${run_root}/far_reference"
  mkdir -p "${control_root}"
  "${athena_exe}" -i "${input_file}" -d "${control_root}" \
    "${control_mesh[@]}" "${common_args[@]}" \
    z4c/boundary_rhs=characteristic_cpbc \
    "z4c/characteristic_bc_source=${characteristic_bc_source}" \
    z4c/characteristic_bc_diagnostics=false \
    >"${control_root}/stdout.log" 2>&1
elif [[ ! -d ${control_root} ]]; then
  echo "Z4C_CONTROL_ROOT does not exist: ${control_root}" >&2
  exit 2
fi

control_final=$(find "${control_root}" -name '*.bin' -type f | sort | tail -1)
if [[ -z ${control_final} ]]; then
  echo "missing far-control final binary" >&2
  exit 1
fi

for repeat in $(seq 1 "${repeat_count}"); do
  run_dir="${run_root}/repeat${repeat}"
  mkdir -p "${run_dir}"
  "${athena_exe}" -i "${input_file}" -d "${run_dir}" \
    "${small_mesh[@]}" "${common_args[@]}" \
    "z4c/boundary_rhs=${boundary_rhs}" \
    "z4c/characteristic_bc_source=${characteristic_bc_source}" \
    >"${run_dir}/stdout.log" 2>&1
  initial=$(find "${run_dir}" -name '*.bin' -type f | sort | head -1)
  final=$(find "${run_dir}" -name '*.bin' -type f | sort | tail -1)
  if [[ -z ${initial} || -z ${final} || ${initial} == "${final}" ]]; then
    echo "missing distinct initial/final binary output" >&2
    exit 1
  fi
  python3 "${checker}" \
    "${initial}" "${final}" "${family}" "${dimensions}" "${side}" \
    --control-final "${control_final}" \
    --boundary-cells "${boundary_cells}" \
    --maximum-ratio "${maximum_ratio}" \
    >"${run_dir}/interior_reflection.txt"
  cat "${run_dir}/interior_reflection.txt"
done

if (( repeat_count >= 2 )); then
  first=$(find "${run_root}/repeat1" -name '*.bin' -type f | sort | tail -1)
  second=$(find "${run_root}/repeat2" -name '*.bin' -type f | sort | tail -1)
  cmp "${first}" "${second}"
  echo "CPBC oblique repeat checksum:"
  cksum "${first}" "${second}"
fi
