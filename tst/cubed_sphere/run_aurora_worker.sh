#!/bin/bash
set -euo pipefail
build="${1:?}"
source_dir="${2:?}"
result="${3:?}"
if [[ "${CUBED_SPHERE_MATRIX:-0}" == 1 || "${CUBED_SPHERE_MATRIX:-0}" == half ]]; then
  cp "$build/src/athena" "$result/athena_pulse_executable"
  sha256sum "$result/athena_pulse_executable" > "$result/executable.sha256"
  resolutions="24 32 16"
  if [[ "$CUBED_SPHERE_MATRIX" == half ]]; then resolutions=16; fi
  for n in $resolutions; do
    label="pulse${n}_t2"
    cfl=0.15
    if [[ "$n" == 16 ]]; then label="pulse16_half_t2"; cfl=0.075; fi
    directory="$result/$label"
    mkdir -p "$directory"
    cp "$source_dir/inputs/z4c/cubed_sphere.athinput" "$directory/base.athinput"
    "$result/athena_pulse_executable" -i "$directory/base.athinput" -d "$directory" \
      mesh/nx1=$((2*n)) mesh/nx2=$((2*n)) mesh/nx3=$((2*n)) \
      cubed_sphere/nr="$n" cubed_sphere/nangular="$n" \
      problem/initial_data=angular_pulse problem/amplitude=0.2 \
      cubed_sphere/outer_boundary=sommerfeld time/tlim=2 time/cfl_number="$cfl" \
      >"$directory/run.log" 2>&1
    echo "$label completed; assess constraints and curvature in its CSV."
  done
  exit 0
fi
ctest --test-dir "$build" --output-on-failure -V --output-junit "$result/ctest.xml"

# After the bounded regressions, extend the substantial regular gauge pulse to
# t=2 at the input's 32^3 bulk / six 16^3 shell resolution. The enclosing timeout
# bounds this entire worker, including tests, to nine minutes.
mkdir -p "$result/pulse16_t2"
cp "$source_dir/inputs/z4c/cubed_sphere.athinput" "$result/pulse16_t2/base.athinput"
"$build/src/athena" -i "$result/pulse16_t2/base.athinput" -d "$result/pulse16_t2" \
  problem/initial_data=angular_pulse problem/amplitude=0.2 \
  cubed_sphere/outer_boundary=sommerfeld time/tlim=2 \
  >"$result/pulse16_t2/run.log" 2>&1
echo "Regular angular pulse completed at t=2; assess constraints in its CSV."
