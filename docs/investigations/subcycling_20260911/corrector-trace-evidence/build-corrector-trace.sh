#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
out="$root/build-corrector-trace"
mkdir "$out"
trap 'echo $? > "$out/status"' EXIT
[ "$(cat "$root/build-output-cadence/status")" = 0 ]
sha256sum -c "$root/build-output-cadence/hashes.sha256" > "$out/base-check.txt"
git -C "$root/source" apply --check "$root/vc-corrector-trace.patch"
git -C "$root/source" apply "$root/vc-corrector-trace.patch"
cp "$root/vc-corrector-trace.patch" "$out/"
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
cmake --build "$root/build" --target athena athena_hierarchy_z4c_test -j4
sha256sum "$root/source/src/driver/hierarchy_rk4.hpp" "$root/build/src/athena" > "$out/hashes.sha256"
