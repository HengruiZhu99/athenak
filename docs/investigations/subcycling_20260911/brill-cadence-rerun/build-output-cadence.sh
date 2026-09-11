#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
out="$root/build-output-cadence"
mkdir "$out"
trap 'echo $? > "$out/status"' EXIT
[ "$(cat "$root/build-live-884f5170/status")" = 0 ]
(cd "$root/source"; sha256sum -c "$root/vc-live-884f5170.files.sha256") > "$out/base-source-check.txt"
git -C "$root/source" apply --check "$root/vc-output-cadence.patch"
git -C "$root/source" apply "$root/vc-output-cadence.patch"
cp "$root/vc-output-cadence.patch" "$out/"
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
cmake --build "$root/build" --target athena -j4
sha256sum "$root/source/src/outputs/outputs.cpp" "$root/build/src/athena" > "$out/hashes.sha256"
