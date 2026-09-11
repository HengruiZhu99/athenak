#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
[ "$(cat "$root/build-38c1fab4.status")" = 0 ]
[ "$(git -C "$root/source" rev-parse --short=8 HEAD)" = 38c1fab4 ]
[ -z "$(git -C "$root/source" status --porcelain)" ]
# Run only after the existing build has terminated and before its GPU tests.
if ps -p 710534 >/dev/null; then echo 'Existing build still present'; exit 1; fi
out="$root/build-live-884f5170"
mkdir "$out"
trap 'echo $? > "$out/status"' EXIT
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
sha256sum "$root/build/src/athena" > "$out/base-executable.sha256"
git -C "$root/source" apply --check "$root/vc-live-884f5170.patch"
git -C "$root/source" apply "$root/vc-live-884f5170.patch"
(cd "$root/source"; sha256sum -c "$root/vc-live-884f5170.files.sha256") > "$out/source-verification.txt"
cp "$root/vc-live-884f5170.patch" "$root/vc-live-884f5170.files.sha256" "$out/"
git -C "$root/source" rev-parse HEAD > "$out/base-source-sha.txt"
# The executable's embedded base SHA remains38c1fab4; this manifest records
# the exact source delta matching884f5170. No production source is touched.
cmake --build "$root/build" --target athena athena_hierarchy_z4c_test -j4
sha256sum "$root/build/src/athena" "$root/build/athena_hierarchy_z4c_test" > "$out/executables.sha256"
