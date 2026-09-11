#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
trap 'echo $? > "$root/corrector-261-build.status"' EXIT
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
[ "$(git -C "$root/source" rev-parse --short=8 HEAD)" = 261b4d25 ]
cmake -S "$root/source" -B "$root/build"
cmake --build "$root/build" --target athena_hierarchy_z4c_test athena_hierarchy_rk4_test athena_vertex_temporal_boundary_test -j 4
sha256sum "$root/build/athena_hierarchy_z4c_test" > "$root/corrector-261-executable.sha256"
