#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
trap 'echo $? > "$root/corrector-103d8052.status"' EXIT
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
[ "$(git -C "$root/source" rev-parse HEAD)" = "$(cat "$root/vc-corrector-103d8052.sha")" ]
cmake -S "$root/source" -B "$root/build"
cmake --build "$root/build" --target athena_hierarchy_z4c_test athena_hierarchy_rk4_test athena_vertex_temporal_boundary_test -j 4
sha256sum "$root/build/athena_hierarchy_z4c_test" > "$root/corrector-103d8052-executable.sha256"
echo BUILD_COMPLETE
salloc --account=m3328_g --qos=shared_interactive --constraint='gpu&hbm80g' --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=00:15:00 --job-name=vc-corrector-gpu bash "$root/run_corrector_gpu_103d8052.sh" > "$root/corrector-103d8052-allocation.log" 2>&1
