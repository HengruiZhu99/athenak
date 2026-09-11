#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
[ "$(cat "$root/build-c204ee04.status")" = 0 ]
[ "$(git -C "$root/source" rev-parse --short=8 HEAD)" = c204ee04 ]
out="$root/gpu_validation_c204ee04"
mkdir "$out"
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
git -C "$root/source" rev-parse HEAD > "$out/source-sha.txt"
sha256sum "$root/build/src/athena" "$root/build/athena_hierarchy_z4c_test" > "$out/executables.sha256"
for variant in adaptive coarse-group global-gauge cpbc rollback; do
  args=(--adaptive)
  case "$variant" in
    coarse-group) args+=(--three-level --coarse-group);;
    global-gauge) args+=(--three-level --global-gauge);;
    cpbc) args+=(--three-level --global-gauge --cpbc);;
    rollback) args+=(--rollback-failure);;
  esac
  export ATHENA_HIERARCHY_TEST_DUMP_PREFIX="$out/$variant-values-"
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/build/athena_hierarchy_z4c_test" "${args[@]}" > "$out/$variant.log" 2>&1
done
unset ATHENA_HIERARCHY_TEST_DUMP_PREFIX
for test in athena_hierarchy_rk4_test athena_vertex_temporal_boundary_test athena_subcycle_schedule_test; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/build/$test" > "$out/$test.log" 2>&1
done
