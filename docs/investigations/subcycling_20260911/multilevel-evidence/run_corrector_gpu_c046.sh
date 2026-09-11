#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
out="$root/gpu_validation_c0464f5c"
mkdir "$out"
[ "$(git -C "$root/source" rev-parse HEAD)" = "$(cat "$root/vc-corrector-cuda-fix.sha")" ]
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
scontrol show job "$SLURM_JOB_ID" > "$out/slurm.txt"
git -C "$root/source" rev-parse HEAD > "$out/source-sha.txt"
sha256sum "$root/build/athena_hierarchy_z4c_test" > "$out/executable.sha256"
for variant in corrector3 three-level coarse-group time-dependent cpbc; do
  args=(--corrector3)
  case "$variant" in
    three-level) args+=(--three-level);;
    coarse-group) args+=(--three-level --coarse-group);;
    time-dependent) args+=(--three-level --time-dependent);;
    cpbc) args+=(--three-level --cpbc);;
  esac
  export ATHENA_HIERARCHY_TEST_DUMP_PREFIX="$out/$variant-values-"
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/build/athena_hierarchy_z4c_test" "${args[@]}" > "$out/$variant.log" 2>&1
done
unset ATHENA_HIERARCHY_TEST_DUMP_PREFIX
for test in athena_hierarchy_rk4_test athena_vertex_temporal_boundary_test; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/build/$test" > "$out/$test.log" 2>&1
done
