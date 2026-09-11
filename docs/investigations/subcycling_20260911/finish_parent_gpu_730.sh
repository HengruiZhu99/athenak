#!/bin/bash
set -euo pipefail
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
cd "$root"
exec 9>parent-gpu-validation.lock
flock -n 9 || exit 1
trap 'echo $? > parent-gpu-validation.status' EXIT
pid=$(cat parent-rhs-build.pid)
while kill -0 "$pid" 2>/dev/null; do
 state=$(ps -p "$pid" -o stat= || true)
 [[ "$state" == Z* ]] && break
 sleep 5
done
grep -q '\[100%\] Built target athena' parent-rhs-build.log
[[ $(git -C source rev-parse --short=8 HEAD) == 730f4d99 ]]
git -C source diff --exit-code
sha256sum -c executable.sha256
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
cmake --build build --target athena_parent_bulk_rhs_test athena_vertex_parent_states_test athena_level_rk_update_test athena_rk4_predictor_states_test athena_vertex_temporal_boundary_test athena_classical_rk4_test athena_rk4_dense_boundary_test -j 4 > parent-gpu-unit-build.log 2>&1
cat > run_parent_gpu_730.sh <<'RUN'
#!/bin/bash
set -euo pipefail
module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
cd "$root"
export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1
mkdir -p gpu_validation_730f4d99
scontrol show job "$SLURM_JOB_ID" > gpu_validation_730f4d99/slurm.txt
git -C source rev-parse HEAD > gpu_validation_730f4d99/source-sha.txt
sha256sum build/src/athena > gpu_validation_730f4d99/executable.sha256
for test in athena_parent_bulk_rhs_test athena_vertex_parent_states_test athena_level_rk_update_test athena_rk4_predictor_states_test athena_vertex_temporal_boundary_test athena_classical_rk4_test athena_rk4_dense_boundary_test; do
 srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/build/$test" > "gpu_validation_730f4d99/$test.log" 2>&1
done
export ATHENA_TEST_SUBCYCLE_PARENTS=1 ATHENA_TEST_RK_PREDICTOR=1
/global/common/software/nersc/pe/conda-envs/24.1.0/python-3.11/nersc-python/bin/python3 "$root/test_classical_cartoon_launcher.py" "$root/build/src/athena" "$root/gpu_validation_730f4d99/cartoon" --static-amr --level-batches --launcher 'srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1' > gpu_validation_730f4d99/cartoon.log 2>&1
RUN
salloc --account=m3328_g --qos=shared_interactive --constraint='gpu&hbm80g' --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=00:15:00 --job-name=vc-rk-gpu bash "$root/run_parent_gpu_730.sh" > parent-gpu-allocation.log 2>&1
