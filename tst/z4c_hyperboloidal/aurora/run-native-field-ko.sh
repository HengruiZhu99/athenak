#!/bin/bash -l
# Same input checkpoint, same whole-cycle timer, fixed C0/K0/K1/C1 ordering.
set -euo pipefail
ulimit -c 0
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
work=$task_root/validation/perf-two-node-v1/field-ko-v1
timing_work=$task_root/validation/perf-two-node-v1/mapped-test-poll-v1
validation=$work/launch
case_dir=$task_root/runs/native-field-ko-abba-v1
audit=$validation/native_field_ko_audit.py
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 2 ]]
[[ ! -e "$case_dir" ]]
cd "$validation"
sha256sum -c launchers.sha256
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
printf '%s\n' "$PBS_JOBID" > "$case_dir/job-id.txt"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export NATIVE_TIMING_WARMUP=10 NATIVE_TIMING_STEPS=20 NATIVE_TIMING_WINDOWS=3
unset MPICH_CH4_IPC_GPU_HANDLE_CACHE MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE
unset MPICH_CH4_IPC_GPU_CACHE_SIZE MPIR_PARAM_CH4_IPC_GPU_CACHE_SIZE
unset MPICH_CH4_IPC_GPU_MAX_CACHE_ENTRIES MPIR_PARAM_CH4_IPC_GPU_MAX_CACHE_ENTRIES
export MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE=specialized
export MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=limited
export MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=128
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
[[ "$wrapper" == /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh ]]
/usr/bin/python3.11 "$audit" guard before
checkpoint=$task_root/runs/native-n12-common-07/rst/native_n12_5M.00007.rst
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
timeout --signal=TERM --kill-after=15s 120s mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$audit" binding "$case_dir/binding.txt"
run_case() {
 local name=$1 executable=$2 out=$case_dir/$1
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 /usr/bin/python3.11 "$audit" environment "$out"
 local probe=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  /usr/bin/python3.11 "$audit" rank-environment)
 printf '%q ' "${probe[@]}" > "$out/rank-cache-command.txt"; printf '\n' >> "$out/rank-cache-command.txt"
 timeout --signal=TERM --kill-after=15s 120s "${probe[@]}" > "$out/rank-cache.txt"
 /usr/bin/python3.11 "$audit" rank-cache "$out/rank-cache.txt"
 local command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$executable"
  -r "$checkpoint" -d "$out" time/nlim=2556 time/tlim=2 time/ndiag=1000000000)
 printf '%q ' "${command[@]}" > "$out/command.txt"; printf '\n' >> "$out/command.txt"
 set +e
 timeout --signal=TERM --kill-after=30s 180s "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 /usr/bin/python3.11 "$audit" completed "$name"
}
run_case timed-C0 "$timing_work/timing-build/athena-cycle-timing"
run_case timed-K0 "$work/build/athena-candidate-timed"
screen=$(/usr/bin/python3.11 "$audit" screen)
if [[ "$screen" == continue ]]; then
 run_case timed-K1 "$work/build/athena-candidate-timed"
 run_case timed-C1 "$timing_work/timing-build/athena-cycle-timing"
else
 [[ "$screen" == reject ]]
 printf 'Prespecified clearly-slower first pair; no improvement claim\n' > "$case_dir/early-rejected.txt"
fi
/usr/bin/python3.11 "$audit" guard after
sha256sum -c launchers.sha256 > "$case_dir/launchers-after.txt"
printf 'Screen complete; full payload and ordinary diagnostics exact; independent timing decision pending\n' > "$case_dir/pipeline-complete.txt"
