#!/bin/bash -l
# Same-allocation native brackets around the frozen five-control legacy suite.
set -euo pipefail
ulimit -c 0
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/perf-two-node-v1/legacy-run-v2
build=$task_root/validation/perf-two-node-v1/legacy
seed=$task_root/validation/perf-two-node-v1/legacy-seed
case_dir=$task_root/runs/legacy-n12-timing-two-v2
audit=$validation/legacy_two_node_timing_audit.py
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 2 && ! -e "$case_dir" ]]
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
unset ZE_AFFINITY_MASK HYPERBOLIC_SPARSE_POISON
unset HYPERBOLIC_PROFILE_SMALL_GROUPS HYPERBOLIC_BULK_STREAM HYPERBOLIC_HOST_SURFACE
unset MPICH_CH4_IPC_GPU_HANDLE_CACHE MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE
unset MPICH_CH4_IPC_GPU_CACHE_SIZE MPIR_PARAM_CH4_IPC_GPU_CACHE_SIZE
unset MPICH_CH4_IPC_GPU_MAX_CACHE_ENTRIES MPIR_PARAM_CH4_IPC_GPU_MAX_CACHE_ENTRIES
export MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE=specialized MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=limited
export MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=128
wrapper=$(command -v gpu_tile_compact.sh)
[[ "$wrapper" == /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh ]]
/usr/bin/python3.11 "$audit" guard before
mkdir -p "$case_dir/native-guards"
/usr/bin/python3.11 "$audit" native-guard before
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
timeout --signal=TERM --kill-after=15s 120s mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$audit" binding "$case_dir/binding.txt"
run_native() {
 local name=$1 out=$case_dir/$1
 local executable=$task_root/validation/perf-two-node-v1/mapped-test-poll-v1/timing-build/athena-cycle-timing
 local checkpoint=$task_root/runs/native-n12-common-07/rst/native_n12_5M.00007.rst
 unset HYPERBOLIC_PROFILE_SMALL_GROUPS HYPERBOLIC_BULK_STREAM HYPERBOLIC_HOST_SURFACE HYPERBOLIC_SPARSE_POISON
 export NATIVE_TIMING_WARMUP=10 NATIVE_TIMING_STEPS=20 NATIVE_TIMING_WINDOWS=3
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 /usr/bin/python3.11 "$audit" native-environment "$name"
 local probe=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  /usr/bin/python3.11 "$validation/native_ipc_cache_audit.py" rank-environment)
 printf '%q ' "${probe[@]}" > "$out/rank-cache-command.txt"; printf '\n' >> "$out/rank-cache-command.txt"
 timeout --signal=TERM --kill-after=15s 120s "${probe[@]}" > "$out/rank-cache.txt"
 /usr/bin/python3.11 "$audit" native-rank-cache "$name"
 local command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$executable"
  -r "$checkpoint" -d "$out" time/nlim=2556 time/tlim=2 time/ndiag=1000000000)
 printf '%q ' "${command[@]}" > "$out/command.txt"; printf '\n' >> "$out/command.txt"
 set +e
 timeout --signal=TERM --kill-after=30s 180s "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 /usr/bin/python3.11 "$audit" native-completed "$name"
 unset NATIVE_TIMING_WARMUP NATIVE_TIMING_STEPS NATIVE_TIMING_WINDOWS
}
run_case() {
 local name=$1 out=$case_dir/$1
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 unset HYPERBOLIC_PROFILE_SMALL_GROUPS HYPERBOLIC_BULK_STREAM HYPERBOLIC_HOST_SURFACE
 case "$name" in v3|v6|v8) export HYPERBOLIC_PROFILE_SMALL_GROUPS=1 HYPERBOLIC_BULK_STREAM=1 ;; esac
 [[ "$name" != v8 ]] || export HYPERBOLIC_HOST_SURFACE=1
 /usr/bin/python3.11 "$audit" environment "$name"
 local command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper")
 if [[ "$name" == immutable ]]; then
  command+=("$task_root/validation/original-eight/evolve-static" "$seed/run.in" "$out/output" "$seed/restart.bin")
 elif [[ "$name" == control ]]; then
  command+=("$build/control-v3/legacy_n12_benchmark" "$seed" "$out/output")
 else
  command+=("$build/$name/legacy_n12_benchmark" "$seed" "$out/output")
 fi
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 timeout --signal=TERM --kill-after=30s 450s "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 [[ "$name" == immutable ]] || cp "$out/exit-code.txt" "$out/output/exit-code.txt"
 /usr/bin/python3.11 "$audit" completed "$name"
}
postprocess_control() {
 local out=$case_dir/control endpoint=$build/endpoint-diagnostics-v1
 local builder=$endpoint/package/legacy_n12_endpoint_diagnostics.py
 mkdir -p "$out/postprocess-cache/neo" "$out/postprocess-cache/sycl"
 export NEO_CACHE_DIR="$out/postprocess-cache/neo" SYCL_CACHE_DIR="$out/postprocess-cache/sycl"
 /usr/bin/python3.11 "$audit" endpoint-environment
 /usr/bin/python3.11 "$builder" inputs --seed "$seed" --accepted "$out/output" \
  --output "$out/endpoint-inputs.json" > "$out/endpoint-inputs-create.txt"
 /usr/bin/python3.11 "$builder" verify-inputs --manifest "$out/endpoint-inputs.json" > "$out/endpoint-inputs-verify-before.txt"
 local command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  "$endpoint/build/legacy_n12_endpoint_diagnostics" "$seed" "$out/output" "$out/diagnostics")
 printf '%q ' "${command[@]}" > "$out/endpoint-command.txt"
 printf '\n' >> "$out/endpoint-command.txt"
 set +e
 timeout --signal=TERM --kill-after=30s 180s "${command[@]}" > "$out/endpoint-run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/endpoint-exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 cp "$out/endpoint-exit-code.txt" "$out/diagnostics/exit-code.txt"
 /usr/bin/python3.11 "$builder" verify-inputs --manifest "$out/endpoint-inputs.json" > "$out/endpoint-inputs-verify-after.txt"
 /usr/bin/python3.11 "$audit" completed-endpoint
}
run_native timed-native-before
run_case immutable
run_case control
postprocess_control
run_case v3
run_case v6
run_case v8
run_native timed-native-after
/usr/bin/python3.11 "$audit" guard after
/usr/bin/python3.11 "$audit" native-guard after
sha256sum -c launchers.sha256 > "$case_dir/launchers-after.txt"
printf 'Two native brackets and five legacy controls complete; full matched science and timing admission pending\n' > "$case_dir/pipeline-complete.txt"
