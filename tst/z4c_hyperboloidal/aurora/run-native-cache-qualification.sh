#!/bin/bash -l
# Existing solver binaries and existing science readers; all timings untimed qualification.
set -euo pipefail
ulimit -c 0
root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
launch=$root/validation/perf-two-node-v1/ipc-cache-v2/launch
case_dir=$root/runs/native-n12-ipc-cache-abba-v2/qualification
audit=$launch/native_cache_qualification.py
cache_audit=$launch/native_ipc_cache_audit.py
[[ ! -e "$case_dir" ]]
mkdir "$case_dir"
module load oneapi/release/2025.3.1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export NATIVE_TIMING_WARMUP=10 NATIVE_TIMING_STEPS=20 NATIVE_TIMING_WINDOWS=3
unset ZE_AFFINITY_MASK
unset MPICH_CH4_IPC_GPU_HANDLE_CACHE MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE
unset MPICH_CH4_IPC_GPU_CACHE_SIZE MPIR_PARAM_CH4_IPC_GPU_CACHE_SIZE
unset MPICH_CH4_IPC_GPU_MAX_CACHE_ENTRIES MPIR_PARAM_CH4_IPC_GPU_MAX_CACHE_ENTRIES
export MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE=specialized MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=limited
export MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=128
/usr/bin/python3.11 "$audit" guard before
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
wrapper=/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh
run_case() {
 local name=$1 cap=$2 out=$case_dir/$1
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 /usr/bin/python3.11 "$audit" environment "$name"
 local probe=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  /usr/bin/python3.11 "$cache_audit" rank-environment)
 printf '%q ' "${probe[@]}" > "$out/rank-cache-command.txt"; printf '\n' >> "$out/rank-cache-command.txt"
 timeout --signal=TERM --kill-after=15s 120s "${probe[@]}" > "$out/rank-cache.txt"
 /usr/bin/python3.11 "$cache_audit" rank-cache "$out/rank-cache.txt" timed-K0
 local command=()
 mapfile -d '' -t command < <(/usr/bin/python3.11 "$audit" command "$name")
 [[ ${#command[@]} -ge 9 ]]
 printf '%q ' "${command[@]}" > "$out/command.txt"; printf '\n' >> "$out/command.txt"
 set +e
 (cd "$out"; timeout --signal=TERM --kill-after=30s "${cap}s" "${command[@]}") > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 /usr/bin/python3.11 "$audit" completed "$name" > "$out/audit.log" 2>&1
}
run_case observer70 600
run_case poison8 180
run_case amr4 300
run_case prefix2 120
/usr/bin/python3.11 "$launch/observer_run_audit.py" check-prefix "$case_dir/prefix2" > "$case_dir/prefix-check.txt"
run_case resume8 180
/usr/bin/python3.11 "$audit" guard after
/usr/bin/python3.11 "$audit" finish
printf 'Five qualification launches completed; independent scientific review pending\n' > "$case_dir/pipeline-complete.txt"
