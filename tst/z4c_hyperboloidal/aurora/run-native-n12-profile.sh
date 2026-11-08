#!/bin/bash -l
set -euo pipefail
ulimit -c 0
side=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
work=$side/validation/perf-two-node-v1/profile-cache128-v1
case=$side/runs/native-n12-profile-cache128-v1
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 2 ]]
[[ ! -e "$case" ]]
cd "$work"
sha256sum -c launchers.sha256
module load oneapi/release/2025.3.1
mkdir -p "$case"
printf '%s\n' "$PBS_JOBID" > "$case/job-id.txt"
cp "$PBS_NODEFILE" "$case/nodes.txt"
module list > "$case/modules.txt" 2>&1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
unset ZE_AFFINITY_MASK
# Exact accepted native-gpu-ipc-cache.env policy, same in both controls.
unset MPICH_CH4_IPC_GPU_HANDLE_CACHE MPIR_PARAM_CH4_IPC_GPU_HANDLE_CACHE
unset MPICH_CH4_IPC_GPU_CACHE_SIZE MPIR_PARAM_CH4_IPC_GPU_CACHE_SIZE
unset MPICH_CH4_IPC_GPU_MAX_CACHE_ENTRIES MPIR_PARAM_CH4_IPC_GPU_MAX_CACHE_ENTRIES
export MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE=specialized
export MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE=limited
export MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES=128
# Existing full frozen-source/object and exact2486 admission guard; no PTI module.
/usr/bin/python3.11 "$work/native_two_node_timing_audit.py" guard --admission "$work/admission.json" --directory "$case" --suffix before
wrapper=$(command -v gpu_tile_compact.sh)
[[ "$wrapper" == /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh ]]
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
for mode in normal profiled; do
 out=$case/$mode
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /usr/bin/python3.11
  "$work/native_n12_profile_rank.py" --mode "$mode" --directory "$out")
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 timeout --signal=TERM --kill-after=30s 660s "${command[@]}" > "$out/run.log" 2>&1
 status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ $status == 0 ]] || exit "$status"
done
/usr/bin/python3.11 "$work/native_two_node_timing_audit.py" guard --admission "$work/admission.json" --directory "$case" --suffix after
sha256sum -c launchers.sha256 > "$case/launchers-after.txt"
/usr/bin/python3.11 "$work/native_n12_profile_audit.py" --directory "$case" --source "$side/runs/native-n12-common-07/rst/native_n12_5M.00007.rst" --output "$case/qualification.json"
