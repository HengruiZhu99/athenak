#!/bin/bash
# Bounded single-hole validation; no submissions, chaining or binary evolution.
set -euo pipefail
root=/lus/flare/projects/CompactBinaryMerger/hzhu/hyperboloidal-binary-20M-20261109
package=$root/single-hole-performance
case_dir=$root/runs/single-hole-performance-${PBS_JOBID%%.*}
checkpoint=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side/runs/native-n12-common-07/rst/native_n12_5M.00007.rst
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} && ! -e "$case_dir" ]]
[[ $(sort -u "$PBS_NODEFILE" | wc -l) == 2 ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
printf '%s\n' "$PBS_JOBID" > "$case_dir/job-id.txt"
module list > "$case_dir/modules.txt" 2>&1
sha256sum -c "$package/executables.sha256" > "$case_dir/executable-check-before.txt"
printf '%s  %s\n' dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec "$checkpoint" | sha256sum -c - > "$case_dir/checkpoint-check-before.txt"
cp "$package"/*.json "$case_dir/"
cp "$package"/*.sh "$package"/*.pbs "$case_dir/"
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
source "$package/mpi-gpu-cache.sh"
unset ZE_AFFINITY_MASK
export NATIVE_TIMING_WARMUP=10 NATIVE_TIMING_STEPS=20 NATIVE_TIMING_WINDOWS=3
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" | sha256sum -c - > "$case_dir/wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s rank=%s tile=%s implicit=%s cache=%s size=%s entries=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling" "$MPIR_CVAR_CH4_IPC_GPU_HANDLE_CACHE" "$MPIR_CVAR_CH4_IPC_GPU_CACHE_SIZE" "$MPIR_CVAR_CH4_IPC_GPU_MAX_CACHE_ENTRIES"' > "$case_dir/rank-environment.txt"
run_case() {
 local name=$1 policy=$2 kind=$3 out=$case_dir/$1
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 local executable=$root/timing-current/athena-cycle-timing
 [[ "$kind" != normal ]] || executable=$root/build-sycl/src/athena
 local args=()
 if [[ "$policy" == saved ]]; then
  args=(-r "$checkpoint" time/nlim=2556)
 else
  /usr/bin/python3.11 - "$package/n12.athinput" "$out/input.athinput" "$policy" <<'PY'
import pathlib,sys
source,output,policy=sys.argv[1:]
text=pathlib.Path(source).read_text()
anchor='<z4c_hyperboloidal>\n'
assert text.count(anchor)==1 and policy in ('origin_r2','regular_c6')
settings='residual_weight = '+policy+'\n'
if policy=='regular_c6':settings+='weight_inner = .30\nweight_outer = .55\n'
pathlib.Path(output).write_text(text.replace(anchor,anchor+settings))
PY
  args=(-i "$out/input.athinput" time/nlim=70)
 fi
 local command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$executable" "${args[@]}" -d "$out" time/tlim=2 time/ndiag=1000000000)
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 # Capture only relevant configuration; never dump an unrestricted environment.
 env | sort | sed -n '/^MPIR_/p;/^MPICH_CH4_IPC_GPU_/p;/^MPIR_PARAM_CH4_IPC_GPU_/p;/^OMP_NUM_THREADS=/p;/^ZE_/p;/^ONEAPI_DEVICE_SELECTOR=/p;/^NATIVE_TIMING_/p;/^NEO_CACHE_DIR=/p;/^SYCL_CACHE_DIR=/p' > "$out/environment.txt"
 set +e
 timeout --signal=TERM --kill-after=15s 180s "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || return "$status"
}
# Historical saved state establishes direct timing comparability. Fresh paired
# cases exercise the changed chi treatment without rewriting restart identity.
run_case saved-a saved timed
run_case origin-a origin_r2 timed
run_case regular-a regular_c6 timed
run_case regular-b regular_c6 timed
run_case origin-b origin_r2 timed
run_case saved-b saved timed
run_case origin-normal origin_r2 normal
run_case regular-normal regular_c6 normal
sha256sum -c "$package/executables.sha256" > "$case_dir/executable-check-after.txt"
printf '%s  %s\n' dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec "$checkpoint" | sha256sum -c - > "$case_dir/checkpoint-check-after.txt"
printf 'All eight bounded single-hole cases completed. External analysis pending.\n' > "$case_dir/completed.txt"
