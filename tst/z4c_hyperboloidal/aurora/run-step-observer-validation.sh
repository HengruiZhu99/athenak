#!/bin/bash -l
# Bounded read-only observer qualification of the original native main.
set -euo pipefail
ulimit -c 0
nodes=${1:?Pass 2 or 8 nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/step-observer
case_dir=$task_root/runs/step-observer-nodes${nodes}
input=$task_root/source/tst/z4c_hyperboloidal/aurora/n12.athinput
executable=$validation/build/athena-step-observer
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
cp "$input" "$case_dir/input.athinput"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
verify() {
 /usr/bin/python3.11 "$task_root/source/tst/z4c_hyperboloidal/aurora/source_manifest.py" "$task_root/source" "$task_root/build-sycl/source-sha256.json" --check
 /usr/bin/python3.11 "$validation/native_step_observer.py" verify --output "$validation/build"
 printf '%s  %s\n' f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c "$task_root/build-sycl/src/athena" | sha256sum -c -
 printf '%s  %s\n' f2c2e6cd910279f0c558164466da8027f0e5b18c5bbd26bb4e36a853dd669f9d "$executable" | sha256sum -c -
}
verify > "$case_dir/provenance-before.txt"
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$validation/observer_run_audit.py" binding "$case_dir" "$nodes"
run_case() {
 local name=$1; shift
 local out=$case_dir/$name
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 local command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$executable" "$@" -d "$out")
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 timeout --kill-after=15 700 "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
}
run_case fresh8 -i "$input" time/nlim=8
/usr/bin/python3.11 "$validation/observer_run_audit.py" run "$case_dir/fresh8" "$ranks" 0 8
if [[ "$nodes" == 2 ]]; then
 run_case prefix2 -i "$input" time/nlim=2
 /usr/bin/python3.11 "$validation/observer_run_audit.py" run "$case_dir/prefix2" "$ranks" 0 2
else
 prefix=$task_root/runs/step-observer-nodes2/prefix2
 /usr/bin/python3.11 "$validation/observer_run_audit.py" check-prefix "$prefix"
 run_case resume8 -r "$prefix/rst/native_n12_5M.00001.rst" time/nlim=8
 /usr/bin/python3.11 "$validation/observer_run_audit.py" run "$case_dir/resume8" "$ranks" 2 8
fi
verify > "$case_dir/provenance-after.txt"
printf 'Observer qualification operationally complete; scientific comparisons pending\n' > "$case_dir/pipeline-complete.txt"
