#!/bin/bash -l
# Seven bounded correctness launches of the separate coherent AMR candidate.
set -euo pipefail
ulimit -c 0
nodes=${1:?Pass 2 or 8 nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/amr-admission-candidate-v1
case_dir=$task_root/runs/amr-candidate-nodes${nodes}
tools=$validation/package/tools
input=$validation/package/inputs/amr_helper.athinput
restart_input=$validation/package/inputs/amr_restart.athinput
ordinary=$validation/coherent/build/src/athena
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
cp "$input" "$restart_input" "$case_dir/"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
verify() {
 /usr/bin/python3.11 "$tools/native_amr_candidate_cmake.py" verify --output "$validation/coherent"
 /usr/bin/python3.11 "$tools/native_amr_gpu.py" verify --output "$validation/lifecycle-build"
 /usr/bin/python3.11 "$tools/native_amr_gpu.py" verify --output "$validation/evolution-build"
 /usr/bin/python3.11 "$task_root/source/tst/z4c_hyperboloidal/aurora/source_manifest.py" "$task_root/source" "$task_root/build-sycl/source-sha256.json" --check
 printf '%s  %s\n' f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c "$task_root/build-sycl/src/athena" \
  442594b847ff0eb56407513da48e96930ec41dbe537a69c202fd802da4ea6a31 "$ordinary" \
  6e9f5ae7cba71f160e0c83012875da69d67a874a1548863f407b9bf44fa63b3b "$validation/lifecycle-build/amr_lifecycle" \
  752357a16cefa1cfc03f89b1337de6dd896542b62f54e81400d11a2bf5a726b7 "$validation/evolution-build/amr_evolution" \
  feb01f57769a884c2ddc87e9f1655a4f8b3e6d99d4a7bf11a82a54da6c8cc05a "$input" \
  a4b3de1310649346ed4206be3a5cb93fe582330c3f7f1e2217cede61505dd621 "$restart_input" | sha256sum -c -
}
verify > "$case_dir/provenance-before.txt"
if [[ "$nodes" == 8 ]]; then
 prefix=$task_root/runs/amr-candidate-nodes2/prefix
 /usr/bin/python3.11 "$validation/amr_run_audit.py" check-prefix "$prefix" > "$case_dir/prefix-admission.txt"
fi
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$validation/amr_run_audit.py" binding "$case_dir" "$nodes"
run_case() {
 local name=$1; shift
 local out=$case_dir/$name
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 local command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$@")
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 (cd "$out"; timeout --kill-after=15 600 "${command[@]}") > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 /usr/bin/python3.11 "$validation/amr_run_audit.py" run "$out" "$ranks" "$name"
}
run_case lifecycle "$validation/lifecycle-build/amr_lifecycle" "$input" --diagnostics
run_case evolution "$validation/evolution-build/amr_evolution" "$input"
if [[ "$nodes" == 2 ]]; then
 run_case prefix "$ordinary" -i "$restart_input" time/nlim=2
else
 run_case continuous "$ordinary" -i "$restart_input"
 run_case resumed "$ordinary" -r "$prefix/rst/amr_restart.00003.rst" time/nlim=4
fi
verify > "$case_dir/provenance-after.txt"
printf 'AMR candidate operational coverage complete; scientific comparisons pending\n' > "$case_dir/pipeline-complete.txt"
