#!/bin/bash -l
# Future qualification only: no submission/chaining, and no use before full3551.
set -euo pipefail
ulimit -c 0
nodes=${1:?Pass 2 or 8 nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/cycle-timing-run
case_dir=$task_root/runs/cycle-timing-nodes${nodes}
audit=$validation/cycle_timing_run_audit.py
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" ]]
cd "$validation"
sha256sum -c launchers.sha256
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
printf '%s\n' "$PBS_JOBID" > "$case_dir/job-id.txt"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export NATIVE_TIMING_WARMUP=10 NATIVE_TIMING_STEPS=40 NATIVE_TIMING_WINDOWS=3
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
[[ "$wrapper" == /opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh ]]
guard() {
 /usr/bin/python3.11 "$audit" guard --root "$task_root" --admission "$validation/admission.json" \
  --wrapper "$wrapper" --verify-log "$case_dir/timing-verify-$1.txt" \
  --output "$case_dir/guard-$1.json"
}
guard before
if [[ "$nodes" == 8 ]]; then
 /usr/bin/python3.11 "$audit" previous-pair --root "$task_root" --output "$case_dir/previous-pair-terminal.txt"
fi
checkpoint=$(/usr/bin/python3.11 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_checkpoint_remote"])' "$case_dir/guard-before.json")
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$audit" binding "$case_dir/binding.txt" "$nodes"
run_case() {
 local name=$1 executable=$2 out=$case_dir/$1
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 /usr/bin/python3.11 "$audit" environment "$out"
 local command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$executable"
  -r "$checkpoint" -d "$out" time/nlim=3681 time/tlim=2 time/ndiag=1000000000)
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 "${command[@]}" > "$out/run.log" 2>&1
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
 [[ -f "$out/rst/native_n12_5M.00011.rst" ]]
 grep -Eq 'cycle=3681([[:space:]]|$)' "$out/run.log"
}
run_case normal "$task_root/build-sycl/src/athena"
run_case timed "$task_root/validation/cycle-timing/build/athena-cycle-timing"
[[ -f "$case_dir/timed/native-cycle-timing.json" ]]
guard after
sha256sum -c launchers.sha256 > "$case_dir/launchers-after.txt"
printf 'Two runs complete; full-state, diagnostic and performance qualification pending\n' > "$case_dir/pipeline-complete.txt"
