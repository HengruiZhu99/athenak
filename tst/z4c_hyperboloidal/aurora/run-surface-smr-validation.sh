#!/bin/bash -l
# One unchanged manufactured-operator invocation; never submit an allocation.
set -euo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
nodes=${1:?Pass 2 or 8 nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/surface-smr-gpu-v1
case_dir=$task_root/runs/surface-smr-gpu-nodes${nodes}
audit=$validation/surface_smr_run_audit.py
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir/cache/neo" "$case_dir/cache/sycl"
printf '%s\n' "$PBS_JOBID" > "$case_dir/job-id.txt"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
/usr/bin/python3.11 "$audit" guard > "$case_dir/guard-before.json"
# Check the successful two-node allocation before any 96-rank MPI invocation.
if [[ "$nodes" == 8 ]]; then
 /usr/bin/python3.11 "$audit" prior > "$case_dir/prior-two.json"
fi
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
unset ZE_AFFINITY_MASK
export NEO_CACHE_DIR="$case_dir/cache/neo" SYCL_CACHE_DIR="$case_dir/cache/sycl"
/usr/bin/python3.11 "$audit" environment > "$case_dir/environment.json"
wrapper=/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 "$audit" binding --nodes "$nodes"
command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" \
 "$validation/build/surface_smr" "$validation/package/surface_smr.athinput")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"
printf '\n' >> "$case_dir/command.txt"
printf '%s\n' "$case_dir" > "$case_dir/cwd.txt"
set +e
(cd "$case_dir"; timeout --kill-after=15 600 "${command[@]}") > "$case_dir/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$case_dir/exit-code.txt"
# Preserve post-run integrity evidence even when the numerical test fails.
/usr/bin/python3.11 "$audit" guard > "$case_dir/guard-after.json"
[[ "$status" == 0 ]] || exit "$status"
/usr/bin/python3.11 "$audit" audit --nodes "$nodes"
printf 'Surface SMR operator gates passed; independent report review pending\n' > "$case_dir/pipeline-complete.txt"
