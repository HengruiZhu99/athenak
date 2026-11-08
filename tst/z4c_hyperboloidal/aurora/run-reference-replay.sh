#!/bin/bash -l
# Exactly one immutable N12 saved-state diagnostic replay; no evolution.
set -euo pipefail
ulimit -c 0
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
reference=$task_root/reference
case_dir=$task_root/runs/reference-step3551-two
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 2 ]]
[[ ! -e "$case_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
sha256sum -c "$reference/immutable.sha256" > "$case_dir/immutable-before.txt"
export LD_LIBRARY_PATH="$reference${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$reference/fastflow_diagnostics" > "$case_dir/dynamic-libraries.txt"
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export MPIR_CVAR_ENABLE_GPU=1
export NEO_CACHE_DIR="$case_dir/cache/neo" SYCL_CACHE_DIR="$case_dir/cache/sycl"
mkdir -p "$NEO_CACHE_DIR" "$SYCL_CACHE_DIR"
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" \
  | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  "$reference/fastflow_diagnostics" "$reference/central_smr20.in"
  "$reference/snapshot-3551.txt" "$case_dir/output")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"
printf '\n' >> "$case_dir/command.txt"
set +e
timeout 840 "${command[@]}" > "$case_dir/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$case_dir/exit-code.txt"
sha256sum -c "$reference/immutable.sha256" > "$case_dir/immutable-after.txt"
[[ $status == 0 ]] || exit "$status"
/usr/bin/python3.11 - "$case_dir/output/result.json" <<'PY'
import json,pathlib,sys
result=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert result['status']=='complete' and result['snapshots']==1 and result['found']==1
assert result['time_M']==5.0009660239551499
print(json.dumps(result,sort_keys=True))
PY
