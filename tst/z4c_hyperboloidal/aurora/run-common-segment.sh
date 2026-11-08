#!/bin/bash -l
# Held until actual observer24/96 and restart gates pass. No automatic chaining.
set -euo pipefail
ulimit -c 0
segment=${1:?Pass common-time segment1..10}
[[ "$segment" =~ ^([1-9]|10)$ ]]
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
scripts=$task_root/validation/full-n12
build=$task_root/build-sycl
observer=$task_root/validation/step-observer
case_dir=$task_root/runs/native-n12-common-$(printf '%02d' "$segment")
reference=$task_root/runs/n12-native-gpu-eight-two/rst/native_n12_5M.00000.rst
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 2 ]]
[[ ! -e "$case_dir" ]]
# Root stages this evidence only after the declared correctness prerequisites.
/usr/bin/python3.11 "$scripts/longrun_admission.py" "$scripts/admission.json" --reference "$reference"
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
cp "$scripts/admission.json" "$case_dir/admission.json"
/usr/bin/python3.11 "$scripts/common_segment_audit.py" verify-runtime "$task_root" > "$case_dir/provenance-before.txt"
/usr/bin/python3.11 "$observer/native_step_observer.py" verify --output "$observer/build" >> "$case_dir/provenance-before.txt"
read -r end_cycle nominal_tlim < <(/usr/bin/python3.11 "$scripts/common_segment_audit.py" limits "$segment")
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export NEO_CACHE_DIR="$case_dir/cache/neo" SYCL_CACHE_DIR="$case_dir/cache/sycl"
mkdir -p "$NEO_CACHE_DIR" "$SYCL_CACHE_DIR"
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" /bin/bash -c \
 'printf "%s local_rank=%s tile=%s implicit_scaling=%s\n" "$(hostname)" "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-unknown}}" "$ZE_AFFINITY_MASK" "$EnableImplicitScaling"' > "$case_dir/binding.txt"
/usr/bin/python3.11 - "$case_dir/binding.txt" <<'PY'
import collections,pathlib,re,sys
hosts=collections.defaultdict(dict)
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
 m=re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line);assert m,line
 host,rank,tile=m.groups();rank=int(rank)
 assert rank not in hosts[host] and tile==f'{rank//2}.{rank%2}'
 hosts[host][rank]=tile
assert len(hosts)==2 and all(set(x)==set(range(12)) for x in hosts.values())
PY
command=(mpiexec -n 24 -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$observer/build/athena-step-observer")
audit_args=()
if [[ "$segment" == 1 ]]; then
  cp "$task_root/source/tst/z4c_hyperboloidal/aurora/n12.athinput" "$case_dir/input.athinput"
  command+=(-i "$case_dir/input.athinput")
else
  predecessor=$task_root/runs/native-n12-common-$(printf '%02d' "$((segment-1))")/rst/native_n12_5M.$(printf '%05d' "$((segment-1))").rst
  # Separate complete-state/reference acceptance is mandatory at each handoff.
  /usr/bin/python3.11 - "$predecessor" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);a=json.loads((p.parent.parent/'continuation-admission.json').read_text())
assert a['passed'] is True and a['checkpoint_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
PY
  command+=(-r "$predecessor")
  audit_args+=(--predecessor "$predecessor")
  sha256sum "$predecessor" > "$case_dir/predecessor.sha256"
fi
# No -t, no input overlay for restart, no output/seed override, full unchanged dt.
command+=(-d "$case_dir" "time/nlim=$end_cycle" "time/tlim=$nominal_tlim")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"; printf '\n' >> "$case_dir/command.txt"
set +e
timeout --kill-after=15 3420 "${command[@]}" > "$case_dir/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$case_dir/exit-code.txt"
/usr/bin/python3.11 "$scripts/common_segment_audit.py" verify-runtime "$task_root" > "$case_dir/provenance-after.txt"
/usr/bin/python3.11 "$observer/native_step_observer.py" verify --output "$observer/build" >> "$case_dir/provenance-after.txt"
[[ "$status" == 0 ]] || exit "$status"
/usr/bin/python3.11 "$scripts/common_segment_audit.py" audit --case "$case_dir" --reference "$reference" \
 --segment "$segment" "${audit_args[@]}" --output "$case_dir/structural-cadence.json" > "$case_dir/audit.log"
