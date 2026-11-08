#!/bin/bash -l
# Authorized96-rank correctness bundle after matched24-rank short gate PASS.
# Saved-state admission first; original native eight-step smoke only after PASS.
set -euo pipefail
ulimit -c 0
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/saved-state
case_dir=$task_root/runs/native-saved3551-eight
smoke_dir=$task_root/runs/n12-native-gpu-eight-eight
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == 8 ]]
[[ ! -e "$case_dir" && ! -e "$smoke_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
module list > "$case_dir/modules.txt" 2>&1
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
qstat -u hzhu > "$case_dir/queue-at-start.txt"
verify() {
  /usr/bin/python3.11 - "$task_root" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);build=root/'build-sycl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((build/'runtime-object-sha256.json').read_text())
assert sha(pathlib.Path(manifest['executable']))==manifest['executable_sha256']
for rel,want in manifest['object_sha256'].items():assert sha(build/rel)==want,rel
for rel,want in json.loads((build/'source-sha256.json').read_text()).items():
 assert sha(root/'source'/rel)==want,rel
p=root/'validation/saved-state/build/provenance.json';provenance=json.loads(p.read_text())
for key in ['sources','objects','legacy_dependencies','reference']:
 for path,want in provenance.get(key,{}).items():assert sha(pathlib.Path(path))==want,path
assert sha(p.parent/'native_saved_state_gpu')==provenance['executable_sha256']
print('Verified frozen native executable/sources/objects and independent replay inputs')
PY
}
verify > "$case_dir/provenance-before.txt"
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
command=(mpiexec -n 96 -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  "$validation/build/native_saved_state_gpu"
  "$task_root/source/tst/z4c_hyperboloidal/n12_initialize.athinput"
  "$task_root/reference/step3551")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"
printf '\n' >> "$case_dir/command.txt"
cd "$case_dir"
set +e
timeout 600 "${command[@]}" > run.log 2>&1
replay_status=$?
printf '%s\n' "$replay_status" > replay-exit-code.txt
/usr/bin/python3.11 "$validation/native_saved_state_gpu.py" audit --output "$case_dir" > audit.log 2>&1
audit_status=$?
printf '%s\n' "$audit_status" > audit-exit-code.txt
set -e
verify > "$case_dir/provenance-after-replay.txt"
[[ $replay_status == 0 && $audit_status == 0 ]] || exit 2
# A missing/failed strict replay gate must never enter the evolution branch.
/usr/bin/python3.11 - "$case_dir/audit.json" <<'PY'
import json,pathlib,sys
result=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert result['passed'] and result['strict_replay_passed']
assert len(result['dump_files_sha256'])==96
print('Strict complete saved-state replay passed; starting authorized eight-step smoke')
PY
NATIVE_STEPS=8 NATIVE_CASE=n12-native-gpu-eight-eight timeout 600 \
  bash "$task_root/source/tst/z4c_hyperboloidal/aurora/run-native.sh" 8 \
  > "$case_dir/eight-step-launch.log" 2>&1
verify > "$case_dir/provenance-after-smoke.txt"
/usr/bin/python3.11 - "$case_dir" "$smoke_dir" <<'PY'
import json,pathlib,sys
case,smoke=map(pathlib.Path,sys.argv[1:]);restarts=sorted(smoke.rglob('*.rst'))
assert len(restarts)>=2 and all(p.stat().st_size>10000 for p in restarts), 'initial/final restarts missing'
result={'saved_state_replay_passed':True,'native_eight_step_smoke_completed':True,
        'restarts':[str(p) for p in restarts], 'complete_state_comparison_pending':True,
        'native_runtime_source_unchanged':True}
(case/'pipeline.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
PY

# Same-allocation unchanged original GPU control; native runtime remains frozen.
timeout 480 bash "$validation/run-original-control-eight.sh" \
  > "$case_dir/original-control-launch.log" 2>&1
verify > "$case_dir/provenance-after-original-control.txt"
/usr/bin/python3.11 - "$case_dir/pipeline.json" <<'PYFINISH'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text())
r['original_gpu_eight_step_control_completed']=True
r['matched_and_cross_rank_state_comparison_pending']=True
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
PYFINISH
