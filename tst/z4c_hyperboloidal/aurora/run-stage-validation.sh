#!/bin/bash -l
# Test-only observers and immutable controls; no production/runtime modification.
set -euo pipefail
ulimit -c 0
nodes=${1:?Pass2or8nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
validation=$task_root/validation/stage-trace
launcher=$task_root/validation/stage-trace-v2
case_dir=$task_root/runs/stage-validation-v2-nodes${nodes}
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" && ! -e "$task_root/runs/original-gpu-one-nodes${nodes}" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
module list > "$case_dir/modules.txt" 2>&1
qstat -u hzhu > "$case_dir/queue-at-start.txt"
verify() {
 /usr/bin/python3.11 "$task_root/source/tst/z4c_hyperboloidal/aurora/source_manifest.py" "$task_root/source" "$task_root/build-sycl/source-sha256.json" --check
 /usr/bin/python3.11 "$validation/native_stage_trace_gpu.py" verify --output "$validation/build"
 /usr/bin/python3.11 "$task_root/validation/legacy-stage-trace/legacy_stage_trace_gpu.py" verify --output "$task_root/validation/legacy-stage-trace/build"
 /usr/bin/python3.11 "$task_root/validation/geometry-map-v2/n12_checkpoint_map_portable.py" verify --output "$task_root/validation/geometry-map-v2/build"
 sha256sum -c "$task_root/validation/original-eight/immutable.sha256"
}
verify > "$case_dir/provenance-before.txt"
/usr/bin/python3.11 - "$task_root" "$case_dir" <<'PY'
import pathlib,sys,hashlib,json
root,case=map(pathlib.Path,sys.argv[1:]);p=root/'source/tst/z4c_hyperboloidal/n12_initialize.athinput'
s=p.read_text();assert s.count('evolution = static')==1
s=s.replace('evolution = static','evolution = dynamic\nintegrator = rk4_classical\ntlim = 1\nnlim = 8\nndiag = 1')
(case/'native-eight.athinput').write_text(s)
(case/'input-origin.json').write_text(json.dumps({'origin':str(p),'origin_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changes':'evolution=dynamic; integrator=rk4_classical;tlim1;nlim8;ndiag1; no physics/mesh changes','generated_sha256':hashlib.sha256(s.encode()).hexdigest()},indent=2)+'\n')
PY
export OMP_NUM_THREADS=1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu MPIR_CVAR_ENABLE_GPU=1
export LD_LIBRARY_PATH="$task_root/reference${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ZE_AFFINITY_MASK
wrapper=$(command -v gpu_tile_compact.sh)
printf '%s  %s\n' 4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd "$wrapper" | sha256sum -c - > "$case_dir/gpu-wrapper-check.txt"
binding='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
run_case() {
 local name=$1 limit=$2 count=$3;shift 3
 local out=$case_dir/$name
 mkdir -p "$out/cache/neo" "$out/cache/sycl"
 export NEO_CACHE_DIR="$out/cache/neo" SYCL_CACHE_DIR="$out/cache/sycl"
 local command=(mpiexec -n "$count" -ppn 12 --cpu-bind=list:"$binding" "$wrapper" "$@")
 printf '%q ' "${command[@]}" > "$out/command.txt"
 printf '\n' >> "$out/command.txt"
 set +e
 (cd "$out"; timeout --kill-after=15 "$limit" "${command[@]}" > run.log 2>&1)
 local status=$?
 set -e
 printf '%s\n' "$status" > "$out/exit-code.txt"
 [[ "$status" == 0 ]] || exit "$status"
}
# Metadata-only independent native/legacy bulk+scri coordinate proof, one rank.
mkdir "$case_dir/geometry-map"
# run_case owns output directory; mkdir-p inside is safe only for this precreated mapper path.
run_case geometry-map 120 1 "$task_root/validation/geometry-map-v2/build/n12_checkpoint_map" \
 "$task_root/source/tst/z4c_hyperboloidal/n12_initialize.athinput" "$case_dir/geometry-map"
run_case native-clean 480 "$ranks" "$validation/build/native_stage_trace_gpu" "$case_dir/native-eight.athinput" clean
run_case native-poison 480 "$ranks" "$validation/build/native_stage_trace_gpu" "$case_dir/native-eight.athinput" poison
/usr/bin/python3.11 "$validation/native_stage_trace_compare.py" --output "$case_dir/native-clean" > "$case_dir/native-clean/audit.log" 2>&1
/usr/bin/python3.11 "$validation/native_stage_trace_compare.py" --output "$case_dir/native-poison" --compare "$case_dir/native-clean" > "$case_dir/native-poison/audit.log" 2>&1
run_case legacy-eight 300 "$ranks" "$task_root/validation/legacy-stage-trace/build/legacy_stage_trace_gpu" \
 "$task_root/validation/original-eight/run-eight.in"
bash "$launcher/run-original-one.sh" "$nodes" > "$case_dir/original-one-launch.log" 2>&1
verify > "$case_dir/provenance-after.txt"
/usr/bin/python3.11 - "$case_dir" "$ranks" <<'PY'
import json,pathlib,sys,struct
root=pathlib.Path(sys.argv[1]);ranks=int(sys.argv[2])
for name in ['native-clean','native-poison','legacy-eight']:
 r=json.loads((root/name/'results.json').read_text())
 assert r['passed'] and r['backend']=='SYCL' and r['ranks']==ranks and r['steps']==8
 assert r['bulk_points']==229392 and r['scri_points']==864 and r['dt']==.00046944203735617445 and r['time']==.0037555362988493960
for name in ['native-clean','native-poison']:
 a=json.loads((root/name/'trace-audit.json').read_text());assert a['passed'] and len(a['labels'])==26
assert json.loads((root/'native-poison/trace-audit.json').read_text())['stage_records_exact']
for region,count in [('bulk',229392),('scri',864)]:
 for label,nv in [('geometry',3),('accepted0',39),('accepted1',39),('accepted8',39),('first-rhs',31),('final-snapshot',39)]:
  files=list((root/'legacy-eight').glob(f'{label}-{region}-rank*.bin'));assert len(files)==ranks
  total=0
  for p in files:
   with p.open('rb') as f:
    assert f.read(16).rstrip(b'\0')==b'LEGACY_TRACE_V1'
    n,nk,nval,reg,version=struct.unpack('<QIIII',f.read(24));assert nk==6 and nval==nv and reg==(region=='scri') and version==1
    assert p.stat().st_size==40+n*(24+8*nv);total+=n
  assert total==count
for name in ['mapping.bin','topology.bin','atlas.txt']:
 assert (root/'geometry-map'/name).is_file()
(root/'pipeline.json').write_text(json.dumps({'observer_bundle_completed':True,'ranks':ranks,'native_clean_poison_bytewise_passed':True,'legacy_first_stage_and_accepted_states_exported':True,'immutable_original_one_step_completed':True,'matched_scientific_comparisons':'pending independent local audit','runtime_sources_unchanged':True},indent=2)+'\n')
PY
