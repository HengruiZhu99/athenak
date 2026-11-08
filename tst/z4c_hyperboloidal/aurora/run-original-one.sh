#!/bin/bash -l
# Unchanged pinned original N12 GPU executable, fresh one-step control only.
set -euo pipefail
ulimit -c 0
task_root=/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side
control=$task_root/validation/original-eight
nodes=${1:?Pass2or8nodes}
case "$nodes" in 2|8) ;; *) exit 2 ;; esac
ranks=$((12*nodes))
case_dir=$task_root/runs/original-gpu-one-nodes${nodes}
[[ -n ${PBS_JOBID:-} && -r ${PBS_NODEFILE:-} ]]
[[ $(sort -u "$PBS_NODEFILE" | awk 'END {print NR}') == "$nodes" ]]
[[ ! -e "$case_dir" ]]
module load oneapi/release/2025.3.1
mkdir -p "$case_dir"
/usr/bin/python3.11 - "$control" "$case_dir" <<'PYINPUT'
import hashlib,pathlib,re,sys
control,case=map(pathlib.Path,sys.argv[1:])
old=(control/'run-eight.in').read_bytes()
assert hashlib.sha256(old).hexdigest()=='3d0ce81d74e7b470272fc938802d505e9b53ec4ab1e3a011c6f3fad96bb74ac6'
new,n=re.subn(rb'(?m)^nlim=8$',b'nlim=1',old);assert n==1
(case/'run-one.in').write_bytes(new)
cfg=(control/'central_smr_constant.cfg').read_bytes()
assert hashlib.sha256(cfg).hexdigest()=='9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363'
(case/'central_smr_constant.cfg').write_bytes(cfg)
(case/'input-change.txt').write_text('Only nlim=8 -> nlim=1; same original binary/configuration; fresh start.\n')
PYINPUT
module list > "$case_dir/modules.txt" 2>&1
cp "$PBS_NODEFILE" "$case_dir/nodes.txt"
qstat -u hzhu > "$case_dir/queue-at-start.txt"
sha256sum -c "$control/immutable.sha256" > "$case_dir/immutable-before.txt"
export LD_LIBRARY_PATH="$task_root/reference${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$control/evolve-static" > "$case_dir/dynamic-libraries.txt"
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
command=(mpiexec -n "$ranks" -ppn 12 --cpu-bind=list:"$binding" "$wrapper"
  "$control/evolve-static" "$case_dir/run-one.in" "$case_dir/output")
printf '%q ' "${command[@]}" > "$case_dir/command.txt"
printf '\n' >> "$case_dir/command.txt"
set +e
timeout 480 "${command[@]}" > "$case_dir/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$case_dir/exit-code.txt"
sha256sum -c "$control/immutable.sha256" > "$case_dir/immutable-after.txt"
[[ $status == 0 ]] || exit "$status"
/usr/bin/python3.11 - "$case_dir/output" "$ranks" <<'PY'
import csv,json,pathlib,sys
out=pathlib.Path(sys.argv[1]);ranks=int(sys.argv[2]);result=json.loads((out/'result.json').read_text())
assert result['status']=='step_limit' and result['steps']==1 and not result['failure']
plan=json.loads((out/'plan.json').read_text())
assert plan['backend']=='SYCL' and plan['ranks']==ranks
assert plan['bulk_points']==229392 and plan['surface_points']==864 and plan['nlim']==1
rows=list(csv.DictReader((out/'steps.csv').open()))
assert [int(row['step']) for row in rows]==[1]
assert all(float(row['dt'])==.00046944203735617445 for row in rows)
assert float(rows[-1]['t'])==.00046944203735617445
for name,step,time in [('initial',0,0.),('final',1,.00046944203735617445)]:
 target=out/name;clock=json.loads((target/'time.json').read_text())
 assert clock['steps']==step and clock['t']==time
 assert (target/'fields.bin').stat().st_size==229392*31*8
 assert (target/'boundary.bin').stat().st_size==864*31*8
 assert (target/'restart.bin').is_file() and (target/'chi-policy.json').is_file()
assert len(list(out.glob('rank-*.txt')))==ranks
print(json.dumps({'original_gpu_one_step_completed':True,'complete_field_comparison_pending':True,
                  'chi_record_scope':'unchanged historical aggregate chi-policy diagnostics; no per-cell ledger export'}))
PY
