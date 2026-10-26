#!/usr/bin/env python3
"""Prepare one immutable read-only CPU audit; never submit implicitly."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';name='cpu-memory-rhs-audit-v1';target=A/name
if target.exists():raise RuntimeError('Existing audit preparation is immutable')
base=A/'cpu-schwarzschild-kappa2-diagnostic-v1';target.mkdir();shutil.copytree(base/'source',target/'source');scripts=Path(__file__).resolve().parent
for name in ['benchmark_frozen_rhs.cpp','cpu_memory_audit_worker.py','audit_ringdown_rhs.cpp','build_audit_executable.py','stage_rhs_inputs.py','scalar_evidence.py']:
 shutil.copy2(scripts/name,target/'source/tst/test_scalar'/name)
for name in ['cpu-admission.json','preserved-cpu-executable']:shutil.copy2(base/name,target/name)
source={name:sha(Path('src/test_scalar')/name) for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']}
cases=[]
for n in [36,48,60]:
 collection=A/('cpu-schwarzschild-commensurate-v1-collected' if n<60 else 'cpu-schwarzschild-finer-n60-resume1-collected');receipt=json.loads((collection/'collection-receipt.json').read_text());label=f'schwarzschild-n{n}-fixed-cube'+('' if n<60 else '-resume1');local=collection/'result'/label
 if not receipt['passed']:raise RuntimeError('Native collection failed')
 cases.append(dict(n=n,run=receipt['remote_root']+'/results/'+receipt['job']+'/'+label,checkpoint_sha256=sha(local/'checkpoint.f64'),collection_sha256=sha(collection/'collection-receipt.json')))
science=dict(kind='read_only_CPU_audit',declared=time.time(),time_steps=0,physical_source_changes=0,fields=5,reference_unchanged=True,inputs=cases,criteria=dict(full_active_RHS_bitwise_identical=True,all_four_geometry_fingerprints_identical=True,active_scalar_state_unchanged=True),performance_is_scientific_admission=False,reference_assessment='Use independently verified six-point interpolation and2400-point radial solution; retain every earlier failed review.',documentation=['https://docs.alcf.anl.gov/aurora/running-jobs-aurora/','https://docs.alcf.anl.gov/aurora/known-issues/'])
(target/'scientific-plan.json').write_text(json.dumps(science,indent=2)+'\n')
plan=dict(kind='cpu_validation',gpu_evolution=False,threads=102,source_sha256=source,compiled_source_sha256=source,binary_sha256=sha(target/'preserved-cpu-executable'),cases=[],audit_cases=cases,scientific_plan_sha256=sha(target/'scientific-plan.json'),limits=dict(queue='debug',account='CompactBinaryMerger',nodes=1,minutes=60,max_task_allocations=1),declared_before_submission=time.time())
(target/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
pbs='''#!/bin/bash -l
#PBS -N frozen-scalar
#PBS -A CompactBinaryMerger
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
set -euo pipefail
ulimit -c 0
cd "${PBS_O_WORKDIR:?}"
result="$PWD/results/${PBS_JOBID:?}"
mkdir -p "$result"
exec >"$result/job.log" 2>&1
module load oneapi/release/2025.3.1
module list
sha256sum -c source.sha256
build=/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/cpu-stable-endpoint-v1/build-cpu
python=/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/validation-python/bin/python
"$python" source/tst/test_scalar/build_audit_executable.py --build "$build" --source source/tst/test_scalar/audit_ringdown_rhs.cpp --output "$result/audit_ringdown_rhs"
"$python" source/tst/test_scalar/build_audit_executable.py --build "$build" --source source/tst/test_scalar/benchmark_frozen_rhs.cpp --output "$result/benchmark_frozen_rhs"
mpiexec -n 1 --cpu-bind list:1-51,53-103 "$python" source/tst/test_scalar/cpu_memory_audit_worker.py --root "$PWD" --output "$result"
sha256sum -c source.sha256
date -u
'''
(target/'run_aurora.pbs').write_text(pbs)
subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(target),'--output',str(A/'cpu-memory-rhs-audit-v1.tar.gz'),'--extra','cpu-plan.json','scientific-plan.json','preserved-cpu-executable'],check=True)
print(target)
