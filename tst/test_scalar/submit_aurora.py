#!/usr/bin/env python3
"""Submit one CPU-admitted scalar allocation, with persistent ambiguity guard."""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
from aurora_resource_policy import validate
ROOT=Path(os.environ.get('SCALAR_ROOT','/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016'))
os.chdir(ROOT)
resources=validate(ROOT)
if (ROOT/'submission.json').exists():raise SystemExit('Existing submission handle; inspect it and never duplicate')
admission=json.loads((ROOT/'cpu-admission.json').read_text())
if not admission.get('passed') or admission.get('stage') not in ['minkowski','schwarzschild','kerr']:raise SystemExit('CPU admission required')
if admission['stage']!='minkowski' and not admission.get('gpu_cases'):raise SystemExit('Exact CPU-admitted black-hole arguments required')
binding=admission.get('delivery_source_sha256')
if not binding:raise SystemExit('A source-bound CPU admission is required for a new allocation')
for name,digest in binding.items():
 if hashlib.sha256((ROOT/'source/src/test_scalar'/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Uploaded scalar operator differs from the CPU-admitted source')
dependencies=admission.get('dependency_source_sha256')
if not dependencies:raise SystemExit('CPU binding of Cartesian/cubed-sphere dependencies is required')
for name,digest in dependencies.items():
 if hashlib.sha256((ROOT/'source'/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Uploaded scalar geometry dependency differs from the CPU admission')
# qselect selects this task's exact name; unrelated held research jobs are untouched.
selected=subprocess.run(['qselect','-u',os.environ['USER'],'-N','frozen-scalar'],capture_output=True,text=True)
(ROOT/'scheduler-before.txt').write_text(selected.stdout+selected.stderr)
if selected.stdout.strip():raise SystemExit('An existing task allocation prevents another submission')
if selected.returncode not in [0,153]:raise SystemExit('Cannot establish scheduler state; submission refused')
lock=ROOT/'.submission-lock'
lock.mkdir() # Unknown prior outcomes leave this guard in place.
with (ROOT/'source-check-before.txt').open('w') as f:
 subprocess.run(['sha256sum','-c','source.sha256'],stdout=f,stderr=subprocess.STDOUT,check=True)
(ROOT/'submission-pending.json').write_text(json.dumps(dict(started=time.time(),stage=admission['stage'],command=['qsub','run_aurora.pbs']),indent=2)+'\n')
result=subprocess.run(['qsub','run_aurora.pbs'],capture_output=True,text=True)
(ROOT/'qsub-output.txt').write_text(result.stdout+result.stderr)
if result.returncode:raise SystemExit('qsub failed; inspect saved output before any retry')
ids=re.findall(r'(?m)^(\d+\.\S+)$',result.stdout.strip())
if len(ids)!=1:raise SystemExit('Ambiguous qsub outcome: preserve lock and inspect scheduler')
record=dict(job_id=ids[0],root=str(ROOT),stage=admission['stage'],**resources,one_task_allocation_at_a_time=True,cpu_admission_sha256=hashlib.sha256((ROOT/'cpu-admission.json').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((ROOT/'source.sha256').read_bytes()).hexdigest(),submitted=time.time())
(ROOT/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
