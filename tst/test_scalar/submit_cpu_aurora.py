#!/usr/bin/env python3
"""Submit one bounded CPU-only scalar validation allocation; guard ambiguity."""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
from aurora_resource_policy import validate
root=Path(os.environ['SCALAR_ROOT']);os.chdir(root)
resources=validate(root)
if (root/'submission.json').exists():raise SystemExit('Existing submission: inspect it; never duplicate')
plan=json.loads((root/'cpu-plan.json').read_text());admission=json.loads((root/'cpu-admission.json').read_text())
previous=plan.get('previous_stage','minkowski')
if plan['kind']!='cpu_validation' or plan['gpu_evolution'] or not admission['passed'] or previous not in ['minkowski','schwarzschild'] or admission['stage']!=previous:raise SystemExit('Declared CPU-only validation and matching previous-stage admission required')
if admission['delivery_source_sha256']!=plan['source_sha256']:raise SystemExit('CPU validation operator differs from previous-stage admission')
for name,digest in plan['source_sha256'].items():
 if hashlib.sha256((root/'source/src/test_scalar'/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Source differs from declared CPU plan')
for name,digest in admission['dependency_source_sha256'].items():
 if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Cartesian/cubed-sphere dependency differs from previous CPU admission')
if plan.get('compiled_source_sha256')!=plan['source_sha256'] or hashlib.sha256((root/'preserved-cpu-executable').read_bytes()).hexdigest()!=plan['binary_sha256']:raise SystemExit('Preserved CPU executable/source binding mismatch')
selected=subprocess.run(['qselect','-u',os.environ['USER'],'-N','frozen-scalar'],capture_output=True,text=True)
(root/'scheduler-before.txt').write_text(selected.stdout+selected.stderr)
if selected.stdout.strip():raise SystemExit('Existing task allocation blocks submission')
if selected.returncode not in [0,153]:raise SystemExit('Cannot establish scheduler state')
(root/'.submission-lock').mkdir()
with (root/'source-check-before.txt').open('w') as log:subprocess.run(['sha256sum','-c','source.sha256'],stdout=log,stderr=subprocess.STDOUT,check=True)
(root/'submission-pending.json').write_text(json.dumps(dict(started=time.time(),command=['qsub','run_aurora.pbs'],plan_sha256=hashlib.sha256((root/'cpu-plan.json').read_bytes()).hexdigest()),indent=2)+'\n')
r=subprocess.run(['qsub','run_aurora.pbs'],capture_output=True,text=True);(root/'qsub-output.txt').write_text(r.stdout+r.stderr)
if r.returncode:raise SystemExit('Submission failed; preserve and inspect the outcome')
ids=re.findall(r'(?m)^(\d+\.\S+)$',r.stdout.strip())
if len(ids)!=1:raise SystemExit('Ambiguous submission outcome: preserve lock and inspect scheduler')
record=dict(job_id=ids[0],root=str(root),kind='cpu_validation',gpu_evolution=False,**resources,one_task_allocation_at_a_time=True,plan_sha256=hashlib.sha256((root/'cpu-plan.json').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((root/'source.sha256').read_bytes()).hexdigest(),submitted=time.time())
(root/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
