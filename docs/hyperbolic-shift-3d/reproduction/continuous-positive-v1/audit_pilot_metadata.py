#!/usr/bin/env python3
"""Qualify only the documented V11 pilot metadata error; never rerun physics.

Preserves the original failed worker/audit. No numerical tolerance is changed.
The corrected counts follow the already fixed V11 mesh and target/dt, not fitting.
"""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--verification',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
v=json.loads(a.verification.read_text());r=v['report'];cp=v['checkpoint']
assert v['verification_passed'] and not v['scientific_checks_passed']
assert r['status']=='completed' and r['checkpoint_admitted'] and r['diagnostics_current']
assert r['checkpoint_schema']=='HYP3D_MPI_V11' and r['n']==16 and r['start_time']==0
assert r['target_time']==r['t']==.15 and r['dt']==.0005
n=r['n'];cells=12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2
t=0.;steps=0
while t<r['target_time']:
    t+=min(r['dt'],r['target_time']-t);steps+=1
assert (cells,steps)==(159200,300)
assert cp['cells']==r['cells']==cells and cp['steps']==r['steps']==r['segment_steps']==steps
failed=[x for x in v['checks'] if not x['passed']]
assert len(failed)==2
expected={('cells',):(189512,cells),('segment_steps',):(2001,steps)}
corrections=[]
for x in failed:
    rule=x['check'];key=tuple(rule['path']);old,new=expected.pop(key)
    assert rule['op']=='eq' and rule['value']==old and x['observed']==new
    corrections.append(dict(path=list(key),original_expected=old,derived_expected=new,observed=x['observed']))
assert not expected
assert r['lapse_floor_node_interventions']==0
result=dict(qualified_pilot=True,original_worker_status=v['worker_status'],
    original_scientific_checks_passed=False,numerical_criteria_unchanged=True,
    original_verification_sha256=hashlib.sha256(a.verification.read_bytes()).hexdigest(),
    checkpoint_sha256=cp['sha256'],source_manifest_sha256=v['source_manifest_sha256'],
    binary_sha256=v['binary_sha256'],plan_sha256=v['plan_sha256'],job_id=v['job_id'],
    corrections=corrections,cell_formula='12*n^3 + 8*(n+3)^3 + 6*(n+3)*(n+6)^2',
    step_rule='Advance from 0 to predeclared .15 in steps min(.0005, remaining).',
    scope='Bounded pilot qualified after explicit correction of contradictory inherited bookkeeping checks. Not a convergence or 3M result.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
