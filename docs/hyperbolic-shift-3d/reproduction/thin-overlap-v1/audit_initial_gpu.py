#!/usr/bin/env python3
"""Verify the recorded failed restart check without promoting it to an admission."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'tst/hyperbolic_shift_3d/aurora'))
import control,preflight
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=a.root/'campaigns/preflight-thin-core-v1';read=lambda name:json.loads((d/name).read_text())
plan=read('plan.json');binding=read('binding.json');worker=read('worker-status.json');collected=read('collected.json')
assert worker['status']=='failed' and len(worker['cases'])==6
assert control.digest(d/'plan.json')==binding['plan_sha256']==worker['plan_sha256']
assert binding['source_manifest_sha256']==worker['source_manifest_sha256']
assert collected['scheduler']['job_state']=='F' and collected['scheduler']['Exit_status']==1
artifacts={};summaries=[]
for command,case in zip(plan['commands'],worker['cases']):
 assert command['name']==case['name'];expected='failed' if case['name']=='oblique-equivalence' else 'passed';assert case['status']==expected
 for path,metadata in case['artifacts'].items():
  f=a.root/path;assert f.stat().st_size==metadata['bytes'] and control.digest(f)==metadata['sha256'],path;artifacts[path]=metadata
 report=json.loads((a.root/command['report']).read_text());checks=[control.evaluate_check(report,c) for c in command['checks']]
 assert all(c['passed'] for c in checks)==(expected=='passed')
 if command['kind']!='check':
  cp=a.root/(command['args'][1]+'.checkpoint');header=preflight.read_checkpoint(cp)
  values=np.fromfile(cp,dtype='<f8',offset=header['header_bytes']);assert np.isfinite(values).all()
  assert header['time']==report['t'] and header['steps']==report['steps']
  summaries.append(dict(name=case['name'],t=report['t'],steps=report['steps'],seconds=report['seconds'],
   max_state_deviation=report['max_state_deviation'],H_L2=report['H_L2'],M_L2=report['M_L2'],
   max_null=max(report['max_null_surface'],report['max_null_ghost']),floor=report['lapse_floor_node_interventions']))
component=json.loads((d/'output/paired-device.report.json').read_text());assert component['placement_passed'] and len(component['ranks'])==12
for rank in component['ranks']:
 r=rank['result'];parsed=preflight.parse_component((a.root/r['log']).read_text(),'paired_trace_device',r['returncode']);assert parsed['passed'] and parsed['maxima']==r['maxima']
comparison=preflight.compare_checkpoints(d/'output/oblique-full.checkpoint',d/'output/oblique-resume.checkpoint')
recorded=json.loads((d/'output/oblique-equivalence.report.json').read_text())
for key,value in comparison.items():assert recorded[key]==value,key
assert not comparison['bitwise_payload_equal'] and not comparison['restart_equivalence']
def schedule(end,dt,start=0):
 time=start;result=[]
 while time<end-1e-14:
  step=min(dt,end-time);result.append(dict(t=time,dt=step));time+=step
 return result
full=schedule(.005,.00025);split=schedule(.0025,.00025)+schedule(.005,.00025,.0025)
changed=[dict(step=i+1,direct=x,split=y) for i,(x,y) in enumerate(zip(full,split)) if x!=y]
assert len(full)==len(split)==20 and changed
binary_full=schedule(20/4096,1/4096);binary_split=schedule(10/4096,1/4096)+schedule(20/4096,1/4096,10/4096)
assert binary_full==binary_split
result=dict(status='verified_recorded_failure',job_id=collected['job_id'],cases=summaries,component_maxima=component['maxima'],
 comparison=comparison,decimal_schedule_differences=changed,dyadic_schedule_exact=True,verified_artifacts=artifacts,
 conclusion='Decimal-time direct/split runs have different clipped timesteps. This is a candidate explanation, not a proven cause until independent direct/full-sparse and dyadic-restart runs complete. No tolerance or scientific acceptance was changed; no GPU puncture ran in this stopped plan.')
with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps({k:result[k] for k in ['status','job_id','cases','component_maxima','conclusion']},indent=2))
