#!/usr/bin/env python3
"""Independently audit a collected V12 preflight mirror; no scheduler actions."""
import argparse,hashlib,json,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(repo/'tst/hyperbolic_shift_3d/aurora'))
import control,preflight

def verify(root,campaign):
 directory=root/campaign
 read=lambda name:json.loads((directory/name).read_text())
 plan=read('plan.json');binding=read('binding.json');worker=read('worker-status.json');collected=read('collected.json')
 assert control.digest(directory/'plan.json')==binding['plan_sha256']
 assert worker['plan_sha256']==binding['plan_sha256'] and worker['source_manifest_sha256']==binding['source_manifest_sha256']
 assert worker['status']=='passed' and len(worker['cases'])==len(plan['commands'])
 assert collected['scheduler']['job_state'] in ['F','X'] and collected['scheduler']['Exit_status']==0
 artifacts={};reports={};summaries=[]
 for command,case in zip(plan['commands'],worker['cases']):
  assert command['name']==case['name'] and case['status']=='passed'
  for name,metadata in case['artifacts'].items():
   path=root/name
   assert path.stat().st_size==metadata['bytes'] and control.digest(path)==metadata['sha256'],name
   artifacts[name]=metadata
  report=json.loads((root/command['report']).read_text());reports[command['name']]=report
  assert all(control.evaluate_check(report,c)['passed'] for c in command['checks']),command['name']
  if command['kind']!='check':
   header=preflight.read_checkpoint(root/(command['args'][1]+'.checkpoint'))
   assert header['time']==report['t'] and header['steps']==report['steps']
   for key in ['core_evolved_layers','shifted_overlap_donors','trace_kappa','trace_normal_ratio']:
    assert header[key]==report[key]
   summaries.append(dict(name=command['name'],t=report['t'],kind=report['kind'],steps=report['steps'],seconds=report['seconds'],
    cells=report['cells'],H_L2=report['H_L2'],M_L2=report['M_L2'],floor=report['lapse_floor_node_interventions'],
    max_null=max(report['max_null_surface'],report['max_null_ghost']),horizon_found=report.get('horizon_found')))
 device=reports['paired-device'];assert device['placement_passed'] and len(device['ranks'])==12
 for rank in device['ranks']:
  log=root/rank['result']['log']
  parsed=preflight.parse_component(log.read_text(),'paired_trace_device',rank['result']['returncode'])
  assert parsed['passed'] and parsed['maxima']==rank['result']['maxima']
 pairs={}
 for name,left,right in [('oblique','oblique-full','oblique-resume'),('puncture','puncture-full','puncture-sparse')]:
  paths=[root/plan['output_dir']/(c+'.checkpoint') for c in (left,right)]
  pairs[name]=preflight.compare_checkpoints(*paths)
  assert pairs[name]['restart_equivalence'] and pairs[name]['bitwise_payload_equal']
 return dict(status='passed',source_manifest_sha256=binding['source_manifest_sha256'],plan_sha256=binding['plan_sha256'],
  job_id=collected['job_id'],cases=summaries,comparisons=pairs,verified_artifacts=artifacts,
  scope='V12 12-tile preflight through 0.15M only. No 3M, reflection, accuracy-matched cost or convergence qualification.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--campaign',default='campaigns/preflight-thin-v1');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();result=verify(a.root,a.campaign)
 with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps({k:result[k] for k in ['status','job_id','scope']},indent=2))
