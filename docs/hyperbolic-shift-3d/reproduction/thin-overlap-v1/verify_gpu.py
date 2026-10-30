#!/usr/bin/env python3
"""Independently audit a collected V12 preflight mirror; no scheduler actions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(repo/'tst/hyperbolic_shift_3d/aurora'))
import control,preflight

def verify(root,campaign,mode='complete-preflight'):
 directory=root/campaign
 read=lambda name:json.loads((directory/name).read_text())
 plan=read('plan.json');binding=read('binding.json');worker=read('worker-status.json');collected=read('collected.json')
 assert control.digest(directory/'plan.json')==binding['plan_sha256']
 assert worker['plan_sha256']==binding['plan_sha256'] and worker['source_manifest_sha256']==binding['source_manifest_sha256']
 assert worker['status']=='passed' and len(worker['cases'])==len(plan['commands'])
 assert collected['scheduler']['job_state'] in ['F','X'] and collected['scheduler']['Exit_status']==0
 for name,digest in binding['inputs'].items():assert control.digest(root/name)==digest,name
 if mode=='clock-isolation':
  assert [c['name'] for c in plan['commands']]==['oblique-sparse-direct','direct-communication','oblique-full-resume','resumed-communication','dyadic-full','dyadic-part','dyadic-resume','dyadic-restart']
 if mode=='puncture-pair':
  assert [c['name'] for c in plan['commands']]==['puncture-full','puncture-sparse','puncture-equivalence']
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
   checkpoint=root/(command['args'][1]+'.checkpoint');header=preflight.read_checkpoint(checkpoint)
   assert np.isfinite(np.fromfile(checkpoint,dtype='<f8',offset=header['header_bytes'])).all()
   assert report['backend']=='SYCL' and report['mpi_size']==12
   assert header['time']==report['t'] and header['steps']==report['steps']
   for key in ['core_evolved_layers','shifted_overlap_donors','trace_kappa','trace_normal_ratio']:
    assert header[key]==report[key]
   summaries.append(dict(name=command['name'],t=report['t'],kind=report['kind'],steps=report['steps'],seconds=report['seconds'],
    cells=report['cells'],H_L2=report['H_L2'],M_L2=report['M_L2'],floor=report['lapse_floor_node_interventions'],
    max_null=max(report['max_null_surface'],report['max_null_ghost']),horizon_found=report.get('horizon_found')))
 if mode=='complete-preflight':
  device=reports['paired-device'];assert device['placement_passed'] and len(device['ranks'])==12
  for rank in device['ranks']:
   log=root/rank['result']['log']
   parsed=preflight.parse_component(log.read_text(),'paired_trace_device',rank['result']['returncode'])
   assert parsed['passed'] and parsed['maxima']==rank['result']['maxima']
 pairs={}
 for command in plan['commands']:
  if command['kind']!='check' or '--reference' not in command['args']:continue
  args=command['args'];name=command['name']
  paths=[root/args[args.index(flag)+1] for flag in ['--reference','--candidate']]
  pairs[name]=preflight.compare_checkpoints(*paths)
  assert pairs[name]['restart_equivalence'] and pairs[name]['bitwise_payload_equal']
  for key,value in pairs[name].items():assert reports[name][key]==value,(name,key)
 if mode=='complete-preflight':assert {'oblique-equivalence','puncture-equivalence'}<=set(pairs)
 elif mode=='clock-isolation':assert set(pairs)=={'direct-communication','resumed-communication','dyadic-restart'}
 else:assert set(pairs)=={'puncture-equivalence'}
 scopes={'complete-preflight':'V12 12-tile preflight through 0.15M only. No 3M, reflection, accuracy-matched cost or convergence qualification.',
         'clock-isolation':'V12 twelve-tile clock/communication isolation only; this batch contains no puncture or device-parity admission.',
         'puncture-pair':'V12 twelve-tile short puncture full/sparse equality through 0.15M only. This batch does not itself qualify 3M stability or full preflight admission.'}
 return dict(status='passed',source_manifest_sha256=binding['source_manifest_sha256'],plan_sha256=binding['plan_sha256'],
  job_id=collected['job_id'],cases=summaries,comparisons=pairs,verified_artifacts=artifacts,
  scope=scopes[mode])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--campaign',default='campaigns/preflight-thin-v1');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--mode',choices=['complete-preflight','clock-isolation','puncture-pair'],default='complete-preflight')
 a=p.parse_args();result=verify(a.root,a.campaign,a.mode)
 with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps({k:result[k] for k in ['status','job_id','scope']},indent=2))
