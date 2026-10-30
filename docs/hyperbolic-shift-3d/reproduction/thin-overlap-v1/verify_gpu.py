#!/usr/bin/env python3
"""Independently audit a collected V12 preflight mirror; no scheduler actions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(repo/'tst/hyperbolic_shift_3d/aurora'))
import control,preflight

def cross_schema_control(root,output):
 paths=[root/output/(name+'.checkpoint') for name in ['pulse-l3-control','pulse-v11-control']]
 headers=[preflight.read_checkpoint(path) for path in paths]
 thin,legacy=headers
 assert thin['magic']=='HYP3D_MPI_V12' and legacy['magic']=='HYP3D_MPI_V11'
 assert thin['core_evolved_layers']==3 and thin['trace_kappa']==0 and not thin['shifted_overlap_donors']
 with paths[0].open('rb') as first,paths[1].open('rb') as second:
  assert first.read(152)[16:]==second.read(152)[16:]
 for key,value in legacy.items():
  if key not in ['magic','header_bytes']:assert thin[key]==value,key
 bits=[np.fromfile(path,dtype='<u8',offset=header['header_bytes']) for path,header in zip(paths,headers)]
 assert bits[0].shape==bits[1].shape
 for values in bits:assert np.isfinite(values.view('<f8')).all()
 difference=float(np.max(abs(bits[0].view('<f8')-bits[1].view('<f8'))))
 equal=bool(np.array_equal(*bits))
 assert equal,('V12/V11 control payload differs',difference)
 return dict(bitwise_payload_equal=equal,maximum_absolute_error=difference,compared_doubles=len(bits[0]),
  excluded_header_bytes=[h['header_bytes'] for h in headers],checkpoint_sha256=[control.digest(p) for p in paths],
  scope='Only the schema tag and deliberate 32-byte V12 descriptor extension differ; all shared header fields and four payload arrays match.')

def verify(root,campaign,mode='complete-preflight'):
 directory=root/campaign
 read=lambda name:json.loads((directory/name).read_text())
 plan=read('plan.json');binding=read('binding.json');worker=read('worker-status.json');collected=read('collected.json')
 assert control.digest(directory/'plan.json')==binding['plan_sha256']
 assert control.digest(root/binding['source_manifest'])==binding['source_manifest_sha256']
 assert worker['plan_sha256']==binding['plan_sha256'] and worker['source_manifest_sha256']==binding['source_manifest_sha256']
 assert worker['status']=='passed' and len(worker['cases'])==len(plan['commands'])
 assert collected['scheduler']['job_state'] in ['F','X'] and collected['scheduler']['Exit_status']==0
 affinity=worker['affinity']['records']
 assert worker['affinity']['passed'] and len(affinity)==12
 assert {r['rank'] for r in affinity}==set(range(12)) and len({r['host'] for r in affinity})==1
 assert all(r['mask']==f"{r['rank']//2}.{r['rank']%2}" and r['hierarchy']=='COMPOSITE' for r in affinity)
 for name,digest in binding['inputs'].items():assert control.digest(root/name)==digest,name
 if mode=='clock-isolation':
  assert [c['name'] for c in plan['commands']]==['oblique-sparse-direct','direct-communication','oblique-full-resume','resumed-communication','dyadic-full','dyadic-part','dyadic-resume','dyadic-restart']
 if mode=='puncture-pair':
  assert [c['name'] for c in plan['commands']]==['puncture-full','puncture-sparse','puncture-equivalence']
 if mode=='matrix':
  assert [c['name'] for c in plan['commands']]==['minkowski-l1-k0','minkowski-l2-k0','minkowski-l2-k1','minkowski-l3-k0','minkowski-l3-k1','pulse-l1-zero','pulse-l1-paired','pulse-l3-control','pulse-v11-control']
 if mode=='minkowski-stability':
  assert [c['name'] for c in plan['commands']]==['minkowski-l1-paired']
  assert plan['phase']=='stability' and plan['n']==16
 if mode=='profile-observer':
  assert [c['name'] for c in plan['commands']]==['old-full','new-disabled','new-enabled','minkowski-enabled','compiled-observer-invariance','enabled-observer-invariance']
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
   if header['magic']=='HYP3D_MPI_V12':
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
 elif mode=='puncture-pair':assert set(pairs)=={'puncture-equivalence'}
 elif mode=='profile-observer':
  assert set(pairs)=={'compiled-observer-invariance','enabled-observer-invariance'}
  for name in ['old-full','new-disabled']:
   assert not (root/plan['output_dir']/(name+'.profiles.csv')).exists()
  assert not reports['new-disabled']['profiles_enabled']
  assert reports['new-disabled']['profile_samples']==0
  for name in ['new-enabled','minkowski-enabled']:
   report=reports[name]
   assert report['profiles_enabled'] and report['profile_samples']==3 and report['profile_rows_per_sample']==1462
   data=np.atleast_1d(np.genfromtxt(root/plan['output_dir']/(name+'.profiles.csv'),delimiter=',',names=True))
   assert len(data)==3*1462 and np.isfinite(data.view(float)).all()
   np.testing.assert_array_equal(np.unique(data['t']),[0,report['t']/2,report['t']])
   geometry=None
   for time in np.unique(data['t']):
    frame=data[data['t']==time]
    assert len(frame)==1462 and len(set(zip(frame['domain'],frame['ray'],frame['r'])))==1462
    current=np.column_stack([frame[k] for k in ['domain','ray','r','x','y','z']])
    if geometry is not None:np.testing.assert_array_equal(current,geometry)
    geometry=current
   if name=='minkowski-enabled':
    assert np.max(abs(data['v0']-1))<1e-11
    assert max(np.max(abs(data[f'v{f}'])) for f in range(1,31))<1e-11
    assert max(np.max(abs(data[f'dr_v{f}'])) for f in range(31))<1e-9
 elif mode=='matrix':
  assert not pairs
  pairs['v12-v11-control']=cross_schema_control(root,plan['output_dir'])
 elif mode=='minkowski-stability':
  assert not pairs
  report=reports['minkowski-l1-paired']
  assert report['t']==1 and report['dt']==1/2048 and report['steps']==2048
  history_path=root/plan['output_dir']/'minkowski-l1-paired.history.csv'
  history=np.atleast_1d(np.genfromtxt(history_path,delimiter=',',names=True))
  np.testing.assert_array_equal(history['t'],np.arange(33)/32)
  np.testing.assert_array_equal(history['steps'],np.arange(33)*64)
  maxima={}
  for key,bound in [('state_deviation',1e-10),('H_L2',1e-10),('M_L2',1e-10),
                    ('null_surface',1e-12),('null_ghost',1e-12),
                    ('determinant_residual',1e-12),('tracefree_residual',1e-12)]:
   assert np.isfinite(history[key]).all() and np.all(abs(history[key])<=bound),key
   maxima[key]=float(np.max(abs(history[key])))
  assert np.all(history['min_chi']>0) and np.all(history['min_lapse']>0)
  assert report['lapse_floor_node_interventions']==0
  pairs['equilibrium-history']=dict(samples=len(history),maxima=maxima,
   scope='All saved equilibrium samples; not a perturbation or puncture stability test.')
 scopes={'complete-preflight':'V12 12-tile preflight through 0.15M only. No 3M, reflection, accuracy-matched cost or convergence qualification.',
         'clock-isolation':'V12 twelve-tile clock/communication isolation only; this batch contains no puncture or device-parity admission.',
         'puncture-pair':'V12 twelve-tile short puncture full/sparse equality through 0.15M only. This batch does not itself qualify 3M stability or full preflight admission.',
         'matrix':'Five remaining short Minkowski configurations and four short spherical pulses; exact V12/V11 matched control. No long interface crossing, 3M stability, reflection or convergence qualification.',
         'profile-observer':'Short 12-tile GPU observer invariance and finite cached time-series/Minkowski profiles. Offline sample reconstruction is a separate audit. No long stability or reflection qualification.',
         'minkowski-stability':'Fixed n16 L1 paired Minkowski preservation to coordinate time 1 only. Does not establish perturbed, puncture, AMR or convergence behavior.'}
 return dict(status='passed',source_manifest_sha256=binding['source_manifest_sha256'],plan_sha256=binding['plan_sha256'],
  job_id=collected['job_id'],cases=summaries,comparisons=pairs,verified_artifacts=artifacts,
  scope=scopes[mode])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--campaign',default='campaigns/preflight-thin-v1');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--mode',choices=['complete-preflight','clock-isolation','puncture-pair','matrix','minkowski-stability','profile-observer'],default='complete-preflight')
 a=p.parse_args();result=verify(a.root,a.campaign,a.mode)
 with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps({k:result[k] for k in ['status','job_id','scope']},indent=2))
