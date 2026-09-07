#!/usr/bin/env python3
"""Uniform 50-field repeated halo/RK/restart tests across block/rank decompositions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
from run_legacy_equivalence import input_text
from intrinsic_restart import read_restart,global_state,ghost_error
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--reference-binary',type=Path,required=True)
p.add_argument('--launcher',default='')
p.add_argument('--ranks',type=int,default=1)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seeded',action='store_true',help='Synthetic intrinsic restart with an independent oblique Fourier mode per field')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
tol=2e-12;records=[];runs=[]

def input_for(order,dim,mb):
 t=input_text(order,dim,'collision_factorized',1).replace('nlim = 1','nlim = 3').replace('tlim = 0.0001','tlim = 0.0003')
 start=t.index('<pc_gh>');end=t.index('<problem>')
 t=t[:start]+f'''<pc_gh>
formulation = intrinsic_clean
spatial_order = {order}
shift_eta = 2
kappa = 1
reduction_rate = 1
reduction_profile = lapse_scaled
dissipation = 0.3
research_dt_ceiling = 0.0001
'''+t[end:].replace('legacy_equivalence','intrinsic_smooth')
 start=t.index('<meshblock>');end=t.index('<time>')
 t=t[:start]+f'<meshblock>\nnx1 = {mb}\nnx2 = {mb}\nnx3 = {mb if dim==3 else 1}\n'+t[end:]
 return t+'\n<output1>\nfile_type = rst\ndt = 0.0001\n'

def run(name,text,binary,launcher,ranks,restart=None):
 d=a.output/name;d.mkdir();f=d/'used.athinput';f.write_text(text)
 command=shlex.split(launcher)+[str(binary.resolve()),'-i',str(f.resolve())]
 if restart is not None:command+=['-r',str(restart.resolve())]
 start=time.time()
 with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 log=(d/'run.log').read_text();reported=re.findall(r'Number of parallel ranks = (\d+)',log)
 ok=r.returncode==0 and reported==[str(ranks)]
 runs.append(dict(name=name,command=command,returncode=r.returncode,expected_ranks=ranks,reported_ranks=reported,wall_seconds=time.time()-start,status='PASS' if ok else 'FAIL',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
 (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
 assert ok,(name,log)
 assert len(list(d.glob('intrinsic-health-rank*.csv')))==ranks
 return sorted((d/'rst').glob('*.rst'))

def seed_restart(source,destination):
 data=read_restart(source);state=data['state'];ng,bx,by,bz=data['mb'][:4]
 nx,ny,nz=data['mesh'][1:4]
 kk,jj,ii=np.indices((nz,ny,nx));global_values=np.empty((50,nz,ny,nx))
 for n in range(50):
  phase=2*np.pi*((n%3+1)*(ii+.5)/nx
   +((n//3)%3+1)*(jj+.5)/ny+(((n//9)%3+1)*(kk+.5)/nz if nz>1 else 0))+.17*n
  global_values[n]=(1. if n<2 else 0.)+.001*np.sin(phase)/(1+.03*n)
 for m,(x,y,z,_) in enumerate(data['locations']):
  iz=(z*bz+np.arange(state.shape[2])-(ng if bz>1 else 0))%nz
  iy=(y*by+np.arange(state.shape[3])-(ng if by>1 else 0))%ny
  ix=(x*bx+np.arange(state.shape[4])-ng)%nx
  state[m]=global_values[:,iz[:,None,None],iy[None,:,None],ix[None,None,:]]
 raw=source.read_bytes();prefix=raw[:-state.nbytes]
 prefix=prefix.replace(b'<par_end>\n',b'<problem>\nrestart_fixture = asymmetric_50_field_v1\n<par_end>\n')
 destination.write_bytes(prefix+state.tobytes())
 checked=read_restart(destination);assert ghost_error(checked,global_values)==0
 return destination

for dim in [2,3]:
 for order in [2,4,6]:
  name=f'fd{order}-{dim}d';text=input_for(order,dim,4)
  reference=run(name+'-single',input_for(order,dim,8),a.reference_binary,'',1)
  target=run(name+'-multi',text,a.binary,a.launcher,a.ranks)
  if a.seeded:
   seed_r=seed_restart(reference[0],a.output/(name+'-seed-single.rst'))
   seed_t=seed_restart(target[0],a.output/(name+'-seed-multi.rst'))
   reference=[seed_r]+run(name+'-seeded-single',input_for(order,dim,8),a.reference_binary,'',1,seed_r)
   target=[seed_t]+run(name+'-seeded-multi',text,a.binary,a.launcher,a.ranks,seed_t)
  assert len(reference)==len(target)==4
  errors=[];ghosts=[];states=[]
  for f,g in zip(reference,target):
   r=read_restart(f);v=read_restart(g)
   assert len(r['locations'])==1 and len(v['locations'])==2**dim
   assert (r['cycle'],r['time'])==(v['cycle'],v['time'])
   expected=global_state(r);actual=global_state(v)
   errors.append(float(np.max(abs(actual-expected)/(1+abs(expected)))))
   ghosts.append(ghost_error(v,expected));states.append(v)
  resumed=run(name+'-resume',text,a.binary,a.launcher,a.ranks,target[1])
  final=read_restart(resumed[-1]);resume_error=float(np.max(abs(final['state']-states[-1]['state'])/(1+abs(states[-1]['state']))))
  rank_change_error=0.
  if a.ranks>1:
   migrated=run(name+'-serial-resume',text,a.reference_binary,'',1,target[1])
   final=read_restart(migrated[-1])
   rank_change_error=float(np.max(abs(final['state']-states[-1]['state'])/(1+abs(states[-1]['state']))))
  record=dict(case=name,status='PASS' if max(errors+ghosts+[resume_error,rank_change_error])<=tol else 'FAIL' ,tolerance=tol,cycles=[0,1,2,3],global_errors=errors,all_ghost_errors=ghosts,restart_error=resume_error,rank_change_restart_error=rank_change_error,blocks=2**dim,ranks=a.ranks,fields=50,global_cells=8**dim)
  records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
  assert record['status']=='PASS'
summary=dict(status='PASS',seeded=a.seeded,cases=len(records),runs=len(runs),binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),reference_binary_sha256=hashlib.sha256(a.reference_binary.read_bytes()).hexdigest(),scope='uniform same-level periodic decomposition, all 50 fields and all stored ghosts at four synchronized cycles, two-step restart continuation; no refinement/physical qualification')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
