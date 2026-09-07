#!/usr/bin/env python3
"""Uniform periodic production RK3 against independent nonlinear Python RHS."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
from run_legacy_equivalence import input_text

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--rate-profile',choices=['constant','lapse_scaled'],default='constant')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('candidate',a.reference)
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records=[];dt=1e-4;tol=2e-12

def run(name,text,extra=()):
 d=a.output/name;d.mkdir();f=d/'used.athinput';f.write_text(text)
 command=[str(a.binary.resolve()),'-i',str(f.resolve()),*extra]
 start=time.time()
 with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 (d/'manifest.json').write_text(json.dumps(dict(command=command,returncode=r.returncode,wall_seconds=time.time()-start,input_sha256=digest(f)),indent=2)+'\n')
 assert r.returncode==0,(name,(d/'run.log').read_text())
 return d

def text(order,dim,steps=1,mb=8,pgen='intrinsic_smooth'):
 t=input_text(order,dim,'collision_factorized',steps)
 start=t.index('<pc_gh>');end=t.index('<problem>')
 t=t[:start]+f'''<pc_gh>
formulation = intrinsic_clean
spatial_order = {order}
shift_eta = 2
kappa = 1
reduction_rate = 1
reduction_profile = {a.rate_profile}
dissipation = 0.3
research_dt_ceiling = {dt}
'''+t[end:].replace('legacy_equivalence',pgen)
 t=t.replace('tlim = 0.0001',f'tlim = {steps*dt}')
 t+='\n<output1>\nfile_type = rst\ndt = 0.0001\n'
 return t

def payload(file,dim):
 raw=file.read_bytes();header=raw[:raw.index(b'<par_end>')].decode()
 assert 'intrinsic_pcgh50' in header
 shape=(50,16 if dim==3 else 1,16,16)
 return np.frombuffer(raw[-np.prod(shape)*8:],dtype=np.float64).reshape(shape).copy()

def active(x,dim):
 return np.moveaxis(x[:,4:12 if dim==3 else 1,4:12,4:12],0,-1) if dim==3 else np.moveaxis(x[:,:,4:12,4:12],0,-1)

def rhs(u,order,dim):
 du=np.zeros((3,*u.shape));ko=np.zeros_like(u)
 weights={2:[.5],4:[2/3,-1/12],6:[3/4,-3/20,1/60]}[order]
 radius=order//2+1
 for d,h in enumerate([1/8,1.3/8,1.7/8][:dim]):
  axis=2-d
  for offset,c in enumerate(weights,1):du[d]+=c*(np.roll(u,-offset,axis)-np.roll(u,offset,axis))/h
  for offset in range(-radius,radius+1):
   ko+=-.3*((-1.)**offset)*math.comb(2*radius,radius+offset)*np.roll(u,-offset,axis)/(2**(2*radius)*h)
 out=np.empty_like(u)
 for idx in np.ndindex(u.shape[:-1]):out[idx]=ref.rhs(u[idx],du[(slice(None),*idx)],rate=u[idx][0]*u[idx][1] if a.rate_profile=='lapse_scaled' else 1,eta=2,kappa=1)
 return out+ko

for dim in [2,3]:
 for order in [2,4,6]:
  name=f'fd{order}-{dim}d';d=run(name,text(order,dim))
  files=sorted((d/'rst').glob('*.rst'));assert len(files)==2
  u=active(payload(files[0],dim),dim);actual=active(payload(files[-1],dim),dim)
  with (d/'intrinsic-health-rank0.csv').open() as health_file:
   health=list(csv.DictReader(health_file))
  assert health and all(np.isfinite(float(v)) for row in health for k,v in row.items() if k!='stage')
  eigen=[];condition=[]
  for index in np.ndindex(u.shape[:-1]):
   g=ref.chart(u[index])[1];eigen.append(np.linalg.eigvalsh(g)[0])
   condition.append(np.linalg.norm(g)*np.linalg.norm(np.linalg.inv(g)))
  health_error=max(abs(float(health[0]['min_metric_eigenvalue'])-min(eigen)),abs(float(health[0]['max_frobenius_condition'])-max(condition)))
  assert health_error<=tol
  v=u+dt*rhs(u,order,dim)
  v=.75*u+.25*v+.25*dt*rhs(v,order,dim)
  v=u/3+2*v/3+2*dt*rhs(v,order,dim)/3
  np.savez(d/'oracle.npz',initial=u,expected=v,actual=actual)
  error=float(np.max(abs(actual-v)/(1+abs(v))))
  padded=np.pad(v,((4,4) if dim==3 else (0,0),(4,4),(4,4),(0,0)),mode='wrap')
  ghost_error=float(np.max(abs(np.moveaxis(payload(files[-1],dim),0,-1)-padded)/(1+abs(padded))))
  record=dict(case=name,status='PASS' if max(error,ghost_error)<=tol and np.isfinite(actual).all() else 'FAIL',max_error=error,all_cells_with_ghosts_error=ghost_error,initial_health_error=health_error,tolerance=tol,fields=50,active_cells=int(np.prod(u.shape[:-1])),dt=dt)
  records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
  assert record['status']=='PASS'
# Actual two-step state vs one-step + tagged restart, including all ghosts.
controls=[]
for dim in [2,3]:
 full=run(f'continuity-{dim}d-full',text(6,dim,steps=2))
 source=sorted((a.output/f'fd6-{dim}d'/'rst').glob('*.rst'))[-1]
 resume=run(f'continuity-{dim}d-resume',text(6,dim,steps=2),['-r',str(source.resolve())])
 f=payload(sorted((full/'rst').glob('*.rst'))[-1],dim)
 r=payload(sorted((resume/'rst').glob('*.rst'))[-1],dim)
 err=float(np.max(abs(f-r)/(1+abs(f))))
 controls.append(dict(case=f'restart-{dim}d',error=err,status='PASS' if err<=tol else 'FAIL'))
 assert err<=tol
 flat=run(f'minkowski-{dim}d',text(6,dim,steps=2,pgen='intrinsic_minkowski'))
 ff=sorted((flat/'rst').glob('*.rst'))
 err=float(np.max(abs(payload(ff[-1],dim)-payload(ff[0],dim))))
 controls.append(dict(case=f'minkowski-{dim}d',error=err,status='PASS' if err==0 else 'FAIL'))
 assert err==0
(a.output/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
summary=dict(status='PASS',rate_profile=a.rate_profile,cases=len(records),binary_sha256=digest(a.binary),reference_sha256=digest(a.reference),scope='actual uniform periodic serial mesh one-step RK3, independent nonlinear Python point RHS and separate FD/KO/RK assembly; no convergence or physical qualification')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
