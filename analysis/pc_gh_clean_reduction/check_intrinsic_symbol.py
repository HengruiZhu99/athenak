#!/usr/bin/env python3
"""Compiled 50-field eigenspaces/projectors at admitted and excluded crossings."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import sympy as sp
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--replay',type=Path,help='Existing batch with input.txt, output.txt, states.json and binary-sha256.txt; analyze without executing')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_dir))
from candidate import chart, switch
from characteristics import blocks
from check_full_symbol import wave_embedding
rng=np.random.default_rng(202609071);cases=[]

def case(kind,alpha,w,u,O,offset=None,ladder=None):
 u=u.copy();u[0]=w;u[1]=alpha/w
 Y,n=wave_embedding(u,O);n=n/np.linalg.norm(n)
 nu=float(np.sqrt(n@np.linalg.inv(chart(u)[1])@n))
 return dict(kind=kind,alpha=alpha,w=w,u=u.tolist(),O=O.tolist(),normal=n.tolist(),nu=nu,
             rate=alpha,eta=2.,kappa=1.,offset=offset,ladder=ladder)

for alpha in [.8,1.4]:
 for family in ['light_shift','lapse_transverse','lapse_longitudinal']:
  u=rng.uniform(-.2,.2,50);O=np.linalg.qr(rng.normal(size=(3,3)))[0]
  w0=1/alpha if family=='light_shift' else 1/np.sqrt(2*alpha) if family=='lapse_transverse' else np.sqrt(4/(alpha*(6+alpha)))
  for offset in [-.01,-.0001,-.000001,0,.000001,.0001,.01]:cases.append(case('admitted',alpha,w0*(1+offset),u,O,offset,f'{family}-a{alpha}'))
for z in [.01,.05,.1,.15,.3,.49]:
 u=rng.uniform(-.2,.2,50);O=np.linalg.qr(rng.normal(size=(3,3)))[0];alpha=rng.uniform(.4,1.6)
 cases.append(case('separated',alpha,np.sqrt(z/alpha),u,O))
u=rng.uniform(-.2,.2,50);O=np.linalg.qr(rng.normal(size=(3,3)))[0]
for alpha in [1.8,1.98,1.998,2.]:cases.append(case('excluded' if alpha==2 else 'boundary_approach',alpha,.5,u,O))
inputs=[]
for c in cases:
 u=np.array(c['u']);n=np.array(c['normal'])
 for col in range(-1,50):
  du=np.zeros((3,50))
  if col>=0:du[:,col]=n
  inputs.append(np.r_[u,du.ravel(),c['rate'],c['eta'],c['kappa']])
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g');(a.output/'states.json').write_text(json.dumps(cases,indent=2)+'\n')
cmd=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
start=time.time()
if a.replay:
 assert (a.output/'input.txt').read_bytes()==(a.replay/'input.txt').read_bytes(), 'replay input differs'
 assert json.loads((a.replay/'states.json').read_text())==cases, 'replay states differ'
 for name in ['output.txt','kokkos.txt','run.log']:shutil.copy2(a.replay/name,a.output/name)
 binary_hash=(a.replay/'binary-sha256.txt').read_text().split()[0]
 cmd=['analysis-only replay',str(a.replay.resolve())]
else:
 with (a.output/'run.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 binary_hash=hashlib.sha256(a.binary.read_bytes()).hexdigest()
raw=np.loadtxt(a.output/'output.txt');assert raw.shape==(len(inputs),128);assert np.isfinite(raw).all()
matrices=np.array([(raw[51*k+1:51*k+51,:50]-raw[51*k,:50]).T for k in range(len(cases))]);np.save(a.output/'matrices.npy',matrices)
mm=lambda x,y:np.einsum('ij,jk->ik',x,y,optimize=False)
norm=lambda x:float(np.linalg.norm(x,2))
rows=[]
for k,c in enumerate(cases):
 u=np.array(c['u']);n=np.array(c['normal']);A=matrices[k]-np.eye(50)*(u[7:10]@n);an=norm(A)
 Y,_=wave_embedding(u,np.array(c['O']));B,V,T=[np.array(x,dtype=float) for x in blocks(c['alpha'],c['w'],float(switch(c['alpha']*c['w']**2)))];B20=np.array(sp.diag(sp.Matrix(B),sp.Matrix(V),sp.Matrix(V),sp.Matrix(T),sp.Matrix(T)),dtype=float)*c['nu']
 block_error=norm(mm(A,Y)-mm(Y,B20))/(1+norm(mm(A,Y)))
 # U*U^T equals Y*pinv(Y), avoiding the platform BLAS matmul path.
 uy,sy,_=np.linalg.svd(Y,full_matrices=False)
 assert sy[-1]>1e-12*sy[0], 'normal embedding loses rank'
 image_error=norm(A-mm(mm(uy,uy.T),A))/(1+an)
 alpha=c['alpha'];w=c['w'];nu=c['nu'];sigma=float(switch(alpha*w*w))
 values=[(0.,30)]
 for speed,multiplicity in [(alpha*w,6),(1.,2),(w*np.sqrt(2*alpha),1),(np.sqrt((4-sigma*alpha*alpha*w*w)/3),1)]:
  values.extend([(-nu*speed,multiplicity),(nu*speed,multiplicity)])
 groups=[]
 for value,multiplicity in sorted(values):
  if groups and abs(value-groups[-1][0])<=1e-10*(1+abs(value)):groups[-1][1]+=multiplicity
  else:groups.append([value,multiplicity])
 projections=[];nullities=[];eigen_error=0.;complete=True
 for value,multiplicity in groups:
  left,s,right=np.linalg.svd(A-value*np.eye(50));nullity=int(np.count_nonzero(s<=1e-11*(1+an)));nullities.append(dict(value=value,expected=multiplicity,actual=nullity))
  if nullity!=multiplicity:complete=False
  if c['kind']=='excluded' or nullity!=multiplicity:continue
  v=right[-nullity:].T;l=left[:,-nullity:];projector=mm(v,np.linalg.solve(mm(l.T,v),l.T));projections.append(projector)
  eigen_error=max(eigen_error,norm(mm(A,projector)-value*projector)/(1+an*norm(projector)))
 algebra_error=0.;raw_algebra_error=0.;max_projector=None
 if complete:
  pn=[norm(x) for x in projections];max_projector=max(pn)
  err=norm(sum(projections)-np.eye(50));raw_algebra_error=err;algebra_error=err/(1+sum(pn))
  for i,x in enumerate(projections):
   for j,y in enumerate(projections):
    err=norm(mm(x,y)-(x if i==j else 0));raw_algebra_error=max(raw_algebra_error,err);algebra_error=max(algebra_error,err/(1+pn[i]*pn[j]))
 if c['kind']=='excluded':
  detected=sorted((g['expected'],g['actual']) for g in nullities)==[(10,9),(10,9),(30,30)]
  status='PASS' if detected and block_error<=2e-11 and image_error<=2e-11 else 'FAIL'
 else:status='PASS' if complete and block_error<=2e-11 and image_error<=2e-11 and eigen_error<=2e-9 and algebra_error<=2e-7 else 'FAIL'
 row=dict(case=k,kind=c['kind'],ladder=c['ladder'],offset=c['offset'],alpha=alpha,w=w,sigma=sigma,block_error=block_error,image_error=image_error,nullities=nullities,eigenspace_complete=complete,max_projector_norm=max_projector,projector_eigen_error=eigen_error,projector_algebra_error=algebra_error,raw_projector_algebra_error=raw_algebra_error,status=status)
 rows.append(row);print(k,c['kind'],status,max_projector,flush=True)
 (a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
ladders=[]
for name in sorted({r['ladder'] for r in rows if r['ladder']}):
 data=[r for r in rows if r['ladder']==name];baseline=max(r['max_projector_norm'] for r in data if abs(r['offset'])==.01);ratio=max(r['max_projector_norm'] for r in data)/baseline;ladders.append(dict(name=name,max_norm_ratio=ratio,status='PASS' if ratio<=4 else 'FAIL'))
summary=dict(status='PASS' if all(r['status']=='PASS' for r in rows+ladders) else 'FAIL',cases=len(cases),compiled_points=len(inputs),ladders=ladders,max_block_error=max(r['block_error'] for r in rows),max_image_error=max(r['image_error'] for r in rows),excluded_control=rows[-1],wall_seconds=time.time()-start,binary_sha256=binary_hash,input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),matrix_sha256=hashlib.sha256((a.output/'matrices.npy').read_bytes()).hexdigest(),command=cmd,scope='compiled full-symbol numerical checks; compact-subset theorem remains the separate analytical proof, not a puncture-uniform claim')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
