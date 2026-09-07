#!/usr/bin/env python3
"""Complete compiled Minkowski Fourier matrices and finite-time amplification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import scipy.linalg as la
import sympy as s
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.reference_dir))
from exact_flat_symbol import matrices
from check_flat_fourier import source_matrix
params=[(1,2,1),(0,2,0),(1,0,0)];eps=[1e-4,5e-5];inputs=[];batches=[]
u0=np.zeros(50);u0[:2]=1
for rate,eta,kap in params:
 batch={'parameters':[rate,eta,kap],'center':len(inputs),'sources':[],'principal':[]}
 inputs.append(np.r_[u0,np.zeros(150),rate,eta,kap])
 for h in eps:
  batch['sources'].append(len(inputs))
  for col in range(50):
   for sign in [-1,1]:
    u=u0.copy();u[col]+=sign*h;inputs.append(np.r_[u,np.zeros(150),rate,eta,kap])
 for axis in range(3):
  batch['principal'].append(len(inputs))
  for col in range(50):
   du=np.zeros((3,50));du[axis,col]=1;inputs.append(np.r_[u0,du.ravel(),rate,eta,kap])
 batches.append(batch)
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g');(a.output/'batches.json').write_text(json.dumps(batches,indent=2)+'\n')
cmd=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())];start=time.time()
if a.replay:
 assert (a.output/'input.txt').read_bytes()==(a.replay/'input.txt').read_bytes()
 for name in ['output.txt','kokkos.txt','run.log']:shutil.copy2(a.replay/name,a.output/name)
 bh=(a.replay/'binary-sha256.txt').read_text().split()[0];cmd=['replay',str(a.replay.resolve())]
else:
 with (a.output/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 bh=hashlib.sha256(a.binary.read_bytes()).hexdigest()
raw=np.loadtxt(a.output/'output.txt');assert raw.shape==(len(inputs),128) and np.isfinite(raw).all();raw=raw[:,:50]
mm=lambda x,y:np.einsum('ij,jk->ik',x,y,optimize=False)
norm=lambda x:float(la.svdvals(x)[0])
err=lambda x,y:float(np.max(abs(x-y)/(1+abs(y))))
n=np.array([1.,2.,3.])/np.sqrt(14);ks=[0,.01,.1,1,10];times=[0,.1,1,5,20];reports=[];archive={}
for ib,batch in enumerate(batches):
 rate,eta,kap=batch['parameters'];Pexact=matrices(s.Integer(rate),s.Integer(eta));Jexact=source_matrix(s.Integer(rate),s.Integer(eta),s.Integer(kap));Pr=np.array([np.array(v,dtype=float) for v in Pexact]);Jr=np.array(Jexact,dtype=float)
 Js=[]
 for h,offset in zip(eps,batch['sources']):Js.append(((raw[offset+1:offset+100:2]-raw[offset:offset+100:2])/(2*h)).T)
 Ps=np.array([raw[o:o+50].T-raw[batch['center']][:,None] for o in batch['principal']]);J=Js[-1]
 je=[err(v,Jr) for v in Js];pe=err(Ps,Pr);zero=float(abs(raw[batch['center']]).max());assert max(*je,pe,zero)<=2e-11
 nullities=[];power=s.eye(50)
 for degree in range(1,5):power=power*Jexact;nullities.append(50-power.rank())
 archive[f'J{ib}']=J;archive[f'P{ib}']=Ps;rows=[]
 for k in ks:
  kn=k*n;A=J+1j*np.einsum('i,ijk->jk',kn,Ps);Ar=Jr+1j*np.einsum('i,ijk->jk',kn,Pr)
  C=np.zeros((30,50),complex);C[:,20:]=np.eye(30)
  for i in range(3):
   C[i,0]=-1j*kn[i];C[3+i,0:2]=-1j*kn[i]
   for v in range(5):C[6+5*i+v,2+v]=-1j*kn[i]
   for v in range(3):C[21+3*i+v,7+v]=-1j*kn[i]
  eig=la.eigvals(A);evol=[]
  for t in times:
   X=la.expm(t*A);Xr=la.expm(t*Ar);xe=norm(X-Xr)/(1+norm(Xr));ce=norm(mm(C,X)-np.exp(-rate*t)*C)/(1+norm(C)*norm(X))
   assert np.isfinite(X).all() and xe<=2e-9 and ce<=2e-9
   evol.append(dict(t=t,norm2=norm(X),reference_error=xe,reduction_closure_error=ce))
  rows.append(dict(k=k,max_eigenvalue_real=float(eig.real.max()),eigenvalues=[[float(v.real),float(v.imag)] for v in eig],times=evol))
 report=dict(parameters=[rate,eta,kap],source_errors=je,principal_error=pe,flat_rhs_max=zero,zero_frequency_nullities=nullities,zero_jordan_chains_at_least_2=nullities[1]-nullities[0],rows=rows)
 reports.append(report);print('completed',batch['parameters'],'nullities',nullities,flush=True)
np.savez(a.output/'matrices.npz',**archive);(a.output/'cases.json').write_text(json.dumps(reports,indent=2)+'\n')
summary=dict(status='PASS',compiled_points=len(inputs),parameter_cases=3,max_source_error=max(max(r['source_errors']) for r in reports),max_principal_error=max(r['principal_error'] for r in reports),max_exponential_error=max(t['reference_error'] for r in reports for row in r['rows'] for t in row['times']),max_reduction_closure_error=max(t['reduction_closure_error'] for r in reports for row in r['rows'] for t in row['times']),max_sampled_norm=max(t['norm2'] for r in reports for row in r['rows'] for t in row['times']),binary_sha256=bh,input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),command=cmd,wall_seconds=time.time()-start,scope='stationary Minkowski full Fourier and sampled transients; norm growth is reported, not ruled out')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
