#!/usr/bin/env python3
"""Actual Dx/KO consumer plus coupled 50-field periodic RK3 amplification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import scipy.linalg as la
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--matrices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-only',action='store_true');p.add_argument('--replay',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cases=[];inputs=[];h=np.array([.125,.2,.3]);amp=1e-5
for order in [2,4,6]:
 for dim in [2,3]:
  for ko in [0.,.3]:
   for v in [0.,.2,.8,1.6,np.pi]:
    theta=v*np.array([1.,.7,-.4]);c=dict(order=order,dimensions=dim,ko=ko,theta=theta.tolist(),start=len(inputs));cases.append(c)
    for phase in [0.,np.pi/2]:
     for col in range(50):
      for sign in [-1,1]:inputs.append(np.r_[order,dim,col,sign*amp,phase,theta,1/h,1,2,1,ko])
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g');(a.output/'states.json').write_text(json.dumps(cases,indent=2)+'\n')
cmd=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())];start=time.time()
if a.replay:
 assert (a.output/'input.txt').read_bytes()==(a.replay/'input.txt').read_bytes()
 for name in ['output.txt','kokkos.txt','run.log','binary-sha256.txt']:shutil.copy2(a.replay/name,a.output/name)
 bh=(a.output/'binary-sha256.txt').read_text().split()[0];cmd=['replay',str(a.replay.resolve())]
else:
 with (a.output/'run.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 bh=hashlib.sha256(a.binary.read_bytes()).hexdigest();(a.output/'binary-sha256.txt').write_text(bh+'\n')
raw=np.loadtxt(a.output/'output.txt');assert raw.shape==(len(inputs),50) and np.isfinite(raw).all()
if a.run_only:
 (a.output/'run-manifest.json').write_text(json.dumps(dict(status='COMPILED_BATCH_ONLY',command=cmd,binary_sha256=bh,wall_seconds=time.time()-start),indent=2)+'\n');raise SystemExit(0)
mat=np.load(a.matrices);J=mat['J0'];Ps=mat['P0'];I=np.eye(50);mm=lambda x,y:np.einsum('ij,jk->ik',x,y,optimize=False);norm=lambda x:float(la.svdvals(x)[0]);R=lambda z:1+z+z*z/2+z*z*z/6

def rk(A,dt):
 stage1=I+dt*A;stage2=.75*I+.25*stage1+.25*dt*mm(A,stage1)
 return I/3+2*stage2/3+2*dt*mm(A,stage2)/3

def power(X,n):
 out=np.eye(50,dtype=complex)
 while n:
  if n%2:out=mm(out,X)
  n//=2
  if n:X=mm(X,X)
 return out
reports=[]
for c in cases:
 order=c['order'];dim=c['dimensions'];theta=np.array(c['theta']);ko=c['ko'];radius=order//2
 coeff={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order]
 keff=np.array([2*sum(cc*np.sin((j+1)*theta[d]) for j,cc in enumerate(coeff))/h[d] if d<dim else 0 for d in range(3)])
 q=-ko*sum(np.sin(theta[d]/2)**(2*(radius+1))/h[d] for d in range(dim))
 A=J+1j*np.einsum('i,ijk->jk',keff,Ps)+q*I
 o=c['start'];re=((raw[o+1:o+100:2]-raw[o:o+100:2])/(2*amp)).T;o+=100
 im=-((raw[o+1:o+100:2]-raw[o:o+100:2])/(2*amp)).T
 matrixerr=float(np.max(abs(re+1j*im-A)/(1+abs(A))))
 C=np.zeros((30,50),complex);C[:,20:]=np.eye(30)
 for d in range(3):
  C[d,0]=-1j*keff[d];C[3+d,0:2]=-1j*keff[d]
  for v in range(5):C[6+5*d+v,2+v]=-1j*keff[d]
  for v in range(3):C[21+3*d+v,7+v]=-1j*keff[d]
 dt=.2*min(h[:dim])/np.sqrt(2);M=rk(A,dt);Z=dt*A;Z2=mm(Z,Z);poly=I+Z+Z2/2+mm(Z2,Z)/6
 stageerr=norm(M-poly)/(1+norm(poly));closure=norm(mm(C,M)-R(dt*(q-1))*C)/(1+norm(C)*norm(M))
 k=float(np.linalg.norm(keff));eigen=[-1+q]
 for damping,scale in [(0,1),(0,2),(2,1),(1,1),(2,1)]:eigen.extend(np.roots([1,damping,scale*k*k])+q)
 radius_rk=float(max(abs(R(dt*np.array(eigen)))))
 exact=la.expm(.5*A);errors=[];norms=[];counts=[];steps=math.ceil(.5/dt)
 for mult in [1,2,4]:
  count=steps*mult;X=power(rk(A,.5/count),count);errors.append(norm(X-exact)/(1+norm(exact)));norms.append(norm(X));counts.append(count)
 orders=np.log2(np.array(errors[:-1])/errors[1:]);passed=matrixerr<=2e-8 and stageerr<=2e-12 and closure<=2e-11 and radius_rk<=1+1e-12 and all(e<=1e-10 or p>=2.8 for e,p in zip(errors,orders))
 row=dict(**c,matrix_error=matrixerr,modified_k=keff.tolist(),ko_symbol=q,dt=dt,stage_error=stageerr,reduction_closure_error=closure,rk_radius=radius_rk,time_errors=errors,time_orders=orders.tolist(),step_counts=counts,power_norms=norms,exponential_norm=norm(exact),status='PASS' if passed else 'FAIL');reports.append(row)
 print(order,dim,ko,c['theta'][0],row['status'],matrixerr,orders,flush=True)
 (a.output/'cases.json').write_text(json.dumps(reports,indent=2)+'\n')
summary=dict(status='PASS' if all(c['status']=='PASS' for c in reports) else 'FAIL',cases=len(cases),compiled_points=len(inputs),max_matrix_error=max(c['matrix_error'] for c in reports),max_rk_radius=max(c['rk_radius'] for c in reports),max_closure_error=max(c['reduction_closure_error'] for c in reports),minimum_time_order=min(min(c['time_orders']) for c in reports),binary_sha256=bh,input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),matrix_archive_sha256=hashlib.sha256(a.matrices.read_bytes()).hexdigest(),wall_seconds=time.time()-start,command=cmd,scope='compiled nonlinear FD consumer linearization and uniform coupled Fourier RK3/KO; no mesh boundaries or evolution')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
