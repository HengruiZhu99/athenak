#!/usr/bin/env python3
"""Independent symbolic geometry and Cholesky finite-radius conversion oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import sympy as sp

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20260907)
a0,c,b,d,e=sp.symbols('a c b d e',real=True); chart=(a0,c,b,d,e)
# Explicit entries, differentiated symbolically independently of C++ dT/ddT.
g=sp.Matrix([[sp.exp(2*a0),sp.exp(a0)*b,sp.exp(a0)*d],
 [sp.exp(a0)*b,b*b+sp.exp(2*c),b*d+sp.exp(c)*e],
 [sp.exp(a0)*d,b*d+sp.exp(c)*e,d*d+e*e+sp.exp(-2*a0-2*c)]])
assert sp.simplify(g.det())==1
gi=g.inv()
for q in chart: assert sp.simplify(sp.trace(gi*g.diff(q)))==0
av=sp.symbols('A:5')
ah_exact=sp.Matrix([[av[0],av[1],av[2]],[av[1],av[3],av[4]],[av[2],av[4],-av[0]-av[3]]])
t_exact=sp.Matrix([[sp.exp(a0),0,0],[b,sp.exp(c),0],[d,e,sp.exp(-a0-c)]])
assert sp.simplify(sp.trace(gi*t_exact*ah_exact*t_exact.T))==0
metric=sp.lambdify(chart,g,'numpy')
jac=sp.lambdify(chart,[list(g.diff(q)) for q in chart],'numpy')
hess=sp.lambdify(chart,[[list(g.diff(q).diff(r)) for r in chart] for q in chart],'numpy')
ij=np.triu_indices(3)

def stf(v):
 x=np.zeros((3,3)); x[ij]=[*v,-v[0]-v[3]]
 return x+x.T-np.diag(x.diagonal())

def full(v):
 x=np.zeros((3,3));x[ij]=v;return x+x.T-np.diag(x.diagonal())

def reference(u):
 s=u[2:7];gm=np.array(metric(*s)); t=np.linalg.cholesky(gm);ti=np.linalg.inv(t)
 j=np.array(jac(*s)).reshape(5,3,3); h=np.array(hess(*s)).reshape(5,5,3,3)
 ah=stf(u[11:16]);ac=t@ah@t.T;q=np.einsum('aij,ka->kij',j,u[26:41].reshape(3,5))
 old=np.zeros(55);old[0]=u[0];old[1:7]=gm[ij];old[7]=u[10];old[8:14]=ac[ij]
 old[14:17]=u[16:19];old[17]=u[19];old[18]=u[1];old[19:22]=u[7:10]
 old[22:25]=u[20:23];old[25:43]=q[:,ij[0],ij[1]].ravel();old[43:46]=2*u[23:26];old[46:55]=u[41:50]
 values=np.concatenate([x.ravel() for x in [t,ti,gm,np.linalg.inv(gm),ah,ac,q,j,h]])
 return values,old

def reverse(old):
 gm=full(old[1:7]);t=np.linalg.cholesky(gm);ti=np.linalg.inv(t)
 u=np.zeros(50);u[0]=old[0];u[1]=old[18];u[2:7]=[np.log(t[0,0]),np.log(t[1,1]),t[1,0],t[2,0],t[2,1]]
 ah=ti@full(old[8:14])@ti.T;u[11:16]=ah[ij][:5]
 u[10]=old[7];u[16:19]=old[14:17];u[19]=old[17];u[7:10]=old[19:22];u[20:23]=old[22:25];u[23:26]=old[43:46]/2;u[41:50]=old[46:55]
 for k in range(3):
  q=full(old[25+6*k:31+6*k]);ad=q[0,0]/(2*gm[0,0]);bd=q[0,1]/t[0,0]-t[1,0]*ad
  cd=(q[1,1]-2*t[1,0]*bd)/(2*t[1,1]**2);dd=q[0,2]/t[0,0]-t[2,0]*ad
  ed=(q[1,2]-bd*t[2,0]-t[1,0]*dd)/t[1,1]-t[2,1]*cd
  u[26+5*k:31+5*k]=[ad,cd,bd,dd,ed]
 return u

inputs=[];expected=[];valid=[]
for n in range(100):
 u=rng.uniform(-.7,.7,50);u[:2]=np.exp(rng.uniform(-3,1,2))
 values,old=reference(u)
 if n>=50:
  # Reverse-map inputs assembled from a separately drawn SPD matrix and tensors.
  rot=np.linalg.qr(rng.normal(size=(3,3)))[0];ev=rng.uniform(-.7,.7,3);ev-=ev.mean()
  gm=rot@np.diag(np.exp(ev))@rot.T;inv=np.linalg.inv(gm)
  old[1:7]=gm[ij]
  for sl in [slice(8,14),slice(25,31),slice(31,37),slice(37,43)]:
   x=rng.normal(size=(3,3));x=(x+x.T)/2;x-=gm*np.sum(inv*x)/3;old[sl]=x[ij]
 back=reverse(old);_,again=reference(back)
 assert np.max(abs(again-old)/(1+abs(old)))<2e-12
 inputs.append(np.r_[u,old]);expected.append(np.r_[values,reference(u)[1],1,back]);valid.append(True)
base=inputs[0].copy()
for name,slot,value in [('negative_w',50,-1),('zero_rho',68,0),('non_spd',51,-1),('determinant',56,base[56]+.2),('curvature_trace',58,base[58]+.2),('gradient_trace',75,base[75]+.2)]:
 row=base.copy();row[slot]=value;inputs.append(row);v=expected[0].copy();v[406]=0;v[407:]=-9876;expected.append(v);valid.append(False)
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g')
start=time.time();command=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
with (a.output/'run.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert r.returncode==0
actual=np.loadtxt(a.output/'output.txt');expected=np.array(expected)
error=abs(actual-expected)/(1+abs(expected));worst=np.unravel_index(np.argmax(error),error.shape)
summary=dict(status='PASS' if np.isfinite(actual).all() and error.max()<=2e-12 else 'FAIL',cases=len(inputs),valid_cases=sum(valid),tolerance=2e-12,max_normalized_error=float(error.max()),worst_index=list(map(int,worst)),wall_seconds=time.time()-start,command=command,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),exact_determinant_and_five_gradient_trace_identities=True)
identities=[]
for row in actual[:100]:
 gm=row[18:27].reshape(3,3); inv=row[27:36].reshape(3,3); ac=row[45:54].reshape(3,3); q=row[54:81].reshape(3,3,3)
 identities.append([abs(np.linalg.det(gm)-1),np.max(abs(inv@gm-np.eye(3))),abs(np.sum(inv*ac))/(1+np.sum(abs(inv*ac))),max(abs(np.sum(inv*x))/(1+np.sum(abs(inv*x))) for x in q)])
summary['compiled_identity_maxima']=dict(zip(['determinant','inverse','curvature_trace','gradient_trace'],np.max(identities,axis=0).tolist()))
assert max(summary['compiled_identity_maxima'].values())<=2e-12
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
