#!/usr/bin/env python3
"""Independent spatial differentiation of compiled nonlinear subsidiary laws."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--replay',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(202609072)
hs=[.02,.01,.005];pairs=[(0,1),(0,2),(1,2)]
idx=np.array([[20+i,23+i,*range(26+5*i,31+5*i),*range(41+3*i,44+3*i)] for i in range(3)])
inputs=[];cases=[]

def jets(c,x):
 u,L,H,T=[np.array(c[k]) for k in ['u','L','H','T']]
 v=u+np.einsum('in,i->n',L,x)+.5*np.einsum('ijn,i,j->n',H,x,x)+np.einsum('in,i->n',T,x**3)/6
 dv=L+np.einsum('ijn,j->in',H,x)+.5*T*x[:,None]**2
 return v,dv

def add(c,x):
 u,du=jets(c,np.array(x));rate=u[0]*u[1]*(1+np.array([.2,-.15,.1])@x)
 inputs.append(np.r_[u,du.ravel(),rate,2.,1.]);return len(inputs)-1

for k in range(12):
 u=rng.uniform(-.25,.25,50);alpha=[.7,1.,1.3][k//4];z=[.05,.2,.3,.7][k%4];u[0]=np.sqrt(z/alpha);u[1]=alpha/u[0]
 L=rng.uniform(-.15,.15,(3,50));H=rng.uniform(-.1,.1,(3,3,50));H=(H+H.swapaxes(0,1))/2;T=rng.uniform(-.1,.1,(3,50))
 c=dict(u=u.tolist(),L=L.tolist(),H=H.tolist(),T=T.tolist());c['center']=add(c,np.zeros(3));c['stencils']=[]
 for h in hs:
  stencil=[]
  for i in range(3):
   indices=[]
   for offset in [-2,-1,1,2]:
    x=np.zeros(3);x[i]=offset*h;indices.append(add(c,x))
   stencil.append(indices)
  c['stencils'].append(stencil)
 cases.append(c)
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g')
(a.output/'states.json').write_text(json.dumps(cases,indent=2)+'\n')
start=time.time()
cmd=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
if a.replay:
 assert (a.output/'input.txt').read_bytes()==(a.replay/'input.txt').read_bytes()
 assert cases==json.loads((a.replay/'states.json').read_text())
 for name in ['output.txt','kokkos.txt','run.log']:shutil.copy2(a.replay/name,a.output/name)
 bh=(a.replay/'binary-sha256.txt').read_text().split()[0];cmd=['replay',str(a.replay.resolve())]
else:
 with (a.output/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 bh=hashlib.sha256(a.binary.read_bytes()).hexdigest()
out=np.loadtxt(a.output/'output.txt');assert out.shape==(len(inputs),128) and np.isfinite(out).all()
inputs=np.array(inputs);ut=out[:,:50]
xt=np.column_stack([ut[:,0],inputs[:,1]*ut[:,0]+inputs[:,0]*ut[:,1],ut[:,2:10]])
gt=ut[:,idx]
rows=[]
for k,c in enumerate(cases):
 u,L,H=[np.array(c[n]) for n in ['u','L','H']];alpha=u[0]*u[1];beta=u[7:10];db=L[:,7:10]
 dx=np.column_stack([L[:,0],u[1]*L[:,0]+u[0]*L[:,1],L[:,2:10]])
 ddx=np.concatenate([H[:,:,:1],(u[1]*H[:,:,0]+u[0]*H[:,:,1]+np.outer(L[:,0],L[:,1])+np.outer(L[:,1],L[:,0]))[:,:,None],H[:,:,2:10]],axis=2)
 G=u[idx];dG=L[:,idx];ddG=H[:,:,idx];E=G-dx;dE=dG-ddx
 omega=np.zeros((3,3,10));dom=np.zeros((3,3,3,10))
 for i in range(3):
  for j in range(3):
   omega[i,j]=dG[i,j]-dG[j,i]
   for m in range(3):dom[m,i,j]=ddG[m,i,j]-ddG[m,j,i]
 rate=alpha;drate=dx[:,1]+alpha*np.array([.2,-.15,.1])
 stretch=np.einsum('ij,ja->ia',db,E)
 et=np.einsum('j,jia->ia',beta,dE)+stretch-rate*E
 omt=np.einsum('k,kija->ija',beta,dom)+np.einsum('ik,kja->ija',db,omega)+np.einsum('jk,ika->ija',db,omega)-rate*omega
 ratecurl=-drate[:,None,None]*E[None,:,:]+drate[None,:,None]*E[:,None,:]
 omt+=ratecurl
 errors=[]
 for h,stencil in zip(hs,c['stencils']):
  dxt=[];dgt=[]
  for ids in stencil:
   coeff=np.array([1.,-8.,8.,-1.])/(12*h)
   dxt.append(np.einsum('r,ra->a',coeff,xt[ids]));dgt.append(np.einsum('r,ria->ia',coeff,gt[ids]))
  dxt=np.array(dxt);dgt=np.array(dgt)
  actualE=gt[c['center']]-dxt
  actualO=np.array([dgt[i,j]-dgt[j,i] for i,j in pairs]);targetO=np.array([omt[i,j] for i,j in pairs])
  ee=abs(actualE-et)/(1+abs(et));oe=abs(actualO-targetO)/(1+abs(targetO))
  errors.append(dict(h=h,reduction_max=float(ee.max()),curl_max=float(oe.max()),reduction_components=ee.tolist(),curl_components=oe.tolist()))
 rows.append(dict(case=k,errors=errors,omitted_stretch_difference=float(abs(stretch).max()),omitted_rate_gradient_difference=float(abs(ratecurl).max())))
maxE=[max(r['errors'][i]['reduction_max'] for r in rows) for i in range(3)]
maxO=[max(r['errors'][i]['curl_max'] for r in rows) for i in range(3)]
orders={};passed=True
for name,values in [('reduction',maxE),('curl',maxO)]:
 order=np.log2(np.array(values[:-1])/values[1:]);orders[name]=order.tolist()
 passed &= values[-1]<=1e-7 and all(v<=1e-10 or o>=3.5 for v,o in zip(values,order))
passed &= all(r['omitted_stretch_difference']>1e-5 and r['omitted_rate_gradient_difference']>1e-5 for r in rows)
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(status='PASS' if passed else 'FAIL',cases=len(cases),compiled_points=len(inputs),h=hs,reduction_errors=maxE,curl_errors=maxO,orders=orders,binary_sha256=bh,input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),wall_seconds=time.time()-start,command=cmd,scope='local nonlinear continuum subsidiary identities by independent spatial differentiation; no evolution')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert passed
