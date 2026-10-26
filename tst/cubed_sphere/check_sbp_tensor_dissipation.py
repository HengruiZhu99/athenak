#!/usr/bin/env python3
"""Constant-coefficient tensor/momentum-dissipation algebra, no native claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);a=p.parse_args();s=json.loads(a.sbp.read_text());rows=np.array(s['left_boundary_rows']);w=np.array(s['norm_boundary_weights']);trace=np.array(s['left_trace']);m=len(w)
def build(n):
 h=np.ones(n);h[:m]=w;h[-m:]=w[::-1];d=np.zeros((n,n));width=rows.shape[1];d[:m,:width]=rows;d[-m:,-width:]=-rows[::-1,::-1]
 for k in range(m,n-m):
  for off,c in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:d[k,k+off]=c
 tl=np.zeros(n);tl[:m]=trace;return h/n,d*n,tl,tl[::-1]
hx,dx,tl,tr=build(24);hy,dy,_,_=build(28);x=(np.arange(24)+.5)/24;y=(np.arange(28)+.5)/28;u=x[:,None]**2*y[None,:]**3
xy=np.einsum('ik,kj,lj->il',dx,u,dy);yx=np.einsum('ik,kj,lj->il',dx,u,dy);# Separate order of application below.
xy=np.einsum('ik,kj->ij',dx,np.einsum('ij,kj->ik',u,dy));yx=np.einsum('ij,kj->ik',np.einsum('ik,kj->ij',dx,u),dy)
comm=float(np.linalg.norm(xy-yx)/max(1,np.linalg.norm(xy)));poly=float(max(abs((xy-6*x[:,None]*y[None,:]**2).ravel())))
adv=[]
for speed in [-1.,1.]:
 t=tl if speed>0 else tr;L=-speed*dx-abs(speed)*np.outer(t,t)/hx[:,None];E=hx[:,None]*L+L.T*hx[None,:];largest=float(np.linalg.eigvalsh(E).max());adv.append(dict(speed=speed,maximum_energy_eigenvalue=largest,passed=largest<=1e-10))
n=24;step=1/n;G=np.zeros((n-3,n))
for i in range(n-3):G[i,i:i+4]=np.array([-1.,3.,-3.,1.])/step**3
Q=-.01*step**5*np.einsum('ji,jk->ik',G*step,G)/hx[:,None];E=hx[:,None]*Q+Q.T*hx[None,:];scale=max(1,np.linalg.norm(E));largest=float(np.linalg.eigvalsh(E).max()/scale);moments=[float(max(abs(np.einsum('ij,j->i',Q,x**k)))/max(1,np.linalg.norm(Q))) for k in range(3)]
r=dict(mixed_derivative_commutator_relative=comm,mixed_polynomial_max_error=poly,advection=adv,dissipation_normalized_max_energy_eigenvalue=largest,dissipation_polynomial_relative_errors=moments,passed=bool(comm<=1e-10 and poly<=1e-8 and all(x['passed'] for x in adv) and largest<=1e-10 and max(moments)<=1e-8),scope='Constant rectangular tensor product only. Q acts on velocity, with energy contribution 2vTHQv<=0. Not actual native KO, curvilinear mapping/variable coefficients, patch seams or nonlinear Z4c validation.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sbp_sha256=hashlib.sha256(a.sbp.read_bytes()).hexdigest());a.output.write_text(json.dumps(r,indent=2)+'\n');print(r)
