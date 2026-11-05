#!/usr/bin/env python3
"""Weighted near-scri quadratic pencils from shared C++ jet Jacobians.

Exploratory leading spatial balance, not a complete indicial theorem. At a
characteristic boundary a rank-deficient leading pencil can require subleading
terms and time dependence; finite-x eigenvalues alone must not be called roots.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from kernel import Kernel
from audit_discrete_symbol import jet_jacobians

INDEPENDENT=np.array([0,1,3,5,6,7,8,9,10,11])
WEIGHTS=np.array([1,1,0,1,1,0,1,1,0,0])
ACTIVE=np.array([0,1,2,3,4,5,6,7,9]) # B=0 is invariant in exact outer gauge.

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--background',choices=('native','frozen-outer'),default='native');ap.add_argument('--metric-value',type=float,default=1.);ap.add_argument('--outer-lapse-damping',type=float,default=1.);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);k=Kernel(a.library,separated=True,compact_end=.97,outer_lapse_damping=a.outer_lapse_damping);records=[];arrays={}
 for x in (.01,.003,.001,.0003,.0001):
  r=1-x;q,source,_=k.initial(np.array([r]));d,dd=k.outer_reference_jets(np.array([r]));mats=[]
  if a.background=='frozen-outer':
   value=np.zeros((1,10));first=value.copy();second=value.copy();value[0,[0,1]]=a.metric_value;value[0,3]=-3;value[0,6]=(1+r*r)/2;value[0,7]=-r;first[0,6]=r;first[0,7]=-1;second[0,6]=1;q,d,dd=k.expand_chart(value,first,second)
  for eps in (1e-5,3e-6):
   A,B,C=jet_jacobians(k,r,q[0],d[0],dd[0],source[0],INDEPENDENT,eps);row=x**(1-WEIGHTS[:,None]);A=row*A*x**WEIGHTS[None,:];B=row*B*x**(WEIGHTS[None,:]-1);C=row*C*x**(WEIGHTS[None,:]-2);w=WEIGHTS[None,:]
   mats.append(np.array([A-B*w+C*w*(w-1),-B+C*(2*w-1),C])[:,ACTIVE][:,:,ACTIVE])
  M0,M1,M2=mats[-1];m=len(ACTIVE);L=np.block([[np.zeros((m,m)),np.eye(m)],[-M0,-M1]]);R=np.block([[np.eye(m),np.zeros((m,m))],[np.zeros((m,m)),M2]])
  values,V=eig(L,R);finite=np.isfinite(values);roots=[]
  for j in np.where(finite)[0]:
   s=values[j];v=V[:m,j];matrix=M0+s*M1+s*s*M2;res=float(np.linalg.norm(matrix@v)/max(1,np.linalg.norm(matrix)*np.linalg.norm(v)));roots.append(dict(real=float(s.real),imag=float(s.imag),residual=res))
  roots.sort(key=lambda y:y['real']);sv={str(s):np.linalg.svd(M0+s*M1+s*s*M2,compute_uv=False).tolist() for s in (0.,.5,.75,1.,1.5,2.)}
  tag=f'x{x}';arrays[tag+'-pencil']=mats[-1];arrays[tag+'-epsilon_difference']=mats[1]-mats[0];arrays[tag+'-roots']=values;arrays[tag+'-vectors']=V
  rec=dict(distance=x,finite_roots=roots,infinite_roots=int(sum(~finite)),singular_values=sv,coefficient_norms=[float(np.linalg.norm(v)) for v in mats[-1]],relative_increment_difference=float(np.linalg.norm(mats[1]-mats[0])/np.linalg.norm(mats[-1])));records.append(rec);print(json.dumps(dict(distance=x,roots=[[y['real'],y['imag']] for y in roots],relative_increment_difference=rec['relative_increment_difference'])),flush=True)
 k.close();np.savez_compressed(a.output/'pencils.npz',**arrays)
 out=dict(outer_lapse_damping=a.outer_lapse_damping,background=a.background,metric_value=a.metric_value,rows=records,weights=WEIGHTS.tolist(),active_chart_columns=ACTIVE.tolist(),ansatz='delta q_j = amplitude_j * (1-r)^(s+weight_j); scale RHS row i by (1-r)^(1-s-weight_i). B omitted only in the exact outer B=0 invariant subsystem.',source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('audit_discrete_symbol.py'))},binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Native stationary jets or explicitly frozen outer-value jets. No evolution boundary projection, penalty, discrete stencil, or KO. A singular leading pencil requires a higher-order reduction; finite-distance generalized eigenvalues are diagnostic only.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
