#!/usr/bin/env python3
"""Construct a numerical cell-centered 4/2 diagonal-norm SBP candidate.
Research algebra experiment only; no native solver integration.
"""
import json
from pathlib import Path
import numpy as np
m=6;pairs=[(i,j) for i in range(m) for j in range(i+1,m)];nv=len(pairs)+m
trace=np.array([15/8,-5/4,3/8,0,0,0]);E=-np.outer(trace,trace)
base=np.zeros((m,m+2));base[:,:m]=E/2
center={-2:1/12,-1:-2/3,1:2/3,2:-1/12}
for i in range(m):
 for j in range(m,m+2):base[i,j]=-center.get(i-j,0)
basis=[]
for i,j in pairs:
 q=np.zeros_like(base);q[i,j]=1;q[j,i]=-1;basis.append(q)
x=np.arange(m+2)+.5;matrix=[];rhs=[]
for i in range(m):
 for degree in range(3):
  row=[sum(q[i]*x**degree) for q in basis]+[0.]*m
  row[len(pairs)+i]=-degree*x[i]**(degree-1) if degree else 0
  matrix.append(row);rhs.append(-sum(base[i]*x**degree))
A=np.array(matrix);b=np.array(rhs);part=np.linalg.lstsq(A,b,rcond=1e-12)[0]
_,s,v=np.linalg.svd(A,full_matrices=True);rank=np.count_nonzero(s>1e-10);null=v[rank:].T
# Choose free coefficients to minimize deviation of norm weights from unity.
coef=np.linalg.lstsq(null[-m:],np.ones(m)-part[-m:],rcond=1e-12)[0]
sol=part+np.einsum('ij,j->i',null,coef);H=sol[-m:]
assert max(abs(np.einsum("ij,j->i",A,sol)-b))<1e-10 and min(H)>0,(H,max(abs(np.einsum("ij,j->i",A,sol)-b)))
Q=base.copy()
for z,q in zip(sol,basis):Q+=z*q
rows=Q/H[:,None]
N=40;h=np.ones(N);h[:m]=H;h[-m:]=H[::-1];D=np.zeros((N,N))
D[:m,:m+2]=rows;D[-m:,-m-2:]=-rows[::-1,::-1]
for i in range(m,N-m):
 for delta,c in center.items():D[i,i+delta]=c
tL=np.zeros(N);tL[:m]=trace;tR=tL[::-1];boundary=np.outer(tR,tR)-np.outer(tL,tL)
identity=h[:,None]*D+D.T*h[None,:]-boundary
xx=np.arange(N)+.5
errors=[float(max(abs(np.einsum("ij,j->i",D,xx**k)-(k*xx**(k-1) if k else 0)))) for k in range(3)]
assert max(abs(identity.ravel()))<1e-12 and max(errors)<1e-9
D2=np.einsum('ij,jk->ik',D,D)
energy2=h[:,None]*D2+np.einsum('ji,j,jk->ik',D,h,D)-np.einsum('ij,jk->ik',boundary,D)
assert np.max(abs(energy2))<1e-12
third=np.zeros((N-3,N))
for i in range(N-3):third[i,i:i+4]=[-1,3,-3,1]
KO=-np.einsum('ji,jk->ik',third,third)/h[:,None]
energies={}
for speed in [-1.,1.]:
 t=tL if speed>0 else tR
 evolution=-speed*D-abs(speed)*np.outer(t,t)/h[:,None]
 energy=h[:,None]*evolution+evolution.T*h[None,:]
 largest=float(np.linalg.eigvalsh(energy).max());assert largest<1e-12
 energies[str(speed)]=largest
ko_energy=h[:,None]*KO+KO.T*h[None,:];assert np.linalg.eigvalsh(ko_energy).max()<1e-12
out=dict(compatible_D2_energy_residual=float(np.max(abs(energy2))),scalar_advection_inflow_SAT_max_energy_eigenvalue=energies,KO_max_energy_eigenvalue=float(np.linalg.eigvalsh(ko_energy).max()),status='numerical first-derivative SBP candidate only; not integrated',norm_boundary_weights=H.tolist(),left_boundary_rows=rows.tolist(),left_trace=trace.tolist(),sbp_identity_max=float(max(abs(identity.ravel()))),degree_0_1_2_max_errors=errors,free_dimension=int(nv-rank),limitations=['Boundary D1 order two, interior order four','D2=D1 squared has only first-order boundary truncation; wider compatible closure not optimized','Current mesh interfaces do not automatically inherit this identity','Scalar SAT examples use homogeneous inflow, not derived Z4c constraint/gauge/physical data','No nonlinear native evolution claim'])
p=Path('docs/validation/cubed-sphere-balanced-20261013/outer-closure/cell-centered-sbp-exploration.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='left_boundary_rows'}))
