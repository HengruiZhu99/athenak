#!/usr/bin/env python3
"""Node-centred diagonal-norm SBP6/3, constructed and audited independently."""
import json
from pathlib import Path
import numpy as np
m=6;pairs=[(i,j) for i in range(m) for j in range(i+1,m)];base=np.zeros((m,m+3));base[0,0]=-.5
center={-3:-1/60,-2:3/20,-1:-3/4,1:3/4,2:-3/20,3:1/60}
for i in range(m):
 for j in range(m,m+3):base[i,j]=-center.get(i-j,0)
basis=[]
for i,j in pairs:
 q=np.zeros_like(base);q[i,j]=1;q[j,i]=-1;basis.append(q)
x=np.arange(m+3);A=[];b=[]
for i in range(m):
 for degree in range(4):
  row=[sum(q[i]*x**degree) for q in basis]+[0.]*m;row[len(pairs)+i]=-degree*x[i]**(degree-1) if degree else 0
  A.append(row);b.append(-sum(base[i]*x**degree))
A=np.array(A);b=np.array(b);part=np.linalg.lstsq(A,b,rcond=1e-12)[0];_,s,v=np.linalg.svd(A,full_matrices=True);rank=np.count_nonzero(s>1e-10);null=v[rank:].T
sol=part;H=sol[-m:];Q=base.copy()
for z,q in zip(sol,basis):Q+=z*q
rows=Q/H[:,None];N=80;h=np.ones(N);h[:m]=H;h[-m:]=H[::-1];D=np.zeros((N,N));D[:m,:m+3]=rows;D[-m:,-m-3:]=-rows[::-1,::-1]
for i in range(m,N-m):
 for d,c in center.items():D[i,i+d]=c
E=np.zeros_like(D);E[0,0]=-1;E[-1,-1]=1;identity=np.max(abs(h[:,None]*D+D.T*h[None,:]-E));poly=max(np.max(abs(D@np.arange(N)**k-(k*np.arange(N)**(k-1) if k else 0))) for k in range(4))
B=np.zeros((N-3,N))
for i in range(N-3):B[i,i:i+4]=[-1,3,-3,1]
KO=-B.T@B/h[:,None];ke=float(np.linalg.eigvalsh(h[:,None]*KO+KO.T*h[None,:]).max());assert identity<1e-10 and poly<1e-6 and min(h)>0 and ke<1e-10
path=Path('tst/test_scalar/sbp63.hpp');path.write_text('#pragma once\nnamespace radial_sbp {\nconstexpr int m=6;\nconstexpr double H[6]={'+','.join(f'{z:.17g}' for z in H)+'};\nconstexpr double left[6][9]={'+','.join('{'+','.join(f'{z:.17g}' for z in r)+'}' for r in rows)+'};\n}\n')
result=dict(identity_max=float(identity),polynomial_error=float(poly),minimum_norm=float(min(h)),KO_max_energy_eigenvalue=ke,passed=True,scope='Node-centred SBP first derivative and scalar KO algebra only; no variable-coefficient or coupled interface energy admission')
Path('docs/validation/test-scalar-20261016/radial-oracle/sbp-algebra.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
