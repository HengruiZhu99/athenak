"""Full regular-center one-step Jacobian around discrete Minkowski equilibrium.

This is a frozen linearized numerical boundary audit, not a time evolution or
nonlinear stability proof. The constrained step map includes stage projection.
"""
import argparse,json
from pathlib import Path
import numpy as np
from candidate import NonadvectiveEvolution
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=25);p.add_argument('--advective',action='store_true');p.add_argument('--reduction-damping',type=float,default=0.);p.add_argument('--disabled-trace',type=int);p.add_argument('--cfl',type=float,default=.025);a=p.parse_args()
rows=[]
for eta,sigma in ([(1,.1)] if a.advective else [(0,0),(1,.1)]):
 if a.advective:
  from evolution import Evolution
  e=Evolution(a.n,a.library,shift_gauge='moving-puncture',gamma_coefficient=4,restoring=4,inner_z4_damping=8);base=np.zeros((a.n,10))
 else:
  e=NonadvectiveEvolution(a.n,a.library,puncture=False,reference_damping=eta,reference_rate=sigma,boundary_reduction_damping=a.reduction_damping,disabled_trace=a.disabled_trace);base=e.initial()
 dt=a.cfl*e.h
 indices=[(i,f) for i in range(a.n) for f in range(10)]+([] if a.advective else [(a.n-1,f) for f in range(10,13)])
 def step(q):
  e.compensation=np.zeros_like(q);return e.step(q,dt)
 baseline=step(base);matrices=[]
 for eps in [1e-6,5e-7]:
  matrix=np.empty((len(indices),len(indices)))
  for col,(i,f) in enumerate(indices):
   plus=base.copy();minus=base.copy();plus[i,f]+=eps;minus[i,f]-=eps
   delta=(step(plus)-step(minus))/(2*eps)
   matrix[:,col]=[delta[j,k] for j,k in indices]
  ev,vectors=np.linalg.eig(matrix);matrices.append(matrix)
  j=int(np.argmax(abs(ev)));mode=np.zeros_like(base,dtype=complex)
  for (i,f),value in zip(indices,vectors[:,j]):mode[i,f]=value
  np.savez_compressed(a.output.with_name(a.output.stem+f"-eta{eta}-eps{eps:g}.npz"),r=e.r,matrix=matrix,eigenvalues=ev,dominant_mode=mode,dt=dt)
  rows.append(dict(n=a.n,eta=eta,sigma=sigma,reduction_damping=a.reduction_damping,disabled_trace=a.disabled_trace,dt=dt,epsilon=eps,spectral_radius=float(max(abs(ev))),unstable_eigenvalues=[dict(real=float(z.real),imag=float(z.imag),modulus=float(abs(z))) for z in ev if abs(z)>1+1e-6]))
 rows[-1]['matrix_relative_difference']=float(np.linalg.norm(matrices[1]-matrices[0])/np.linalg.norm(matrices[0]))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(rows=rows,scope=__doc__),indent=2)+'\n');print(json.dumps(rows,indent=2))
