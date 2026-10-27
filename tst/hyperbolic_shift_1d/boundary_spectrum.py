"""Linearized method-of-lines spectrum with eliminated boundary slots removed.

The finite-difference Jacobian is a diagnostic, not a continuum energy proof.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from evolution import Evolution
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[];arrays={}
for n in [33,65]:
 for restoring in [0.,1.,4.]:
  e=Evolution(n,a.library,restoring=restoring);base=np.zeros((n,10))
  indices=np.arange(n*10);indices=indices[~np.isin(indices,[(n-1)*10+6,(n-1)*10+7,(n-1)*10+9])]
  columns=[];eps=1e-6;start=time.monotonic()
  for j in indices:
   q=base.copy();q.flat[j]=eps;plus=e.rhs(q).ravel()[indices]
   q.flat[j]=-eps;minus=e.rhs(q).ravel()[indices];columns.append((plus-minus)/(2*eps))
  J=np.asarray(columns).T;values,vectors=np.linalg.eig(J);i=int(np.argmax(values.real))
  mode=np.zeros(n*10,complex);mode[indices]=vectors[:,i];mode=mode.reshape(n,10)
  amplitude=np.linalg.norm(mode,axis=1);key=f'n{n}_s{int(restoring)}'
  arrays[key+'_r']=e.r;arrays[key+'_eigenvalues']=values;arrays[key+'_mode']=amplitude/max(amplitude)
  rec=dict(n=n,restoring=restoring,max_real=float(values[i].real),max_imag=float(values[i].imag),
           h_times_growth=float(values[i].real*e.h),peak_radius=float(e.r[np.argmax(amplitude)]),
           fraction_outer=float(sum(amplitude[e.r>.8]**2)/sum(amplitude**2)),wall_seconds=time.monotonic()-start)
  records.append(rec);print(json.dumps(rec),flush=True)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'spectrum.json').write_text(json.dumps(records,indent=2)+'\n');np.savez_compressed(a.output/'spectrum.npz',**arrays)
