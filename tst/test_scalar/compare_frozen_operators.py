#!/usr/bin/env python3
"""Locate backend differences in fixed interpolation and scalar operators."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidates',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing comparison is immutable')
records=[]
for d in a.candidates:
 r=dict(candidate=str(d),metadata=json.loads((d/'operator.json').read_text()),arrays=[])
 for name,cols,dtype in [('transfer.i32',16,np.int32),('transfer.f64',50,np.float64),('kernel.i32',15,np.int32),('kernel.f64',56,np.float64),('initial.f64',5,np.float64),('initial-rhs.f64',5,np.float64)]:
  x=np.memmap(a.reference/name,dtype=dtype,mode='r').reshape(-1,cols);y=np.memmap(d/name,dtype=dtype,mode='r').reshape(-1,cols)
  if x.shape!=y.shape:raise RuntimeError('Operator shapes differ')
  mismatch=np.any(x!=y,axis=1);indices=np.flatnonzero(mismatch);entry=dict(name=name,rows=len(x),different_rows=int(mismatch.sum()),first_rows=indices[:12].tolist())
  if np.issubdtype(dtype,np.integer):
   entry['examples']=[dict(row=int(i),reference=x[i].tolist(),candidate=y[i].tolist()) for i in indices[:12]]
   if name=='transfer.i32':
    xyz=np.memmap(a.reference/'transfer-positions.f64',dtype=np.float64,mode='r').reshape(-1,3);entry['positions']=[xyz[i].tolist() for i in indices[:12]]
  else:
   delta=abs(x-y);idx=np.unravel_index(delta.argmax(),delta.shape);entry.update(max_absolute_difference=float(delta[idx]),max_scaled_difference=float(np.max(delta/(1+abs(x)))),maximum_row=int(idx[0]),maximum_column=int(idx[1]))
  r['arrays'].append(entry)
 records.append(r)
result=dict(reference=str(a.reference),records=records,scope='Frozen operators and initial RHS only; no convergence admission and no threshold relaxation.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
