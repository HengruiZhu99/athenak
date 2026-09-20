"""Recompute saved finite-matrix results from the parameter fields in their JSON.

Usage: OPENBLAS_NUM_THREADS=1 python3 reproduce_selected.py coefficient-isolation.json
Requires this folder to reside anywhere within the AthenaK review repository.
Outputs a separate *.reproduced.json; never overwrites retained evidence.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from fourier_boundary import full_matrix

p=argparse.ArgumentParser();p.add_argument('table',type=Path);args=p.parse_args()
allowed=set(inspect.signature(full_matrix).parameters)
rows=json.loads(args.table.read_text());out=[]
for row in rows:
    params={k:v for k,v in row.items() if k in allowed}
    L,_=full_matrix(**params);eig=np.linalg.eigvals(L);value=eig[np.argmax(eig.real)]
    prior=row.get('gamma',row.get('max_real'))
    out.append(dict(parameters=params,gamma=float(value.real),omega=float(value.imag),
                    prior_gamma=prior,difference=float(value.real-prior)))
result=args.table.with_suffix('.reproduced.json');result.write_text(json.dumps(out,indent=2)+'\n')
print(f'{len(rows)} cases; maximum gamma difference {max(abs(r["difference"]) for r in out):.3e}; {result}')
