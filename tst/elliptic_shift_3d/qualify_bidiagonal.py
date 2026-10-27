#!/usr/bin/env python3
"""Independent SVD controls, including a stationary nonzero residual."""
import argparse,json
from pathlib import Path
import numpy as np
from bounded_bidiagonal import investigate
from compatible_sbp import mm

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
rng=np.random.default_rng(8127);U=np.linalg.qr(rng.normal(size=(35,35)))[0];V=np.linalg.qr(rng.normal(size=(24,24)))[0]
A=mm(U[:,:24]*np.geomspace(1,1e-5,24),V.T);M=np.zeros((35,35));M[:,:24]=A
truth=rng.normal(size=24);rows=[]
for inconsistent in [False,True]:
    rhs=mm(A,truth)+(.25*U[:,30] if inconsistent else 0)
    x,w,r=investigate(lambda v:mm(M,v),lambda v:mm(M.T,v),rhs,max_iterations=24)
    independent=np.linalg.lstsq(A,rhs,rcond=None)[0];residual=rhs-mm(M,x)
    rows.append({'inconsistent':inconsistent,'independent_solution_error':float(np.linalg.norm(x[:24]-independent)),
                 'residual_norm':float(np.linalg.norm(residual)),
                 'normal_residual_norm':float(np.linalg.norm(mm(M.T,residual))),
                 'reported_converged':r['converged'],'diagnostic':r['history'][-1]})
gates={'consistent_solution':rows[0]['reported_converged'] and rows[0]['independent_solution_error']<1e-8,
       'stationary_residual_not_accepted':not rows[1]['reported_converged'] and abs(rows[1]['residual_norm']-.25)<1e-12 and rows[1]['normal_residual_norm']<1e-12,
       'known_condition':all(abs(r['diagnostic']['conditioning_lower_bound']/1e5-1)<1e-8 for r in rows)}
result={'scope':__doc__,'rows':rows,'gates':gates}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if all(gates.values()) else 1)
