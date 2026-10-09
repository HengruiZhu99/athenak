"""Independent canonical Taylor checks of the non-authoritative Arnoldi pilot."""
from pathlib import Path
import json,time
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply
w=Path(__file__).resolve().parent;g='production';J=load_npz(w/f'{g}-projected-J20.npz');v=np.load(w/f'{g}-projected-krylov-m50-80-h0.1-t0.05.npz');B=v['values'][0];res={'gauge':g,'checks':[]};saved={};t0=time.monotonic()
for t in [.025,.05]:
 begin=time.monotonic();y=expm_multiply(t*J,B,traceA=float(t*J.diagonal().sum()));idx=np.argmin(abs(v['times']-t));k=v['values'][idx];r={'time':t,'relative_state_l2':(np.linalg.norm(y-k,axis=0)/np.linalg.norm(y,axis=0)).tolist(),'linf_difference':abs(y-k).max(axis=0).tolist(),'canonical_Taylor_seconds':time.monotonic()-begin};res['checks'].append(r);saved[str(t)]=y;print(r,flush=True)
res['total_seconds']=time.monotonic()-t0;(w/'production-krylov-pilot-independent-validation.json').write_text(json.dumps(res,indent=2)+'\n');np.savez_compressed(w/'production-krylov-pilot-canonical-vectors.npz',**saved);print('done',res,flush=True)
