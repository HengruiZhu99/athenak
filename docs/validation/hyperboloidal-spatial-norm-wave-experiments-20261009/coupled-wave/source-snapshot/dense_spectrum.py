from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.io import mmread
from scipy.linalg import eigvals
w=Path(__file__).resolve().parent
result=[]
for closure in ['ray','mls']:
 p=w/f'N16-span2.2-{closure}-ko0.1.mtx';A=mmread(p).toarray();begin=time.monotonic();ev=eigvals(A,check_finite=False);order=np.argsort(ev.real)[::-1];ev=ev[order];r={'closure':closure,'unknowns':len(ev),'scope':'full dense eigenvalue spectrum','matrix_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':time.monotonic()-begin,'positive_real_count_gt_1e-8':int((ev.real>1e-8).sum()),'max_real':float(ev.real.max()),'rightmost':[{'real':float(v.real),'imag':float(v.imag)} for v in ev[:12]]};np.savez_compressed(w/f'N16-{closure}-full-spectrum.npz',eigenvalues=ev);result.append(r);(w/'dense-spectrum.json').write_text(json.dumps(result,indent=2)+'\n');print(r,flush=True)
