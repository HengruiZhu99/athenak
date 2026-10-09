from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply
w=Path(__file__).resolve().parent;assert json.loads((w/'diagnostic-gate.json').read_text())['all_pass'];assert json.loads((w/'spatialnorm-pre-pilot-gate.json').read_text())['all_consistency_checks_pass'];A=load_npz(w/'spatialnorm-projected-J20.npz');d=np.load(w/'spatialnorm-validation-vectors.npz');names=['gauge_pulse','shell_random'];v=np.column_stack([d[n]/np.linalg.norm(d[n]) for n in names]);t=time.monotonic();result=expm_multiply(A,v,start=0,stop=.05,num=3,traceA=float(A.diagonal().sum()));np.savez_compressed(w/'spatialnorm-canonical-short.npz',times=[0,.025,.05],names=names,values=result);out={'semantics':'SciPy expm_multiply independent short action of fresh P J22 Lift; no full PDE or long native claim','seconds':time.monotonic()-t,'matrix_sha256':hashlib.sha256((w/'spatialnorm-projected-J20.npz').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'finite':bool(np.isfinite(result).all())};(w/'canonical-short.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
