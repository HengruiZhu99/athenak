"""Independent short canonical action with exact constant similarity for units."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.sparse import load_npz,diags
from scipy.sparse.linalg import expm_multiply
w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();meta=json.loads((w/'spatialnorm-cache0.0001-metadata.json').read_text());N=meta['points'];h=meta['spacing'];J=load_npz(w/'spatialnorm-projected-J20.npz');data=np.load(w/'spatialnorm-projected-krylov-m50-80-h0.1-t0.05.npz');d=np.ones((N,20));d[:,[0,1,2,3,4,5,16,17,18,19]]=1/h;d=d.ravel();B=data['values'][0];A=diags(d)@J@diags(1/d);rng=np.random.default_rng(228);x=rng.normal(size=len(d));similarity_err=float(np.linalg.norm((A@(d*x))/d-J@x)/np.linalg.norm(J@x));assert similarity_err<1e-13
result={'scope':'independent short Taylor canonical exp(tJ), exact constant diagonal similarity only; no eigenvalue/runtime/norm change','formula':'A=TJT^-1; exp(tJ)B=T^-1 exp(tA) T B','T_configuration_fields_1_over_h':[0,1,2,3,4,5,16,17,18,19],'T_momenta':1,'random_matvec_similarity_relative_l2':similarity_err,'matrix_sha256':sha(w/'spatialnorm-projected-J20.npz'),'checks':[]};t0=time.monotonic()
for t in [.025,.05]:
 start=time.monotonic();y=expm_multiply(t*A,d[:,None]*B,traceA=float(t*A.diagonal().sum()))/d[:,None];k=data['values'][np.argmin(abs(data['times']-t))];err=(np.linalg.norm(y-k,axis=0)/np.linalg.norm(y,axis=0)).tolist();result['checks'].append({'time':t,'relative_state_l2':err,'seconds':time.monotonic()-start});assert max(err)<1e-10;print(result['checks'][-1],flush=True)
result['seconds']=time.monotonic()-t0;(w/'short-canonical-validation.json').write_text(json.dumps(result,indent=2)+'\n');print('done',result['seconds'])
