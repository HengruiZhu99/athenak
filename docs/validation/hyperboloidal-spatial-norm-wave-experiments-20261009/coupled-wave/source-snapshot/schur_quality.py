from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.io import mmread
from scipy.linalg import schur,eigvals
w=Path(__file__).resolve().parent;results=[]
for closure in ['ray','mls']:
 p=w/f'N16-span2.2-{closure}-ko0.1.mtx';A=mmread(p).tocsr();begin=time.monotonic();T,Z=schur(A.toarray(),output='real',check_finite=False);TZ=Z@T;AZ=A@Z;delta=AZ-TZ;ortho=Z.T@Z-np.eye(len(Z));ev=eigvals(T,check_finite=False);old=np.load(w/f'N16-{closure}-full-spectrum.npz')['eigenvalues'];r={'closure':closure,'matrix_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'full_unknowns':len(Z),'schur_backward_residual_fro_over_A_fro':float(np.linalg.norm(delta,'fro')/np.linalg.norm(A.data)),'orthogonality_residual_fro_over_sqrtN':float(np.linalg.norm(ortho,'fro')/np.sqrt(len(Z))),'max_schur_eigenvalue_real':float(ev.real.max()),'max_dense_eigenvalue_real':float(old.real.max()),'positive_real_count_gt_1e-8':int((ev.real>1e-8).sum()),'seconds':time.monotonic()-begin};results.append(r);(w/'schur-quality.json').write_text(json.dumps(results,indent=2)+'\n');print(r,flush=True)
