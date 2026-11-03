"""Read-only finite/metric audit of the hash-bound BC2 endpoint."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3];D=P/'completed'
cp=ROOT/'output/boosted-refined-core/aurora/campaigns/n16-boosted-refined-core-cfl-v2-c16-evolution-segment01/output/puncture.checkpoint'
assert hashlib.sha256(cp.read_bytes()).hexdigest()==json.loads((D/'local-verification.json').read_text())['checkpoint']['sha256']
u=np.fromfile(cp,dtype='<f8',offset=208,count=31*521056).reshape(-1,31);g=np.zeros((len(u),3,3))
for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=u[:,f+1]+(i==j)
e=np.linalg.eigvalsh(g);assert e.min()>0 and np.isfinite(u).all()
v=dict(checkpoint_admitted=True,volume_metric_min_eigenvalue=float(e[:,0].min()),minimum_cell=int(np.argmin(e[:,0])),chi_min=float(u[:,0].min()),all_finite=True,maximum_abs_shift_deviation_components=np.max(abs(u[:,22:25]),axis=0).tolist(),maximum_abs_D_components=np.max(abs(u[:,25:28]),axis=0).tolist(),maximum_abs_D_minus_Lambda_components=np.max(abs(u[:,25:28]-u[:,15:18]),axis=0).tolist(),reported_cartesian_dt_over_dx=.022222222222222223,actual_cartesian_dx=.36/64,actual_dt_over_dx=.00025/(.36/64),metadata_note='Frozen report cartesian_dt_over_dx uses legacy .18/n rather than .36/(4*n); actual core geometry and measured CFL are used. This metadata issue does not change the evolution.')
(D/'endpoint-state-audit.json').write_text(json.dumps(v,indent=2)+'\n')
