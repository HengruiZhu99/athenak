"""One controlled pulse and leading mode, independent exponential propagation."""
import json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from screen import defaults,sectors
from strip_model import full_matrix

tick=time.monotonic()
p=defaults(n=128,h=32,width=512,rate=.001,kappa=0,angle=0)
L,_=full_matrix(**p);ix=sectors(p['n'])[0];A=L[np.ix_(ix,ix)]
w,v=eig(A,check_finite=False);j=np.argmax(w.real);lam=w[j];mode=v[:,j]
mode/=np.linalg.norm(mode)
n=p['n'];x=(np.arange(n)+.5)*p['h'];r=(x-n*p['h']/2)/128
pulse=np.zeros((20,n),complex);mask=abs(r)<1;pulse[16,mask]=np.cos(np.pi*r[mask]/2)**4
u=pulse.reshape(-1)[ix];u/=np.linalg.norm(u)
B=np.column_stack([u,mode])
# Exact stored matrix is used, including sub-ulp entries; sparse storage only.
S=csr_matrix(A)
out=expm_multiply(S,B,start=0,stop=50000,num=6,endpoint=True,traceA=S.diagonal().sum())
records=[]
for t,states in zip(np.linspace(0,50000,6),out):
    row={'time':float(t),'cases':[]}
    for k,label in enumerate(['compact_lapse','leading_mode']):
        full=np.zeros((20,n),complex);full.reshape(-1)[ix]=states[:,k]
        scaled=full.copy();scaled[6:16]*=p['h']
        row['cases'].append(dict(label=label,state_norm=float(np.linalg.norm(states[:,k])),weighted_state_norm=float(np.linalg.norm(scaled)),theta_max=float(np.max(abs(full[7]))),lapse_max=float(np.max(abs(full[16]))),metric_max=float(np.max(abs(full[:6]))),eigenmode_relative_error=float(np.linalg.norm(states[:,k]-np.exp(lam*t)*B[:,k])/np.linalg.norm(np.exp(lam*t)*B[:,k])) if k==1 else None))
    records.append(row)
    print(json.dumps(row),flush=True)
result=dict(parameters=p,eigenvalue=[float(lam.real),float(lam.imag)],sparse_nnz=S.nnz,records=records,wall_seconds=time.monotonic()-tick,scope='Two initial directions, not an induced-norm bound or all-perturbation stability proof; unit-normalized initial states, linear homogeneity.')
Path('results/candidate-transient-50000.json').write_text(json.dumps(result,indent=2)+'\n')
