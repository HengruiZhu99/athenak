"""Compatible compact-pulse finite-time checks of the two all-q controls."""
import json,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from configuration_boundary_model import matrix,QP,PP
from corner_model import build
n=12;N=n*n;h=32.;x,y=np.meshgrid(np.arange(n),np.arange(n));r=np.hypot(x-5.5,y-5.5)
pulse=np.zeros((n,n));inside=r<1.25;pulse[inside]=np.exp(-1/(1-(r[inside]/1.25)**2))
u=np.zeros(20*N);u[16*N:17*N]=1e-8*pulse.ravel()
boundary=(x==0)|(x==n-1)|(y==0)|(y==n-1);bp=np.flatnonzero(boundary)
_,aux=build(n=n,h=h,return_aux=True);V=aux['V'];db=aux['dbs']
records=[]
for mode in ['dirichlet','outgoing']:
 L=matrix(n,h,mode);initial_rate=(V@u).reshape(20,N)
 compatibility=max(abs(initial_rate[f,bp]).max() for f in QP)
 assert compatibility==0
 tic=time.monotonic();states=expm_multiply(csr_matrix(L),u,start=0,stop=5000,num=21,endpoint=True)
 rows=[]
 for t,state in zip(np.linspace(0,5000,21),states):
  a=state.reshape(20,N);weighted=a.copy();weighted[6:16]*=h
  rows.append(dict(time=float(t),weighted_norm=float(np.linalg.norm(weighted)),theta_max=float(max(abs(a[7]))),metric_max=float(abs(a[1:6]).max()),q_boundary_max=float(abs(a[QP][:,bp]).max())))
 rec=dict(mode=mode,initial_boundary_q_rate=compatibility,initial_boundary_q=float(abs(u.reshape(20,N)[QP][:,bp]).max()),wall_seconds=time.monotonic()-tic,rows=rows)
 records.append(rec);print(mode,rec['wall_seconds'],rows[-1],flush=True)
Path('configuration-transients.json').write_text(json.dumps(records,indent=2)+'\n')
