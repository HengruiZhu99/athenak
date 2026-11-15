from pathlib import Path
import json,sys
import numpy as np
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'analysis-source/vis/python'))
from check_scri_weyl import curvature
from scri_geometry import spacetime_derivatives
from scri_news import stencil
from scri_angular import spin_basis
cache=root/'metric-cache';grid=np.load(cache/'grid.npz');x=grid['x'];w=grid['weights'];a=np.load(cache/'metric.npy',mmap_mode='r');records=json.loads((cache/'records.json').read_text());times=np.array([v['time'] for v in records]);basis,labels=spin_basis(x,8);y20=basis[:,labels.index((2,0))]
mu=x[:,0];sn=np.sqrt(1-mu*mu);e=(x*mu[:,None]-np.array([1.,0,0]))/sn[:,None];f=np.column_stack((np.zeros(len(x)),-x[:,2],x[:,1]))/sn[:,None]
rows=[]
for q in range(3,len(times)-3,10):
 window,center=stencil(times,q);g,dg,ddg=spacetime_derivatives(a[window],times[window],center);c,ric,inv=curvature(g,dg,ddg)
 n=np.einsum('qab,qb->qa',inv,np.column_stack((np.zeros(len(x)),-x)))
 gamma=g[:,1:,1:];ee=np.einsum('qi,qij,qj->q',e,gamma,e);ef=np.einsum('qi,qij,qj->q',e,gamma,f);ff=np.einsum('qi,qij,qj->q',f,gamma,f)
 dyad=np.column_stack((np.zeros(len(x)),(e/np.sqrt(ee)[:,None]-1j*(f-(ef/ee)[:,None]*e)/np.sqrt(ff-ef*ef/ee)[:,None])/np.sqrt(2)))
 value=np.einsum('qa,qb,qc,qd,qabcd->q',n,dyad,n,dyad,c)
 mode=np.sum(w*value*y20.conj())
 rows.append(dict(native_time_M0=3*float(times[q]),max_abs=float(np.max(abs(value))),rms=float(np.sqrt(np.sum(w*abs(value)**2)/sum(w))),native_20_real=float(mode.real),native_20_imag=float(mode.imag)))
result=dict(kind='Conformal Weyl numerator at scri; zero target, not Psi4 waveform',projection='Cbar_abcd n^a mbar^b n^c mbar^d; n=grad(Omega), native orthonormal angular dyad',samples=rows,max_abs=max(r['max_abs'] for r in rows),max_rms=max(r['rms'] for r in rows))
(root/'weyl-numerator-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2));print('last',rows[-1])
