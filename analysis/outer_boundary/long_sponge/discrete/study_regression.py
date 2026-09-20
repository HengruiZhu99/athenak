"""Checks the analysis-only parameterization against the preserved operator."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from strip_model import full_matrix
from strip_reference import full_matrix as reference
from screen import sectors

a=dict(n=16,h=32,angle_y=np.pi/8,degree=1,damping=True,sponge_rate=.02,sponge_cells=4)
L,_=full_matrix(**a);R,_=reference(**a)
x={'default_bitwise_match':bool(np.array_equal(L,R)),'max_difference':float(np.max(abs(L-R)))}
params=dict(a);params['sponge_rate']=.003
A,_=full_matrix(**params,kappa=.03);Z,_=full_matrix(**params,kappa=0);K,_=full_matrix(**params,kappa=.1)
x['kappa_affinity_error']=float(np.max(abs(A-(.3*K+.7*Z))))
ix=sectors(16);full=eigvals(L);part=np.concatenate([eigvals(L[np.ix_(i,i)]) for i in ix])
x['parity_max_real_difference']=float(abs(max(full.real)-max(part.real)))
x['parity_cross_max']=float(np.max(abs(L[np.ix_(ix[0],ix[1])])))

# Isolate variable-kappa volume source against its defining local equations.
p=dict(a,sponge_rate=0,kappa_taper_cells=6.,kappa_taper_end_cells=2.)
_,taper=full_matrix(**p,kappa=.1);_,zero=full_matrix(**p,kappa=0.)
dv=taper['volume']-zero['volume'];n=a['n'];depth=np.minimum(np.arange(n)+.5,n-np.arange(n)-.5)
s=np.clip((6-depth)/4,0,1);kap=.1*(1-s**3*(10-15*s+6*s*s))
err=[]
for j in range(n):
    err.extend([abs(dv[6*n+j,7*n+j]-kap[j]),abs(dv[7*n+j,7*n+j]+2*kap[j])])
    for f in [13,14,15]:err.append(abs(dv[f*n+j,f*n+j]+2*kap[j]))
x['taper_local_source_error']=float(max(err))
_,oldg=full_matrix(**a,kappa=0.)
low=dict(a,lapse_damping=.01)
_,newg=full_matrix(**low,kappa=0.,eta=.02)
diff=newg['volume']-oldg['volume'];expected=np.zeros_like(diff)
for f,delta in [(16,.09),(17,1.98),(18,1.98),(19,1.98)]:
    expected[f*n:(f+1)*n,f*n:(f+1)*n]=np.eye(n)*delta
x['gauge_source_parameterization_error']=float(np.max(abs(diff-expected)))
x['exact_zero']=bool(np.all(L@np.zeros(L.shape[0])==0))
assert x['default_bitwise_match'] and x['exact_zero']
assert x['kappa_affinity_error']<1e-15 and x['taper_local_source_error']<1e-15
assert x['gauge_source_parameterization_error']<1e-15
assert x['parity_max_real_difference']<1e-12
Path('default-regression.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
