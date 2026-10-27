"""Verified complete trajectories for analysis, including checkpoint ancestry."""
import json,hashlib
from pathlib import Path
import numpy as np
from functools import lru_cache
from numerics import weights

@lru_cache(maxsize=8)
def first_derivative_stencil(nodes):
 r=np.array(nodes);n=len(r);indices=[];coefficients=[]
 for i,x in enumerate(r):
  start=min(max(i-3,0),n-7);idx=np.arange(start,start+7)
  indices.append(idx);coefficients.append(weights(r[idx],x,1))
 return np.array(indices),np.array(coefficients)

def physical_lapse_gradient(d):
 """Differentiate the stored chart and analytic AF factor, including r=0.

 Differencing physical alpha directly creates a false nonzero endpoint slope
 when the shrinking AF layer becomes underresolved. The analytic factor has
 exactly alpha'(0)=0 at every finite time in this chart.
 """
 r=d['r']
 idx,coeff=first_derivative_stencil(tuple(r));q=d['q'][:,6]
 qp=np.sum(coeff*(q[idx]-q[:,None]),axis=1)
 O=(1-r*r)/2;A=(1+r*r)/2;den=r*r+O**4
 g=r*r/den;gp=(2*r*O**4+4*r**3*O**3)/den**2
 alpha=A*g*np.exp(q);ap=np.exp(q)*(r*g+A*gp+A*g*qp)
 gradient=np.full_like(r,np.nan);inside=O>0
 gradient[inside]=ap[inside]/O[inside]+alpha[inside]*r[inside]/O[inside]**2
 return gradient

def trajectory(folder,seen=None):
 """Load verified checkpoint ancestry, rejecting conflicting overlap states."""
 seen=set() if seen is None else seen
 folder=folder.resolve();assert folder not in seen,'Cyclic checkpoint ancestry';seen.add(folder)
 m=json.loads((folder/'manifest.json').read_text());data={};ancestry=[]
 if 'checkpoint_parent' in m:
  parent=m['checkpoint_parent'];path=Path(parent['path'])
  assert hashlib.sha256(path.read_bytes()).hexdigest()==parent['sha256'],'Changed parent checkpoint'
  data,ancestry=trajectory(path.parent,seen)
 for path in sorted(folder.glob('profile-*.npz')):
  with np.load(path,allow_pickle=False) as z:
   t=round(float(z['t']),9);d={k:z[k].copy() for k in ['r','q','q_physical','diagnostic','rays','B','beta','gauge_speeds']}
   d['grid_jacobian']=z['grid_jacobian'].copy() if 'grid_jacobian' in z else np.ones_like(d['r'])
  if t in data:
   for k in d:assert np.array_equal(data[t][k],d[k],equal_nan=True),f'Inconsistent restart overlap: {path}, {k}'
  data[t]=d
 ancestry.append(dict(path=str(folder),manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()))
 return data,ancestry


def validate_telegraph_profile(d):
    for name in ['r','grid_jacobian','q','q_physical','B','beta','gauge_speeds','rays']:
        if not np.all(np.isfinite(d[name])):raise ValueError('Nonfinite profile array: '+name)
    if np.any(np.diff(d['r'])<=0) or np.any(d['grid_jacobian']<=0):raise ValueError('Invalid radial geometry')
    diag=d['diagnostic']
    if not np.all(np.isfinite(np.delete(diag,10,axis=1))):raise ValueError('Nonfinite non-mass diagnostic')
    # The implementation deliberately leaves only the actual-scri mass undefined.
    if not np.all(np.isfinite(diag[d['r']<1,10])):raise ValueError('Nonfinite interior mass diagnostic')
