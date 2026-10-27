#!/usr/bin/env python3
"""Independent action-equivalence check of the cached full tensor variation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from amr_shell_sbp_probe import AMRShellGrid
from compatible_tensor_gauge import TensorGauge
from cached_tensor_gauge import CachedTensorGauge

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
grid=AMRShellGrid(10);X=grid.coordinates;W=grid.weights
g=np.broadcast_to(np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]]),W.shape+(3,3)).copy()
g[...,0,0]+=.08*np.sin(X[...,0]+X[...,2]);g[...,1,1]+=.05*np.cos(2*X[...,1]);g[...,0,2]+=.03*np.sin(X[...,1]);g[...,2,0]=g[...,0,2]
old=TensorGauge(grid,g);new=CachedTensorGauge(grid,g);rng=np.random.default_rng(271130);rows=[]
for name,h in [('general_symmetric',rng.normal(size=g.shape)),('isotropic',np.broadcast_to(np.eye(3),g.shape).copy())]:
    h=(h+np.swapaxes(h,-1,-2))/2
    t=time.monotonic();expected=old.variation(h);old_time=time.monotonic()-t
    t=time.monotonic();actual=new.variation(h);new_time=time.monotonic()-t
    delta=actual-expected
    rows.append({'case':name,'maximum':float(np.max(abs(delta))),'scaled_rms':float(np.sqrt(np.sum(W[...,None]*delta**2))/max(1,np.sqrt(np.sum(W[...,None]*expected**2)))),'uncached_seconds':old_time,'cached_seconds':new_time})
result={'scope':__doc__,'cells':grid.cells,'coefficient_bytes':new.local.nbytes+new.principal.nbytes,'rows':rows,'gates':{'same_full_variation':all(r['scaled_rms']<1e-12 and r['maximum']<1e-9 for r in rows)}}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
