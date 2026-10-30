#!/usr/bin/env python3
"""Locate nonpositive metric tensors in a captured rejected volume candidate."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('coordinates',type=Path);a=p.parse_args()
r=json.loads((a.directory/'output/puncture.report.json').read_text());n=r['cells'];capture=a.directory/'output/puncture.failed-stage.bin'
u=np.fromfile(capture,dtype='<f8');v=u[:31*n].reshape(n,31);s=u[31*n:].reshape(-1,31)
def check(v):
    g=np.zeros((len(v),3,3))
    for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=v[:,k+1]+(i==j)
    finite=np.isfinite(g).all(axis=(1,2));eig=np.full((len(v),3),np.nan)
    eig[finite]=np.linalg.eigvalsh(g[finite]);return eig,finite
ve,vf=check(v);se,sf=check(s);bad=np.where((ve[:,0]<=0)|~vf)[0]
coords=np.genfromtxt(a.coordinates,delimiter=',',names=True);assert len(coords)>=n
result=dict(volume_points=n,volume_nonfinite_metric_points=int((~vf).sum()),volume_nonpositive_metric_points=int(np.sum(ve[:,0]<=0)),volume_min_metric_eigenvalue=float(np.nanmin(ve)),bad_indices=bad.tolist(),bad_radii_range=[float(np.min(coords['r'][bad])),float(np.max(coords['r'][bad]))] if len(bad) else None,bad_radii_unique=np.unique(np.round(coords['r'][bad],12)).tolist(),scri_nonpositive_metric_points=int(np.sum(se[:,0]<=0)),scri_min_metric_eigenvalue=float(np.nanmin(se)),last_accepted_t_over_M=r['t_over_M'],coordinate_file_sha256=hashlib.sha256(a.coordinates.read_bytes()).hexdigest(),failed_stage_sha256=hashlib.sha256(capture.read_bytes()).hexdigest())
(a.directory/'failed-metric-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='bad_indices'},indent=2))
