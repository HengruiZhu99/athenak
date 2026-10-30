#!/usr/bin/env python3
from pathlib import Path
import argparse,json,struct
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory/'output'
with (d/'puncture.checkpoint').open('rb') as f:h=f.read(72)
_,n,nf,kind,admitted,cells,surface,steps,t,dt=struct.unpack('<16s4i3Q2d',h)
u=np.fromfile(d/'puncture.failed-stage.bin',dtype='<f8');edge=u[cells*31:].reshape(-1,31)
assert len(edge)*31==surface
metric=np.zeros((len(edge),3,3))
for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
    metric[:,i,j]=metric[:,j,i]=edge[:,k+1]+(i==j)
eig=np.linalg.eigvalsh(metric)
result=dict(scri_rows=len(edge),nonpositive_metric_rows=int(np.sum(eig[:,0]<=0)),minimum_eigenvalue=float(eig.min()),first_row_eigenvalues=eig[0].tolist(),first_row_chi=float(edge[0,0]),first_row_metric=metric[0].tolist(),last_accepted_t_over_M=3*t)
(a.directory/'failed-metric-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
