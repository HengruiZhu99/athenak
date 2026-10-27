#!/usr/bin/env python3
"""Can the global skew affine update be replaced by independent block updates?

With each block's current H and boundary form fixed, P*F must be skew for an
affine-exact skew coefficient update to exist. This diagnostic changes no
derivative, geometry, state or source. Failure excludes this particular local
update, not all possible local SBP constructions with revised metric data.
"""
import argparse,json
from pathlib import Path
import numpy as np
from compatible_sbp import mm
from endpoint_sbp import shell_grid
from affine_sbp_probe import AffineConsistentGrid

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');rows=[]
for n in [10,12]:
    base=shell_grid(n);grid=AffineConsistentGrid(base);X=base.coordinates;W=base.weights
    V=np.concatenate((np.ones(W.shape+(1,)),X),axis=-1)
    derivatives=[base.derivative(V,axis,0) for axis in range(3)]
    blocks=[]
    for b,(patch,layer) in enumerate(base.blocks):
        H=W[b].ravel();v=V[b].reshape(-1,4);gram=mm(v.T*H[None,:],v)
        T=np.linalg.inv(np.linalg.cholesky(gram).T);P=mm(v,T);defects=[]
        for axis in range(3):
            target=np.zeros(4);target[axis+1]=1
            F=mm(target[None,:]-derivatives[axis][b].reshape(-1,4),T)
            S=mm(P.T*H[None,:],F)
            defects.append(float(np.max(abs(S+S.T))))
        blocks.append({'block':b,'patch':patch,'layer':layer,'compatibility_max':max(defects)})
    rows.append({'n':n,'global_compatibility_max':max(x['moment_compatibility_max'] for x in grid.diagnostic['axes']),
                 'local_compatibility_max':max(x['compatibility_max'] for x in blocks),
                 'local_failing_blocks':sum(x['compatibility_max']>1e-10 for x in blocks),'blocks':blocks})
r={'scope':__doc__,'rows':rows,'adopted':False}
(a.output/'assessment.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2))
