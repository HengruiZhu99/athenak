#!/usr/bin/env python3
"""Diagnose affine-field errors of mapped central and paired derivatives.

Centered variants are diagnostic controls only, not proposed replacements:
removing the paired high-mode form can restore unwanted near-null modes.
No field, source or residual is reset to its analytic expected value.
"""
import argparse,json
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');rows=[]
for n in [10,12,14]:
    grid=ShellGrid(n,radial=(0.,.45,1.),transfer_seed='zero');X=grid.coordinates;W=grid.weights
    boundary=np.zeros(W.shape,dtype=bool)
    for b,axis,side in grid.outer:
        index=[b,slice(None),slice(None),slice(None)];index[axis+1]=-1 if side else 0;boundary[tuple(index)]=True
    for rate_sign in [0,-1]:
        gradient=np.stack([grid.derivative(-X,j,rate_sign) for j in range(3)],axis=-2)
        divergence=np.einsum('...ii->...',gradient,optimize=False)
        h=gradient+np.swapaxes(gradient,-1,-2)-(2/3)*np.eye(3)*divergence[...,None,None]
        for connection_sign in [0,1]:
            dh=np.stack([grid.derivative(h,j,connection_sign) for j in range(3)],axis=-3)
            rate=np.einsum('...jij->...i',dh,optimize=False)-.5*np.einsum('...ijj->...i',dh,optimize=False)
            rows.append({'n':n,'rate_derivative_sign':rate_sign,'connection_derivative_sign':connection_sign,
                         'affine_gradient_max':float(np.max(abs(gradient+np.eye(3)))),
                         'metric_rate_max':float(np.max(abs(h))),
                         'metric_rate_scri_max':float(np.max(abs(h[boundary]))),
                         'connection_rate_all_node_rms':float(np.sqrt(np.sum(W[...,None]*rate**2)/np.sum(W))),
                         'connection_rate_scri_max':float(np.max(abs(rate[boundary])))})
result={'scope':__doc__,'rows':rows,'interpretation':'Uniform physical affine fields are represented without puncture pinning; current curved derivative coefficients do not differentiate them exactly. This is a spatial consistency failure to repair, not a reason to subtract a known physical RHS.'}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
