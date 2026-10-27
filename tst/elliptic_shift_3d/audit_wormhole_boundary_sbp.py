#!/usr/bin/env python3
"""Separate geometry/source sampling and solved-shift contributions at scri."""
import argparse,hashlib,json,sys,warnings
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from compatible_tensor_gauge import TensorGauge

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
root=Path(__file__).resolve().parents[2];ref=root/'docs/elliptic-shift-3d/evidence/takeover-spherical-reference-20261027'
receipt=json.loads((ref/'receipt.json').read_text())
for name in ['wormhole_data.py','wormhole_isotropic.py']:
    if hashlib.sha256((ref/'source'/name).read_bytes()).hexdigest()!=receipt['source_hashes'][name]:raise ValueError('reference source mismatch')
sys.path.insert(0,str(ref/'source'));from wormhole_isotropic import IsotropicWormhole
profile=np.load(ref/'short-n2049/profile-0001.npz')
with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
    warnings.simplefilter('always');constructor=IsotropicWormhole(mass=1/3,quadrature_order=96)
rows=[]
for radial in [(0.,.45,1.),(0.,.45,.8,1.),(0.,.45,.55,.60,.65,.70,.75,1.)]:
    grid=ShellGrid(10,radial=radial,transfer_seed='zero');X=grid.coordinates;W=grid.weights;r=np.linalg.norm(X,axis=-1)
    boundary=np.zeros(r.shape,dtype=bool)
    for b,axis,side in grid.outer:
        idx=[b,slice(None),slice(None),slice(None)];idx[axis+1]=-1 if side else 0;boundary[tuple(idx)]=True
    r[boundary]=1;unique,index=np.unique(r,return_inverse=True)
    with warnings.catch_warnings(record=True) as data_warnings,np.errstate(under='warn'):
        warnings.simplefilter('always');q,_,_,metadata=constructor.data(unique)
    alpha=((1+unique**2)/2+q[:,6])[index].reshape(r.shape);arr=q[:,2][index].reshape(r.shape);n=X/r[...,None]
    C=-3*(alpha*arr)[...,None,None]*(n[..., :,None]*n[...,None,:]-np.eye(3)/3)
    metric=np.broadcast_to(np.eye(3),r.shape+(3,3)).copy();gauge=TensorGauge(grid,metric)
    beta_ref=np.interp(r,profile['r'],profile['beta'])[...,None]*n
    ref_h=gauge.rate(beta_ref);affine_h=gauge.rate(-X)
    source=gauge.variation(C);shift_ref=gauge.variation(ref_h);reference=source+shift_ref
    affine=gauge.variation(affine_h)
    row={'cells':grid.cells,'partition_s':list(radial),'constructor_underflow_warning_count':len(caught)+len(data_warnings),
         'physical_source_scri_max':float(np.max(abs(source[boundary]))),
         'reference_shift_scri_max':float(np.max(abs(shift_ref[boundary]))),
         'reference_total_rate_scri_max':float(np.max(abs(reference[boundary]))),
         'reference_total_rate_bulk_rms':float(np.sqrt(np.sum(W[~boundary,None]*reference[~boundary]**2)/np.sum(W[~boundary]))),
         'reference_metric_rate_scri_max':float(np.max(abs(ref_h[boundary]))),
         'affine_metric_rate_scri_max':float(np.max(abs(affine_h[boundary]))),
         'affine_connection_rate_scri_max':float(np.max(abs(affine[boundary])))}
    rows.append(row)
result={'scope':__doc__,'rows':rows,'interpretation':'Diagnostic only; changing shell partition is not a fix unless a fresh solve and independent boundary checks pass.'}
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
