#!/usr/bin/env python3
"""Saved-field boundary decomposition; no source/field correction or new solve."""
import argparse,hashlib,json,sys,warnings
from pathlib import Path
import numpy as np
from endpoint_sbp import shell_grid
from affine_sbp_probe import AffineConsistentGrid
from compatible_tensor_gauge import TensorGauge


def run():
    root=Path(__file__).resolve().parents[2];base=root/'docs/elliptic-shift-3d/evidence'
    ref=base/'takeover-spherical-reference-20261027';receipt=json.loads((ref/'receipt.json').read_text())
    for name in ['wormhole_data.py','wormhole_isotropic.py']:
        if hashlib.sha256((ref/'source'/name).read_bytes()).hexdigest()!=receipt['source_hashes'][name]:raise ValueError('reference mismatch')
    sys.path.insert(0,str(ref/'source'));from wormhole_isotropic import IsotropicWormhole
    rows=[]
    for n in [10,12]:
        case=base/f'compatible-endpoint-fd-wormhole-n{n}-continued-20261027'
        profile=np.load(case/'result/profile.npz');assessment=json.loads((case/'result/assessment.json').read_text())
        original_grid=shell_grid(n,radial=tuple(assessment['radial_partition_s']));grid=AffineConsistentGrid(original_grid)
        if not np.array_equal(grid.coordinates,profile['coordinates']):raise ValueError('saved grid mismatch')
        X=grid.coordinates;W=grid.weights;boundary=profile['boundary'];radius=profile['radius'];normal=X/radius[...,None]
        unique,index=np.unique(radius,return_inverse=True)
        with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
            warnings.simplefilter('always');q,_,_,_=IsotropicWormhole(mass=1/3,quadrature_order=96).data(unique)
        alpha=((1+unique**2)/2+q[:,6])[index].reshape(W.shape);arr=q[:,2][index].reshape(W.shape)
        C=-3*(alpha*arr)[...,None,None]*(normal[..., :,None]*normal[...,None,:]-np.eye(3)/3)
        metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
        old=TensorGauge(original_grid,metric);corrected=TensorGauge(grid,metric)
        for label,beta in [('saved_solve',profile['beta']),('sampled_1d',profile['reference_beta'][...,None]*normal)]:
            h0=C+old.rate(beta);h= C+corrected.rate(beta)
            uncorrected=old.variation(h0);rate=corrected.variation(h)
            shift_part=corrected.variation(h-h0)
            metric_part=corrected.variation(h0)-uncorrected
            if label=='saved_solve' and np.max(abs(rate-profile['actual_metric_rate']))>1e-7:raise ValueError('saved actual rate mismatch')
            maxb=lambda x:float(np.max(abs(x[boundary])))
            area=[];nr=[];tr=[];classes=[]
            for b,_,_ in grid.outer:
                loc=grid.locals[b];u,v=np.meshgrid(2*loc['x']-1,2*loc['x']-1,indexing='ij')
                area.append(loc['H'][:,None]*loc['H'][None,:]*4/(1+u*u+v*v)**1.5)
                nrate=np.sum(rate[b,-1]*normal[b,-1],axis=-1);nr.append(nrate)
                tr.append(np.sum((rate[b,-1]-nrate[...,None]*normal[b,-1])**2,axis=-1))
                j,k=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
                classes.append(((j==0)|(j==n-1)).astype(int)+((k==0)|(k==n-1)).astype(int))
            area=np.array(area);nr=np.array(nr);tr=np.array(tr);classes=np.array(classes)
            vector_squared=nr*nr+tr
            rows.append({'n':n,'field':label,'scri_max':maxb(rate),
                'uncorrected_operator_same_field_scri_max':maxb(uncorrected),
                'shift_coefficient_change_scri_max':maxb(shift_part),
                'metric_coefficient_change_scri_max':maxb(metric_part),
                'decomposition_closure_max':float(np.max(abs(rate-uncorrected-shift_part-metric_part))),
                'surface_normal_mean':float(np.sum(area*nr)/np.sum(area)),
                'surface_normal_rms':float(np.sqrt(np.sum(area*nr*nr)/np.sum(area))),
                'surface_tangential_rms':float(np.sqrt(np.sum(area*tr)/np.sum(area))),
                'surface_region_rms':{name:float(np.sqrt(np.sum(area[classes==i]*vector_squared[classes==i])/np.sum(area[classes==i]))) for i,name in enumerate(['face_interior','face_edge','face_corner'])},
                'constructor_underflow_warnings':len(caught)})
    return {'scope':__doc__,'rows':rows,'interpretation':'Terms can cancel. Evaluating another operator on a saved field is not a solved baseline or a proposed subtraction. All physical runs retain their original residuals.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run();(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
