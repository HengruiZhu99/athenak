#!/usr/bin/env python3
"""Frozen 1D field sampled on 3D layouts; no evolution or solved-field claim.

Compare truncation and affine-moment localization before expensive solves.
Every active node and every scri node is retained in the saved diagnostics.
"""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
from compatible_sbp import mm
from endpoint_sbp import shell_grid
from affine_sbp_probe import AffineConsistentGrid
from compatible_tensor_gauge import TensorGauge


def run():
    root=Path(__file__).resolve().parents[2]
    ref=root/'docs/elliptic-shift-3d/evidence/takeover-spherical-reference-20261027'
    receipt=json.loads((ref/'receipt.json').read_text());hashes={}
    for name in ['wormhole_data.py','wormhole_isotropic.py']:
        path=ref/'source'/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[name]!=receipt['source_hashes'][name]:raise ValueError('reference mismatch')
    sys.path.insert(0,str(ref/'source'));from wormhole_isotropic import IsotropicWormhole
    profile_path=ref/'short-n2049/profile-0001.npz';profile=np.load(profile_path)
    hashes['profile']=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if float(profile['t'])!=0 or int(profile['step'])!=0:raise ValueError('expected initial reference')
    rows=[]
    for radial in [(0.,.45,1.),(0.,.45,.8,1.),(0.,.45,.55,.60,.65,.70,.75,1.)]:
        base=shell_grid(10,radial=radial,transfer_seed='zero');grid=AffineConsistentGrid(base)
        X=grid.coordinates;W=grid.weights;radius=np.linalg.norm(X,axis=-1)
        boundary=np.zeros(W.shape,dtype=bool)
        for b,_,_ in grid.outer:boundary[b,-1]=True
        radius[boundary]=1
        unique,index=np.unique(radius,return_inverse=True)
        with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
            warnings.simplefilter('always');q,_,_,_=IsotropicWormhole(mass=1/3,quadrature_order=96).data(unique)
        normal=X/radius[...,None]
        alpha=((1+unique**2)/2+q[:,6])[index].reshape(W.shape);arr=q[:,2][index].reshape(W.shape)
        C=-3*(alpha*arr)[...,None,None]*(normal[..., :,None]*normal[...,None,:]-np.eye(3)/3)
        beta=np.interp(radius,profile['r'],profile['beta'])[...,None]*normal
        metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
        values=np.concatenate((np.ones(W.shape+(1,)),X),axis=-1)
        derivatives=[base.derivative(values,axis,0) for axis in range(3)]
        groups=[]
        # Restrictions of the global affine defect, with no altered interface
        # rows, test whether a block-diagonal skew correction could exist.
        for label,ids in [('core',[0])]+[(f'layer-{k}',[b for b,(p,l) in enumerate(base.blocks) if p>=0 and l==k]) for k in range(len(radial)-1)]:
            H=W[ids].ravel();v=values[ids].reshape(-1,4)
            transform=np.linalg.inv(np.linalg.cholesky(mm(v.T*H[None,:],v)).T)
            P=mm(v,transform);defects=[]
            for axis in range(3):
                target=np.zeros(4);target[axis+1]=1
                F=mm(target[None,:]-derivatives[axis][ids].reshape(-1,4),transform)
                S=mm(P.T*H[None,:],F);defects.append(float(np.max(abs(S+S.T))))
            groups.append({'group':label,'blocks':ids,'compatibility_max':max(defects)})
        fields=[]
        for name,g in [('base',base),('global_affine',grid)]:
            gauge=TensorGauge(g,metric);rate=gauge.variation(C+gauge.rate(beta))
            fields.append({'operator':name,'scri_max':float(np.max(abs(rate[boundary]))),
                          'all_node_rms':float(np.sqrt(np.sum(W[...,None]*rate**2)/np.sum(W))),
                          'bulk_rms':float(np.sqrt(np.sum(W[~boundary,None]*rate[~boundary]**2)/np.sum(W[~boundary]))),
                          'layer_rms':[float(np.sqrt(np.sum(W[[b for b,(p,l) in enumerate(base.blocks) if p>=0 and l==k],...,None]*rate[[b for b,(p,l) in enumerate(base.blocks) if p>=0 and l==k]]**2)/np.sum(W[[b for b,(p,l) in enumerate(base.blocks) if p>=0 and l==k]]))) for k in range(len(radial)-1)]})
        rows.append({'n':10,'radial_partition_s':radial,'cells':grid.cells,'sampled_reference':fields,
                     'shell_group_affine_moments':groups,'constructor_underflow_warnings':len(caught)})
    return {'scope':__doc__,'reference_hashes':hashes,'rows':rows,'adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run();(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
