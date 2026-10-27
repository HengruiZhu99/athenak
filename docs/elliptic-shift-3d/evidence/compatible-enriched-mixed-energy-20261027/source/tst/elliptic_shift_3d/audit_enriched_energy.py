#!/usr/bin/env python3
"""Independent physical-volume energy falsifier for outer shift bubbles.

Bubble values vanish at the metric nodes, so using the old nodal mass would
give a meaningless zero energy. Instead evaluate the actual off-node trial
polynomial and the polynomial interpolant of its unmodified collocation
operator at an overintegrated Gauss grid, with the analytic map Jacobian.
This tests a stated continuous pairing, not an established discrete SBP form.
A positive result in this finite trial space cannot prove coercivity; a
negative witness excludes that pairing as a positive energy argument.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from compatible_sbp import basis,mm
from mapped_sbp_probe import ShellGrid,axis_apply
from affine_sbp_probe import AffineConsistentGrid
from cached_tensor_gauge import CachedTensorGauge
from enriched_outer_gauge import EnrichedOuterGauge


def run(args):
    start=time.monotonic();grid=AffineConsistentGrid(ShellGrid(10,radial=(0.,.45,.8,1.),transfer_seed='zero'))
    metric=np.broadcast_to(np.eye(3),grid.weights.shape+(3,3)).copy()
    enriched=EnrichedOuterGauge(CachedTensorGauge(grid,metric));boundary=enriched.boundary
    q=grid.locals[1]['x'];V=basis(q,grid.n-1)[0];inverse=np.linalg.inv(V)
    modes=[];labels=[];nvec=grid.coordinates/np.linalg.norm(grid.coordinates,axis=-1)[...,None]
    # Continuous physical angular polynomial amplitudes, face jumps and random
    # angular content are all retained in separate labeled trial directions.
    for component in range(3):
        for scalar in range(4):
            v=np.zeros(grid.weights.shape+(3,));v[...,component][boundary]=1 if scalar==0 else nvec[...,scalar-1][boundary]
            modes.append(v);labels.append(f'global_component_{component}_angular_{scalar}')
    for face in grid.outer:
        for component in range(3):
            v=np.zeros(grid.weights.shape+(3,));v[face[0],-1,...,component]=1
            # Face constants sum to the global constant already included.
            if face==grid.outer[-1]:continue
            modes.append(v);labels.append(f'face_{face[0]}_component_{component}')
    rng=np.random.default_rng(104729)
    for k in range(18):
        v=np.zeros(grid.weights.shape+(3,));v[boundary]=rng.normal(size=(int(np.sum(boundary)),3))
        modes.append(v);labels.append(f'random_{k}')
    if args.include_nodal:
        for degree in [0,3,7]:
            envelope=q*(1-q)*leg.legval(2*q-1,np.eye(8)[degree])
            for component in range(3):
                for scalar in range(4):
                    v=np.zeros(grid.weights.shape+(3,))
                    for b,_,_ in grid.outer:
                        amplitude=np.ones((grid.n,grid.n)) if scalar==0 else nvec[b,-1,...,scalar-1]
                        v[b,...,component]=envelope[:,None,None]*amplitude[None,...]
                    modes.append(v);labels.append(f'nodal_degree_{degree}_component_{component}_angular_{scalar}')
    modes=np.array(modes);actions=np.array([enriched.apply(v) for v in modes])
    def quadrature(order):
        z,w=leg.leggauss(order);x=(z+1)/2;w=w/2
        I=mm(basis(x,grid.n-1)[0],inverse)
        primitive=leg.legint(np.eye(grid.n)[-1])/2
        B=leg.legval(2*x-1,primitive)-leg.legval(-1,primitive)
        trial=[];residual=[];weights=[]
        for b,_,_ in grid.outer:
            # Leading direction is the trial index; interpolation acts only on
            # the three local physical parameter axes after moving it last.
            a=np.moveaxis(modes[:,b,-1],0,-1)
            a=axis_apply(axis_apply(a,0,I),1,I)
            value=B[:,None,None,None,None]*a[None,...]
            nodal=np.moveaxis(modes[:,b].copy(),0,-1);nodal[-1]=0
            for axis in range(3):nodal=axis_apply(nodal,axis,I)
            value+=nodal
            r=np.moveaxis(actions[:,b],0,-1)
            for axis in range(3):r=axis_apply(r,axis,I)
            patch,layer=grid.blocks[b];s0,s1=grid.radial[layer:layer+2]
            u=2*x[None,:,None]-1;v=2*x[None,None,:]-1
            R=1/np.sqrt(1+u*u+v*v);s=s0+(s1-s0)*x[:,None,None]
            d=(1-s)*grid.half_width+s*R
            J=4*(s1-s0)*(R-grid.half_width)*d*d
            W=w[:,None,None]*w[None,:,None]*w[None,None,:]*J
            trial.append(value.reshape(-1,len(modes)));residual.append(r.reshape(-1,len(modes)))
            weights.append(np.repeat(W.ravel(),3))
        T=np.concatenate(trial);A=np.concatenate(residual);W=np.concatenate(weights)
        gram=mm(T.T*W[None,:],T);energy=mm(T.T*W[None,:],A)
        L=np.linalg.cholesky(gram);inv=np.linalg.inv(L)
        E=mm(mm(inv,(energy+energy.T)/2),inv.T)
        eigen,vectors=np.linalg.eigh(E);c=mm(inv.T,vectors[:,0])
        witness=np.einsum('i,ibjklc->bjklc',c,modes,optimize=False)
        return dict(order=order,minimum_rayleigh=float(eigen[0]),maximum_rayleigh=float(eigen[-1]),
                    gram_condition=float(np.linalg.cond(gram)),
                    witness_coefficients=c.tolist()),witness
    rows=[]
    for order in [12,16]:
        row,witness=quadrature(order);rows.append(row)
    np.savez_compressed(args.output/'witness.npz',coefficients=witness,metric_rate=enriched.rate(witness),operator=enriched.apply(witness))
    return {'scope':__doc__,'trial_dimension':len(modes),'labels':labels,'rows':rows,
            'seconds':time.monotonic()-start,'negative_energy_witness':rows[-1]['minimum_rayleigh']<0,
            'energy_qualified':False,'evolution_adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-nodal',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run(args);(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
