#!/usr/bin/env python3
"""Falsifiable full-tensor radial-bubble experiment; not evolution acceptance.

All original metric-built connection-rate equations, INCLUDING scri, are
evaluated. Fixed physical shift values are carried by a separate lift. New
unknowns represent off-node radial polynomial content. Energy/interface
qualification is explicitly outstanding even if the collocation solve passes.
"""
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from affine_sbp_probe import AffineConsistentGrid
from compatible_tensor_gauge import TensorGauge
from cached_tensor_gauge import CachedTensorGauge
from mapped_sbp_preconditioner import MappedPreconditioner
from enriched_outer_gauge import EnrichedOuterGauge,EnrichedPreconditioner
from bounded_krylov import gmres


def run(args):
    start=time.monotonic();root=Path(__file__).resolve().parents[2]
    ref=root/'docs/elliptic-shift-3d/evidence/takeover-spherical-reference-20261027'
    receipt=json.loads((ref/'receipt.json').read_text());hashes={}
    for name in ['wormhole_isotropic.py','wormhole_data.py']:
        digest=hashlib.sha256((ref/'source'/name).read_bytes()).hexdigest()
        if digest!=receipt['source_hashes'][name]:raise ValueError('reference source mismatch')
        hashes[name]=digest
    sys.path.insert(0,str(ref/'source'))
    from wormhole_isotropic import IsotropicWormhole
    grid=AffineConsistentGrid(ShellGrid(args.n,radial=tuple(args.radial),transfer_seed='zero'))
    X=grid.coordinates;W=grid.weights
    metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
    full=TensorGauge(grid,metric);cached=CachedTensorGauge(grid,metric)
    enriched=EnrichedOuterGauge(cached);independent=EnrichedOuterGauge(full)
    boundary=enriched.boundary;radius=np.linalg.norm(X,axis=-1);radius[boundary]=1
    unique,index=np.unique(radius,return_inverse=True)
    with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
        warnings.simplefilter('always')
        q,_,_,metadata=IsotropicWormhole(mass=1/3,quadrature_order=96).data(unique)
    alpha=((1+unique**2)/2+q[:,6])[index].reshape(W.shape)
    arr=q[:,2][index].reshape(W.shape);normal=X/radius[...,None]
    curvature=1.5*arr[...,None,None]*(normal[..., :,None]*normal[...,None,:]-np.eye(3)/3)
    C=-2*alpha[...,None,None]*curvature
    lift=-X.copy();fixed_rate=full.rate(lift)+C
    rhs=full.variation(fixed_rate)
    def base_apply(beta):
        result=cached.apply(beta);result[boundary]=beta[boundary]
        return result
    base_pre=MappedPreconditioner(grid,metric,base_apply,degree=1,balanced=False)
    pre=EnrichedPreconditioner(enriched,base_pre)
    def progress(row):
        with (args.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
    setup=time.monotonic()-start;progress({'stage':'setup_complete','seconds':setup})
    coefficients,result=gmres(enriched.apply,rhs,pre,W,args.max_iterations,args.restart,progress=progress)
    np.savez_compressed(args.output/'solution.npz',coefficients=coefficients,coordinates=X,weights=W,boundary=boundary)
    beta=lift+enriched.nodal(coefficients)
    actual=full.variation(fixed_rate+independent.rate(coefficients))
    radial=np.sum(beta*normal,axis=-1);tangent=beta-radial[...,None]*normal
    profile_path=ref/'short-n2049/profile-0001.npz';reference=np.load(profile_path)
    if float(reference['t'])!=0 or int(reference['step'])!=0:raise ValueError('not t=0 reference')
    reference_beta=np.interp(radius,reference['r'],reference['beta'])
    rms=lambda v:float(np.sqrt(np.sum(W*v*v)/np.sum(W)))
    inner=radius<.15
    # This is an independent finite variation of the actual metric connection,
    # not only a cached/uncached comparison of the linear action.
    h=independent.rate(coefficients);epsilon=1e-5/max(1.,float(np.max(abs(h))))
    finite=(full.connection(metric+epsilon*h)-full.connection(metric-epsilon*h))/(2*epsilon)
    exact=full.variation(h)
    result.update({'scope':__doc__,'time':0.,'mass':1/3,'cells':grid.cells,'n':args.n,
        'radial_partition_s':args.radial,'setup_seconds':setup,'total_seconds':time.monotonic()-start,
        'source_hashes':hashes,'constructor_warnings':[str(w.message) for w in caught],
        'preconditioner':dict(base=base_pre.diagnostic,enriched=pre.diagnostic),
        'bulk_metric_rate_rms':float(np.sqrt(np.sum(W[~boundary,None]*actual[~boundary]**2)/np.sum(W[~boundary]))),
        'all_node_metric_rate_rms':rms(np.linalg.norm(actual,axis=-1)),
        'all_node_metric_rate_max':float(np.max(abs(actual))),
        'scri_metric_rate_max':float(np.max(abs(actual[boundary]))),
        'radial_shift_reference_rms':rms(radial-reference_beta),
        'tangential_shift_rms':rms(np.linalg.norm(tangent,axis=-1)),
        'inner_shift_reference_rms':float(np.sqrt(np.sum(W[inner]*(radial[inner]-reference_beta[inner])**2)/np.sum(W[inner]))),
        'physical_boundary_shift_error':float(np.max(abs(beta[boundary]-lift[boundary]))),
        'bubble_coefficient_max':float(np.max(abs(coefficients[boundary]))),
        'bubble_nodal_value_max':max(p[-1] for p in enriched.plans),
        'finite_epsilon':epsilon,'finite_variation_error_max':float(np.max(abs(finite-exact))),
        'finite_variation_scaled_rms':rms(np.linalg.norm(finite-exact,axis=-1))/max(1.,rms(np.linalg.norm(exact,axis=-1))),
        'energy_qualified':False,'off_node_angular_interfaces_qualified':False,'evolution_adopted':False})
    result['gates']={'full_collocation_residual':result['converged'] and result['all_node_metric_rate_rms']<1e-9 and result['all_node_metric_rate_max']<1e-7,
                     'unchanged_boundary_values':result['physical_boundary_shift_error']<1e-12,
                     'independent_finite_variation':result['finite_variation_scaled_rms']<1e-7}
    np.savez_compressed(args.output/'profile.npz',coordinates=X,weights=W,radius=radius,beta=beta,reference_beta=reference_beta,coefficients=coefficients,actual_metric_rate=actual,boundary=boundary)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.])
    p.add_argument('--max-iterations',type=int,default=900);p.add_argument('--restart',type=int,default=100)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run(args);(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
