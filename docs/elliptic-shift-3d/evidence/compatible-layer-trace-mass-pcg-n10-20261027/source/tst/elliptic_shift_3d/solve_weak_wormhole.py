#!/usr/bin/env python3
"""Physical t=0 flat-metric weak control with full strong residual disclosure.

This is not a full variable-metric gauge implementation or an evolution.
All pointwise residuals and puncture cells are retained. Boundary bubble
moments are weak equations and are not relabeled strong scri collocation.
"""
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from affine_sbp_probe import AffineConsistentGrid
from compatible_tensor_gauge import TensorGauge
from cached_tensor_gauge import CachedTensorGauge
from enriched_outer_gauge import EnrichedOuterGauge
from mapped_sbp_preconditioner import MappedPreconditioner
from weak_enriched_gauge import WeakEnrichedGauge,WeakBlockPreconditioner,conjugate_gradient


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
    X=grid.coordinates;W=grid.weights;metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
    full=TensorGauge(grid,metric);cached=CachedTensorGauge(grid,metric)
    enriched=EnrichedOuterGauge(cached);independent=EnrichedOuterGauge(full)
    weak=WeakEnrichedGauge(enriched);boundary=enriched.boundary
    radius=np.linalg.norm(X,axis=-1);radius[boundary]=1;unique,index=np.unique(radius,return_inverse=True)
    with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
        warnings.simplefilter('always');q,_,_,metadata=IsotropicWormhole(mass=1/3,quadrature_order=96).data(unique)
    alpha=((1+unique**2)/2+q[:,6])[index].reshape(W.shape);arr=q[:,2][index].reshape(W.shape)
    normal=X/radius[...,None];curvature=1.5*arr[...,None,None]*(normal[..., :,None]*normal[...,None,:]-np.eye(3)/3)
    C=-2*alpha[...,None,None]*curvature;lift=-X.copy();fixed_rate=C+full.rate(lift)
    rhs=weak.source(fixed_rate)
    def base_apply(value):
        v=value.copy();v[boundary]=0;out=weak.apply(v);out[boundary]=value[boundary];return out
    base=MappedPreconditioner(grid,metric,base_apply,degree=1,balanced=False)
    pre=WeakBlockPreconditioner(enriched,base)
    dot=lambda a,b:float(np.sum(W.reshape(W.shape+(1,)*(a.ndim-4))*a*b))
    rng=np.random.default_rng(10691);u=rng.normal(size=rhs.shape);v=rng.normal(size=rhs.shape)
    Au=weak.apply(u);Av=weak.apply(v);strain=enriched.rate(u)
    energy=dot(u,Au);strain_energy=.5*dot(strain,strain)
    checks={'energy_relative':abs(energy-strain_energy)/max(1,abs(energy),strain_energy),
            'symmetry_relative':abs(dot(u,Av)-dot(v,Au))/max(1,abs(dot(u,Av)),abs(dot(v,Au))),
            'preconditioner_symmetry_relative':abs(dot(u,pre(v))-dot(v,pre(u)))/max(1,abs(dot(u,pre(v))),abs(dot(v,pre(u)))),
            'energy':energy,'strain_energy':strain_energy}
    if max(checks[k] for k in ['energy_relative','symmetry_relative','preconditioner_symmetry_relative'])>1e-9:
        raise ValueError('weak operator/preconditioner failed energy or symmetry check')
    def progress(row,x):
        with (args.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
        np.savez_compressed(args.output/'latest-iterate.npz',coefficients=x,iteration=row['iteration'])
    setup=time.monotonic()-start
    coefficients,result=conjugate_gradient(weak.apply,rhs,pre,W,args.max_iterations,progress)
    beta=lift+enriched.nodal(coefficients);h=fixed_rate+independent.rate(coefficients)
    actual=full.variation(h);radial=np.sum(beta*normal,axis=-1);tangent=beta-radial[...,None]*normal
    reference=np.load(ref/'short-n2049/profile-0001.npz')
    if float(reference['t'])!=0 or int(reference['step'])!=0:raise ValueError('reference is not t=0')
    reference_beta=np.interp(radius,reference['r'],reference['beta'])
    rms=lambda a:float(np.sqrt(np.sum(W*a*a)/np.sum(W)))
    inner=radius<.15;strong_bulk=actual[~boundary]
    result.update({'scope':__doc__,'n':args.n,'cells':grid.cells,'time':0.,'mass':1/3,
        'radial_partition_s':args.radial,'source_hashes':hashes,'constructor_warnings':[str(w.message) for w in caught],
        'setup_seconds':setup,'total_seconds':time.monotonic()-start,'energy_checks':checks,
        'preconditioner':dict(base=base.diagnostic,bubble=pre.diagnostic),
        'all_node_strong_rate_rms':rms(np.linalg.norm(actual,axis=-1)),
        'all_node_strong_rate_max':float(np.max(abs(actual))),
        'bulk_strong_rate_rms':float(np.sqrt(np.sum(W[~boundary,None]*strong_bulk**2)/np.sum(W[~boundary]))),
        'scri_strong_rate_max':float(np.max(abs(actual[boundary]))),
        'radial_shift_reference_rms':rms(radial-reference_beta),'tangential_shift_rms':rms(np.linalg.norm(tangent,axis=-1)),
        'inner_shift_reference_rms':float(np.sqrt(np.sum(W[inner]*(radial[inner]-reference_beta[inner])**2)/np.sum(W[inner]))),
        'physical_boundary_shift_error':float(np.max(abs(beta[boundary]-lift[boundary]))),
        'bubble_coefficient_max':float(np.max(abs(coefficients[boundary]))),
        'metric_rate_rms':rms(np.sqrt(np.sum(h*h,axis=(-2,-1)))),
        'variable_metric_implemented':False,'evolution_adopted':False})
    result['gates']={'weak_solve':result['converged'],'original_strong_rate':result['all_node_strong_rate_rms']<1e-8 and result['all_node_strong_rate_max']<1e-7,
                     'physical_boundary':result['physical_boundary_shift_error']<1e-12}
    np.savez_compressed(args.output/'profile.npz',coordinates=X,weights=W,radius=radius,beta=beta,reference_beta=reference_beta,
                        coefficients=coefficients,actual_metric_rate=actual,metric_rhs=h,boundary=boundary)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.]);p.add_argument('--max-iterations',type=int,default=800)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run(args);(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
