#!/usr/bin/env python3
"""Local Hermite all-node tensor gauge solve; energy/seam adoption gates open."""
import argparse,json,time
from pathlib import Path
import numpy as np
from endpoint_sbp import shell_grid
from layer_affine_sbp import LayerAffineGrid
from local_hermite_gauge import LocalHermiteGauge
from compatible_tensor_gauge import TensorGauge
from cached_tensor_gauge import CachedTensorGauge
from mapped_sbp_preconditioner import MappedPreconditioner,homogeneous_outer
from enriched_outer_gauge import EnrichedPreconditioner
from bounded_krylov import gmres


def manufactured(args):
    start=time.monotonic();grid=LayerAffineGrid(shell_grid(args.n,radial=tuple(args.radial),transfer_seed='zero'))
    X=grid.coordinates;W=grid.weights
    metric=np.broadcast_to(np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]]),W.shape+(3,3)).copy()
    metric[...,0,0]+=.08*np.sin(X[...,0]+X[...,2]);metric[...,1,1]+=.05*np.cos(2*X[...,1])
    metric[...,0,2]+=.03*np.sin(X[...,1]);metric[...,2,0]=metric[...,0,2]
    full=TensorGauge(grid,metric);cached=CachedTensorGauge(grid,metric)
    e=LocalHermiteGauge(cached,args.moments);independent=LocalHermiteGauge(full,args.moments);boundary=e.boundary
    r2=np.sum(X*X,axis=-1);rng=np.random.default_rng(2732)
    exact=homogeneous_outer((np.sin(2*X)+.001*rng.normal(size=X.shape))*(1-r2)[...,None]**2,grid)
    exact[boundary]=.003*(np.sin(X[boundary])+X[boundary])
    lift=np.array([.01,-.02,.015])+.03*X
    rhs=e.apply(exact)
    def base_apply(beta):
        out=cached.apply(beta);out[boundary]=beta[boundary];return out
    base=MappedPreconditioner(grid,metric,base_apply,degree=1,balanced=False);pre=EnrichedPreconditioner(e,base)
    setup=time.monotonic()-start
    def progress(row):
        with (args.output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
    progress({'stage':'setup_complete','setup_seconds':setup})
    coef,result=gmres(e.apply,rhs,pre,W,args.max_iterations,args.restart,progress=progress)
    actual=independent.apply(exact)-independent.apply(coef);error=coef-exact
    beta=lift+e.nodal(coef);h=independent.rate(coef);epsilon=1e-5/max(1.,float(np.max(abs(h))))
    finite=-(full.connection(metric+epsilon*h)-full.connection(metric-epsilon*h))/(2*epsilon)
    action=independent.apply(coef);norm=lambda v:float(np.sqrt(np.sum(W[...,None]*v*v)/np.sum(W)))
    result.update({'scope':__doc__,'case':'variable_metric_mms','n':args.n,'cells':grid.cells,'radial_partition_s':args.radial,
        'setup_seconds':setup,'total_seconds':time.monotonic()-start,'preconditioner':{'base':base.diagnostic,'enriched':pre.diagnostic},
        'all_node_metric_rate_rms':norm(actual),'all_node_metric_rate_max':float(np.max(abs(actual))),
        'scri_metric_rate_max':float(np.max(abs(actual[boundary]))),'coefficient_error_rms':norm(error),
        'physical_nodal_shift_error_rms':norm(e.nodal(error)),
        'physical_boundary_shift_error':float(np.max(abs(beta[boundary]-lift[boundary]))),
        'finite_variation_scaled_rms':norm(finite-action)/max(1.,norm(action)),
        'seams':e.seam_diagnostic(coef),'exact_coefficient_seams':e.seam_diagnostic(exact),
        'energy_qualified':False,'off_node_angular_interfaces_qualified':False,'evolution_adopted':False})
    result['gates']={'full_collocation_residual':result['converged'] and result['all_node_metric_rate_rms']<1e-9 and result['all_node_metric_rate_max']<1e-7,
                     'discrete_solution':result['coefficient_error_rms']<1e-7,
                     'unchanged_boundary_values':result['physical_boundary_shift_error']<1e-12,
                     'independent_finite_variation':result['finite_variation_scaled_rms']<1e-7}
    np.savez_compressed(args.output/'solution.npz',coefficients=coef,exact=exact,coordinates=X,weights=W,boundary=boundary)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['physical','mms'],default='physical');p.add_argument('--n',type=int,choices=[10,12],default=10)
    p.add_argument('--moments',type=int,choices=[-1,0,1,2],default=2)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.])
    p.add_argument('--max-iterations',type=int,default=800);p.add_argument('--restart',type=int,default=120)
    args=p.parse_args();args.enriched_coarse_degree=0
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    if args.case=='physical':
        import solve_enriched_wormhole as experiment
        experiment.ShellGrid=shell_grid;experiment.AffineConsistentGrid=LayerAffineGrid
        experiment.EnrichedOuterGauge=lambda gauge:LocalHermiteGauge(gauge,args.moments)
        result=experiment.run(args)
        saved=np.load(args.output/'solution.npz');grid=LayerAffineGrid(shell_grid(args.n,radial=tuple(args.radial),transfer_seed='zero'))
        metric=np.broadcast_to(np.eye(3),grid.weights.shape+(3,3)).copy()
        e=LocalHermiteGauge(TensorGauge(grid,metric),args.moments)
        result['seams']=e.seam_diagnostic(saved['coefficients'])
    else:result=manufactured(args)
    result['hermite_scope']=__doc__;result['hermite_moments']=args.moments
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
