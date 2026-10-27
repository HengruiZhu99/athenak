#!/usr/bin/env python3
"""Bounded full variable-metric gauge solve on the mapped spatial candidate.

Discrete manufactured source with explicit Dirichlet rows at the actual sphere.
These are not the physical hyperboloidal source/scri equations or an evolution.
All volume cells and boundary rows are assessed; no puncture neighborhood is
excluded. Nonzero affine boundary data test retained translation/linear modes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from compatible_tensor_gauge import TensorGauge
from mapped_sbp_preconditioner import MappedPreconditioner,homogeneous_outer
from bounded_krylov import gmres


def run(args):
    start=time.monotonic()
    if args.amr_core:
        from amr_shell_sbp_probe import AMRShellGrid
        grid=AMRShellGrid(args.n)
    else:grid=ShellGrid(args.n,transfer_seed=args.mortar_seed)
    X=grid.coordinates;W=grid.weights
    metric=np.broadcast_to(np.array([[1.3,.13,-.07],[.13,.9,.08],[-.07,.08,1.1]]),W.shape+(3,3)).copy()
    amplitude=args.metric_amplitude
    metric[...,0,0]+=amplitude*np.sin(X[...,0]+X[...,2])
    metric[...,1,1]+=.625*amplitude*np.cos(2*X[...,1])
    metric[...,0,2]+=.375*amplitude*np.sin(X[...,1]);metric[...,2,0]=metric[...,0,2]
    reference_gauge=TensorGauge(grid,metric)
    if args.cached:
        from cached_tensor_gauge import CachedTensorGauge
        gauge=CachedTensorGauge(grid,metric)
    else:gauge=reference_gauge
    boundary=np.zeros(W.shape,dtype=bool)
    for b,a,s in grid.outer:
        index=[b,slice(None),slice(None),slice(None)];index[a+1]=-1 if s else 0
        boundary[tuple(index)]=True
    def apply(beta):
        result=gauge.apply(beta);result[boundary]=beta[boundary]
        return result
    r2=np.sum(X*X,axis=-1)
    lift=np.array([.01,-.02,.015])+.03*X
    smooth=np.sin(2*X)*(1-r2)[...,None]**2
    random=.001*np.random.default_rng(271129).normal(size=smooth.shape)*(1-r2)[...,None]**2
    correction=homogeneous_outer(smooth+random,grid);exact=lift+correction
    rhs=apply(exact)-apply(lift)
    # The linearity residual at the prescribed boundary should be exactly zero.
    if np.max(abs(rhs[boundary]))!=0:raise ValueError('inconsistent lifted boundary data')
    pre=MappedPreconditioner(grid,metric,apply,args.coarse_degree,args.balanced)
    setup=time.monotonic()-start;solve_start=time.monotonic()
    def progress(row):
        with (args.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
    progress({'stage':'setup_complete','setup_seconds':setup})
    delta,result=gmres(apply,rhs,pre,W,args.max_iterations,args.restart,progress=progress)
    solution=lift+delta
    original=reference_gauge.apply(exact)-reference_gauge.apply(solution)
    original[boundary]=exact[boundary]-solution[boundary]
    error=solution-exact
    result.update({'scope':__doc__,'n':args.n,'cells':grid.cells,'amr_core':args.amr_core,
                   'mortar_seed':'zero' if args.amr_core else args.mortar_seed,'cached_coefficients':args.cached,
                   'metric_amplitude':amplitude,'metric_determinant_min':float(np.min(np.linalg.det(metric))),
                   'metric_determinant_max':float(np.max(np.linalg.det(metric))),
                   'preconditioner':pre.diagnostic,'setup_seconds':setup,
                   'solve_seconds':time.monotonic()-solve_start,
                   'actual_solution_rms':float(np.sqrt(np.sum(W[...,None]*error**2)/np.sum(W))),
                   'boundary_error_max':float(np.max(abs(error[boundary]))),
                   'independently_recomputed_full_residual_rms':float(np.sqrt(np.sum(W[...,None]*original**2)/np.sum(W))),
                   'independently_recomputed_full_residual_max':float(np.max(abs(original))),
                   'boundary_nodes':int(np.sum(boundary)),
                   'interior_metric_connection_rate_max':float(np.max(abs(original[~boundary])))})
    result['gates']={'original_residual':result['converged'] and result['independently_recomputed_full_residual_rms']<1e-9 and result['independently_recomputed_full_residual_max']<1e-7,
                     'discrete_solution':result['actual_solution_rms']<1e-7,
                     'prescribed_boundary':result['boundary_error_max']<1e-12}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--metric-amplitude',type=float,default=.08)
    p.add_argument('--coarse-degree',type=int,choices=[0,1,2],default=1)
    p.add_argument('--balanced',action='store_true');p.add_argument('--max-iterations',type=int,default=600)
    p.add_argument('--restart',type=int,default=60);p.add_argument('--amr-core',action='store_true');p.add_argument('--mortar-seed',choices=['prolongation','zero'],default='prolongation');p.add_argument('--cached',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run(args);(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
