#!/usr/bin/env python3
"""Frozen physical wormhole initial-state gauge test on the compatible grid.

Uses the successful spherical run's actual Schwarzschild constructor and t=0
shift profiles. This is an initial elliptic solve, not a time evolution. The
bulk source is DV[-2 alpha A], with physical future-null fixed-angle boundary
data at actual scri. Evolved scri metric-rate regularity is not supplied here;
the raw boundary connection rate is reported separately, not declared zero.
"""
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from compatible_tensor_gauge import TensorGauge
from mapped_sbp_preconditioner import MappedPreconditioner
from bounded_krylov import gmres


def run(args):
    start=time.monotonic();root=Path(__file__).resolve().parents[2]
    reference=root/'docs/elliptic-shift-3d/evidence/takeover-spherical-reference-20261027'
    receipt=json.loads((reference/'receipt.json').read_text());hashes={}
    for name in ['wormhole_isotropic.py','wormhole_data.py']:
        path=reference/'source'/name;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=receipt['source_hashes'][name]:raise ValueError('spherical source does not match duration-run receipt')
        hashes[name]=digest
    sys.path.insert(0,str(reference/'source'))
    from wormhole_isotropic import IsotropicWormhole
    if args.amr_core:
        from amr_shell_sbp_probe import AMRShellGrid
        grid=AMRShellGrid(args.n)
    else:grid=ShellGrid(args.n,transfer_seed='zero')
    X=grid.coordinates;W=grid.weights;radius=np.linalg.norm(X,axis=-1)
    boundary=np.zeros(W.shape,dtype=bool)
    for b,a,s in grid.outer:
        index=[b,slice(None),slice(None),slice(None)];index[a+1]=-1 if s else 0
        boundary[tuple(index)]=True
    # These are topologically known r=1 surface nodes. This assignment avoids
    # querying the constructor at 1+roundoff; it does not clamp interior radii.
    radius[boundary]=1
    if np.min(radius)<=0 or np.max(radius)>1:raise ValueError('invalid physical sampling location')
    unique,index=np.unique(radius,return_inverse=True)
    # The frozen constructor normally uses NumPy's default underflow behavior.
    # Its compact transition can square an exponentially tiny number to zero.
    # Preserve those values and record warnings; division, overflow and invalid
    # operations remain fatal, and solver arithmetic remains fully trapped.
    with warnings.catch_warnings(record=True) as caught,np.errstate(under='warn'):
        warnings.simplefilter('always')
        constructor=IsotropicWormhole(mass=1/3,quadrature_order=96)
        q,_,_,metadata=constructor.data(unique)
    constructor_warnings=[{'message':str(w.message),'file':Path(w.filename).name,'line':w.lineno} for w in caught]
    chi=np.array([m['chi_total'] for m in metadata])[index].reshape(radius.shape)
    alpha=((1+unique**2)/2+q[:,6])[index].reshape(radius.shape)
    arr=q[:,2][index].reshape(radius.shape)
    normal=X/radius[...,None]
    curvature=1.5*arr[...,None,None]*(normal[..., :,None]*normal[...,None,:]-np.eye(3)/3)
    C=-2*alpha[...,None,None]*curvature
    metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy();reference_gauge=TensorGauge(grid,metric)
    if args.cached:
        from cached_tensor_gauge import CachedTensorGauge
        gauge=CachedTensorGauge(grid,metric)
    else:gauge=reference_gauge
    physical_source=reference_gauge.variation(C)
    def apply(beta):
        result=gauge.apply(beta);result[boundary]=beta[boundary]
        return result
    # Initial g=I, alpha=chi=1 at scri: beta=-x is the prescribed future branch.
    lift=-X.copy();rhs=physical_source.copy();rhs[boundary]=lift[boundary]
    correction_rhs=rhs-apply(lift)
    pre=MappedPreconditioner(grid,metric,apply,degree=1,balanced=False)
    setup=time.monotonic()-start;solve_start=time.monotonic()
    def progress(row):
        with (args.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(row,elapsed_seconds=time.monotonic()-start))+'\n')
    progress({'stage':'setup_complete','setup_seconds':setup})
    delta,result=gmres(apply,correction_rhs,pre,W,args.max_iterations,args.restart,progress=progress)
    beta=lift+delta
    np.savez_compressed(args.output/'solution.npz',beta=beta,coordinates=X,weights=W,radius=radius,boundary=boundary)
    original=rhs-reference_gauge.apply(beta);original[boundary]=lift[boundary]-beta[boundary]
    actual_rate=reference_gauge.variation(C+reference_gauge.rate(beta))
    reference_values=[]
    for n in [1025,2049]:
        path=reference/f'short-n{n}/profile-0001.npz';hashes[f'profile-n{n}']=hashlib.sha256(path.read_bytes()).hexdigest()
        profile=np.load(path)
        if float(profile['t'])!=0 or int(profile['step'])!=0:raise ValueError('reference profile is not initial state')
        reference_values.append(np.interp(radius,profile['r'],profile['beta']))
    reference_beta=reference_values[1];radial=np.sum(beta*normal,axis=-1)
    tangential=beta-radial[...,None]*normal
    rms=lambda value:float(np.sqrt(np.sum(W*value**2)/np.sum(W)))
    # Equal-radius groups measure actual angular scatter, without feeding a
    # spherical average back into either the state or the elliptic action.
    group_r,groups=np.unique(np.round(radius,13),return_inverse=True);groups=groups.ravel()
    weights=np.bincount(groups,weights=W.ravel());mean=np.bincount(groups,weights=(W*radial).ravel())/weights
    spread=np.sqrt(np.bincount(groups,weights=(W*(radial-mean[groups].reshape(radius.shape))**2).ravel())/weights)
    samples=np.bincount(groups)
    inner=radius<.15
    clock=alpha[boundary];ch=chi[boundary];bn=np.sum(beta[boundary]*normal[boundary],axis=-1)
    null=bn**2-clock**2*ch
    result.update({'scope':__doc__,'n':args.n,'cells':grid.cells,'amr_core':args.amr_core,'cached_coefficients':args.cached,
                   'reference_source_hashes':hashes,'mass':1/3,'time':0.,
                   'constructor_warnings':constructor_warnings,
                   'setup_seconds':setup,'solve_seconds':time.monotonic()-solve_start,
                   'preconditioner':pre.diagnostic,
                   'full_original_rms':float(np.sqrt(np.sum(W[...,None]*original**2)/np.sum(W))),
                   'full_original_point':float(np.max(abs(original))),
                   'bulk_actual_metric_rate_rms':float(np.sqrt(np.sum(W[~boundary,None]*actual_rate[~boundary]**2)/np.sum(W[~boundary]))),
                   'bulk_actual_metric_rate_max':float(np.max(abs(actual_rate[~boundary]))),
                   'raw_scri_metric_rate_max':float(np.max(abs(actual_rate[boundary]))),
                   'radial_shift_reference_rms':rms(radial-reference_beta),
                   'radial_shift_reference_max':float(np.max(abs(radial-reference_beta))),
                   'reference_1025_2049_rms':rms(reference_values[0]-reference_values[1]),
                   'tangential_shift_rms':rms(np.linalg.norm(tangential,axis=-1)),
                   'angular_radial_scatter_rms':float(np.sqrt(np.sum(weights*spread**2)/np.sum(weights))),
                   'inner_shift_reference_rms':float(np.sqrt(np.sum(W[inner]*(radial[inner]-reference_beta[inner])**2)/np.sum(W[inner]))),
                   'inner_cells':int(np.sum(inner)),'minimum_radius':float(np.min(radius)),
                   'null_boundary_max':float(np.max(abs(null))),
                   'fixed_angle_boundary_max':float(np.max(np.linalg.norm(tangential[boundary],axis=-1))),
                   'future_branch_max_beta_dot_normal':float(np.max(bn)),
                   'boundary_clock_error_max':float(np.max(abs(clock-1)))})
    result['gates']={'original_gauge_residual':result['converged'] and result['full_original_rms']<1e-9 and result['full_original_point']<1e-7,
                     'direct_metric_rate':result['bulk_actual_metric_rate_rms']<1e-8 and result['bulk_actual_metric_rate_max']<1e-7,
                     'physical_boundary_data':result['null_boundary_max']<1e-12 and result['fixed_angle_boundary_max']<1e-12 and result['future_branch_max_beta_dot_normal']<0 and result['boundary_clock_error_max']<1e-12}
    np.savez_compressed(args.output/'profile.npz',coordinates=X,weights=W,beta=beta,radius=radius,reference_beta=reference_beta,actual_metric_rate=actual_rate,boundary=boundary,group_radius=group_r,group_mean=mean,group_scatter=spread,group_count=samples)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--amr-core',action='store_true');p.add_argument('--max-iterations',type=int,default=900);p.add_argument('--restart',type=int,default=100);p.add_argument('--cached',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    result=run(args);(args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
