#!/usr/bin/env python3
"""Hermite basis, scalar conditioning and independent full-metric checks.

These component checks cannot establish enriched-space energy stability or
off-node angular-interface consistency. Both remain explicit adoption gates.
"""
import argparse,json
from pathlib import Path
import numpy as np
from compatible_sbp import mm
from endpoint_sbp import local,shell_grid
from layer_affine_sbp import LayerAffineGrid
from local_hermite_gauge import slopes,hermite,LocalHermiteGauge
from compatible_tensor_gauge import TensorGauge
from cached_tensor_gauge import CachedTensorGauge
from adjoint_tensor_gauge import EnrichedGaugeAdjoint


def run():
    scalar=[]
    for n in [10,12,16,24,40]:
        loc=local(n);H=loc['H'];x=loc['x'];D=loc['D'];R=loc['R']/H[:,None]
        for moment in [-1,0,1,2]:
            high=slopes(loc,moment);B,dB=hermite(x,high,x)
            G=np.column_stack(((D-R)[:,1:-1],high));A=-mm(D+R,G)
            exact=x*(1-x)*np.exp(x);source=(1-x-x*x)*np.exp(x)
            rhs=-mm(D+R,source)
            scaled=np.sqrt(H)[:,None]*A;singular=np.linalg.svd(scaled,compute_uv=False)
            coef=np.linalg.lstsq(scaled,np.sqrt(H)*rhs,rcond=1e-12)[0]
            values=np.r_[0.,coef[:-1],0.]
            points=np.linspace(0,1,1001);bubble,_=hermite(x,high,points)
            scalar.append({'n':n,'moments':moment,'nodal_bubble_max':float(np.max(abs(B))),
                'derivative_match_max':float(np.max(abs(dB-high))),
                'quadrature_moments':[float(np.sum(H*x**k*high)) for k in range(3)],
                'scalar_singular_min':float(singular[-1]),'scalar_condition':float(singular[0]/singular[-1]),
                'scalar_residual_rms':float(np.sqrt(np.sum(H*(mm(A,coef)-rhs)**2))),
                'nodal_solution_rms':float(np.sqrt(np.sum(H*(values-exact)**2))),
                'bubble_coefficient':float(coef[-1]),'off_node_bubble_max':float(np.max(abs(coef[-1]*bubble)))})
    grid=LayerAffineGrid(shell_grid(10,radial=(0.,.45,.8,1.),transfer_seed='zero'))
    X=grid.coordinates;W=grid.weights;rng=np.random.default_rng(2731);rows=[]
    for variable in [False,True]:
        metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
        if variable:
            metric[...,0,0]=1.3+.08*np.sin(X[...,0]+X[...,2]);metric[...,1,1]=.9+.05*np.cos(2*X[...,1]);metric[...,2,2]=1.1
            metric[...,0,2]=metric[...,2,0]=.03*np.sin(X[...,1]);metric[...,0,1]=metric[...,1,0]=.1
        full=TensorGauge(grid,metric);cached=CachedTensorGauge(grid,metric)
        for moment in [-1,2]:
            e=LocalHermiteGauge(cached,moment);independent=LocalHermiteGauge(full,moment);adjoint=EnrichedGaugeAdjoint(e)
            coef=.01*rng.normal(size=W.shape+(3,));test=rng.normal(size=coef.shape)
            h=e.rate(coef);action=e.apply(coef)
            left=float(np.sum(W[...,None]*test*action));right=float(np.sum(W[...,None]*coef*adjoint.apply(test)))
            eps=1e-5/max(1.,float(np.max(abs(h))))
            finite=-(full.connection(metric+eps*h)-full.connection(metric-eps*h))/(2*eps)
            norm=lambda v:float(np.sqrt(np.sum(W[...,None]*v*v)/np.sum(W)))
            pure=np.zeros_like(coef);pure[e.boundary]=coef[e.boundary]
            energy=float(np.sum(W[...,None]*pure*e.apply(pure)))
            rows.append({'variable_metric':variable,'moments':moment,'coefficient_adjoint_relative':abs(left-right)/max(1,abs(left),abs(right)),
                         'uncached_action_max':float(np.max(abs(action-independent.apply(coef)))),
                         'finite_variation_scaled_rms':norm(finite-action)/max(1.,norm(action)),
                         'bubble_only_coefficient_energy':energy,
                         'energy_interpretation':'Chosen coefficient norm is not a qualified physical enriched-space norm.',
                         'random_coefficient_seams':e.seam_diagnostic(coef)})
    gates={'basis_and_quadrature':all(r['nodal_bubble_max']<1e-14 and r['derivative_match_max']<1e-12 and (r['moments']<0 or max(abs(v) for v in r['quadrature_moments'][:r['moments']+1])<1e-12) for r in scalar),
           'scalar_rank_and_residual':all(r['scalar_singular_min']>1e-9 and r['scalar_residual_rms']<1e-8 for r in scalar),
           'full_metric_consistency':all(r['coefficient_adjoint_relative']<1e-10 and r['uncached_action_max']<1e-8 and r['finite_variation_scaled_rms']<1e-7 for r in rows)}
    return {'scope':__doc__,'scalar':scalar,'tensor':rows,'component_gates':gates,
            'energy_qualified':False,'off_node_angular_interfaces_qualified':False,'evolution_adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');result=run()
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['component_gates'].values()) else 1)
