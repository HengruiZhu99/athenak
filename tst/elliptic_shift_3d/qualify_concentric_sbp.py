#!/usr/bin/env python3
"""Concentric shell geometry, derivative, energy and full tensor checks."""
import argparse,json
from pathlib import Path
import numpy as np
from concentric_sbp import ConcentricGrid
from layer_affine_sbp import LayerAffineGrid
from qualify_mapped_sbp import tensor_checks
import amr_shell_sbp_probe as combined
import qualify_amr_shell_sbp as amr_experiment
from endpoint_sbp import local


def amr_grid(n):
    old,old_rule=combined.ShellGrid,combined.lobatto
    try:
        combined.ShellGrid=ConcentricGrid;combined.lobatto=local
        return LayerAffineGrid(combined.AMRShellGrid(n,radial=(0.,.4,1.),half_width=.2))
    finally:combined.ShellGrid=old;combined.lobatto=old_rule


def run():
    rows=[]
    for n in [10,11,12]:
        grid=LayerAffineGrid(ConcentricGrid(n));X=grid.coordinates;W=grid.weights
        coef=np.array([.3,.2,-.1]);u=np.exp(np.einsum('...i,i->...',X,coef,optimize=False))
        rng=np.random.default_rng(4732);f=rng.normal(size=W.shape);g=rng.normal(size=W.shape)
        errors=[];affine=0.;adjoint=0.
        for a in range(3):
            errors.append(grid.derivative(u,a,-1)-coef[a]*u)
            for sign in [-1,0,1]:
                affine=max(affine,float(np.max(abs(grid.derivative(X,a,sign)-np.eye(3)[a]))),float(np.max(abs(grid.derivative(np.ones_like(W),a,sign)))))
            left=float(np.sum(W*f*grid.derivative(g,a,-1)));right=float(np.sum(W*g*grid.derivative(f,a,1)));b=grid.boundary_integral(f,g,a)
            adjoint=max(adjoint,abs(left+right-b)/max(1,abs(left),abs(right),abs(b)))
        error2=np.sum(np.array(errors)**2,axis=0)
        outer=np.array([grid.trace(X,face) for face in grid.outer])
        row={'n':n,'cells':grid.cells,'affine_max':affine,'adjoint_relative_max':adjoint,
             'smooth_rms':float(np.sqrt(np.sum(W*error2)/np.sum(W))),
             'jacobian_min':float(np.min(grid.jacobians)),'metric_identity_max':grid.metric_identity_max,
             'normal_mismatch':grid.normal_mismatch,'actual_scri_radius_error':float(np.max(abs(np.linalg.norm(outer,axis=-1)-1))),
             'mortar_amplification':max(float(np.max(np.sum(abs(T),axis=1))) for link in grid.links for T in link[2:4] if T is not None)}
        if n==10:row['tensor']=tensor_checks(grid)
        rows.append(row)
    amr_experiment.AMRShellGrid=amr_grid;amr=amr_experiment.run()
    orders=[float(np.log(rows[i-1]['smooth_rms']/rows[i]['smooth_rms'])/np.log((rows[i]['n']-1)/(rows[i-1]['n']-1))) for i in [1,2]]
    t=rows[0]['tensor']
    gates={'geometry':all(r['jacobian_min']>0 and r['normal_mismatch']<1e-9 and r['actual_scri_radius_error']<1e-14 and r['metric_identity_max']<1e-9 and r['mortar_amplification']<16 for r in rows),
           'affine_and_adjoint':all(r['affine_max']<1e-9 and r['adjoint_relative_max']<1e-10 for r in rows),
           'smooth_refinement':min(orders)>1.5,
           'constant_metric_energy':all(r['energy_relative_defect']<1e-9 and r['symmetry_relative_defect']<1e-9 and r['rayleigh']>0 for r in t['constant_metric'].values()),
           'variable_metric_variation':min(r['scaled_rms'] for r in t['variable_metric_frechet'])<1e-7,
           'amr_components':all(amr['gates'].values())}
    return {'scope':__doc__,'half_width':.2,'physical_radial_knots':[0.,.4,.55,.7,1.],'rows':rows,'orders':orders,'amr':amr,'gates':gates,'evolution_adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');result=run()
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
