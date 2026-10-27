#!/usr/bin/env python3
"""Layer-local coefficient checks, with unaltered interface jumps and all nodes."""
import argparse,json
from pathlib import Path
import numpy as np
from endpoint_sbp import shell_grid,local
from layer_affine_sbp import LayerAffineGrid
from qualify_mapped_sbp import tensor_checks
import mapped_sbp_probe as mapped
import amr_shell_sbp_probe as combined
import qualify_amr_shell_sbp as amr_experiment


def amr_grid(n):
    m,c=mapped.lobatto,combined.lobatto
    try:
        mapped.lobatto=local;combined.lobatto=local
        return LayerAffineGrid(combined.AMRShellGrid(n))
    finally:mapped.lobatto=m;combined.lobatto=c


def run():
    rows=[]
    for n in [10,12,14]:
        base=shell_grid(n,radial=(0.,.45,1.),transfer_seed='zero');grid=LayerAffineGrid(base)
        X=grid.coordinates;W=grid.weights;coef=np.array([.3,.2,-.1])
        u=np.exp(np.einsum('...i,i->...',X,coef,optimize=False));errors=[];affine=0.;adjoint=0.
        rng=np.random.default_rng(4731);f=rng.normal(size=W.shape);g=rng.normal(size=W.shape)
        jump=np.zeros_like(W);jump[1:7]=1.;jump_defect=0.;jump_energies=[]
        for a in range(3):
            errors.append(grid.derivative(u,a,-1)-coef[a]*u)
            for sign in [-1,0,1]:
                affine=max(affine,float(np.max(abs(grid.derivative(X,a,sign)-np.eye(3)[a]))),float(np.max(abs(grid.derivative(np.ones_like(W),a,sign)))))
            left=float(np.sum(W*f*grid.derivative(g,a,-1)));right=float(np.sum(W*g*grid.derivative(f,a,1)));b=grid.boundary_integral(f,g,a)
            adjoint=max(adjoint,abs(left+right-b)/max(1,abs(left),abs(right),abs(b)))
            # Constants on each block have zero volume high-mode action.
            # Thus these energies isolate the retained interface jump form.
            old=.5*(base.derivative(jump,a,1)-base.derivative(jump,a,-1))
            new=.5*(grid.derivative(jump,a,1)-grid.derivative(jump,a,-1))
            jump_defect=max(jump_defect,float(np.max(abs(old-new))))
            jump_energies.append(float(np.sum(W*jump*new)))
        error2=np.sum(np.array(errors)**2,axis=0)
        row={'n':n,'cells':grid.cells,'affine_max':affine,'adjoint_relative_max':adjoint,
             'smooth_rms':float(np.sqrt(np.sum(W*error2)/np.sum(W))),
             'jump_action_change_max':jump_defect,'jump_energies':jump_energies,'moments':grid.diagnostic}
        if n==10:row['tensor']=tensor_checks(grid)
        rows.append(row)
    # A core-supported field cannot alter the far outer layer through K.
    base=shell_grid(10,radial=(0.,.45,.8,1.),transfer_seed='zero');grid=LayerAffineGrid(base)
    v=np.zeros_like(base.weights);v[0]=np.sin(base.coordinates[0,...,0])
    locality=max(float(np.max(abs(grid.derivative(v,a,s)[13:]))) for a in range(3) for s in [-1,0,1])
    amr_experiment.AMRShellGrid=amr_grid;amr=amr_experiment.run()
    orders=[float(np.log(rows[i-1]['smooth_rms']/rows[i]['smooth_rms'])/np.log((rows[i]['n']-1)/(rows[i-1]['n']-1))) for i in [1,2]]
    t=rows[0]['tensor']
    gates={'affine_and_adjoint':all(r['affine_max']<1e-9 and r['adjoint_relative_max']<1e-10 for r in rows),
           'interface_jump_preserved':all(r['jump_action_change_max']<1e-9 and min(r['jump_energies'])>0 for r in rows),
           'radial_locality':locality<1e-12,'smooth_refinement':min(orders)>1.5,
           'constant_metric_energy':all(r['energy_relative_defect']<1e-9 and r['symmetry_relative_defect']<1e-9 and r['rayleigh']>0 for r in t['constant_metric'].values()),
           'variable_metric_variation':min(r['scaled_rms'] for r in t['variable_metric_frechet'])<1e-7,
           'amr_components':all(amr['gates'].values())}
    return {'scope':__doc__,'rows':rows,'orders':orders,'core_to_outer_derivative_max':locality,'amr':amr,'gates':gates,'evolution_adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');result=run()
    (a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
