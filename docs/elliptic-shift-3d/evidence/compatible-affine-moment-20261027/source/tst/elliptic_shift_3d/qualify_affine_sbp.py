#!/usr/bin/env python3
"""Qualify a derivative-coefficient update, without evolving or resetting fields."""
import argparse,json,time
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from amr_shell_sbp_probe import AMRShellGrid
from affine_sbp_probe import AffineConsistentGrid
from qualify_mapped_sbp import tensor_checks
from compatible_tensor_gauge import TensorGauge


def run():
    report={'scope':__doc__,'rows':[]}
    for amr,n in [(False,10),(False,12),(False,14),(True,10)]:
        base=AMRShellGrid(n) if amr else ShellGrid(n,transfer_seed='zero')
        grid=AffineConsistentGrid(base);X=grid.coordinates;W=grid.weights
        coeff=np.array([.3,.2,-.1]);exponential=np.exp(np.einsum('...i,i->...',X,coeff,optimize=False))
        quadratic=X[...,0]*X[...,1]+.2*X[...,2]**2
        qtruth=[X[...,1],X[...,0],.4*X[...,2]]
        trig=np.sin(2*X[...,0])*np.cos(X[...,1])+.3*np.sin(3*X[...,2])
        ttruth=[2*np.cos(2*X[...,0])*np.cos(X[...,1]),-np.sin(2*X[...,0])*np.sin(X[...,1]),.9*np.cos(3*X[...,2])]
        rng=np.random.default_rng(271131);u=rng.normal(size=W.shape);v=rng.normal(size=W.shape)
        affine=0.;adjoint=0.;errors={'exponential':[],'quadratic':[],'trigonometric':[]}
        corrections=[];r_energy=[]
        for axis in range(3):
            for sign in [-1,0,1]:
                affine=max(affine,float(np.max(abs(grid.derivative(X,axis,sign)-np.eye(3)[axis]))),float(np.max(abs(grid.derivative(np.ones_like(W),axis,sign)))))
            dm=grid.derivative(v,axis,-1);dp=grid.derivative(u,axis,1)
            left=float(np.sum(W*u*dm));right=float(np.sum(W*v*dp));boundary=grid.boundary_integral(u,v,axis)
            adjoint=max(adjoint,abs(left+right-boundary)/max(1,abs(left),abs(right),abs(boundary)))
            errors['exponential'].append(grid.derivative(exponential,axis,-1)-coeff[axis]*exponential)
            errors['quadratic'].append(grid.derivative(quadratic,axis,-1)-qtruth[axis])
            errors['trigonometric'].append(grid.derivative(trig,axis,-1)-ttruth[axis])
            corrections.append(grid.derivative(trig,axis,-1)-base.derivative(trig,axis,-1))
            r_energy.append(float(np.sum(W*v*(grid.derivative(v,axis,1)-grid.derivative(v,axis,0)))))
        rms=lambda e:float(np.sqrt(np.sum(W*np.sum(np.array(e)**2,axis=0))/np.sum(W)))
        metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy();gauge=TensorGauge(grid,metric)
        h=gauge.rate(-X);rate=gauge.variation(h)
        row={'n':n,'amr':amr,'cells':grid.cells,'affine_derivative_max':affine,'adjoint_relative_max':adjoint,'smooth_rms':{k:rms(v) for k,v in errors.items()},'trigonometric_correction_rms':rms(corrections),'high_mode_energy_min':min(r_energy),'affine_metric_rate_max':float(np.max(abs(h))),'affine_connection_rate_max':float(np.max(abs(rate))),'coefficient_diagnostics':grid.diagnostic}
        report['rows'].append(row)
        if n==10:row['tensor']=tensor_checks(grid)
    family=report['rows'][:3];orders={}
    for name in errors:
        orders[name]=[float(np.log(family[i-1]['smooth_rms'][name]/family[i]['smooth_rms'][name])/np.log(family[i]['n']/family[i-1]['n'])) for i in [1,2]]
    report['orders']=orders
    report['gates']={'affine_exactness':all(r['affine_derivative_max']<1e-9 and r['affine_connection_rate_max']<1e-8 for r in report['rows']),
                     'paired_identity':all(r['adjoint_relative_max']<1e-10 and r['high_mode_energy_min']>0 for r in report['rows']),
                     'unrelated_smooth_refinement':min(min(v) for v in orders.values())>2,
                     'correction_decreases':all(family[i]['trigonometric_correction_rms']<family[i-1]['trigonometric_correction_rms'] for i in [1,2]),
                     'tensor_energy':all(c['energy_relative_defect']<1e-9 and c['symmetry_relative_defect']<1e-9 and c['rayleigh']>0 for r in report['rows'] if 'tensor' in r for c in r['tensor']['constant_metric'].values()),
                     'full_metric_variation':all(min(c['scaled_rms'] for c in r['tensor']['variable_metric_frechet'])<1e-7 for r in report['rows'] if 'tensor' in r)}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    r=run();r['seconds']=time.monotonic()-start;(a.output/'assessment.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if all(r['gates'].values()) else 1)
