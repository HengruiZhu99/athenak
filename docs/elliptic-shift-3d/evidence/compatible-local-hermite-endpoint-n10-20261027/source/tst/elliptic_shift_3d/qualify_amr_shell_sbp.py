#!/usr/bin/env python3
"""Combined AMR/core/shell spatial checks; no scri or evolution acceptance."""
import argparse,json,time
from pathlib import Path
import numpy as np
from amr_shell_sbp_probe import AMRShellGrid
from qualify_mapped_sbp import tensor_checks


def run():
    report={'scope':__doc__,'rows':[]}
    for n in [10,11,12]:
        grid=AMRShellGrid(n);X=grid.coordinates;W=grid.weights
        coefficients=np.array([.3,.2,-.1]);f=np.exp(np.einsum('...i,i->...',X,coefficients,optimize=False))
        rng=np.random.default_rng(427);u=rng.normal(size=f.shape);v=rng.normal(size=f.shape)
        errors=[];adjoint=[];constant=0.
        for a in range(3):
            errors.append(grid.derivative(f,a,-1)-coefficients[a]*f)
            constant=max(constant,float(np.max(abs(grid.derivative(np.ones_like(f),a,-1)))))
            left=float(np.sum(W*u*grid.derivative(v,a,-1)));right=float(np.sum(W*v*grid.derivative(u,a,1)))
            boundary=grid.boundary_integral(u,v,a)
            adjoint.append(abs(left+right-boundary)/max(1,abs(left),abs(right),abs(boundary)))
        error2=np.sum(np.array(errors)**2,axis=0);nc=grid.core_count
        report['rows'].append({'n':n,'cells':grid.cells,'core_leaves':nc,'shell_blocks':len(grid.blocks)-nc,
                              'rms':float(np.sqrt(np.sum(W*error2)/np.sum(W))),
                              'maximum':float(np.max(abs(np.array(errors)))),
                              'core_rms':float(np.sqrt(np.sum(W[:nc]*error2[:nc])/np.sum(W[:nc]))),
                              'shell_rms':float(np.sqrt(np.sum(W[nc:]*error2[nc:])/np.sum(W[nc:]))),
                              'constant_derivative_max':constant,'adjoint_relative_max':max(adjoint),
                              'interface_value_L1_max':grid.amplification,
                              'transfer_adjoint_max':grid.transfer_adjoint_max,
                              'transfer_composition_max':grid.transfer_composition_max,
                              'transfer_tensor_polynomial_max':grid.transfer_polynomial_max})
        if n==10:report['tensor']=tensor_checks(grid)
    r=report['rows'];report['orders']=[float(np.log(r[i-1]['rms']/r[i]['rms'])/np.log(r[i]['n']/r[i-1]['n'])) for i in range(1,len(r))]
    report['gates']={'all_cell_refinement':min(report['orders'])>2,
                     'constant_and_adjoint':all(x['constant_derivative_max']<1e-8 and x['adjoint_relative_max']<1e-10 for x in r),
                     'interface_amplification':max(x['interface_value_L1_max'] for x in r)<16,
                     'interface_polynomials':max(x['transfer_tensor_polynomial_max'] for x in r)<1e-9,
                     'constant_metric_energy':all(x['energy_relative_defect']<1e-9 and x['symmetry_relative_defect']<1e-9 and x['rayleigh']>0 for x in report['tensor']['constant_metric'].values()),
                     'full_metric_variation':min(x['scaled_rms'] for x in report['tensor']['variable_metric_frechet'])<1e-7}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    result=run();result['seconds']=time.monotonic()-start;(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
