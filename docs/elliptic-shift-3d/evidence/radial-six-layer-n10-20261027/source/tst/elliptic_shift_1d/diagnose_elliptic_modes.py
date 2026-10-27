#!/usr/bin/env python3
"""Finite-grid linear mode diagnostic for shift discretizations, not a theorem."""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evolve import Evolution
from fast import FastEngine
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--resolutions',type=int,nargs='+',default=[32,64,128]);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);records=[]
for n in a.resolutions:
 for method,bias in [('quadrature','centered'),('discrete-ode','centered'),('discrete-ode','right')]:
  for epsilon in [1e-6,1e-7]:
    opts=SimpleNamespace(n=n,origin_grid=False,puncture=False,origin_variables=False,backend='spherical-g1',
       width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,theta='appendix-c',
       slicing=1.,lapse_restoring=1.,endpoint='analytic-g1',gauge_corner_sat=0.,advection='centered',
       scri_sat=4.,regularity_sat=8.,origin_ko=.5,origin_ko_radius=.4,
       metric_variable='log',lapse_variable='log-reference',elliptic_method=method,elliptic_refinements=2,
       elliptic_quadrature='inward',elliptic_bias=bias)
    ev=Evolution(opts);engine=FastEngine(ev,8,a.fast_library);indices=[i*8+f for i in range(n) for f in [0,2,3,4,5,6] if not(i==n-1 and f==6)]
    J=np.empty((len(indices),len(indices)));start=time.perf_counter()
    for col,k in enumerate(indices):
        q=np.zeros((n,8));q.flat[k]=epsilon;plus=engine.rhs(q)[0].ravel()[indices]
        q.flat[k]=-epsilon;minus=engine.rhs(q)[0].ravel()[indices];J[:,col]=(plus-minus)/(2*epsilon)
    values,vectors=np.linalg.eig(J);k=np.argmax(values.real);mode=np.zeros((n,8),dtype=complex);mode.flat[indices]=vectors[:,k]
    energy=np.sum(abs(mode)**2);record=dict(n=n,method=method,bias=bias,epsilon=epsilon,dominant_real=float(values[k].real),
        dominant_imaginary=float(values[k].imag),h_real=float(ev.h*values[k].real),
        inner_fraction=float(np.sum(abs(mode[ev.r<.1])**2)/energy),endpoint_fraction=float(np.sum(abs(mode[ev.r>.9])**2)/energy),
        field_energy_fraction=(np.sum(abs(mode)**2,axis=0)/energy).tolist(),runtime_seconds=time.perf_counter()-start)
    np.savez(a.output/f'n{n}-{method}-{bias}-eps{epsilon}.npz',J=J,r=ev.r,eigenvalues=values,dominant_mode=mode)
    records.append(record);engine.close();print(json.dumps(record),flush=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=dict(scope='Discrete Jacobian on the full finite-grid phase. Compatible continuum weighted jets are not imposed; this does not prove or refute nonlinear continuum stability.',
    library_sha256=sha(a.library),fast_library_sha256=sha(a.fast_library),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},records=records)
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
