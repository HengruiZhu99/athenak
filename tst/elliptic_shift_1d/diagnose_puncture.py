#!/usr/bin/env python3
"""Instantaneous AF-chart Jacobian; not a continuum/nonlinear stability theorem."""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--n',type=int,default=129);p.add_argument('--ko',type=float,default=.5)
p.add_argument('--mass',type=float,default=.1);p.add_argument('--epsilon',type=float,default=1e-6)
p.add_argument('--state',type=Path)
p.add_argument('--damping-form',choices=['constant','reference-lapse'],default='constant');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
opts=SimpleNamespace(n=a.n,origin_grid=False,puncture=True,puncture_chart=True,origin_variables=False,
    backend='spherical-g1',width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,
    theta='appendix-c',slicing=1.,lapse_restoring=1.,lapse_restoring_form='multiplicative',damping_form=a.damping_form,
    endpoint='raw-lhopital',scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,
    puncture_ko=a.ko,puncture_ko_radius=.8,advection='upwind',eta=0.,gauge_corner_sat=0.)
ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole(a.mass))
if a.state:
    with np.load(a.state,allow_pickle=False) as z:q=z['q'].copy()
    if q.shape!=(a.n,8):raise ValueError('Snapshot shape mismatch')
indices=np.array([i*8+f for i in range(a.n) for f in [0,2,3,4,5,6] if not (i==a.n-1 and f==6)])
J=np.empty((len(indices),len(indices)));start=time.perf_counter()
for col,index in enumerate(indices):
    plus=q.copy();minus=q.copy();plus.flat[index]+=a.epsilon;minus.flat[index]-=a.epsilon
    J[:,col]=(ev.rhs(plus).ravel()[indices]-ev.rhs(minus).ravel()[indices])/(2*a.epsilon)
values,vectors=np.linalg.eig(J);order=np.argsort(values.real)[::-1];records=[]
for k in order[:8]:
    mode=np.zeros_like(q,dtype=complex);mode.flat[indices]=vectors[:,k];energy=np.sum(abs(mode)**2,axis=1)
    records.append(dict(real=float(values[k].real),imag=float(values[k].imag),
        roughness=float(np.sum(abs(np.diff(mode,axis=0))**2)/(4*np.sum(energy))),
        inner_energy_fraction=float(np.sum(energy[ev.r<.1])/np.sum(energy)),
        transition_energy_fraction=float(np.sum(energy[(ev.r>.1)&(ev.r<.4)])/np.sum(energy)),
        scri_energy_fraction=float(np.sum(energy[ev.r>.9])/np.sum(energy))))
dominant=np.zeros_like(q,dtype=complex);dominant.flat[indices]=vectors[:,order[0]]
np.savez(a.output/'jacobian.npz',J=J,eigenvalues=values,dominant=dominant,r=ev.r,background=q)
record=dict(n=a.n,ko=a.ko,mass=a.mass,epsilon=a.epsilon,damping_form=a.damping_form,
    modes=records,runtime_seconds=time.perf_counter()-start,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    background_snapshot=None if a.state is None else dict(path=str(a.state),sha256=hashlib.sha256(a.state.read_bytes()).hexdigest()),
    interpretation='Instantaneous nonlinear RHS derivative about nonstationary initial data; positive real parts alone do not prove instability. Resolution, frequency and localization must be assessed.')
(a.output/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
