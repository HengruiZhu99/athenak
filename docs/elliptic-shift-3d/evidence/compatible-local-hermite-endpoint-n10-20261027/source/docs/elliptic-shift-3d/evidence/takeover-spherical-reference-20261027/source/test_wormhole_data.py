#!/usr/bin/env python3
"""Exact-jet constraints, mass, actual-scri and inner-end geometry of wormhole data."""
import argparse,json
from pathlib import Path
import numpy as np
from wormhole_data import data,point
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path);args=p.parse_args()
r=np.unique(np.r_[np.linspace(.005,.99,401),1.,1e-3,2e-3,4e-3])
q,d,dd,meta=data(r,mass=.1)
R,S,c=Kernel(args.library).parts(r,q,d,dd,-r,-np.ones_like(r),np.zeros_like(r),total_chi=np.array([x['chi_total'] for x in meta]))
# The arbitrary beta=-r here serves diagnostics only, not a shift solve/evolution.
assert np.max(abs(c[:,0]))<5e-11, np.max(abs(c[:,0]))
assert np.max(abs(c[:,1]))<5e-11, np.max(abs(c[:,1]))
assert np.max(abs(c[:,3]))<2e-13
assert np.max(abs(c[r<.95,10]-.1))<2e-10
# Near scri the generic mass formula subtracts terms of size R^2; report its
# roundoff amplification separately rather than claiming uniform mass precision.
assert np.max(abs(c[:-1,10]-.1))<1e-8
assert np.max(abs(S[-1,:7]))<2e-12, S[-1]
assert abs(c[-1,6])<2e-14
inner=[]
for x in [1e-3,5e-4,2.5e-4]:
 v,_,_,m=point(x,mass=.1);rho=x/(1-x);psi=1+.1/(2*rho)
 assert abs(m['physical_lapse']-psi**-2)<1e-17
 assert m['branch']=='isotropic' and m['K_phys']==0
 inner.append(dict(r=x,R=m['R'],lapse_phys=m['physical_lapse'],sqrt_gamma_rr=np.sqrt(m['physical_radial_metric']),chi=1+v[0]))
# R~1/r and proper-distance density~1/r^2 distinguish this data from a trumpet.
assert inner[-1]['R']/inner[-2]['R']>1.9
assert inner[-1]['sqrt_gamma_rr']/inner[-2]['sqrt_gamma_rr']>3.9
record={'kind':'initial-data validation only; no wormhole-to-trumpet evolution',
 'mass':.1,'a':1.,'transition_R':[.4,.8],'n_points':len(r),
 'max_H':float(max(abs(c[:,0]))),'max_M':float(max(abs(c[:,1]))),
 'max_Z':float(max(abs(c[:,3]))),'max_mass_error':float(max(abs(c[:-1,10]-.1))),
 'scri_poles':S[-1].tolist(),'scri_null':float(c[-1,6]),'puncture':inner,
 'embedding':'T=t+h(R); h_R=s(R)(R/a)/(F sqrt(F+(R/a)^2)); unchanged Einstein-Rosen interior',
 'physical_lapse':'psi^-2 in isotropic interior; smooth conformal lapse (1-s)Omega psi^-2+s A',
 'shift':'beta=-r used for gauge-independent diagnostics only; puncture elliptic boundary conditions not yet admitted'}
if args.output:args.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
