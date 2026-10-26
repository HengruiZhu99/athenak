#!/usr/bin/env python3
"""Independent native rates for a positive, reference-weighted lapse variable.

The variable profile is independent of the selected physical lapse gauge.
No metric/curvature or constraint field is slaved to this profile.
"""
import argparse,json
from itertools import product
from types import SimpleNamespace
import numpy as np
from evolve import Evolution
from trumpet_data import CMCTrumpet
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args();records=[]
for reference,metric in product(['minkowski','CMC-trumpet'],['deviation','log-trumpet']):
 args=SimpleNamespace(n=129,origin_grid=False,puncture=True,origin_variables=False,backend='spherical-g1',
    width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,theta='appendix-c',
    slicing=1.,lapse_restoring=1.,lapse_restoring_form='multiplicative',damping_form='constant',endpoint='raw-lhopital',
    gauge_corner_sat=0.,advection='centered',scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,
    initial_data='trumpet',mass=1/3,initial_quadrature_order=96,lapse_reference=reference,lapse_variable='log-trumpet',metric_variable=metric)
 ev=Evolution(args);initial,*_=CMCTrumpet(args.mass).data(ev.r);state=ev.from_physical(initial)
 np.testing.assert_allclose(ev.to_physical(state),initial,rtol=0,atol=2e-14)
 f=(1-ev.r)**6
 for field,amplitude in [(0,.00001),(2,.001),(3,.0015),(4,.0001),(5,.0002),(6,.002)]:state[:,field]+=amplitude*f
 rate=ev.rhs(state);R,S,diag,b=ev.parts(state);q=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second
 r=ev.r;alpha=ev.total_alpha;chi=ev.total_chi
 db=b/r+1.5*alpha*q[:,2];ddb=1.5*((r+d[:,6])*q[:,2]+alpha*d[:,2])+1.5*alpha*q[:,2]/r
 nR,nS,native_diag=Kernel(a.library).parts(r,q,d,dd,b,db,ddb,total_chi=chi)
 nS[:,6]-=2*alpha*q[:,6];nS[:,6]+=ev.lapse_source
 native=nR[:-1]+nS[:-1]/ev.omega[:-1,None]
 physical=rate.copy();physical[:,6]*=alpha
 if metric=='log-trumpet':physical[:,0]*=chi
 error=float(max(abs(physical[:-1]-native).ravel()));assert error<2e-9,error
 assert all(max(abs(diag[:,i]))>0 for i in range(4))
 diagnostic_error=float(max(abs(diag[:,:4]-native_diag[:,:4]).ravel()));assert diagnostic_error<2e-9,diagnostic_error
 records.append(dict(lapse_reference=reference,metric_variable=metric,native_physical_rate_error=error,native_constraint_error=diagnostic_error,
    independent_constraints=True,min_positive_lapse=float(min(alpha)),
    scope='Invertible lapse-variable/analytic-derivative audit only; no nonlinear stability admission'))
print(json.dumps(records))
