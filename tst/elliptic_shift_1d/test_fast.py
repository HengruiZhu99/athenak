#!/usr/bin/env python3
"""Cross-implementation RHS/RK/flux checks before any long compiled runs."""
import argparse,json,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from evolve import Evolution,CoupledEvolution
from fast import FastEngine

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True)
p.add_argument('--output',type=Path);a=p.parse_args();cases=[]
for origin,puncture in [(False,False),(True,False),(False,True)]:
 for scalar in [False,True]:
  for appendix in [False,True]:
   for ko,endpoint,corner,advection,form in [(k,e,g,adv,f) for k in ([0.] if puncture else [0.,.5]) for e in ['raw-lhopital','analytic-g1'] for g in [0.,4.] for adv in ['centered','upwind'] for f in ['symmetric','multiplicative']]:
    opts=SimpleNamespace(n=65,origin_grid=origin,puncture=puncture,origin_variables=False,backend='spherical-g1',
       width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,
       theta='appendix-c' if appendix else 'repository',slicing=1.,lapse_restoring=1.,lapse_restoring_form=form,
       endpoint=endpoint,gauge_corner_sat=corner,advection=advection,scri_sat=4.,regularity_sat=8.,origin_ko=ko,origin_ko_radius=.4)
    ev=(CoupledEvolution if scalar else Evolution)(opts);r=ev.r;nf=10 if scalar else 8
    rng=np.random.default_rng(601);q=np.zeros((65,nf))
    for f,power in [(0,0),(2,2),(3,0),(4,0),(5,1),(6,0)]+([(8,0),(9,0)] if scalar else []):
     q[:,f]=r**power*sum(rng.normal()*.001*r**(2*k) for k in range(4))
    q[-1,6]=0
    fast=FastEngine(ev,nf,a.fast_library);expected=ev.rhs(q);got,flux=fast.rhs(q)
    error=float(np.max(abs(got-expected)));assert error<2e-11,(origin,scalar,appendix,ko,error)
    if scalar:np.testing.assert_allclose(flux,4*np.pi*ev.scalar_scri_rate**2,rtol=2e-12,atol=1e-20)
    # Use compatible, smooth, small data for the step comparison.
    q[:]=0;q[:,6]=.001*np.exp(-((r*r-.25**2)**2)/(4*.1**4));q[-1,6]=0
    if scalar:q[:,8]=.0001*(1-r*r)**4;q[:,9]=2*q[:,8]
    fast.close();fast=FastEngine(ev,nf,a.fast_library)
    observed=[]
    if scalar and not origin and appendix and ko==.5 and endpoint=='analytic-g1':
     original=ev.rhs
     def monitored(state):
      rate=original(state);observed.append(ev.report(state,0.)[0]);return rate
     ev.rhs=monitored
    reference=q.copy();dt=.02*ev.h;energy=0
    for _ in range(20):
     k1=ev.rhs(reference);f1=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0
     k2=ev.rhs(reference+dt*k1/2);f2=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0
     k3=ev.rhs(reference+dt*k2/2);f3=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0
     k4=ev.rhs(reference+dt*k3);f4=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0
     reference+=dt*(k1+2*k2+2*k3+k4)/6;energy+=dt*(f1+2*f2+2*f3+f4)/6
    done,ef,failed=fast.advance(q,20,dt)
    assert done==20 and not failed
    step_error=float(np.max(abs(q-reference)));assert step_error<2e-13,step_error
    np.testing.assert_allclose(ef,energy,rtol=1e-10,atol=1e-20)
    if observed:
     stats=fast.diagnostics()
     for key,field in [('max_scri_pole','scri_pole'),('max_lapse_pole','scri_lapse_pole'),
       ('max_H_rms','H_rms'),('max_M_rms','M_rms'),('max_Theta_rms','Theta_rms'),('max_Z_rms','Z_rms')]:
      np.testing.assert_allclose(stats[key],max(x[field] for x in observed),atol=2e-12,rtol=1e-8)
     for key in ['min_chi','min_alpha']:
      np.testing.assert_allclose(stats[key],min(x[key] for x in observed),atol=2e-13)
    fast.close();cases.append(dict(origin=origin,puncture=puncture,scalar=scalar,appendix=appendix,ko=ko,endpoint=endpoint,corner=corner,advection=advection,restoring_form=form,rhs_error=error,step_error=step_error))
record=dict(cases=cases,scope='Compiled/Python agreement; no new continuum equations or stability claim')
if a.output:a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),max_rhs_error=max(x['rhs_error'] for x in cases),max_step_error=max(x['step_error'] for x in cases))))
