#!/usr/bin/env python3
"""Compiled/Python agreement for independent positive-field flags and KO."""
import argparse,json
from itertools import product
from types import SimpleNamespace
import numpy as np
from evolve import Evolution,CoupledEvolution
from fast import FastEngine
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args();records=[]
for index,(origin,scalar,variables,endpoint,ko) in enumerate(product([False,True],[False,True],
    [('log','deviation'),('deviation','log-reference'),('log','log-reference')],['raw-lhopital','analytic-g1'],[0.,.5])):
    opts=SimpleNamespace(n=65,origin_grid=origin,puncture=False,origin_variables=False,backend='spherical-g1',
        width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,theta='appendix-c' if index%2 else 'repository',
        slicing=1.,lapse_restoring=1.,lapse_restoring_form='multiplicative' if index%3 else 'symmetric',
        damping_form='reference-lapse' if index%5 else 'constant',endpoint=endpoint,gauge_corner_sat=0.,advection='centered',
        scri_sat=4.,regularity_sat=8.,origin_ko=ko,origin_ko_radius=.4,metric_variable=variables[0],lapse_variable=variables[1])
    opts.elliptic_quadrature='inward' if index%2 else 'prefix'
    opts.elliptic_method='discrete-ode' if index%4==0 else 'quadrature'
    opts.elliptic_bias='right'
    opts.slicing_family=['linear','rational','blended-log'][index%3];opts.slicing_scale=.1
    ev=(CoupledEvolution if scalar else Evolution)(opts);r=ev.r;nf=10 if scalar else 8;q=np.zeros((len(r),nf));f=(1-r*r)**4
    for field,amplitude,power in [(0,.002,0),(2,.001,2),(3,.0015,0),(4,.0001,0),(5,.0002,1),(6,.002,0)]:q[:,field]=amplitude*r**power*f
    if scalar:q[:,8]=.0001*f;q[:,9]=.0002*f
    engine=FastEngine(ev,nf,a.fast_library);expected=ev.rhs(q,accepted=True);got,flux=engine.rhs(q)
    error=float(max(abs(expected-got).ravel()));assert error<2e-11,error
    if scalar:np.testing.assert_allclose(flux,4*np.pi*ev.scalar_scri_rate**2,rtol=1e-11,atol=1e-20)
    state=q.copy();energy=0.;dt=.01*ev.h
    for _ in range(20):
        k1=ev.rhs(state,accepted=True);f1=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
        k2=ev.rhs(state+dt*k1/2);f2=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
        k3=ev.rhs(state+dt*k2/2);f3=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
        k4=ev.rhs(state+dt*k3);f4=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
        state+=dt*(k1+2*k2+2*k3+k4)/6;energy+=dt*(f1+2*f2+2*f3+f4)/6
    done,got_energy,failed=engine.advance(q,20,dt);assert done==20 and not failed
    step_error=float(max(abs(q-state).ravel()));assert step_error<2e-13,step_error
    np.testing.assert_allclose(energy,got_energy,rtol=1e-10,atol=1e-20)
    stats=engine.diagnostics()
    for key in ['max_scri_pole','max_lapse_pole','max_regularity_trace','min_chi','min_alpha',
        'max_H_rms','max_M_rms','max_Theta_rms','max_Z_rms','max_scri_null','max_scri_clock','max_Bondi_nonaffinity',
        'max_accepted_H_rms','max_accepted_M_rms','max_accepted_Theta_rms','max_accepted_Z_rms',
        'max_accepted_H_bulk','max_accepted_M_bulk','max_accepted_H_endpoint','max_accepted_M_endpoint']:
        np.testing.assert_allclose(stats[key],ev.stage_metrics[key],atol=2e-11,rtol=1e-8,err_msg=key)
    engine.close();records.append(dict(origin=origin,scalar=scalar,variables=variables,endpoint=endpoint,KO=ko,rhs_error=error,step_error=step_error))
# A separate RHS-only test checks total fields below reference relative
# precision. Such highly off-constraint data can be stiff and are not evolved.
opts.origin_ko=0.;ev=CoupledEvolution(opts);q=np.zeros((65,10));q[:,0]=q[:,6]=-40*(1-ev.r**2)**4;q[:,9]=1.
engine=FastEngine(ev,10,a.fast_library);expected=ev.rhs(q);got,_=engine.rhs(q)
np.testing.assert_allclose(got,expected,atol=2e-10,rtol=2e-12)
np.testing.assert_allclose(got[:,8],ev.total_alpha,atol=0,rtol=2e-13)
engine.close()
q[:]=0;epsilon=1e-25;q[:,6]=epsilon*(1-ev.r**2)**2
engine=FastEngine(ev,10,a.fast_library);expected=ev.rhs(q);got,_=engine.rhs(q)
tiny_error=float(max(abs(got-expected).ravel())/epsilon)
assert tiny_error<1e-8 and max(abs(got[:,2]))>epsilon*.1,tiny_error
engine.close()
print(json.dumps(dict(cases=len(records),max_rhs_error=max(z['rhs_error'] for z in records),
    max_step_error=max(z['step_error'] for z in records),all_stage_diagnostics_and_RK_flux_match=True,
    below_reference_relative_precision_RHS_agreement=True,
    tiny_deviation_RHS_error_relative_to_amplitude=tiny_error,
    scope='Cross-implementation audit, not nonlinear stability evidence')))
