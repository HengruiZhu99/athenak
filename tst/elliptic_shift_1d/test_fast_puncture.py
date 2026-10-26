#!/usr/bin/env python3
"""Independent Python/compiled AF chart RHS, RK and all-stage monitors."""
import argparse,json
from itertools import product
from types import SimpleNamespace
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from fast import FastEngine
from compensated_update import accumulate
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args();records=[]
for n,endpoint,advection,family,damping,profile in product([33,65],['raw-lhopital','analytic-g1'],['centered','upwind'],['linear','rational','blended-log'],['constant','reference-lapse'],['constant','physical-log']):
 opts=SimpleNamespace(n=n,origin_grid=False,puncture=True,puncture_chart=True,origin_variables=False,
  backend='spherical-g1',width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,
  theta='appendix-c' if n==33 else 'repository',slicing=1.,lapse_restoring=1.,lapse_restoring_form='multiplicative',
  endpoint=endpoint,scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,
  advection=advection,eta=0.,gauge_corner_sat=0.,damping_form=damping,slicing_family=family,slicing_scale=.1,
  lapse_background_source='collapsed-inner' if profile=='physical-log' else 'reference',lapse_restoring_profile=profile,
  puncture_ko=.5 if advection=='upwind' else 0.,puncture_ko_radius=.8)
 opts.compensated_update=n==65
 ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole());f=(1-ev.r)**6
 for field,amplitude in [(0,.001),(2,.001),(3,.002),(4,.003),(5,.004),(6,.0015)]:q[:,field]+=amplitude*f
 engine=FastEngine(ev,8,a.fast_library);expected=ev.rhs(q,accepted=True);got,_=engine.rhs(q)
 error=float(max(abs(got-expected).ravel()));np.testing.assert_allclose(got,expected,atol=2e-10,rtol=2e-11)
 assert all(ev.stage_metrics['max_'+name+'_rms']>0 for name in ['H','M','Theta','Z'])
 for key,value in engine.diagnostics().items():
  np.testing.assert_allclose(value,ev.stage_metrics[key],atol=2e-10,rtol=2e-9,err_msg=key)
 state=q.copy();correction=np.zeros_like(state);compiled=q.copy();dt=.002*ev.h
 for _ in range(8):
  k1=ev.rhs(state,accepted=True);k2=ev.rhs(state+dt*k1/2);k3=ev.rhs(state+dt*k2/2);k4=ev.rhs(state+dt*k3)
  increment=dt*(k1+2*k2+2*k3+k4)/6
  if opts.compensated_update:state,correction=accumulate(state,correction,increment)
  else:state+=increment
 done,energy,failed=engine.advance(compiled,8,dt);assert done==8 and not failed,(done,engine.failure_message)
 step_error=float(max(abs(state-compiled).ravel()));np.testing.assert_allclose(compiled,state,atol=2e-12,rtol=2e-12)
 assert energy==0
 for key,value in engine.diagnostics().items():np.testing.assert_allclose(value,ev.stage_metrics[key],atol=2e-9,rtol=2e-8,err_msg=key)
 engine.close();records.append(dict(n=n,endpoint=endpoint,advection=advection,family=family,damping=damping,profile=profile,rhs_error=error,step_error=step_error))
# Collapsed and tiny-deviation states must keep total fields and independent
# curvature/constraint coefficients. These stiff states are RHS-only controls.
ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole());q[:,6]-=30*(1-ev.r)**6
engine=FastEngine(ev,8,a.fast_library);expected=ev.rhs(q);got,_=engine.rhs(q)
np.testing.assert_allclose(got,expected,atol=2e-9,rtol=2e-10);engine.close()
print(json.dumps(dict(cases=len(records),max_rhs_error=max(v['rhs_error'] for v in records),max_step_error=max(v['step_error'] for v in records),independent_constraints_retained=True,all_stage_monitors_match=True,scope='AF chart cross-implementation audit, not nonlinear stability admission')))
