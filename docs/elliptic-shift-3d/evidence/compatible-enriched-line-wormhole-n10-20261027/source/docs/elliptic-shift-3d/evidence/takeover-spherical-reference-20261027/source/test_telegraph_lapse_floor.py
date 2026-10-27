#!/usr/bin/env python3
"""Physical-lapse floor, stage projection, disable switch, and compensation tests."""
import argparse,json
import numpy as np
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from compensated_update import accumulate
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
opts=options(65,a.library);opts.grid_stretch=4.;ev=PunctureChart(opts)
q,*_=ev.initial_wormhole(IsotropicWormhole(1/3,96));r=ev.r;O=(1-r*r)/2;A=(1+r*r)/2;ref=A*r*r/(r*r+O**4);positive=(r>0)&(r<1)
def engine(floor=1e-10):return TelegraphEngine(ev,a.fast_library,.005,.02,inner_profile='af-regular',lapse_floor=floor)
e=engine();before=q.copy();low=q.copy();low[1,6]=-1000.;comp=np.arange(9*len(r),dtype=float)*1e-18;e.compensation(comp);e.apply_lapse_floor(low)
assert low[0,6]==before[0,6] and low[-1,6]==before[-1,6]
assert ref[1]*np.exp(low[1,6])/O[1]>=1e-10
np.testing.assert_array_equal(low[:,[0,1,2,3,4,5,7]],before[:,[0,1,2,3,4,5,7]])
after=e.compensation();assert after[14]==0;comp[14]=0;np.testing.assert_array_equal(comp,after);e.close()
e=engine(0);disabled=before.copy();disabled[1,6]=-1000.;copy=disabled.copy();e.apply_lapse_floor(disabled);np.testing.assert_array_equal(disabled,copy);e.close()
for value in [-1,float('nan'),float('inf')]:
 try:e=engine(value)
 except ValueError:pass
 else:e.close();raise AssertionError('Accepted invalid floor')
# Independently project every stage in a Python RK4 oracle. Native evolution
# must agree even when the source repeatedly tries to decrease the lapse.
floor=.001;e=engine(floor);state=before.copy();mask=(r>0)&(r<.2);state[mask,6]=np.log(floor*O[mask]/ref[mask]);B=-100*r*np.exp(-(r/.2)**2);dt=1e-6
threshold=np.full_like(r,-np.inf);threshold[positive]=np.log(floor*O[positive]/ref[positive])
def project(x):
 y=x.copy();y[positive,6]=np.maximum(y[positive,6],threshold[positive]);return y
q0=project(state);k1,b1,_=e.rhs(q0,B);k2,b2,_=e.rhs(project(q0+dt*k1/2),B+dt*b1/2);k3,b3,_=e.rhs(project(q0+dt*k2/2),B+dt*b2/2);k4,b4,_=e.rhs(project(q0+dt*k3),B+dt*b3)
expected=project(q0+dt*(k1+2*k2+2*k3+k4)/6);expectedB=B+dt*(b1+2*b2+2*b3+b4)/6
actual=state.copy();actualB=B.copy();done,error=e.advance(actual,actualB,np.empty((0,2)),1,dt,0.)
assert done==1 and error is None,error
np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-13);np.testing.assert_allclose(actualB,expectedB,rtol=3e-13,atol=3e-13)
assert np.all(ref[positive]*np.exp(actual[positive,6])/O[positive]>=floor)
assert e.diagnostics()['max_lapse_floor_clamps_per_call']>0;e.close()
print(json.dumps(dict(default=1e-10,physical_lapse=True,AF_and_scri_preserved=True,only_lapse_clamped=True,compensation_reset=True,disabled_and_invalid_inputs=True,independent_projected_RK4_oracle=True)))
