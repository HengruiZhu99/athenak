#!/usr/bin/env python3
"""Independent total-lapse RK4 oracle and compensated split-step check."""
import argparse,json
import numpy as np
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
opts=options(65,a.library);opts.grid_stretch=4.;ev=PunctureChart(opts)
q,*_=ev.initial_wormhole(IsotropicWormhole(1/3,96));r=ev.r;O=(1-r*r)/2;ref=(1+r*r)/2*r*r/(r*r+O**4);inside=(r>0)&(r<1);floor=.001
def engine():return TelegraphEngine(ev,a.fast_library,.005,.02,inner_profile='af-regular',lapse_floor=floor,lapse_update='total')
e=engine();e.apply_lapse_floor(q);B=-100*r*np.exp(-(r/.2)**2);dt=1e-6
def to_total(q):
 y=q.copy();y[inside,6]=ref[inside]*np.exp(q[inside,6]);return y
def to_chart(y):
 x=y.copy();x[inside,6]=np.log(np.maximum(y[inside,6],floor*O[inside])/ref[inside]);return x
def rate(y,b):
 x=to_chart(y);dq,db,_=e.rhs(x,b);dq[inside,6]*=ref[inside]*np.exp(x[inside,6]);return dq,db
y=to_total(q);k1,b1=rate(y,B);k2,b2=rate(y+dt*k1/2,B+dt*b1/2);k3,b3=rate(y+dt*k2/2,B+dt*b2/2);k4,b4=rate(y+dt*k3,B+dt*b3)
expected=to_chart(y+dt*(k1+2*k2+2*k3+k4)/6);expectedB=B+dt*(b1+2*b2+2*b3+b4)/6
actual=q.copy();flux=B.copy();done,error=e.advance(actual,flux,np.empty((0,2)),1,dt,0.);assert done==1 and error is None,error
np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12);np.testing.assert_allclose(flux,expectedB,rtol=3e-12,atol=3e-12)
assert np.all(ref[inside]*np.exp(actual[inside,6])/O[inside]>=floor);e.close()
whole=engine();split=engine();x=q.copy();y=q.copy();bx=B.copy();by=B.copy();empty=np.empty((0,2))
assert whole.advance(x,bx,empty,10,dt/10,0.)==(10,None)
assert split.advance(y,by,empty,3,dt/10,0.)==(3,None)
saved=split.compensation();split.close();split=engine();split.compensation(saved)
assert split.advance(y,by,empty,7,dt/10,3*dt/10)==(7,None)
for aa,bb in [(x,y),(bx,by),(whole.compensation(),split.compensation())]:np.testing.assert_array_equal(aa,bb)
whole.close();split.close()
print(json.dumps(dict(independent_total_RK4=True,physical_floor=True,split_restart_exact=True)))
