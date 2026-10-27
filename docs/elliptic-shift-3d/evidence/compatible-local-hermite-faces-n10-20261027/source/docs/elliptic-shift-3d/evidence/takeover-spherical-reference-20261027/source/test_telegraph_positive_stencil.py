#!/usr/bin/env python3
"""Positive-node closure: polynomial derivatives and endpoint isolation."""
import argparse,json
import numpy as np
from telegraph import TelegraphEngine,options,python_rhs,window
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
opts=options(65,a.library);opts.grid_stretch=4.;opts.puncture_stencil='positive';opts.inner_total_derivatives_radius=.1;ev=PunctureChart(opts);r=ev.r
assert (ev.op.index[1:]>ev.op.width).all() and (ev.op.aindex[:,1:]>ev.op.width).all()
values=1+3*r+2*r*r;d,dd=ev.op.derivatives(values,1)
np.testing.assert_allclose(d[1:],3+4*r[1:],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(dd[1:],4,rtol=1e-8,atol=1e-8)
q,*_=ev.initial_wormhole(IsotropicWormhole(1/3,96))
# A trumpet-like chi ~ r^2 and constant lapse must have the direct polynomial
# derivatives at the first nodes, regardless of their singular log chart.
polynomial=q.copy();positive=r>0
polynomial[positive,0]=np.log(r[positive]**2/ev.refs['chi'][positive,0])
polynomial[positive,6]=np.log(1e-6/ev.refs['alpha'][positive,0])
cd,cdd=ev.op.derivatives(polynomial,np.ones(8));_,_,_,_,pd,pdd=ev.reconstruct(polynomial,cd,cdd)
np.testing.assert_allclose(pd[1:4,0],2*r[1:4],rtol=1e-11,atol=1e-13)
np.testing.assert_allclose(pdd[1:4,0],2,rtol=1e-11,atol=1e-11)
np.testing.assert_allclose(pd[1:4,6]+r[1:4],0,atol=1e-17)
np.testing.assert_allclose(pdd[1:4,6]+1,0,atol=1e-14)
B=1e-6*r*np.exp(-(r/.2)**2)
engine=lambda:TelegraphEngine(ev,a.fast_library,.005,.02,inner_profile='af-regular',lapse_update='total',lapse_attenuation=.001)
e=engine();e.apply_lapse_floor(q);dq,db,s=e.rhs(q,B);p1,p2,p3=python_rhs(ev,q,B,.005,.02,window(r,inner_profile='af-regular'),lapse_attenuation=.001)
np.testing.assert_allclose(dq,p1,rtol=2e-8,atol=2e-8);np.testing.assert_allclose(db,p2,rtol=2e-8,atol=2e-8);np.testing.assert_allclose(s,p3,rtol=2e-10,atol=2e-10)
other=q.copy();other[0,0]+=1000;other[0,6]+=1000
dx,dy,ds=e.rhs(other,B)
np.testing.assert_array_equal(dx[1:],dq[1:]);np.testing.assert_array_equal(dy,db);np.testing.assert_array_equal(ds,s)
assert np.isfinite(ev.rhs(other)).all();e.close()
# A passive endpoint coefficient must not contaminate even a full RK step.
first=engine();second=engine();q1=q.copy();q2=other.copy();b1=B.copy();b2=B.copy();empty=np.empty((0,2))
assert first.advance(q1,b1,empty,5,1e-6,0.)==(5,None)
assert second.advance(q2,b2,empty,5,1e-6,0.)==(5,None)
np.testing.assert_array_equal(q1[1:],q2[1:]);np.testing.assert_array_equal(b1,b2);first.close();second.close()
print(json.dumps(dict(positive_node_polynomial_derivatives=True,independent_python_native_rates=True,large_AF_coefficient_no_bulk_feedback=True)))
