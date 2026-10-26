#!/usr/bin/env python3
"""Nonzero-flux cross-implementation, radial tensor, baseline and restart checks."""
import argparse,json
from itertools import product
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from fast import FastEngine
from telegraph import TelegraphEngine,options,window,python_rhs
from compensated_update import accumulate
from telegraph_analysis_io import physical_lapse_gradient
from wormhole_data import Jet
from numerics import puncture_dissipation,RadialOperators
from radial_grid import mapped_grid
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args()
errors=[]
for n,inner_profile,stretch,kind in product([33,65,129],['collar','af-regular'],[0.,4.,6.],['sinh','inner-c7']):
 opts=options(n,a.library);opts.grid_stretch=stretch;opts.grid_map=kind
 ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole());r=ev.r
 support=window(r,inner_profile=inner_profile);w,wp=support;B=.001*r*(1-r)**8
 for j,amp in [(0,.001),(2,.001),(3,.002),(4,.003),(5,.004),(6,.0015)]:q[:,j]+=amp*(1-r)**6
 engine=TelegraphEngine(ev,a.fast_library,.02,.03,nray=2,start=0.,inner_profile=inner_profile)
 expected,eb,es=python_rhs(ev,q,B,.02,.03,support,True);got,gb,gs=engine.rhs(q,B)
 np.testing.assert_allclose(got,expected,rtol=2e-11,atol=2e-10)
 np.testing.assert_allclose(gb,eb,rtol=2e-11,atol=2e-11);np.testing.assert_allclose(gs,es,rtol=2e-13,atol=2e-13)
 errors.append(float(np.max(abs(got-expected))))
 for name in ['max_accepted_H_rms','max_accepted_M_rms','max_accepted_Theta_rms','max_accepted_Z_rms']:
  np.testing.assert_allclose(engine.diagnostics()[name],ev.stage_metrics[name],rtol=2e-10,atol=2e-10)
 # Independent Cartesian tensor derivative for B_i=x_i*r^2:
 # delta^ij partial_i B_j=5*r^2, including two angular directions.
 xx=np.array([.13,.21,.34]);radius=np.linalg.norm(xx);tensor=2*np.outer(xx,xx)+radius**2*np.eye(3)
 assert abs(np.trace(tensor)-(3*radius**2+2*radius**2))<1e-15
 # Actual implementation versus Cartesian radial tensor identity, B_i=x_i*r^2.
 zero=np.zeros_like(B);base,_,_=engine.rhs(q,zero);withB,_,_=engine.rhs(q,r**3)
 alpha=ev.total_alpha;chi=ev.total_chi;expected_delta=np.zeros(n)
 expected_delta[1:]=chi[1:]*(5*w[1:]*r[1:]**2+wp[1:]*r[1:]**3)/alpha[1:]
 np.testing.assert_allclose(withB[:,6]-base[:,6],expected_delta,rtol=3e-11,atol=3e-11)
 # Nonzero flux, free-constraint RK stages against Python without any projection.
 state=q.copy();flux=B.copy();qc=np.zeros_like(q);bc=np.zeros_like(B);dt=.002*ev.h
 expected_rays=np.array([[.25,0.],[.25,0.]]);rc=np.zeros_like(expected_rays)
 def ray_rate(y,speeds):
  out=np.zeros_like(y)
  for j in range(len(y)):
   i=np.clip(np.searchsorted(r,y[j,0],side='right')-1,0,len(r)-2)
   slope=(speeds[i+1,j]-speeds[i,j])/(r[i+1]-r[i])
   out[j]=[speeds[i,j]+slope*(y[j,0]-r[i]),slope]
  return out
 for _ in range(4):
  k1,b1,s1=python_rhs(ev,state,flux,.02,.03,support);j1=ray_rate(expected_rays,s1)
  k2,b2,s2=python_rhs(ev,state+dt*k1/2,flux+dt*b1/2,.02,.03,support);j2=ray_rate(expected_rays+dt*j1/2,s2)
  k3,b3,s3=python_rhs(ev,state+dt*k2/2,flux+dt*b2/2,.02,.03,support);j3=ray_rate(expected_rays+dt*j2/2,s3)
  k4,b4,s4=python_rhs(ev,state+dt*k3,flux+dt*b3,.02,.03,support);j4=ray_rate(expected_rays+dt*j3,s4)
  expected_rays,rc=accumulate(expected_rays,rc,dt*(j1+2*j2+2*j3+j4)/6)
  state,qc=accumulate(state,qc,dt*(k1+2*k2+2*k3+k4)/6)
  flux,bc=accumulate(flux,bc,dt*(b1+2*b2+2*b3+b4)/6)
 cq=q.copy();cb=B.copy();rays=np.array([[.25,0.],[.25,0.]])
 done,failure=engine.advance(cq,cb,rays,4,dt,0.);assert done==4 and failure is None,failure
 np.testing.assert_allclose(cq,state,rtol=3e-12,atol=3e-12);np.testing.assert_allclose(cb,flux,rtol=3e-12,atol=3e-12)
 np.testing.assert_allclose(rays,expected_rays,rtol=2e-12,atol=2e-12)
 # Save/reconstruct all geometry, auxiliary and ray compensated state.
 correction=engine.compensation();q2=cq.copy();B2=cb.copy();ray2=rays.copy()
 restarted=TelegraphEngine(ev,a.fast_library,.02,.03,nray=2,start=0.,inner_profile=inner_profile);restarted.compensation(correction)
 assert engine.advance(cq,cb,rays,5,dt,4*dt)[1] is None
 assert restarted.advance(q2,B2,ray2,5,dt,4*dt)[1] is None
 assert np.array_equal(cq,q2) and np.array_equal(cb,B2) and np.array_equal(rays,ray2)
 assert np.array_equal(engine.compensation(),restarted.compensation())
 restarted.close();engine.close()
 # kappa=0, B=0 must reproduce the actual existing solver bitwise.
 ref=FastEngine(ev,8,a.fast_library);new=TelegraphEngine(ev,a.fast_library,0.,.03,inner_profile=inner_profile)
 cq=q.copy();rq=q.copy();cb=np.zeros(n);rays=np.zeros((0,2))
 done,_,failed=ref.advance(rq,8,dt);assert not failed
 done,failure=new.advance(cq,cb,rays,8,dt,0.);assert failure is None
 assert np.array_equal(cq,rq) and np.array_equal(cb,np.zeros(n))
 ref.close();new.close()
 event=TelegraphEngine(ev,a.fast_library,0.,.03,nray=2,start=2*dt,inner_profile=inner_profile)
 cq=q.copy();cb=np.zeros(n);ray=np.array([[.25,0.],[.25,0.]]);seed=ray.copy()
 assert event.advance(cq,cb,ray,2,dt,0.)[1] is None
 assert np.array_equal(ray,seed),'Last RK stage before injection must not advance rays early'
 assert event.advance(cq,cb,ray,1,dt,2*dt)[1] is None
 assert not np.array_equal(ray,seed)
 event.close()
# Differentiation of the support function uses an independent numerical oracle.
r=np.linspace(.015,.79,1000);w,wp=window(r);eps=1e-6
np.testing.assert_allclose(wp,(window(r+eps)[0]-window(r-eps)[0])/(2*eps),atol=2e-6,rtol=2e-6)
# The new forcing vanishes on exact Minkowski u=0,B=0. Both endpoint
# collars have identically zero coupling, including all support derivatives.
w,wp=window(np.array([0.,.005,.9,1.]));assert np.array_equal(w,np.zeros(4)) and np.array_equal(wp,np.zeros(4))
# AF asymptotic extension: chi=X*r^4, alpha=a*r^2, B=b*r.
# Added stored log-alpha rate is 3 X b r^2/a, so the leading AF
# coefficient equation is unchanged even though w(0)=1.
r=np.array([1e-2,5e-3,2.5e-3]);X=2.;alpha_coeff=3.;b=.4
w,wp=window(r,inner_profile='af-regular');assert np.array_equal(w,np.ones(3))
correction=X*r**4*(w*b+(wp+2*w/r)*(b*r))/(alpha_coeff*r*r)
np.testing.assert_allclose(correction,3*X*b*r*r/alpha_coeff,rtol=2e-15)
np.testing.assert_allclose(correction[:-1]/correction[1:],4.,rtol=2e-15)
# Frozen coupled dispersion: independently computed roots must be damped.
for k in [.01,1,100]:
 cg2=.2;ct2=.3;tau=.03
 roots=np.roots([1,1/tau,(cg2+ct2)*k*k,cg2*k*k/tau]);assert np.max(roots.real)<0
# Independent analytic derivative of an AF lapse with a polynomial log chart.
r=np.linspace(0,1,129);q=np.zeros((129,8));q[:,6]=.1+.2*r+.03*r*r
actual=physical_lapse_gradient(dict(r=r,q=q));expected=[]
for x in r[:-1]:
 z=Jet(x,1);O=(1-z*z)/2;A=(1+z*z)/2;ref=A*z*z/(z*z+O**4)/O
 expected.append(np.exp(.1+.2*x+.03*x*x)*(ref.d+ref.v*(.2+.06*x)))
np.testing.assert_allclose(actual[:-1],expected,rtol=5e-14,atol=5e-14);assert actual[0]==0
# Mapped dissipation is negative semidefinite in the physical dr measure.
x=np.linspace(0,1,129);r=np.sinh(6*x)/np.sinh(6);J=6*np.cosh(6*x)/np.sinh(6);dx=x[1]
v=np.random.default_rng(314159).normal(size=(len(x),1));Q=puncture_dissipation(v,r,dx,.5,.8)/J[:,None]
third=np.diff(v[1:,0],n=3);mid=(r[1:-3]+r[4:])/2;W=np.maximum(1-(mid/.8)**2,0)**6
np.testing.assert_allclose(np.sum(J[:,None]*v*Q),-.5*np.sum(W*third**2)/(64*dx),rtol=3e-15)
# An independently specified smooth physical function has the expected order
# on the nested mapped grids; polynomial quadrature also remains exact.
de=[]
for n in [65,129,257]:
 x=np.linspace(0,1,n);r=np.sinh(6*x)/np.sinh(6)
 op=RadialOperators(r,puncture=True,allow_puncture_node=True,subtract_local=True)
 df,_=op.derivatives(np.sin(2*r),1);de.append(max(abs(df-2*np.cos(2*r))))
 np.testing.assert_allclose(op.integral(r**4),r**5/5,rtol=3e-13,atol=3e-15)
assert np.log2(de[-2]/de[-1])>5.,de
for kind in ['sinh','inner-c7']:
 x=np.linspace(.001,.999,1000);r,J=mapped_grid(x,6,kind);eps=1e-6
 np.testing.assert_allclose(J,(mapped_grid(x+eps,6,kind)[0]-mapped_grid(x-eps,6,kind)[0])/(2*eps),rtol=2e-8,atol=2e-9)
 assert min(J)>0 and np.all(np.diff(r)>0)
 if kind=='inner-c7':assert np.array_equal(r[x>=.4],x[x>=.4]) and np.array_equal(J[x>=.4],np.ones(sum(x>=.4)))
print(json.dumps(dict(max_python_compiled_rate_difference=max(errors),resolutions=[33,65,129],nonzero_flux_RK=True,free_constraints=True,disabled_bitwise_equivalence=True,restart_bitwise=True,stage_characteristic_oracle=True,ray_activation=True,scope='Implementation validation only; nonlinear convergence remains unproven.')))
