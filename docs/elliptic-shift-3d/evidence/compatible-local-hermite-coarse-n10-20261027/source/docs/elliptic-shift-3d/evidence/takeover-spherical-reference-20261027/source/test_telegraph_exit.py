#!/usr/bin/env python3
"""Exact-flow event convergence and independent coupled-stage exit verification."""
import argparse,ctypes as C,json,sys
import numpy as np
from numpy.polynomial import Polynomial as P
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from telegraph import TelegraphEngine,options,window,python_rhs
from compensated_update import accumulate

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);p.add_argument('--scipy-path');a=p.parse_args()
if a.scipy_path:
 sys.path.insert(0,a.scipy_path)
 from scipy.interpolate import PchipInterpolator
lib=C.CDLL(a.fast_library);fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.telegraph_dense_ray_step.argtypes=[fp,fp,C.c_double,C.c_double,C.c_double,fp]
def step(y,k,h):
 out=np.zeros(3);lib.telegraph_dense_ray_step(np.ascontiguousarray(y),np.ascontiguousarray(k),h,0.,1.,out);return out
# Independent polynomial representation, order conditions through order three.
b=[P([0,1,-1.5,2/3]),P([0,0,1,-2/3]),P([0,0,1,-2/3]),P([0,0,-.5,2/3])]
c=np.array([0,.5,.5,1]);Ac=np.array([0,0,.25,.5])
for theta in np.linspace(0,1,31):
 weights=np.array([v(theta) for v in b])
 np.testing.assert_allclose([sum(weights),weights@c,weights@(c*c),weights@Ac],[theta,theta**2/2,theta**3/3,theta**3/6],atol=5e-16,rtol=0)
np.testing.assert_allclose([v(1) for v in b],[1/6,1/3,1/3,1/6],atol=2e-16,rtol=0)
np.testing.assert_equal([v.deriv()(0) for v in b],[1,0,0,0])
# Crossing and returning inside a single step: both boundaries and first root.
for direction in [1,-1]:
 seed=np.array([.8 if direction==1 else .2,0.]);rates=np.c_[direction*np.array([1.2,0,0,-1.2]),np.ones(4)]
 got=step(seed,rates,1);theta=(1-np.sqrt(1-4*.2/1.2))/2
 np.testing.assert_allclose(got,[1 if direction==1 else 0,theta,theta],atol=3e-16,rtol=0)
# Analytic affine, then time-dependent affine velocities c=(a0+a1*t)*(x+1).
# x+1=(x0+1)*exp(a0*t+a1*t^2/2), logJ=a0*t+a1*t^2/2.
exact_results=[]
for varying in [False,True]:
 errors=[]
 for h in [.08,.04,.02,.01]:
  peaks=np.zeros(2)
  for direction in [1,-1]:
   a0=direction*.6;a1=direction*.2 if varying else 0.;wall=1 if direction==1 else 0
   for seed in np.linspace(.23,.77,17):
    target=np.log((wall+1)/(seed+1))
    te=target/a0 if not varying else 2*target/(a0+direction*np.sqrt(a0*a0+2*a1*target))
    state=np.array([seed,0.]);time=0.
    def rhs(t,y):return np.array([(a0+a1*t)*(y[0]+1),a0+a1*t])
    for count in range(1000):
     k1=rhs(time,state);k2=rhs(time+h/2,state+h*k1/2);k3=rhs(time+h/2,state+h*k2/2);k4=rhs(time+h,state+h*k3)
     out=step(state,np.array([k1,k2,k3,k4]),h)
     if out[2]>=0:
      peaks=np.maximum(peaks,[abs(time+h*out[2]-te),abs(out[1]-target)]);break
     state=out[:2];time+=h
    else:raise AssertionError('Exact-flow ray did not exit')
  errors.append(peaks.tolist())
 errors=np.array(errors);orders=np.log2(errors[:-1]/errors[1:])
 assert np.all(orders>3.3),(varying,errors,orders)
 assert np.all(errors[-1]<2e-10),(varying,errors)
 exact_results.append(dict(time_dependent=varying,dt=[.08,.04,.02,.01],errors=errors.tolist(),orders=orders.tolist()))
# Terminal records remain exact even with nonzero supplied stage rates.
for x in [0.,1.]:np.testing.assert_equal(step(np.array([x,2.5]),np.ones((4,2)),.1),[x,2.5,0.])
# Coupled geometry, including real near-scri exits in the first RK step.
opts=options(129,a.library);opts.grid_stretch=16;opts.grid_map='inner-exp-scri';opts.ko_order=8;opts.mapped_ko='local'
ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole());r=ev.r;flux=.001*r*(1-r)**8
for j,amp in [(0,.001),(2,.001),(3,.002),(4,.003),(5,.004),(6,.0015)]:q[:,j]+=amp*(1-r)**6
seed=np.array([[.25,0.],[.999999,0.],[.26,0.],[.9999995,0.]])
kwargs=dict(nray=4,start=0.,inner_profile='af-regular',ray_interpolation='pchip')
legacy=TelegraphEngine(ev,a.fast_library,.005,.02,**kwargs);dense=TelegraphEngine(ev,a.fast_library,.005,.02,ray_exit='dense',**kwargs)
dt=.002*ev.h;states=[]
for eng in [legacy,dense]:
 cq=q.copy();cb=flux.copy();y=seed.copy();assert eng.advance(cq,cb,y,1,dt,0)==(1,None);states.append((cq,cb,y,eng.compensation()))
for j in [0,1]:assert np.array_equal(states[0][j],states[1][j])
assert np.array_equal(states[0][3][:9*len(r)],states[1][3][:9*len(r)])
assert np.all(states[1][2][1::2,0]==1.)
assert not np.array_equal(states[0][2][1::2,1],states[1][2][1::2,1]),'Exit repair did not alter terminal compression'
if a.scipy_path:
 support=window(r,inner_profile='af-regular')
 def rhs(q,B,y):
  dq,db,c=python_rhs(ev,q,B,.005,.02,support);ip=PchipInterpolator(r,c,axis=0,extrapolate=True)
  dy=np.array([[ip(v[0])[j%2],ip(v[0],1)[j%2]] for j,v in enumerate(y)])
  return dq,db,dy
 k1=rhs(q,flux,seed);k2=rhs(*(v+dt*k/2 for v,k in zip((q,flux,seed),k1)))
 k3=rhs(*(v+dt*k/2 for v,k in zip((q,flux,seed),k2)));k4=rhs(*(v+dt*k for v,k in zip((q,flux,seed),k3)))
 expected=seed.copy();events=[]
 for j in range(4):
  polynomials=[P([seed[j,f]])+dt*sum((poly*k[2][j,f] for poly,k in zip(b,[k1,k2,k3,k4])),P([0])) for f in range(2)]
  roots=[]
  for wall in [0.,1.]:
   roots += [(float(root.real),wall) for root in (polynomials[0]-wall).roots() if abs(root.imag)<1e-10 and 0<root.real<=1]
  theta,wall=min(roots) if roots else (1.,None)
  expected[j]=[polynomials[0](theta) if wall is None else wall,polynomials[1](theta)];events.append(theta)
 np.testing.assert_allclose(states[1][2],expected,rtol=2e-12,atol=2e-12)
 # Independently verified geometry and auxiliary stage sums remain unchanged.
 for field,index in [(0,0),(1,1)]:
  expected_field=(q if field==0 else flux)+dt*sum(weight*k[index] for weight,k in zip([1,2,2,1],[k1,k2,k3,k4]))/6
  np.testing.assert_allclose(states[1][field],expected_field,rtol=3e-12,atol=3e-12)
# Restart with already-exited plus rays and active minus rays, all low parts.
q1,B1,y1,comp=states[1];q2,B2,y2=q1.copy(),B1.copy(),y1.copy();terminal=y1[1::2].copy()
restart=TelegraphEngine(ev,a.fast_library,.005,.02,ray_exit='dense',**kwargs);restart.compensation(comp)
assert dense.advance(q1,B1,y1,13,dt,dt)==(13,None);assert restart.advance(q2,B2,y2,13,dt,dt)==(13,None)
for x,y in [(q1,q2),(B1,B2),(y1,y2),(dense.compensation(),restart.compensation())]:assert np.array_equal(x,y)
assert np.array_equal(terminal,y1[1::2]),'Terminal compression changed after exit'
for eng in [legacy,dense,restart]:eng.close()
print(json.dumps(dict(exact_flows=exact_results,continuous_order_conditions=True,cross_and_return=True,no_feedback_bitwise=True,terminal_freezing_bitwise=True,restart_with_exited_rays_bitwise=True,independent_coupled_stage_oracle=bool(a.scipy_path))))
