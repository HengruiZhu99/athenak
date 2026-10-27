#!/usr/bin/env python3
"""Diagnostic-only interpolation checks; optional independent SciPy stage oracle.

Install SciPy only in an isolated build-directory target for the external oracle.
The normal CTest entry has no new runtime dependency.
"""
import argparse,ctypes as C,json,sys
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from telegraph import TelegraphEngine,options,window,python_rhs
from compensated_update import accumulate

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True)
p.add_argument('--scipy-path');a=p.parse_args()
if a.scipy_path:
 sys.path.insert(0,a.scipy_path)
 from scipy.interpolate import PchipInterpolator
 import scipy
rng=np.random.default_rng(1729);maximum_error=0.
for grid,stretch in [('sinh',0.),('sinh',4.),('inner-exp-scri',16.)]:
 opts=options(129,a.library);opts.grid_map=grid;opts.grid_stretch=stretch;opts.mapped_ko='local';opts.ko_order=8
 ev=PunctureChart(opts);q,*_=ev.initial_wormhole(IsotropicWormhole());r=ev.r;B=.001*r*(1-r)**8
 for j,amp in [(0,.001),(2,.001),(3,.002),(4,.003),(5,.004),(6,.0015)]:q[:,j]+=amp*(1-r)**6
 # Interior nodes, both floating neighbors, and points in every interval.
 x=np.r_[r[1:-1],np.nextafter(r[1:-1],0),np.nextafter(r[1:-1],1),r[:-1]+.37*np.diff(r)]
 rays=np.zeros((2*len(x),2));rays[:,0]=np.repeat(x,2)
 engine=TelegraphEngine(ev,a.fast_library,.005,.02,nray=len(rays),start=0.,inner_profile='af-regular',ray_interpolation='pchip')
 fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 engine.base.lib.telegraph_ray_rates.argtypes=[C.c_void_p,fp,fp,fp]
 def rates(speeds,y=rays):
  out=np.zeros_like(y);engine.base.lib.telegraph_ray_rates(engine.handle,np.ascontiguousarray(speeds),y,out);return out
 for data in [np.c_[2*r+1,3-4*r],np.c_[r*r,np.exp(r)],rng.normal(size=(len(r),2)),np.zeros((len(r),2)),np.c_[np.maximum(r-.3,0),np.minimum(r,.6)]]:
  got=rates(data)
  i=np.clip(np.searchsorted(r,x,side='right')-1,0,len(r)-2)
  for f in range(2):
   lo=np.minimum(data[i,f],data[i+1,f]);hi=np.maximum(data[i,f],data[i+1,f])
   assert np.all(got[f::2,0]>=lo-2e-14) and np.all(got[f::2,0]<=hi+2e-14),'Overshoot'
  if a.scipy_path:
   oracle=PchipInterpolator(r,data,axis=0)
   expected=np.empty_like(got)
   for f in range(2):expected[f::2]=np.c_[oracle(x)[:,f],oracle(x,1)[:,f]]
   relative=np.max(abs(got-expected)/(1+abs(expected)));maximum_error=max(maximum_error,float(relative))
   np.testing.assert_allclose(got[:,0],expected[:,0],rtol=3e-12,atol=2e-13)
   # Near a zero nodal derivative, expanded polynomials cancel terms of
   # size delta-c/h; random data on tiny mapped cells require that scale.
   secants=np.diff(data,axis=0)/np.diff(r)[:,None]
   local_scale=np.maximum(abs(secants[i]),np.maximum(abs(secants[np.maximum(i-1,0)]),abs(secants[np.minimum(i+1,len(r)-2)]))).ravel()
   bound=3e-12*abs(expected[:,1])+64*np.finfo(float).eps*(1+local_scale)
   assert np.all(abs(got[:,1]-expected[:,1])<=bound),float(np.max(abs(got[:,1]-expected[:,1])/bound))
 affine=rates(np.c_[2*r+1,3-4*r]);np.testing.assert_allclose(affine[0::2],np.c_[2*x+1,np.full(len(x),2)],atol=2e-8,rtol=0)
 np.testing.assert_allclose(affine[1::2],np.c_[3-4*x,np.full(len(x),-4)],atol=2e-8,rtol=0)
 # C1 matching at all nodes, normalized by local curvature scales.
 smooth=rates(np.c_[r*r,np.exp(r)])[:,1].reshape(-1,2);m=len(r)-2
 np.testing.assert_allclose(smooth[:m],smooth[m:2*m],rtol=2e-11,atol=2e-11)
 np.testing.assert_allclose(smooth[:m],smooth[2*m:3*m],rtol=2e-11,atol=2e-11)
 engine.close()
 # Actual stage evolution, nonzero flux, exact no-feedback and restart checks.
 seed=np.array([[.25,0.],[.25,0.],[.41,0.],[.41,0.]])
 engines=[TelegraphEngine(ev,a.fast_library,.005,.02,nray=4,start=0.,inner_profile='af-regular',ray_interpolation=mode) for mode in ['linear','pchip']]
 result=[];dt=.002*ev.h
 for eng in engines:
  cq=q.copy();cb=B.copy();y=seed.copy();assert eng.advance(cq,cb,y,4,dt,0)==(4,None)
  result.append((cq,cb,y,eng.compensation()))
 assert np.array_equal(result[0][0],result[1][0]) and np.array_equal(result[0][1],result[1][1])
 assert np.array_equal(result[0][3][:9*len(r)],result[1][3][:9*len(r)])
 assert not np.array_equal(result[0][2],result[1][2]),'Interpolation control had no effect'
 if a.scipy_path:
  state=q.copy();flux=B.copy();y=seed.copy();qc=np.zeros_like(q);bc=np.zeros_like(B);yc=np.zeros_like(y)
  support=window(r,inner_profile='af-regular')
  def rhs(q,b,y):
   dq,db,c=python_rhs(ev,q,b,.005,.02,support);ip=PchipInterpolator(r,c,axis=0)
   dy=np.array([[ip(v[0])[j%2],ip(v[0],1)[j%2]] for j,v in enumerate(y)])
   return dq,db,dy
  for step in range(4):
   k1=rhs(state,flux,y);k2=rhs(*(v+dt*k/2 for v,k in zip((state,flux,y),k1)))
   k3=rhs(*(v+dt*k/2 for v,k in zip((state,flux,y),k2)));k4=rhs(*(v+dt*k for v,k in zip((state,flux,y),k3)))
   inc=[dt*(v1+2*v2+2*v3+v4)/6 for v1,v2,v3,v4 in zip(k1,k2,k3,k4)]
   state,qc=accumulate(state,qc,inc[0]);flux,bc=accumulate(flux,bc,inc[1]);y,yc=accumulate(y,yc,inc[2])
  for got,expected in zip(result[1][:3],(state,flux,y)):np.testing.assert_allclose(got,expected,atol=3e-12,rtol=3e-12)
 eng=engines[1];restart=TelegraphEngine(ev,a.fast_library,.005,.02,nray=4,start=0.,inner_profile='af-regular',ray_interpolation='pchip')
 q1,b1,y1,comp=result[1];q2,b2,y2=q1.copy(),b1.copy(),y1.copy();restart.compensation(comp)
 assert eng.advance(q1,b1,y1,7,dt,4*dt)==(7,None);assert restart.advance(q2,b2,y2,7,dt,4*dt)==(7,None)
 for first,second in [(q1,q2),(b1,b2),(y1,y2),(eng.compensation(),restart.compensation())]:assert np.array_equal(first,second)
 # Terminal positions stay fixed for both interpolation modes.
 terminal=np.array([[0.,-1.],[1.,-2.],[0.,3.],[1.,4.]])
 terminal_engine=TelegraphEngine(ev,a.fast_library,.005,.02,nray=4,start=0.,inner_profile='af-regular',ray_interpolation='pchip')
 assert terminal_engine.advance(q1,b1,terminal,1,dt,11*dt)==(1,None)
 assert np.array_equal(terminal[:,0],[0.,1.,0.,1.])
 for obj in [*engines,restart,terminal_engine]:obj.close()
print(json.dumps(dict(grids=['uniform','sinh','inner-exp-scri'],C1=True,no_overshoot=True,affine_exactness=True,geometry_flux_bitwise=True,restart_bitwise=True,scipy_version=scipy.__version__ if a.scipy_path else None,independent_stage_oracle=bool(a.scipy_path),max_normalized_oracle_difference=maximum_error)))
