#!/usr/bin/env python3
"""Native and independently compiled rates for coordinate-only metric profiles."""
import argparse,json
from itertools import product
from types import SimpleNamespace
import numpy as np
from evolve import Evolution,CoupledEvolution
from fast import FastEngine
from kernel import Kernel
from trumpet_data import CMCTrumpet
from numerics import puncture_dissipation
import scalar_rhs
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--fast-library',required=True);a=p.parse_args();records=[]
for profile,scalar,advection,ko,endpoint,form,damping in product(['log-trumpet','log-inner-trumpet'],[False,True],['centered','upwind'],[0.,.5],['raw-lhopital','analytic-g1'],['symmetric','multiplicative'],['constant','reference-lapse']):
 opts=SimpleNamespace(n=65,inner_radius=.03,origin_grid=False,puncture=True,puncture_chart=False,origin_variables=False,backend='spherical-g1',width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,theta='appendix-c' if form=='multiplicative' else 'repository',slicing=.25,lapse_restoring=1.,lapse_restoring_form=form,endpoint=endpoint,scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,advection=advection,eta=0.,gauge_corner_sat=0.,damping_form=damping,lapse_background_source='collapsed-inner',lapse_restoring_profile='physical-log',initial_data='trumpet',mass=1/3,metric_variable=profile,lapse_variable='log-reference',puncture_ko=ko,puncture_ko_radius=.8,puncture_ko_inner_width=.03,compensated_update=False)
 ev=(CoupledEvolution if scalar else Evolution)(opts);r=ev.r;O=(1-r*r)/2;A=(1+r*r)/2
 geom,_,_,meta=CMCTrumpet(1/3).data(r);chi=np.array([v['chi_total'] for v in meta]);alpha=np.r_[[v['physical_lapse'] for v in meta[:-1]],0.]*O;alpha[-1]=1.
 state=ev.from_physical(geom,total_chi=chi);state[:,6]=np.log(alpha/A)
 if scalar:state=np.c_[state,np.zeros((65,2))]
 f=(1-r*r)**4
 for field,amplitude in [(0,.0001),(2,.0002),(3,.0003),(4,.00004),(5,.00005),(6,.0001)]:state[:,field]+=amplitude*f
 if scalar:state[:,8]=1e-5*f;state[:,9]=2e-5*f+1e-5
 expected=ev.rhs(state,accepted=True);R,S,diag,beta=ev.parts(state);q=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second;chi=ev.total_chi;alpha=ev.total_alpha
 db=beta/r+1.5*alpha*q[:,2];ddb=1.5*((r+d[:,6])*q[:,2]+alpha*d[:,2])+1.5*alpha*q[:,2]/r
 nR,nS,nDiag=Kernel(a.library).parts(r[:-1],q[:-1],d[:-1],dd[:-1],beta[:-1],db[:-1],ddb[:-1],slicing=.25,total_chi=chi[:-1],appendix_c=opts.theta=='appendix-c')
 strength=1-.25*(1-r*r)**4;nS[:,6]-=strength[:-1]*(2*alpha[:-1] if form=='multiplicative' else alpha[:-1]+A[:-1])*q[:-1,6]
 if damping=='reference-lapse':
  D=-4*q[:-1,6];nS[:,3]+=D*1.99*q[:-1,4];nS[:,4]-=D*1.01*q[:-1,4];nS[:,5]-=D*q[:-1,5]
 nS[:,6]+=(1-r[:-1]**2)**4*r[:-1]**2*q[:-1,6]/A[:-1]
 if scalar:scalar_rhs.add_sources(r[:-1],q[:-1],nR,nDiag,ev.scalar_phir[:-1],ev.scalar_normal[:-1],total_chi=chi[:-1],total_alpha=alpha[:-1])
 reference=expected.copy()
 if advection=='upwind':
  parity=np.r_[[1,1,1,1,1,-1,1,1],np.ones(2)] if scalar else np.array([1,1,1,1,1,-1,1,1])
  first,_=ev.op.derivatives(state,parity);reference-=beta[:,None]*(ev.op.advection_derivative(state,parity,beta)-first)
 if ko:reference-=puncture_dissipation(state,r,ev.h,ko,include_first=True,inner_width=.03)
 physical=reference[:,:8].copy();physical[:,0]*=chi;physical[:,6]*=alpha
 native_error=float(max(abs(physical[:-1]-nR-nS/O[:-1,None]).ravel()));assert native_error<3e-10,native_error
 np.testing.assert_allclose(diag[:-1,:4],nDiag[:,:4],atol=3e-10,rtol=1e-10)
 assert all(max(abs(diag[:,i]))>0 for i in range(4))
 engine=FastEngine(ev,state.shape[1],a.fast_library);got,flux=engine.rhs(state);error=float(max(abs(got-expected).ravel()))
 np.testing.assert_allclose(got,expected,atol=3e-10,rtol=2e-11)
 if scalar:np.testing.assert_allclose(flux,4*np.pi*ev.scalar_scri_rate**2,rtol=1e-10,atol=1e-20)
 python=state.copy();compiled=state.copy();energy=0.;dt=.002*ev.h
 for _ in range(8):
  k1=ev.rhs(python,accepted=True);f1=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
  k2=ev.rhs(python+dt*k1/2);f2=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
  k3=ev.rhs(python+dt*k2/2);f3=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
  k4=ev.rhs(python+dt*k3);f4=4*np.pi*ev.scalar_scri_rate**2 if scalar else 0.
  python+=dt*(k1+2*k2+2*k3+k4)/6;energy+=dt*(f1+2*f2+2*f3+f4)/6
 done,compiled_energy,failed=engine.advance(compiled,8,dt);np.testing.assert_allclose(compiled_energy,energy,rtol=1e-10,atol=1e-20);assert done==8 and not failed,engine.failure_message
 step_error=float(max(abs(compiled-python).ravel()));np.testing.assert_allclose(compiled,python,atol=3e-12,rtol=2e-12)
 for key,value in engine.diagnostics().items():np.testing.assert_allclose(value,ev.stage_metrics[key],atol=3e-10,rtol=1e-7,err_msg=key)
 engine.close();records.append(dict(profile=profile,scalar=scalar,advection=advection,KO=ko,endpoint=endpoint,form=form,damping=damping,native_error=native_error,compiled_error=error,step_error=step_error))
print(json.dumps(dict(cases=len(records),max_native_error=max(v['native_error'] for v in records),max_compiled_error=max(v['compiled_error'] for v in records),max_step_error=max(v['step_error'] for v in records),independent_H_M_Theta_Lambda_retained=True,scope='Invertible variable/physical tensor and compiled scalar/vacuum audit, not nonlinear stability or wormhole admission')))
