#!/usr/bin/env python3
"""Positive regular-center variables against independent native tensor rates."""
import argparse,json
from types import SimpleNamespace
import numpy as np
from evolve import Evolution,CoupledEvolution
from kernel import Kernel
import scalar_rhs
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args();records=[]
for origin in [False,True]:
 for scalar in [False,True]:
    opts=SimpleNamespace(n=129,origin_grid=origin,puncture=False,origin_variables=False,backend='spherical-g1',
        width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,eta=0.,theta='appendix-c',
        slicing=1.,lapse_restoring=1.,lapse_restoring_form='symmetric',damping_form='constant',endpoint='raw-lhopital',
        gauge_corner_sat=0.,advection='centered',scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,
        metric_variable='log',lapse_variable='log-reference',elliptic_quadrature='inward')
    ev=(CoupledEvolution if scalar else Evolution)(opts);r=ev.r;q=np.zeros((len(r),10 if scalar else 8));f=(1-r*r)**4
    assert max(abs(ev.rhs(q)).ravel())<2e-13
    for field,amplitude,power in [(0,.002,0),(2,.001,2),(3,.0015,0),(4,.0001,0),(5,.0002,1),(6,.002,0)]:q[:,field]=amplitude*r**power*f
    if scalar:q[:,8]=.0005*f;q[:,9]=.001*f
    rate=ev.rhs(q);_,_,diag,b=ev.parts(q);geom=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second
    chi=ev.total_chi;alpha=ev.total_alpha;positive=(r>0)&(r<1)
    db=np.zeros_like(r);ddb=np.zeros_like(r);db[positive]=b[positive]/r[positive]+1.5*alpha[positive]*geom[positive,2]
    ddb[positive]=1.5*((r[positive]+d[positive,6])*geom[positive,2]+alpha[positive]*d[positive,2])+1.5*alpha[positive]*geom[positive,2]/r[positive]
    nR,nS,nDiag=Kernel(a.library).parts(r[positive],geom[positive],d[positive],dd[positive],b[positive],db[positive],ddb[positive],total_chi=chi[positive])
    nS[:,6]-=(alpha[positive]+(1+r[positive]**2)/2)*geom[positive,6]
    if scalar:scalar_rhs.add_sources(r[positive],geom[positive],nR,nDiag,ev.scalar_phir[positive],ev.scalar_normal[positive])
    native=nR+nS/ev.omega[positive,None]
    physical=rate[:,:8].copy();physical[:,0]*=chi;physical[:,6]*=alpha
    error=float(max(abs(physical[positive]-native).ravel()));assert error<2e-10,error
    diagnostic_error=float(max(abs(diag[positive,:4]-nDiag[:,:4]).ravel()));assert diagnostic_error<2e-10,diagnostic_error
    assert all(max(abs(diag[:,i]))>0 for i in range(4))
    records.append(dict(origin_on_grid=origin,scalar=scalar,native_rate_error=error,native_constraint_error=diagnostic_error,
        scope='Invertible positive fields and regular-center chain derivatives; nonlinear evolution gates remain independent'))
# Total fields must not be recovered from exp(U)-1 when they are below the
# relative precision of an order-one reference. This is a representation test,
# not physically constraint-satisfying collapse data.
q[:]=0;q[:,0]=q[:,6]=-40*(1-ev.r**2)**4;q[:,9]=1
_,_,_,_=ev.parts(q)
np.testing.assert_allclose(ev.scalar_normal,ev.omega*np.sqrt(ev.total_chi),rtol=2e-14,atol=0)
df,ddf=ev.op.derivatives(q[:,8:],np.ones(2))
wave,_=scalar_rhs.rhs(ev.op,ev.geometry_state,ev.geometry_first,q[:,8:],df,ddf,
    -ev.r+ev.shift_deviation,ev.shift_deviation,geometry_second=ev.geometry_second,
    origin_beta_derivative=ev.origin_shift_slope,total_chi=ev.total_chi,total_alpha=ev.total_alpha)
np.testing.assert_allclose(wave[:,0],ev.total_alpha,rtol=2e-14,atol=0)
assert np.all(np.isfinite(wave)) and min(ev.total_alpha)>0 and min(ev.total_chi)>0
records.append(dict(totals_below_reference_relative_precision=True,
    scope='Positive total chi/lapse are supplied consistently to geometry and matter; no floor or constraint projection'))
# Independently differentiate the physical deviation polynomial. A total-alpha
# chain evaluated by subtracting its reference derivative loses this signal.
q[:]=0;epsilon=1e-25;q[:,6]=epsilon*(1-ev.r**2)**2
ev.parts(q);physical=ev.to_physical(q);direct_first,direct_second=ev.op.derivatives(physical[:,6],1)
first_error=float(max(abs(ev.geometry_first[:,6]-direct_first))/epsilon)
second_error=float(max(abs(ev.geometry_second[:,6]-direct_second))/epsilon)
assert first_error<1e-10 and second_error<1e-8,(first_error,second_error)
records.append(dict(tiny_deviation_derivative_errors_relative_to_amplitude=[first_error,second_error],
    amplitude=epsilon,scope='Independent direct-deviation differentiation checks cancellation-free lapse chains'))
print(json.dumps(records))
