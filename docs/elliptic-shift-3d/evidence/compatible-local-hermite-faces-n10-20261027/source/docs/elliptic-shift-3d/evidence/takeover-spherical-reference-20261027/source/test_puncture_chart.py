#!/usr/bin/env python3
"""Compare the AF chart with independent native tensor rates off constraints."""
import argparse,json
from types import SimpleNamespace
import numpy as np
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--damping-form',choices=['constant','reference-lapse'],default='constant');p.add_argument('--slicing-family',choices=['linear','rational','blended-log'],default='linear');p.add_argument('--radial-operator',choices=['fd','sbp63'],default='fd');p.add_argument('--lapse-background-source',choices=['reference','collapsed-inner'],default='reference');p.add_argument('--lapse-restoring-profile',choices=['constant','physical-log','physical-local'],default='constant');a=p.parse_args()
args=SimpleNamespace(n=129,origin_grid=False,puncture=True,puncture_chart=True,origin_variables=False,
    backend='spherical-g1',width=7,quadrature_width=7,library=a.library,kappa1=4.,kappa2=-.99,
    theta='appendix-c',slicing=1.,lapse_restoring=1.,lapse_restoring_form='multiplicative',
    endpoint='raw-lhopital',scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,
    advection='centered',eta=0.,gauge_corner_sat=0.,damping_form=a.damping_form,slicing_family=a.slicing_family,slicing_scale=.1,radial_operator=a.radial_operator,lapse_background_source=a.lapse_background_source,lapse_restoring_profile=a.lapse_restoring_profile)
ev=PunctureChart(args);state,initial,*_=ev.initial_wormhole(IsotropicWormhole())
np.testing.assert_allclose(ev.to_physical(state),initial,atol=3e-14,rtol=3e-14)
f=(1-ev.r)**6
for field,amplitude in [(0,.001),(2,.001),(3,.002),(4,.003),(5,.004),(6,.0015)]:state[:,field]+=amplitude*f
rate=ev.rhs(state);R,S,diag,beta=ev.parts(state);q=ev.geometry_state;d=ev.geometry_first;dd=ev.geometry_second
r=ev.r[1:];alpha=ev.total_alpha[1:];Ar=q[1:,2]
db=beta[1:]/r+1.5*alpha*Ar
ddb=1.5*((r+d[1:,6])*Ar+alpha*d[1:,2])+1.5*alpha*Ar/r
nativeR,nativeS,_=Kernel(a.library).parts(r,q[1:],d[1:],dd[1:],beta[1:],db,ddb,total_chi=ev.total_chi[1:])
if a.damping_form=='reference-lapse':
    native=[];kernel=Kernel(a.library);effective=4*((1+r*r)/2)/alpha
    for i in range(len(r)):
        native.append(kernel.parts(r[i:i+1],q[i+1:i+2],d[i+1:i+2],dd[i+1:i+2],beta[i+1:i+2],db[i:i+1],ddb[i:i+1],kappa1=effective[i],total_chi=ev.total_chi[i+1:i+2]))
    nativeR=np.concatenate([z[0] for z in native]);nativeS=np.concatenate([z[1] for z in native])
nu_inner=6. if a.slicing_family=='rational' else 1.
strength=1+(3*nu_inner-1)*(1-r*r)**4 if a.lapse_restoring_profile in ('physical-log','physical-local') else 1.
if a.lapse_restoring_profile=='physical-local' and a.slicing_family=='rational':strength=1+(3*((1+r*r)/2+.1)/(alpha+.1)-1)*(1-r*r)**4
nativeS[:,6]-=2*strength*alpha*q[1:,6]
if a.slicing_family!='linear':
    # Independent replacement of the principal lapse factor in the native RHS.
    O=(1-r*r)/2;A=(1+r*r)/2
    oldF=alpha*alpha+4*O*O*alpha
    newF=alpha*alpha+4*O*O*alpha*(A+.1)/(alpha+.1) if a.slicing_family=='rational' else (1-4*O*O)*alpha*alpha+8*O**3*alpha
    nativeS[:,6]-=(newF-oldF)*q[1:,3]
if a.lapse_background_source=='collapsed-inner':
    O=(1-r*r)/2;Aref=(1+r*r)/2
    nativeS[:,6]+=(2*O)**4*r*r*(alpha/Aref-1)
native=nativeR[:-1]+nativeS[:-1]/ev.omega[1:-1,None]
physical=rate.copy();physical[:,0]*=ev.total_chi;physical[:,6]*=ev.total_alpha
for field in [2,3,4]:physical[:,field]*=ev.refs['w'][:,0]
error=float(max(abs(physical[1:-1]-native).ravel()));assert error<2e-10,error
assert max(abs(diag[:,0]))>1e-3 and max(abs(diag[:,1]))>1e-3
assert max(abs(diag[:,2]))>0 and max(abs(diag[:,3]))>0
assert ev.chart_first[0,0]!=0 # No origin parity condition.
print(json.dumps(dict(native_tensor_rate_error=error,independent_H_M_Theta_Z_errors=True,
    actual_inner_asymptotic_endpoint=True,scope='Transformation and asymptotic ODE audits; stability requires evolutions')))
