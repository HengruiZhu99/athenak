#!/usr/bin/env python3
"""Independent native-tensor audit of the compactified AF inner-end limits."""
import argparse,json
import numpy as np
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--damping-form',choices=['constant','reference-lapse'],default='constant');p.add_argument('--slicing-family',choices=['linear','rational','blended-log'],default='linear');p.add_argument('--lapse-background-source',choices=['reference','collapsed-inner'],default='reference');p.add_argument('--lapse-restoring-profile',choices=['constant','physical-log','physical-local'],default='constant');a=p.parse_args()
r=np.array([.004,.002,.001]);X,A,B,k,tau,L,b1=2.,.7,.03,.02,.01,.04,.05
if a.slicing_family=='blended-log' or a.lapse_restoring_profile in ('physical-log','physical-local'):r/=2
q=np.zeros((len(r),8));d=q.copy();dd=q.copy();chi=X*r**4;alpha=A*r*r
q[:,0]=chi-1;d[:,0]=4*X*r**3;dd[:,0]=12*X*r*r
q[:,2]=B*r*r;d[:,2]=2*B*r;dd[:,2]=2*B
q[:,3]=3+k*r*r;d[:,3]=2*k*r;dd[:,3]=2*k
q[:,4]=tau*r*r;d[:,4]=2*tau*r;dd[:,4]=2*tau
q[:,5]=L;q[:,6]=alpha-(1+r*r)/2;d[:,6]=(2*A-1)*r;dd[:,6]=2*A-1
beta=b1*r+3*A*B*r**5/8;db=b1+15*A*B*r**4/8;ddb=15*A*B*r**3/2
R,S,_=Kernel(a.library).parts(r,q,d,dd,beta,db,ddb,total_chi=chi)
if a.damping_form=='reference-lapse':
 D=4*((1+r*r)/2-alpha)
 S[:,3]+=D*1.99*q[:,4];S[:,4]-=D*1.01*q[:,4];S[:,5]-=D*q[:,5]
nu_inner=6. if a.slicing_family=='rational' else 1.
strength=1+(3*nu_inner-1)*(1-r*r)**4 if a.lapse_restoring_profile in ('physical-log','physical-local') else 1.
if a.lapse_restoring_profile=='physical-local' and a.slicing_family=='rational':strength=1+(3*((1+r*r)/2+.1)/(alpha+.1)-1)*(1-r*r)**4
S[:,6]-=2*strength*alpha*q[:,6] # Explicit multiplicative restoring source.
nu=1.
if a.slicing_family!='linear':
 O=(1-r*r)/2;Aref=(1+r*r)/2
 oldF=alpha*alpha+4*O*O*alpha
 newF=alpha*alpha+4*O*O*alpha*(Aref+.1)/(alpha+.1) if a.slicing_family=='rational' else (1-4*O*O)*alpha*alpha+8*O**3*alpha
 S[:,6]-=(newF-oldF)*q[:,3];nu=6. if a.slicing_family=='rational' else 1.
rates=R+S/((1-r*r)/2)[:,None]
if a.lapse_background_source=='collapsed-inner':
 O=(1-r*r)/2;Aref=(1+r*r)/2
 rates[:,6]+=(2*O)**4*r*r*(alpha/Aref-1)/O
scaled=np.c_[rates[:,0]/chi,rates[:,6]/alpha,rates[:,2]/r**2,rates[:,3]/r**2,rates[:,4]/r**2,rates[:,5]]
expected=np.array([2*b1,2*b1-6*nu+2*(3*nu if a.lapse_restoring_profile in ('physical-log','physical-local') else 1)+2/A,2*b1*B,2*b1*k,2*b1*tau,2*b1*L])
if a.lapse_background_source=='collapsed-inner':expected[1]-=2/A
if a.damping_form=='reference-lapse':expected[3]+=4*1.99*tau;expected[4]-=4*1.01*tau;expected[5]-=4*L
errors=np.max(abs(scaled-expected),axis=1);orders=np.log2(errors[:-1]/errors[1:])
assert errors[-1]<(4e-4 if a.slicing_family=='rational' else 2e-5) and min(orders)>1.8,(errors,orders)
print(json.dumps(dict(sample_r=r.tolist(),errors=errors.tolist(),orders=orders.tolist(),limits=expected.tolist(),
    scope='AF puncture coefficient ODE limits, retaining independent curvature/Khat/Theta/Lambda coefficients; not a trumpet evolution')))
