#!/usr/bin/env python3
"""Independent native-tensor audit of the compactified AF inner-end limits."""
import argparse,json
import numpy as np
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
r=np.array([.004,.002,.001]);X,A,B,k,tau,L,b1=2.,.7,.03,.02,.01,.04,.05
q=np.zeros((len(r),8));d=q.copy();dd=q.copy();chi=X*r**4;alpha=A*r*r
q[:,0]=chi-1;d[:,0]=4*X*r**3;dd[:,0]=12*X*r*r
q[:,2]=B*r*r;d[:,2]=2*B*r;dd[:,2]=2*B
q[:,3]=3+k*r*r;d[:,3]=2*k*r;dd[:,3]=2*k
q[:,4]=tau*r*r;d[:,4]=2*tau*r;dd[:,4]=2*tau
q[:,5]=L;q[:,6]=alpha-(1+r*r)/2;d[:,6]=(2*A-1)*r;dd[:,6]=2*A-1
beta=b1*r+3*A*B*r**5/8;db=b1+15*A*B*r**4/8;ddb=15*A*B*r**3/2
R,S,_=Kernel(a.library).parts(r,q,d,dd,beta,db,ddb,total_chi=chi)
S[:,6]-=2*alpha*q[:,6] # The explicitly disclosed multiplicative restoring variant.
rates=R+S/((1-r*r)/2)[:,None]
scaled=np.c_[rates[:,0]/chi,rates[:,6]/alpha,rates[:,2]/r**2,rates[:,3]/r**2,rates[:,4]/r**2,rates[:,5]]
expected=np.array([2*b1,2*b1-6+2+2/A,2*b1*B,2*b1*k,2*b1*tau,2*b1*L])
errors=np.max(abs(scaled-expected),axis=1);orders=np.log2(errors[:-1]/errors[1:])
assert errors[-1]<2e-5 and min(orders)>1.8,(errors,orders)
print(json.dumps(dict(errors=errors.tolist(),orders=orders.tolist(),limits=expected.tolist(),
    scope='AF puncture coefficient ODE limits, retaining independent curvature/Khat/Theta/Lambda coefficients; not a trumpet evolution')))
