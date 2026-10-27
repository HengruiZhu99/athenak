#!/usr/bin/env python3
"""Independent physical ADM audit on a nonlinear embedded Schwarzschild slice.

Use physical Ricci/Hessian formulas rather than the conformal Z4c kernels.
The slice has nonzero radial tracefree curvature through the transition.
This tests continuum rates, not numerical long-time stability.
"""
import argparse,json
from types import SimpleNamespace
import numpy as np
from wormhole_isotropic import IsotropicWormhole
from wormhole_data import Jet
from numerics import RadialOperators
from elliptic import solve
from kernel import Kernel
import g1_rhs

p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
r=np.linspace(.008,.97,769);initial=IsotropicWormhole();q,d,dd,meta=initial.data(r)
chi=np.array([z['chi_total'] for z in meta]);alpha=(1+r*r)/2+q[:,6]
op=RadialOperators(r,puncture=True)
beta,db,ddb,_=solve(op,np.ones_like(r),d[:,1],dd[:,1],alpha,r+d[:,6],q[:,2],d[:,2],-1.)
# Use a supplied common shift; its elliptic truncation error cannot contaminate
# the comparison between ADM and conformal rates.
integrand=1.5*alpha*q[:,2]/r;integral=op.integral(integrand)
sub=SimpleNamespace(r=r,integral=lambda _:integral)
R,S,*_=g1_rhs.parts(sub,q,d,dd,total_chi=chi)
bg1=-r+r*(integral-integral[-1]-(q[-1,0]+(2*q[-1,6]+q[-1,6]**2)*chi[-1])/(1+alpha[-1]*np.sqrt(chi[-1])))
dbg1=bg1/r+1.5*alpha*q[:,2]
ddbg1=1.5*((r+d[:,6])*q[:,2]+alpha*d[:,2])+1.5*alpha*q[:,2]/r
nR,nS,diag=Kernel(a.library).parts(r,q,d,dd,bg1,dbg1,ddbg1,total_chi=chi)
O=(1-r*r)/2;rates=R+S/O[:,None];native=nR+nS/O[:,None]
expected=np.zeros((len(r),3))
for i,x in enumerate(r):
    rad=Jet(float(x),1);omega=(1-rad*rad)/2;ref=(1+rad*rad)/2
    c=Jet(chi[i],d[i,0],dd[i,0]);al=Jet(alpha[i],x+d[i,6],1+dd[i,6]);f=al/omega
    up=omega.d/omega.v+.5*c.d/c.v
    upp=omega.dd/omega.v-(omega.d/omega.v)**2+.5*(c.dd/c.v-c.d*c.d/c.v**2)
    e2u=omega.v**2*c.v
    ricdiff=e2u*(upp-up/x+up*up)
    hessdiff=e2u*(f.dd+(2*up-1/x)*f.d)
    lap=e2u*(f.dd+(2/x-up)*f.d)
    K=q[i,3]-3;Ar=q[i,2];b=bg1[i]
    expected[i,0]=b*c.d-al.v*c.v*Ar+2*al.v*c.v*K/(3*omega.v)-2*b*c.v*ref.v/(x*omega.v)
    expected[i,1]=b*d[i,2]+(b*omega.d/omega.v+al.v*K/omega.v)*Ar+2*(-hessdiff+f.v*ricdiff)/(3*omega.v)
    expected[i,2]=b*d[i,3]-lap+f.v*(K*K/3+1.5*omega.v**2*Ar*Ar)
selected=[0,2,3];err=np.max(abs(rates[:,selected]-expected),axis=0)
nerr=np.max(abs(native[:,selected]-expected),axis=0)
assert min(np.max(abs(q[:,2])),np.max(abs(d[:,3])))>1
assert max(err)<2e-9 and max(nerr)<2e-9,(err,nerr)
print(json.dumps(dict(ADM_vs_canceled_Z4c=err.tolist(),ADM_vs_native_Z4c=nerr.tolist(),
    max_initial_constraints=np.max(abs(diag[:,:4]),axis=0).tolist(),
    scope='Nonlinear on-constraint physical ADM rates including nonzero tracefree curvature; no stationarity subtraction')))
