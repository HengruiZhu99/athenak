#!/usr/bin/env python3
"""Independent regular-center limits; these checks do not assert stability."""
import numpy as np
from numerics import RadialOperators
from kernel import PARITY
import g1_rhs,scalar_rhs

r=np.linspace(0,1,129);op=RadialOperators(r)
for parity,powers in [(1,[0,2,4,6]),(-1,[1,3,5])]:
    for k in powers:
        value=r**k;d,dd=op.derivatives(value,parity)
        exact1=np.zeros_like(r) if k==0 else k*r**(k-1)
        exact2=np.zeros_like(r) if k<2 else k*(k-1)*r**(k-2)
        np.testing.assert_allclose(d,exact1,atol=4e-12,rtol=1e-11)
        np.testing.assert_allclose(dd,exact2,atol=4e-9,rtol=1e-9)
        np.testing.assert_allclose(op.integral(value),r**(k+1)/(k+1),atol=2e-15)

q=np.zeros((len(r),8));d,dd=op.derivatives(q,PARITY)
R,S,diag,beta,*_=g1_rhs.parts(op,q,d,dd,restoring=1)
assert np.max(abs(R))==np.max(abs(S))==np.max(abs(diag[:,:6]))==0
assert diag[0,10]==0 and np.isposinf(diag[0,11])

# Exact flat-space scalar limit from the regular radial wave operator:
# f_t(0)=-f(0)+P(0)/2; P_t(0)=-2P(0)+3 f_2+f(0).
f0,f2,f4,P0,P2=.7,-.23,.11,.31,-.17
fields=np.c_[f0+f2*r*r+f4*r**4,P0+P2*r*r]
df,ddf=op.derivatives(fields,np.ones(2))
wave,info=scalar_rhs.rhs(op,q,d,fields,df,ddf,beta,np.zeros(len(r)),
    geometry_second=dd,origin_beta_derivative=-1.)
np.testing.assert_allclose(wave[0],[-f0+.5*P0,-2*P0+3*f2+f0],atol=2e-11)
assert np.max(abs(info['potential']))==0

# Independent regular-center coefficients of off-constraint geometric data.
c0,c2,u0,u2,p0,T0,ell=.01,.003,.02,-.004,.007,-.002,.005
q[:,0]=c0+c2*r*r;q[:,2]=.006*r*r;q[:,3]=p0
q[:,4]=T0;q[:,5]=ell*r;q[:,6]=u0+u2*r*r
d,dd=op.derivatives(q,PARITY)
R,S,diag,*_=g1_rhs.parts(op,q,d,dd,restoring=1)
rate=R[0]+2*S[0];alpha=.5+u0;K=-3+p0+2*T0;chi=1+c0
np.testing.assert_allclose(rate[[2,5]],[0,0],atol=2e-12)
expected_T=alpha/4*(12*c2+3*chi*ell)+2*alpha*((K+3)*(K-3)/3-3*c0-4*1.01*T0)
np.testing.assert_allclose(rate[4],expected_T,atol=2e-12)
expected_H=2*(K+3)*(K-3)/3-6*c0+3*c2
np.testing.assert_allclose(diag[0,0],expected_H,atol=2e-12)
print('Origin polynomial, quadrature, scalar and free-Z4c limits passed; stability is a separate evolution gate.')
