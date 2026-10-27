#!/usr/bin/env python3
"""Compatible initial lapse: stationary chi and exact-vacuum mass identity."""
import numpy as np
from scalar_initial_data import solve_data,stationary_chi_lapse
from numerics import RadialOperators
from kernel import PARITY
import g1_rhs
errors=[];mass_errors=[]
for n in [128,256,512]:
    r=(np.arange(n)+.5)/(n-.5);initial=solve_data(n,.058,1e-11,grid=np.r_[0.,r],precision='decimal')
    assert initial['converged'];O=(1-r*r)/2;A=(1+r*r)/2
    q=np.zeros((n,8));q[:,0]=np.expm1(-4*np.log1p(O**3*initial['w'][1:]))
    q[:,6]=stationary_chi_lapse(initial)[1:]
    op=RadialOperators(r);d,dd=op.derivatives(q,PARITY)
    R,S,*_=g1_rhs.parts(op,q,d,dd,restoring=1)
    rate=R[:-1,0]+S[:-1,0]/O[:-1]
    errors.append(float(np.sqrt(np.mean(rate*rate))))
    chi=1+q[:,0];mass=8*initial['w'][-1];mask=r>.8
    identity=2*A*q[:,6]+q[:,6]**2+r*r*q[:,0]/chi+2*mass*O**3*np.sqrt(chi)/r
    mass_errors.append(float(max(abs(identity[mask]))))
    assert min(A+q[:,6])>0
orders=np.log2(np.array(errors[:-1])/np.array(errors[1:]));assert min(orders)>4,orders
assert mass_errors[-1]<1e-12,mass_errors
print(dict(chi_rate_errors=errors,orders=orders.tolist(),exterior_stationary_lapse_identity=mass_errors))
