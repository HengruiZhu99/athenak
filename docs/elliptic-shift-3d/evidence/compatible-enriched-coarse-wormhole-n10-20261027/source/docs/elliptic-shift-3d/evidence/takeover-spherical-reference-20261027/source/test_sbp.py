#!/usr/bin/env python3
"""SBP algebra, compatible second derivatives, and norm-weighted KO only."""
import json
import numpy as np
from sbp import SBPOperators
from numerics import puncture_dissipation
errors=[];identities=[];compatible=[]
for n in [33,65,129]:
    r=np.linspace(0,1,n);op=SBPOperators(r);h=r[1];I=np.eye(n)
    D=op.first(I);D2=op.first(D);E=np.zeros((n,n));E[0,0]=-1;E[-1,-1]=1
    norm=h*op.norm;identity=max(abs(norm[:,None]*D+D.T*norm[None,:]-E).ravel())
    identities.append(float(identity));assert identity<1e-11,identity
    comp=max(abs(norm[:,None]*D2+np.einsum('ij,jk->ik',D.T,norm[:,None]*D)-np.einsum('ij,jk->ik',E,D)).ravel())
    compatible.append(float(comp));assert comp<1e-9,comp
    for k in range(4):
        first,second=op.derivatives(r**k,1)
        np.testing.assert_allclose(first,k*r**(k-1) if k else 0,atol=1e-10,rtol=0)
        np.testing.assert_allclose(second,k*(k-1)*r**(k-2) if k>=2 else 0,atol=1e-8,rtol=0)
    _,dd=op.derivatives(np.sin(r),1);errors.append(float(max(abs(dd+np.sin(r)))))
    rng=np.random.default_rng(810);q=rng.normal(size=(n,8));diss=puncture_dissipation(q,r,h,.5)/op.norm[:,None]
    assert np.sum(norm[:,None]*q*diss)<0 and np.all(diss[0]==0)
    # Scalar incoming-right SAT has the exact energy estimate in the H norm.
    v=np.einsum('ij,j->i',D,q[:,0]);v[-1]-=q[-1,0]/norm[-1]
    energy=np.dot(norm*q[:,0],v);expected=-(q[0,0]**2+q[-1,0]**2)/2
    assert abs(energy-expected)<1e-10
orders=np.log2(np.array(errors[:-1])/errors[1:]);assert min(orders)>1.8,orders
print(json.dumps(dict(SBP_identity=identities,compatible_D2_identity=compatible,
    second_derivative_max_errors=errors,second_derivative_orders=orders.tolist(),
    KO_negative_in_H_norm=True,scalar_SAT_energy_identity=True,
    scope='Operator algebra only; no Z4c stability or endpoint convergence admission')))
