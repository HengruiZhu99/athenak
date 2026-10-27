#!/usr/bin/env python3
"""Independent 2D Christoffel audit of the Bondi-frame nonaffinity."""
import json
import numpy as np
from bondi import nonaffinity
rng=np.random.default_rng(9823);errors=[]
for _ in range(50):
    chi,alpha=np.exp(rng.normal(size=2));cr,ar,br,ct,at=rng.normal(size=5)
    b=-alpha*np.sqrt(chi);bt=b*(at/alpha+.5*ct/chi)
    metric=np.array([[0.,b],[b,1.]])
    # g_Btt=-chi alpha^2+beta^2, g_Btr=beta, g_Brr=1.
    dt=np.array([[-ct*alpha*alpha-2*chi*alpha*at+2*b*bt,bt],[bt,0.]])
    dr=np.array([[-cr*alpha*alpha-2*chi*alpha*ar+2*b*br,br],[br,0.]])
    dg=np.array([dt,dr]);inverse=np.linalg.inv(metric);Gamma=np.zeros((2,2,2))
    for a in range(2):
      for j in range(2):
       for k in range(2):
        Gamma[a,j,k]=sum(inverse[a,l]*(dg[j,l,k]+dg[k,l,j]-dg[l,j,k])/2 for l in range(2))
    n=np.array([1/alpha,0.]);dn=np.array([-at/alpha**3,0.])
    acceleration=dn+np.einsum('ajk,j,k->a',Gamma,n,n)
    kappa=nonaffinity(chi,alpha,cr,ar,br,ct)
    errors.append(float(max(abs(acceleration-kappa*n))))
assert max(errors)<1e-11,errors
assert nonaffinity(1.,1.,0.,1.,-1.,0.)==0.
print(json.dumps(dict(Christoffel_max_error=max(errors),samples=len(errors),Minkowski_affine=True,
    scope='Local scri-frame geometry; accumulated Bondi time still requires trajectory integration')))
