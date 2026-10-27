#!/usr/bin/env python3
"""Principal constraint-free gauge lift and frozen Robin/SBP energy algebra."""
import json
import numpy as np
from gauge import robin_penalty
rng=np.random.default_rng(741);errors=[]
for _ in range(100):
    O,alpha,chi=np.exp(rng.normal(size=3));F=alpha*alpha+alpha
    c=np.sqrt(F*chi);beta=c*rng.uniform(0,1);a=F/O;b=O*chi
    P=np.zeros((6,6)) # [dchi,Ar,Khat,Theta,Lambda,dalpha]
    P[0,[1,2,3]]=[-alpha*chi,2*alpha*chi/(3*O),4*alpha*chi/(3*O)]
    P[1,[0,4,5]]=[alpha/3,2*alpha*chi/3,-2*chi/3]
    P[2,5]=-O*chi;P[3,[0,4]]=[O*alpha,O*alpha*chi/2]
    P[4,[1,2,3]]=[2*alpha,-4*alpha/(3*O),-2*alpha/(3*O)];P[5,2]=-F/O
    mode=np.array([0,2/(3*O),1,0,0,c/(O*chi)])
    errors.append(float(max(abs(np.einsum('ij,j->i',P,mode)+c*mode))))
    # Independent principal Hamiltonian/momentum/Theta/connection errors.
    assert mode[0]==mode[3]==mode[4]==0 and abs(O*mode[1]-2*mode[2]/3)<1e-13
    A,K=rng.normal(size=2);norm=.02
    penalty,_,incoming=robin_penalty(.01,O,alpha,chi,beta,A,F,norm,lapse_power=0)
    left_flux=-beta*(A*A+a/b*K*K)/2+a*K*A
    energy=left_flux+norm*a/b*K*penalty
    expected=-beta*(A-np.sqrt(a/b)*K)**2/2
    np.testing.assert_allclose(energy,expected,atol=1e-10,rtol=1e-10)
    inactive,_,incoming=robin_penalty(.01,O,alpha,chi,2*c,A,F,norm)
    assert inactive==incoming==0
assert max(errors)<1e-10,errors
print(json.dumps(dict(principal_gauge_eigenvector_error=max(errors),samples=len(errors),
    incoming_penalty_homogeneous_Neumann_energy=True,physical_outflow_required=True,
    scope='Frozen principal gauge/constraint lift and energy identity. Robin lower-order terms and full nonlinear Z4c require evolution tests.')))
