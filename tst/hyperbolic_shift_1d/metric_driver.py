"""Transport rate of the metric-derived reference connection V.

This is a gauge-driver source only. The Z4 connection Lambda still evolves
independently in the unmodified tensor equations. No constraint projection.
"""
import numpy as np


def transport_rate(r,g,dg,alpha,da,Arr,dArr,beta,db,ddb):
    # G = (partial_t-beta partial_r) g from the metric evolution equation.
    shear=db-beta/r
    G=-2*alpha*Arr+4*g*shear/3
    Gp=-2*(da*Arr+alpha*dArr)+4*(dg*shear+g*(ddb-shear/r))/3
    root=np.sqrt(g)
    difference=(g-1)*(g+root+1)/(g*(root+1)) # sqrt(g)-1/g
    coefficient=-2*dg/g**3+(1/root+2/g**2)/r
    return (Gp+db*dg)/g**2+coefficient*G+2*beta*difference/r**2
