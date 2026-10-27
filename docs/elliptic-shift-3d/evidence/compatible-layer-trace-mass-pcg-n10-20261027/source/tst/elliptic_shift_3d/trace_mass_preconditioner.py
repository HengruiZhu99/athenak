"""Use the actual endpoint trace norm in the approximate block inverse.

The full gauge, coarse operator, source and boundary equations are unchanged.
For a collocated endpoint, the trace norm is 1/H_endpoint, scaling as N for
uniform finite-difference nodes rather than N^2 for Lobatto nodes. For a core
extrapolation trace, its complete weighted norm is used instead.
"""
import numpy as np
from compatible_sbp import mm
from mapped_sbp_preconditioner import MappedPreconditioner


class TraceMassPreconditioner(MappedPreconditioner):
    def __init__(self,grid,metric,apply,degree=1,balanced=False):
        super().__init__(grid,metric,apply,degree,balanced)
        gi=np.linalg.inv(metric);self.plans=[];penalties=[]
        for b,loc in enumerate(grid.locals):
            H=loc['H'];D=loc['D']-loc['R']/H[:,None];t=loc['trace']
            penalty=float(np.max(np.sum(t*t/H[None,:],axis=1)));penalties.append(penalty)
            S=mm(D.T*H[None,:],D)+penalty*mm(t.T,t)
            F=S/np.sqrt(H)[:,None]/np.sqrt(H)[None,:]
            eigen,U=np.linalg.eigh((F+F.T)/2)
            if min(eigen)<=0:raise ValueError('nonpositive trace-norm block inverse')
            C=grid.cofactors[b];J=grid.jacobians[b];coefficients=[]
            for a in range(3):
                values=np.einsum('...i,...ij,...j->...',C[...,a,:],gi[b],C[...,a,:],optimize=False)/J**2
                coefficients.append(float(np.sum(self.W[b]*values)/np.sum(self.W[b])))
            denominator=coefficients[0]*eigen[:,None,None]+coefficients[1]*eigen[None,:,None]+coefficients[2]*eigen[None,None,:]
            self.plans.append((U,denominator))
        self.diagnostic.update(trace_mass_penalties=penalties,trace_mass_scope=__doc__)
