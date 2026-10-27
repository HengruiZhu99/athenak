"""Exact weighted adjoints of the declared full metric variation and shift rate.

These are diagnostic/solver actions, not a replacement gauge or a source
projection. The evolved connection remains independent. The boundary term of
the paired derivative is included at actual scri; dropping it would give the
wrong transpose and an invalid conditioning diagnosis.
"""
import numpy as np


class GaugeAdjoint:
    def __init__(self,cached_gauge):
        self.gauge=cached_gauge;self.grid=cached_gauge.grid

    def derivative(self,value,axis,sign):
        result=-self.grid.derivative(value,axis,-sign)
        for face in self.grid.outer:
            b,a,side=face
            index=[b,slice(None),slice(None),slice(None)];index[a+1]=-1 if side else 0
            index=tuple(index)
            J=self.grid.trace(np.array(self.grid.jacobians),face)
            H=self.grid.locals[b]['H'][-1 if side else 0]
            coefficient=self.grid.normal(face)[...,axis]/(J*H)
            result[index]+=coefficient.reshape(coefficient.shape+(1,)*(value.ndim-4))*value[index]
        return result

    def variation(self,vector):
        gauge=self.gauge
        out=np.einsum('...iab,...i->...ab',gauge.local,vector,optimize=False)
        for a in range(3):
            flux=np.einsum('...ibc,...i->...bc',gauge.principal[...,a,:,:],vector,optimize=False)
            out+=self.derivative(flux,a,1)
        return out

    def rate_flux(self,tensor):
        g=self.gauge.g
        out=(np.einsum('...kj,...ij->...ik',g,tensor,optimize=False)
             +np.einsum('...kj,...ji->...ik',g,tensor,optimize=False))
        trace=np.einsum('...ij,...ij->...',g,tensor,optimize=False)
        return out-(2/3)*trace[...,None,None]*np.eye(3)

    def rate(self,tensor):
        out=np.einsum('...kij,...ij->...k',self.gauge.advective_dg,tensor,optimize=False)
        flux=self.rate_flux(tensor)
        for axis in range(3):out+=self.derivative(flux[...,axis,:],axis,-1)
        return out


class EnrichedGaugeAdjoint:
    """Transpose in the old positive nodal coefficient norm, including bubbles."""
    def __init__(self,enriched):
        self.enriched=enriched;self.full=GaugeAdjoint(enriched.gauge)

    def rate(self,tensor):
        e=self.enriched;out=self.full.rate(tensor);out[e.boundary]=0
        flux=self.full.rate_flux(tensor);W=e.grid.weights
        for b,high,gradq,_ in e.plans:
            integrand=np.einsum('...ik,...i->...k',flux[b],gradq,optimize=False)*high[:,None,None,None]
            out[b,-1]=np.sum(W[b,...,None]*integrand,axis=0)/W[b,-1,...,None]
        return out

    def apply(self,vector):return -self.rate(self.full.variation(vector))


class EnrichedPreconditionerAdjoint:
    """Transpose of the existing block-triangular preconditioner, unchanged."""
    def __init__(self,preconditioner,operator_adjoint):
        self.pre=preconditioner;self.adjoint=operator_adjoint

    def __call__(self,value):
        p=self.pre;base=p.base;e=p.e
        if base.balanced:raise ValueError('this transpose requires the additive base preconditioner')
        v=value.copy();v[e.boundary]=0
        coefficients=np.einsum('ij,j->i',base.inverse.T,base.coarse_rhs(v),optimize=False)
        coefficients=coefficients.reshape(len(e.grid.blocks),base.count,3)
        out=np.einsum('bmc,bijkm->bijkc',coefficients,base.modes,optimize=False)+base.block_inverse(v)
        remainder=value-self.adjoint.apply(out)
        out[e.boundary]=np.einsum('nji,nj->ni',p.inverse,remainder[e.boundary],optimize=False)
        return out
