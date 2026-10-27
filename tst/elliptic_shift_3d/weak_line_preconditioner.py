"""Positive approximate radial strain-energy blocks, not strong row sums.

The tensor principal energy couples all Cartesian components and the extra
radial bubble within each angular line. Angular stiffness is a positive
diagonal estimate. Summed with the existing positive nodal inverse, these
blocks are an additive Schwarz preconditioner only. All equations are unchanged.
"""
import numpy as np
from compatible_sbp import mm


class WeakLinePreconditioner:
    def __init__(self,enriched,base):
        self.e=enriched;self.base=base;self.inverse={};grid=enriched.grid;n=grid.n
        self.diagnostic={'line_dimension':3*n,'blocks':0,'minimum_eigenvalue':float('inf'),
                         'maximum_condition':0.,'operator_changed':False}
        for b,high,gradq,_ in enriched.plans:
            loc=grid.locals[b];H=loc['H'];D=loc['D']-loc['R']/H[:,None]
            radial=D.copy();radial[:,-1]=high
            # The final coefficient is a bubble, not a nodal endpoint value.
            norms=np.sum(H[:,None]*D*D,axis=0)+n*n*np.sum(loc['trace']**2,axis=0)
            norms/=H
            C=grid.cofactors[b]/grid.jacobians[b][...,None,None]
            K=np.zeros((n,n,n,3,n,3))
            for j in range(n):
                for k in range(n):
                    W=grid.weights[b,:,j,k];normal=gradq[:,j,k]
                    tensor=np.sum(normal**2,axis=-1)[:,None,None]*np.eye(3)+normal[:,:,None]*normal[:,None,:]/3
                    local=np.einsum('i,ij,ik,iab->jakb',W,radial,radial,tensor,optimize=False)
                    for q in range(n-1):
                        for a,index in [(1,j),(2,k)]:
                            v=C[q,j,k,a];T=np.dot(v,v)*np.eye(3)+np.outer(v,v)/3
                            local[q,:,q,:]+=W[q]*norms[index]*T
                    # A local trace penalty models the neighboring radial
                    # block's energy contribution, without changing its field.
                    local[0,:,0,:]+=W[0]*(n*n/H[0])*tensor[0]
                    K[j,k]=local
            K=K.reshape(n,n,3*n,3*n);eig=np.linalg.eigvalsh(K)
            if np.min(eig)<=0:raise ValueError('nonpositive approximate radial strain block')
            self.inverse[b]=np.linalg.inv(K)
            self.diagnostic['blocks']+=n*n
            self.diagnostic['minimum_eigenvalue']=min(self.diagnostic['minimum_eigenvalue'],float(np.min(eig)))
            self.diagnostic['maximum_condition']=max(self.diagnostic['maximum_condition'],float(np.max(eig[...,-1]/eig[...,0])))

    def __call__(self,rhs):
        v=rhs.copy();v[self.e.boundary]=0;out=self.base(v);n=self.e.grid.n
        for b,inverse in self.inverse.items():
            weighted=self.e.grid.weights[b,...,None]*rhs[b]
            line=np.moveaxis(weighted,0,2).reshape(n,n,3*n)
            correction=np.einsum('jkab,jkb->jka',inverse,line,optimize=False)
            out[b]+=np.moveaxis(correction.reshape(n,n,n,3),2,0)
        return out
