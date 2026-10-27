"""Coupled radial nodal/bubble relaxation for the unadopted trial space.

Columns use angularly constant amplitudes on all outer shells. The resulting
line blocks approximate angular row sums, not the exact block diagonal. They
retain all radial and Cartesian-vector coupling. This approximate inverse is
only a preconditioner; the full tensor action verifies every Krylov residual.
"""
import numpy as np


class EnrichedLinePreconditioner:
    def __init__(self,enriched,base):
        self.e=enriched;self.base=base;n=enriched.grid.n
        self.blocks=[b for b,_,_ in enriched.grid.outer]
        matrix=np.zeros((len(self.blocks),n,n,3*n,3*n))
        for q in range(n):
            for component in range(3):
                v=np.zeros(enriched.grid.weights.shape+(3,))
                for b in self.blocks:v[b,q,...,component]=1
                out=enriched.apply(v)
                for k,b in enumerate(self.blocks):
                    matrix[k,...,3*q+component]=np.moveaxis(out[b],0,2).reshape(n,n,3*n)
        singular=np.linalg.svd(matrix,compute_uv=False)
        self.inverse=np.linalg.inv(matrix)
        self.diagnostic={'line_blocks':len(self.blocks)*n*n,'line_dimension':3*n,
                         'minimum_singular':float(np.min(singular)),
                         'maximum_condition':float(np.max(singular[...,0]/singular[...,-1])),
                         'operator_changed':False}

    def __call__(self,rhs):
        interior=rhs.copy();interior[self.e.boundary]=0
        value=self.base(interior);residual=rhs-self.e.apply(value);n=self.e.grid.n
        for k,b in enumerate(self.blocks):
            r=np.moveaxis(residual[b],0,2).reshape(n,n,3*n)
            correction=np.einsum('jkab,jkb->jka',self.inverse[k],r,optimize=False)
            value[b]+=np.moveaxis(correction.reshape(n,n,n,3),2,0)
        return value
