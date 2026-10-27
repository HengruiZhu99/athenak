"""Unadopted outer radial bubble representation for the full tensor gauge.

For n Lobatto metric nodes, B(q)=integral_0^q P_(n-1)(2s-1) ds
vanishes at every metric node, but B'=P_(n-1) does not. A vector
coefficient per outer-shell angular node adds this off-node shift content.
This changes the trial representation, not the prescribed curvature source.
No solved metric rate, source or connection residual is filtered.

The old boundary slots store bubble coefficients, NOT boundary shift values.
Physical Dirichlet values are supplied by a separate lift. All metric nodes,
including scri, receive their original connection-variation equation. This
square collocation experiment has NO established SBP energy identity for the
enriched space and NO qualified off-node angular-interface coupling. It must
not be adopted into evolution on the strength of a successful solve.
"""
import numpy as np
from numpy.polynomial import legendre as leg


class EnrichedOuterGauge:
    def __init__(self, gauge):
        self.gauge=gauge;self.grid=gauge.grid
        self.boundary=np.zeros(self.grid.weights.shape,dtype=bool)
        self.plans=[]
        for b,axis,side in self.grid.outer:
            if axis!=0 or side!=1:raise ValueError('outer radial shell faces required')
            self.boundary[b,-1]=True
            q=self.grid.locals[b]['x'];coeff=np.eye(self.grid.n)[-1]
            high=leg.legval(2*q-1,coeff)
            primitive=leg.legint(coeff)/2
            B=leg.legval(2*q-1,primitive)-leg.legval(-1,primitive)
            if np.max(abs(B))>1e-13:raise ValueError('bubble is not zero at metric nodes')
            # Jacobian/cofactors are the same declared geometric coefficients
            # as the base discretization. No analytic reference RHS is removed.
            gradq=self.grid.cofactors[b][...,0,:]/self.grid.jacobians[b][...,None]
            self.plans.append((b,high,gradq,float(np.max(abs(B)))))

    def nodal(self,coefficients):
        value=coefficients.copy();value[self.boundary]=0
        return value

    def bubble_rate(self,coefficients):
        rate=np.zeros(self.grid.weights.shape+(3,3))
        for b,high,gradq,_ in self.plans:
            db=high[:,None,None,None,None]*gradq[..., :,None]*coefficients[b,-1][None,...,None,:]
            g=self.gauge.g[b];div=np.einsum('...ii->...',db,optimize=False)
            rate[b]=(np.einsum('...kj,...ik->...ij',g,db,optimize=False)
                     +np.einsum('...ik,...jk->...ij',g,db,optimize=False)
                     -(2/3)*g*div[...,None,None])
        return rate

    def rate(self,coefficients):
        return self.gauge.rate(self.nodal(coefficients))+self.bubble_rate(coefficients)

    def apply(self,coefficients):return -self.gauge.variation(self.rate(coefficients))


class EnrichedPreconditioner:
    """Block-triangular approximate inverse, not a change to the equations."""
    def __init__(self,enriched,base_preconditioner):
        self.e=enriched;self.base=base_preconditioner
        boundary=enriched.boundary;shape=enriched.grid.weights.shape
        blocks=np.zeros((int(np.sum(boundary)),3,3))
        for j in range(3):
            a=np.zeros(shape+(3,));a[...,j][boundary]=1
            blocks[...,j]=enriched.apply(a)[boundary]
        self.inverse=np.linalg.inv(blocks)
        singular=np.linalg.svd(blocks,compute_uv=False)
        self.diagnostic={'boundary_block_min_singular':float(np.min(singular)),
                         'boundary_block_max_condition':float(np.max(singular[:,0]/singular[:,-1])),
                         'boundary_coefficients':3*int(np.sum(boundary))}

    def __call__(self,rhs):
        bubble=np.zeros_like(rhs)
        bubble[self.e.boundary]=np.einsum('nij,nj->ni',self.inverse,rhs[self.e.boundary],optimize=False)
        residual=rhs-self.e.apply(bubble);residual[self.e.boundary]=0
        return bubble+self.base(residual)
