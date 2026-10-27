"""Global radial trial functions as a full-operator Krylov preconditioner.

No spherical averaging is applied to the state or the equation. Every fine
unknown and angular residual remains. The columns include three Cartesian
vector directions and a radial vector, with unconstrained fine puncture cells.
This is not a CFC solve or a replacement radial evolution. The full tensor
action, including variable metric and boundary bubble coupling, builds K.
"""
import numpy as np
from numpy.polynomial import chebyshev as cheb
from compatible_sbp import mm
from enriched_outer_gauge import EnrichedPreconditioner


class RadialCoarsePreconditioner:
    def __init__(self,enriched,base,degree=16):
        if not 2<=degree<=20:raise ValueError('bounded radial coarse degree')
        self.e=enriched;self.fine=EnrichedPreconditioner(enriched,base)
        W=enriched.grid.weights;X=enriched.grid.coordinates;r=np.linalg.norm(X,axis=-1)
        polynomials=cheb.chebvander(2*r-1,degree);columns=[]
        for k in range(degree+1):
            profile=(1-r*r)*polynomials[...,k]
            for component in range(3):
                v=np.zeros(W.shape+(3,));v[...,component]=profile;v[enriched.boundary]=0;columns.append(v)
            v=X*profile[...,None];v[enriched.boundary]=0;columns.append(v)
        for kind in range(4):
            v=np.zeros(W.shape+(3,))
            for b,_,gradq,_ in enriched.plans:
                width=1/np.linalg.norm(gradq[-1],axis=-1)
                if kind<3:v[b,-1,...,kind]=width
                else:v[b,-1]=width[...,None]*X[b,-1]/r[b,-1,...,None]
            columns.append(v)
        P=np.array(columns);self.W=W
        gram=np.einsum('bijk,mbijkc,nbijkc->mn',W,P,P,optimize=False)
        factor=np.linalg.cholesky(gram);inverse=np.linalg.inv(factor)
        self.P=np.einsum('mn,nbijkc->mbijkc',inverse,P,optimize=False)
        self.AP=np.array([enriched.apply(v) for v in self.P])
        K=np.einsum('bijk,mbijkc,nbijkc->mn',W,self.P,self.AP,optimize=False)
        singular=np.linalg.svd(K,compute_uv=False)
        if min(singular)<1e-12*max(singular):raise ValueError('singular global radial coarse matrix')
        self.inverse=np.linalg.inv(K)
        self.diagnostic={'degree':degree,'dimension':len(columns),'gram_condition':float(np.linalg.cond(gram)),
                         'minimum_singular':float(min(singular)),'maximum_singular':float(max(singular)),
                         'condition':float(max(singular)/min(singular)),'operator_changed':False,
                         'fine_unknowns_removed':0,'angular_state_averaged':False}

    def coefficients(self,value):
        return mm(self.inverse,np.einsum('bijk,mbijkc,bijkc->m',self.W,self.P,value,optimize=False))

    def prolong(self,c):return np.einsum('m,mbijkc->bijkc',c,self.P,optimize=False)

    def __call__(self,rhs):
        c=self.coefficients(rhs);coarse=self.prolong(c)
        image=np.einsum('m,mbijkc->bijkc',c,self.AP,optimize=False)
        fine=self.fine(rhs-image)
        return coarse+fine-self.prolong(self.coefficients(self.e.apply(fine)))
