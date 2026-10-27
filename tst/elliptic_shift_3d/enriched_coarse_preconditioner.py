"""Full-action coarse correction including the new outer bubble unknowns.

This is solely a Krylov preconditioner. Its coefficient-space inner product is
not asserted to be a physical energy norm. No residual equations are removed.
"""
import numpy as np
from compatible_sbp import mm


class EnrichedCoarsePreconditioner:
    def __init__(self,enriched,base,triangular,degree=1):
        self.e=enriched;self.base=base;self.triangular=triangular;self.modes=[]
        n=enriched.grid.n;W=enriched.grid.weights
        if degree not in [1,2]:raise ValueError('bounded outer coarse degree')
        for b,_,_ in enriched.grid.outer:
            q=2*enriched.grid.locals[b]['x']-1;u,v=np.meshgrid(q,q,indexing='ij')
            modes=[np.ones((n,n)),u,v]
            if degree==2:modes.extend([u*u,u*v,v*v])
            modes=np.stack(modes,axis=-1)
            gram=np.einsum('jk,jkm,jkn->mn',W[b,-1],modes,modes,optimize=False)
            inverse=np.linalg.inv(np.linalg.cholesky(gram).T)
            modes=np.einsum('jkm,mn->jkn',modes,inverse,optimize=False)
            self.modes.append((b,modes))
        self.base_count=3*len(enriched.grid.blocks)*base.count
        dimension=self.base_count+sum(3*m.shape[-1] for _,m in self.modes)
        if dimension>600:raise ValueError('enriched coarse dimension budget')
        K=np.zeros((dimension,dimension))
        for col in range(dimension):
            unit=np.zeros(dimension);unit[col]=1
            K[:,col]=self.restrict(enriched.apply(self.prolong(unit)))
        singular=np.linalg.svd(K,compute_uv=False)
        if min(singular)<1e-12*max(singular):raise ValueError('singular enriched coarse matrix')
        self.inverse=np.linalg.inv(K)
        self.diagnostic={'dimension':dimension,'outer_degree':degree,'condition':float(max(singular)/min(singular)),
                         'minimum_singular':float(min(singular)),'maximum_singular':float(max(singular)),
                         'balanced':True,'operator_changed':False}

    def restrict(self,value):
        out=[self.base.coarse_rhs(value)]
        for b,m in self.modes:
            out.append(np.einsum('jk,jkm,jkc->mc',self.e.grid.weights[b,-1],m,value[b,-1],optimize=False).ravel())
        return np.concatenate(out)

    def prolong(self,coefficients):
        old=coefficients[:self.base_count].reshape(len(self.e.grid.blocks),self.base.count,3)
        out=np.einsum('bmc,bijkm->bijkc',old,self.base.modes,optimize=False)
        pos=self.base_count
        for b,m in self.modes:
            count=3*m.shape[-1]
            out[b,-1]=np.einsum('jkm,mc->jkc',m,coefficients[pos:pos+count].reshape(-1,3),optimize=False)
            pos+=count
        return out

    def coarse(self,rhs):return self.prolong(mm(self.inverse,self.restrict(rhs)))

    def __call__(self,rhs):
        coarse=self.coarse(rhs)
        fine=self.triangular(rhs-self.e.apply(coarse))
        return coarse+fine-self.coarse(self.e.apply(fine))
