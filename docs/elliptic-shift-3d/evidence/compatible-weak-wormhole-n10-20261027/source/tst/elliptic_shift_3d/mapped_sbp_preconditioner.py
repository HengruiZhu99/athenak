"""Separable mapped-block inverse and full-operator coarse correction.

Research preconditioner only. Coarse matrices use the exact nonsymmetric tensor
action. Homogeneous outer Dirichlet correction values are part of the declared
linear boundary problem, never a puncture constraint or evolved-field reset.
"""
import numpy as np
from compatible_sbp import mm
from mapped_sbp_probe import axis_apply


def homogeneous_outer(value,grid):
    result=value.copy()
    for b,a,s in grid.outer:
        index=[b,slice(None),slice(None),slice(None)];index[a+1]=-1 if s else 0
        result[tuple(index)]=0
    return result


class MappedPreconditioner:
    def __init__(self,grid,metric,apply,degree=1,balanced=False):
        self.grid=grid;self.apply=apply;self.balanced=balanced
        self.W=grid.weights;self.roots=np.sqrt(self.W)[...,None];self.plans=[]
        gi=np.linalg.inv(metric);n=grid.n
        for b,loc in enumerate(grid.locals):
            H=loc['H'];D=loc['D']-loc['R']/H[:,None];t=loc['trace']
            penalty=n if grid.blocks[b][0]<0 else n*n
            S=mm(D.T*H[None,:],D)+penalty*mm(t.T,t)
            F=S/np.sqrt(H)[:,None]/np.sqrt(H)[None,:]
            eigen,U=np.linalg.eigh((F+F.T)/2)
            if min(eigen)<=0:raise ValueError('nonpositive mapped block inverse')
            C=grid.cofactors[b];J=grid.jacobians[b]
            coefficients=[]
            for a in range(3):
                values=np.einsum('...i,...ij,...j->...',C[...,a,:],gi[b],C[...,a,:],optimize=False)/J**2
                coefficients.append(float(np.sum(self.W[b]*values)/np.sum(self.W[b])))
            denominator=(coefficients[0]*eigen[:,None,None]+coefficients[1]*eigen[None,:,None]+coefficients[2]*eigen[None,None,:])
            self.plans.append((U,denominator))
        if degree not in [0,1,2]:raise ValueError('coarse polynomial degree budget')
        self.modes=[]
        for b,loc in enumerate(grid.locals):
            q=np.meshgrid(2*loc['x']-1,2*loc['x']-1,2*loc['x']-1,indexing='ij')
            modes=[np.ones((n,n,n))]
            if degree>=1:modes.extend(q)
            if degree>=2:modes.extend([q[i]*q[j] for i in range(3) for j in range(i,3)])
            modes=np.stack(modes,axis=-1)
            for face in grid.outer:
                if face[0]==b:
                    index=[slice(None)]*3;index[face[1]]=-1 if face[2] else 0
                    modes[tuple(index)]=0
            gram=np.einsum('ijk,ijkm,ijkn->mn',self.W[b],modes,modes,optimize=False)
            factor=np.linalg.cholesky(gram)
            modes=np.einsum('ijkm,mn->ijkn',modes,np.linalg.inv(factor.T),optimize=False)
            self.modes.append(modes)
        self.modes=np.array(self.modes);self.count=self.modes.shape[-1]
        dimension=3*len(grid.blocks)*self.count
        if dimension>600:raise ValueError('mapped coarse direct solve budget')
        K=np.zeros((dimension,dimension))
        for col in range(dimension):
            basis=np.zeros(grid.weights.shape+(3,))
            b=col//(3*self.count);mode=(col//3)%self.count
            basis[b,...,col%3]=self.modes[b,...,mode]
            K[:,col]=self.coarse_rhs(apply(basis))
        singular=np.linalg.svd(K,compute_uv=False)
        if min(singular)<1e-12*max(singular):raise ValueError('singular mapped Dirichlet coarse problem')
        self.inverse=np.linalg.inv(K)
        self.diagnostic={'coarse_dimension':dimension,'coarse_degree':degree,
                         'coarse_singular_min':float(min(singular)),
                         'coarse_singular_max':float(max(singular)),
                         'coarse_condition':float(max(singular)/min(singular)),
                         'balanced':balanced,'operator_changed':False}

    def coarse_rhs(self,value):
        return np.einsum('bijk,bijkm,bijkc->bmc',self.W,self.modes,value,optimize=False).ravel()

    def coarse(self,value):
        coefficients=mm(self.inverse,self.coarse_rhs(value)).reshape(len(self.grid.blocks),self.count,3)
        return np.einsum('bmc,bijkm->bijkc',coefficients,self.modes,optimize=False)

    def block_inverse(self,value):
        out=np.empty_like(value)
        for b,(U,denominator) in enumerate(self.plans):
            v=value[b]*self.roots[b]
            for a in range(3):v=axis_apply(v,a,U.T)
            v/=denominator[...,None]
            for a in range(3):v=axis_apply(v,a,U)
            out[b]=v/self.roots[b]
        return homogeneous_outer(out,self.grid)

    def __call__(self,value):
        coarse=self.coarse(value)
        if not self.balanced:return coarse+self.block_inverse(value)
        fine=self.block_inverse(value-self.apply(coarse))
        return coarse+fine-self.coarse(self.apply(fine))
