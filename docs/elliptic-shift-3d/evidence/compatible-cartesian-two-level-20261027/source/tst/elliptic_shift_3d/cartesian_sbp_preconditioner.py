"""Research SPD block tensor inverse plus an exact block-constant coarse space.

This changes only a linear-solve preconditioner. The original gauge operator,
residual, states and field equations are unchanged. Periodic constant modes
remain null; the coarse pseudoinverse verifies exactly those three modes.
"""
import numpy as np
from compatible_sbp import mm

class Preconditioner:
    def __init__(self,grid,metric,apply):
        self.grid=grid;self.metric=metric;n=grid.n;H=grid.local['H'];D=grid.local['Dminus'];trace=grid.local['trace']
        S=mm(D.T*H[None,:],D)+n*mm(trace.T,trace)
        F=S/np.sqrt(H)[:,None]/np.sqrt(H)[None,:]
        self.eigen,self.U=np.linalg.eigh((F+F.T)/2)
        if min(self.eigen)<=0:raise ValueError('block inverse is not positive')
        self.rootH=np.sqrt(H[:,None,None]*H[None,:,None]*H[None,None,:])[...,None]
        self.denominators=[self.eigen[:,None,None]/width[0]**2+self.eigen[None,:,None]/width[1]**2+self.eigen[None,None,:]/width[2]**2 for _,width in grid.blocks]
        dimension=3*len(grid.blocks);K=np.zeros((dimension,dimension))
        for col in range(dimension):
            basis=np.zeros(grid.weights.shape+(3,));basis[col//3,...,col%3]=1
            K[:,col]=self.coarse_rhs(apply(basis))
        defect=float(np.max(abs(K-K.T)))/max(1,float(np.max(abs(K))))
        if defect>1e-10:raise ValueError('coarse Galerkin symmetry failed')
        eigen,U=np.linalg.eigh((K+K.T)/2);threshold=1e-10*max(eigen)
        if min(eigen)<-threshold or sum(abs(eigen)<threshold)!=3:raise ValueError('unexpected coarse nullspace')
        modes=np.tile(np.eye(3),(len(grid.blocks),1))
        null_error=float(np.max(abs(mm(K,modes))))/max(1,float(np.max(abs(K))))
        if null_error>1e-10:raise ValueError('nullspace is not periodic translations')
        inverse=np.where(eigen>threshold,1/np.maximum(eigen,threshold),0)
        self.coarse_inverse=mm(U*inverse[None,:],U.T)
        self.diagnostic={'block_inverse_min_eigenvalue':float(min(self.eigen)),'block_inverse_max_eigenvalue':float(max(self.eigen)),'coarse_dimension':dimension,'coarse_null_count':3,'coarse_constant_defect':null_error,'coarse_symmetry_defect':defect,'coarse_smallest_positive':float(eigen[3]),'coarse_largest':float(eigen[-1]),'boundary_penalty':n,'physical_operator_changed':False}

    def coarse_rhs(self,value):
        lowered=np.einsum('ij,...j->...i',self.metric,value,optimize=False)
        return np.sum(self.grid.weights[...,None]*lowered,axis=(1,2,3)).ravel()

    def __call__(self,rhs):
        result=np.empty_like(rhs)
        for b in range(len(self.grid.blocks)):
            v=rhs[b]*self.rootH
            for axis in range(3):v=self.grid.axis_apply(v,axis,self.U.T)
            v=v/self.denominators[b][...,None]
            for axis in range(3):v=self.grid.axis_apply(v,axis,self.U)
            result[b]=v/self.rootH
        correction=mm(self.coarse_inverse,self.coarse_rhs(rhs)).reshape(len(self.grid.blocks),3)
        return result+correction[:,None,None,None,:]
