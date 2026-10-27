"""Exact per-face bubble blocks for the unchanged full tensor action.

Only a preconditioner. All cross-face and bulk couplings stay in the fine
operator and every reported residual. Blocks have at most 600 unknowns; eight
columns are evaluated together, without angular symmetry or row-sum shortcuts.
"""
import numpy as np


class HermiteFacePreconditioner:
    def __init__(self,enriched,base):
        self.e=enriched;self.base=base;grid=enriched.grid;gauge=enriched.gauge
        n=grid.n;dimension=3*n*n
        if dimension>600:raise ValueError('per-face direct block budget')
        self.inverse={};diagnostics=[]
        for b,high,gradq,_ in enriched.plans:
            matrix=np.zeros((dimension,dimension))
            for first in range(0,dimension,8):
                count=min(8,dimension-first);coef=np.zeros((n,n,3,count))
                flat=coef.reshape(dimension,count);flat[first+np.arange(count),np.arange(count)]=1.
                db=high[:,None,None,None,None,None]*gradq[..., :,None,None]*coef[None,...,None,:,:]
                metric=gauge.g[b];div=np.einsum('...iiz->...z',db,optimize=False)
                h=np.zeros(grid.weights.shape+(3,3,count))
                h[b]=(np.einsum('...kj,...ikz->...ijz',metric,db,optimize=False)
                      +np.einsum('...ik,...jkz->...ijz',metric,db,optimize=False)
                      -(2/3)*metric[...,None]*div[...,None,None,:])
                action=-np.einsum('...iab,...abz->...iz',gauge.local,h,optimize=False)
                for axis in range(3):
                    derivative=grid.derivative(h,axis,1)
                    action-=np.einsum('...ibc,...bcz->...iz',gauge.principal[...,axis,:,:],derivative,optimize=False)
                matrix[:,first:first+count]=action[b,-1].reshape(dimension,count)
            singular=np.linalg.svd(matrix,compute_uv=False)
            condition=float(singular[0]/singular[-1])
            if singular[-1]<1e-12*singular[0]:raise ValueError(f'singular face bubble block {b}: condition={condition}')
            # Independently traverse the unbatched full action, including all
            # geometry and interfaces, for selected actual basis columns.
            defect=0.
            for col in [0,dimension//2,dimension-1]:
                coefficient=np.zeros(grid.weights.shape+(3,));coefficient[b,-1].reshape(-1)[col]=1
                expected=enriched.apply(coefficient)[b,-1].ravel()
                defect=max(defect,float(np.max(abs(expected-matrix[:,col]))))
            if defect>1e-8:raise ValueError('batched face block differs from full action')
            self.inverse[b]=np.linalg.inv(matrix)
            diagnostics.append({'block':b,'dimension':dimension,'minimum_singular':float(singular[-1]),
                                'condition':condition,'independent_column_max':defect})
        self.diagnostic={'scope':__doc__,'blocks':diagnostics,'operator_changed':False}

    def __call__(self,rhs):
        bubble=np.zeros_like(rhs)
        for b,inverse in self.inverse.items():
            bubble[b,-1]=np.einsum('ij,j->i',inverse,rhs[b,-1].ravel(),optimize=False).reshape(rhs[b,-1].shape)
        residual=rhs-self.e.apply(bubble);residual[self.e.boundary]=0
        return bubble+self.base(residual)
