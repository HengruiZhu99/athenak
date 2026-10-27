"""Algebraically cached full Frechet coefficients; no field/operator change.

The independent uncached TensorGauge remains the reference and finite-epsilon
connection evaluator. Caching is at one fixed metric stage only; reconstruct
this object after any metric update.
"""
import numpy as np
from compatible_tensor_gauge import TensorGauge


class CachedTensorGauge(TensorGauge):
    def __init__(self,grid,metric):
        super().__init__(grid,metric)
        gi=self.gi;T=self.dg-.5*np.swapaxes(self.dg,4,5)
        lowered=np.einsum('...il,...jlk->...ijk',gi,T,optimize=False)
        first=-np.einsum('...ja,...ijk,...bk->...iab',gi,lowered,gi,optimize=False)
        q=np.einsum('...jk,...jlk->...l',gi,T,optimize=False)
        w=np.einsum('...bl,...l->...b',gi,q,optimize=False)
        self.local=first-gi[..., :, :,None]*w[...,None,None,:]
        self.principal=(np.einsum('...ac,...ib->...iabc',gi,gi,optimize=False)
                        -.5*np.einsum('...bc,...ia->...iabc',gi,gi,optimize=False))

    def variation(self,h):
        return (np.einsum('...iab,...ab->...i',self.local,h,optimize=False)
                +np.einsum('...iabc,...abc->...i',self.principal,self.gradient(h,1),optimize=False))
