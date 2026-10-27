"""Exact metric-connection variation of the declared paired discretization.

Common Cartesian components, complete variable metric, non-unit determinant
allowed. This does not use the independently evolved Z4 connection. No continuum
product rule or expanded second derivative is substituted for the variation.
"""
import numpy as np


class TensorGauge:
    def __init__(self,grid,metric):
        self.grid=grid;self.g=metric;self.gi=np.linalg.inv(metric)
        if np.min(np.linalg.eigvalsh(metric))<=0:raise ValueError('metric must be SPD')
        self.dg=self.gradient(metric,1);self.advective_dg=self.gradient(metric,-1)

    def gradient(self,value,sign):
        return np.stack([self.grid.derivative(value,j,sign) for j in range(3)],axis=4)

    @staticmethod
    def contraction(left,right,gradient):
        return np.einsum('...jk,...il,...jlk->...i',left,right,gradient,optimize=False)-.5*np.einsum('...jk,...il,...ljk->...i',left,right,gradient,optimize=False)

    def connection(self,metric=None):
        if metric is None:return self.contraction(self.gi,self.gi,self.dg)
        inv=np.linalg.inv(metric)
        return self.contraction(inv,inv,self.gradient(metric,1))

    def rate(self,beta):
        db=self.gradient(beta,-1);div=np.einsum('...ii->...',db,optimize=False)
        return (np.einsum('...k,...kij->...ij',beta,self.advective_dg,optimize=False)
                +np.einsum('...kj,...ik->...ij',self.g,db,optimize=False)
                +np.einsum('...ik,...jk->...ij',self.g,db,optimize=False)
                -(2/3)*self.g*div[...,None,None])

    def variation(self,h):
        inverse_rate=-np.einsum('...ia,...ab,...bj->...ij',self.gi,h,self.gi,optimize=False)
        return (self.contraction(inverse_rate,self.gi,self.dg)
                +self.contraction(self.gi,inverse_rate,self.dg)
                +self.contraction(self.gi,self.gi,self.gradient(h,1)))

    def apply(self,beta):return -self.variation(self.rate(beta))
