"""Cartesian core, cube-to-sphere transition, then concentric spherical shells.

Knots after zero are physical compactified radii, not the old blended map s.
The same discrete-curl geometry, SBP norms and norm-adjoint mortars are used.
No physical source, puncture cell, evolved field or boundary value is changed.
"""
import numpy as np
import mapped_sbp_probe as mapped
from endpoint_sbp import local


class ConcentricGrid(mapped.ShellGrid):
    def __init__(self,n,radial=(0.,.4,.55,.7,1.),half_width=.2,transfer_seed='zero'):
        if radial[1]<=np.sqrt(3)*half_width:raise ValueError('first sphere must enclose the entire Cartesian core')
        old=mapped.lobatto
        try:
            mapped.lobatto=local
            super().__init__(n,radial,half_width,transfer_seed)
        finally:mapped.lobatto=old

    def map(self,patch,layer,q):
        if patch<0:return self.half_width*(2*q-1)
        a=patch//2;b=(a+1)%3;c=(a+2)%3;sgn=1 if patch%2==0 else -1
        u=2*q[...,1]-1;v=2*q[...,2]-1;norm=np.sqrt(1+u*u+v*v)
        if layer==0:d=(1-q[...,0])*self.half_width+q[...,0]*self.radial[1]/norm
        else:d=(self.radial[layer]+np.diff(self.radial)[layer]*q[...,0])/norm
        xyz=np.zeros(q.shape);xyz[...,a]=sgn*d;xyz[...,b]=d*u;xyz[...,c]=d*v
        return xyz
