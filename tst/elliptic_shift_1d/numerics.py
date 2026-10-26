"""Radial differentiation and quadrature; no Einstein constraint projection."""
import math
import numpy as np


def weights(nodes, target, derivative=0, interval=None):
    """Local polynomial weights in scaled coordinates (avoid Vandermonde scaling)."""
    scale = max(abs(nodes-target))
    x = (nodes-target)/scale
    rhs = np.zeros(len(x))
    if interval is None:
        rhs[derivative] = math.factorial(derivative)/scale**derivative
    else:
        lo, hi = (np.asarray(interval)-target)/scale
        rhs = scale*(hi**(np.arange(len(x))+1)-lo**(np.arange(len(x))+1))/(np.arange(len(x))+1)
    return np.linalg.solve(x[None, :]**np.arange(len(x))[:, None], rhs)


class RadialOperators:
    def __init__(self, r, width=7,quadrature_width=None,puncture=False):
        self.r = np.asarray(r)
        self.width = width
        self.origin_on_grid=bool(r[0]==0)
        self.puncture=puncture
        if puncture and self.origin_on_grid:raise ValueError('A puncture is excluded from the grid')
        if len(r) < width or not np.all(np.diff(r) > 0) or r[0] < 0:
            raise ValueError('Need increasing positive grid and enough points')
        # Reflection only applies to regular-center data. Never use for a puncture.
        self.ext = np.r_[-r[1:width+1][::-1],r] if self.origin_on_grid else np.r_[-r[:width][::-1], r]
        self.index = []
        self.d1 = []; self.d2 = []
        for i, x in enumerate(r):
            k = i+width
            start = min(max(k-width//2, width if puncture else 0), len(self.ext)-width)
            idx = np.arange(start, start+width)
            self.index.append(idx)
            self.d1.append(weights(self.ext[idx], x, 1))
            self.d2.append(weights(self.ext[idx], x, 2))
        self.index = np.array(self.index)
        self.d1 = np.array(self.d1); self.d2 = np.array(self.d2)
        self.aindex=[];self.ad1=[]
        for offset in [-1,1]:
            indices=[];coefficients=[]
            for i,x in enumerate(r):
                start=min(max(i+width-width//2+offset,width if puncture else 0),len(self.ext)-width)
                idx=np.arange(start,start+width);indices.append(idx)
                coefficients.append(weights(self.ext[idx],x,1))
            self.aindex.append(indices);self.ad1.append(coefficients)
        self.aindex=np.asarray(self.aindex);self.ad1=np.asarray(self.ad1)
        # Quadrature is independently refinable with fixed differentiation.
        width=quadrature_width or width
        if len(r)+1<width:raise ValueError("Not enough quadrature nodes")
        # Include the analytically regular origin value in integration.
        self.nodes = r if self.origin_on_grid or puncture else np.r_[0., r]
        self.qindex = []; self.qweights = []
        for i in range(len(self.nodes)-1):
            start = min(max(i-width//2, 0), len(self.nodes)-width)
            idx = np.arange(start, start+width)
            self.qindex.append(idx)
            self.qweights.append(weights(self.nodes[idx], self.nodes[i],
                interval=(self.nodes[i], self.nodes[i+1])))
        self.qindex = np.array(self.qindex); self.qweights = np.array(self.qweights)

    def derivatives(self, values, parity):
        values = np.asarray(values)
        reflected=values[1:self.width+1][::-1] if self.origin_on_grid else values[:self.width][::-1]
        ext = np.concatenate([reflected*parity, values], axis=0)
        sample = ext[self.index]
        if self.puncture:sample=sample-(values[:,None] if values.ndim==1 else values[:,None,:])
        if values.ndim == 1:
            first=np.sum(self.d1*sample,axis=1);second=np.sum(self.d2*sample,axis=1)
            if self.origin_on_grid:
                if parity>0:first[0]=0.
                else:second[0]=0.
            return first,second
        first=np.einsum('ij,ijf->if',self.d1,sample);second=np.einsum('ij,ijf->if',self.d2,sample)
        if self.origin_on_grid:
            first[0,np.asarray(parity)>0]=0.;second[0,np.asarray(parity)<0]=0.
        return first,second

    def integral(self, values, origin=0.):
        """Cumulative degree-(width-1) interpolatory quadrature."""
        f=np.asarray(values) if self.origin_on_grid or self.puncture else np.r_[origin,values]
        result=np.cumsum(np.sum(self.qweights*f[self.qindex],axis=1))
        return np.r_[0.,result] if self.origin_on_grid or self.puncture else result

    def advection_derivative(self,values,parity,beta):
        """One-point off-centered derivative for u_t=beta*u_r.

        beta<0 carries information to larger r and uses the left-biased stencil.
        Other derivatives and the elliptic solve keep their central operators.
        """
        reflected=values[1:self.width+1][::-1] if self.origin_on_grid else values[:self.width][::-1]
        ext=np.concatenate([reflected*parity,values],axis=0)
        direction=np.where(beta<0,0,1);row=np.arange(len(self.r))
        indices=self.aindex[direction,row];coefficients=self.ad1[direction,row]
        sample=ext[indices]
        if self.puncture:sample=sample-values[:,None,:]
        return np.einsum('ij,ijf->if',coefficients,sample)
