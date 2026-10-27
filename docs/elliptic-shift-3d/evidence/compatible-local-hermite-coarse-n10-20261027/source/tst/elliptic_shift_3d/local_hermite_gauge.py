"""Unadopted local Hermite shift enrichment at the actual outer boundary.

The physical shift trace is carried by a separate Dirichlet lift. Boundary
coefficient slots specify a radial Hermite bubble with zero value at every
metric node. The bubble and its derivative vanish before the last few cells.
Moment-balanced derivatives annihilate low polynomial quadrature moments;
the endpoint-only control deliberately does not. No source or state is reset.

The old nodal norm is not a qualified norm for the enriched physical space.
Off-node angular seams are diagnosed, not silently assumed continuous.
"""
import numpy as np
from compatible_sbp import mm
from enriched_outer_gauge import EnrichedOuterGauge


def slopes(local,moments=2):
    n=len(local['x']);high=np.zeros(n)
    if moments==-1:high[-1]=1.;return high
    if moments not in [0,1,2]:raise ValueError('local Hermite moment budget')
    ids=np.arange(n-moments-2,n);x=local['x'][ids]
    z=(x-x[0])/(x[-1]-x[0]);V=np.column_stack([z**k for k in range(moments+1)])
    A=(local['H'][ids,None]*V).T
    high[ids[:-1]]=np.linalg.solve(A[:,:-1],-A[:,-1]);high[-1]=1.
    return high


def hermite(q,high,points):
    """Piecewise cubic values and q derivatives; every nodal value is zero."""
    points=np.asarray(points);index=np.clip(np.searchsorted(q,points,side='right')-1,0,len(q)-2)
    h=q[index+1]-q[index];s=(points-q[index])/h
    left=high[index];right=high[index+1]
    value=h*((s**3-2*s*s+s)*left+(s**3-s*s)*right)
    derivative=(3*s*s-4*s+1)*left+(3*s*s-2*s)*right
    return value,derivative


class LocalHermiteGauge(EnrichedOuterGauge):
    def __init__(self,gauge,moments=2):
        self.gauge=gauge;self.grid=gauge.grid;self.moments=moments
        self.boundary=np.zeros(self.grid.weights.shape,dtype=bool);self.plans=[]
        for b,axis,side in self.grid.outer:
            if axis!=0 or side!=1:raise ValueError('outer radial shell required')
            self.boundary[b,-1]=True;loc=self.grid.locals[b];high=slopes(loc,moments)
            value,derivative=hermite(loc['x'],high,loc['x'])
            if np.max(abs(value))>1e-14 or np.max(abs(derivative-high))>1e-12:raise ValueError('invalid Hermite endpoint representation')
            gradq=self.grid.cofactors[b][...,0,:]/self.grid.jacobians[b][...,None]
            self.plans.append((b,high,gradq,float(np.max(abs(value)))))

    def seam_diagnostic(self,coefficients):
        """Actual off-node value jumps along outer angular faces, without repair."""
        repeated=np.zeros_like(coefficients)
        outer={b for b,_,_,_ in self.plans}
        amplitudes={}
        for b,high,_,_ in self.plans:
            repeated[b]=coefficients[b,-1][None,...]
            q=self.grid.locals[b]['x'];points=np.linspace(q[0],q[-1],32*(len(q)-1)+1)
            amplitudes[b]=float(np.max(abs(hermite(q,high,points)[0])))
        maximum=0.;coefficient_jump=0.
        for link in self.grid.links:
            L,R=link[:2]
            if L[0] not in outer or R[0] not in outer or L[1]==0 or R[1]==0:continue
            jump=self.grid.trace(repeated,L)-self.grid.transfer(self.grid.trace(repeated,R),link,True)
            j=float(np.max(abs(jump)));coefficient_jump=max(coefficient_jump,j)
            maximum=max(maximum,amplitudes[L[0]]*j)
        return {'outer_seam_coefficient_jump_max':coefficient_jump,
                'sampled_off_node_shift_jump_max':maximum,
                'maximum_bubble_shape':max(amplitudes.values()),
                'off_node_interface_coupling_implemented':False}
