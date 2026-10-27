"""Constraint-satisfying Schwarzschild wormhole with a hyperboloidal outer end.

Construct an embedded slice in exact Schwarzschild, not a compactification of a
wholly time-symmetric slice. The interior has physical lapse psi**(-2).
This module provides initial data only. Its puncture end is not a regular center.
"""
import math
import numpy as np


class Jet:
    def __init__(self,v,d=0.,dd=0.):self.v=float(v);self.d=float(d);self.dd=float(dd)
    @staticmethod
    def of(x):return x if isinstance(x,Jet) else Jet(x)
    def __add__(self,x):
        x=self.of(x);return Jet(self.v+x.v,self.d+x.d,self.dd+x.dd)
    __radd__=__add__
    def __neg__(self):return Jet(-self.v,-self.d,-self.dd)
    def __sub__(self,x):return self+-self.of(x)
    def __rsub__(self,x):return self.of(x)+-self
    def __mul__(self,x):
        x=self.of(x);return Jet(self.v*x.v,self.d*x.v+self.v*x.d,self.dd*x.v+2*self.d*x.d+self.v*x.dd)
    __rmul__=__mul__
    def __pow__(self,p):
        v=self.v**p;f=p*self.v**(p-1);ff=p*(p-1)*self.v**(p-2)
        return Jet(v,f*self.d,ff*self.d**2+f*self.dd)
    def __truediv__(self,x):return self*self.of(x)**-1
    def __rtruediv__(self,x):return self.of(x)*self**-1
    def exp(self):
        v=math.exp(self.v);return Jet(v,v*self.d,v*(self.dd+self.d**2))


def transition(x):
    if x.v<=0:return Jet(0)
    if x.v>=1:return Jet(1)
    # Logistic of reciprocal distances: infinitely flat at both ends. Avoid
    # overflow when a point lies extremely close to a transition endpoint.
    exponent=1/x-1/(1-x)
    if exponent.v>700:return Jet(0)
    if exponent.v< -700:return Jet(1)
    return 1/(1+exponent.exp())


def point(radius,mass=.1,curvature_radius=1.,R0=None,R1=None):
    if mass<=0 or curvature_radius!=1. or not 0<radius<=1:
        raise ValueError('Currently requires M>0, a=S=1, 0<r<=1')
    a=curvature_radius;R0=4*mass if R0 is None else R0;R1=8*mass if R1 is None else R1
    if not 2*mass<R0<R1:raise ValueError('Transition must lie strictly outside the horizon')
    r=Jet(radius,1);omega=(1-r*r)/(2*a);A=(1+r*r)/(2*a)
    # Y=Omega R, W=Omega^2 dR/dr are rational and regular at actual scri.
    Y=(1+r)/2*(r+(mass/a)*(1-r)+(mass*mass/(4*a*a))*(1-r)**2/r)
    W=(1+r)**2/(4*a)*(1-(mass*mass/(4*a*a))*(1-r)**2/(r*r))
    F=1-2*mass*omega/Y
    if radius==1:
        s=Jet(1);branch='hyperboloidal'
    else:
        rho=a*r/(1-r);R=rho+mass+mass*mass/(4*rho)
        # Never transition the second asymptotically flat end, where R also grows.
        exterior=rho.v>mass/2
        s=transition((R-R0)/(R1-R0)) if exterior else Jet(0)
        branch='hyperboloidal' if s.v==1 else ('isotropic' if s.v==0 else 'transition')
    if branch=='hyperboloidal':
        bar_rr=W*W/(omega*omega*F+Y*Y/(a*a))
        K=Jet(-3/a);Kr=Jet(-1/a)
        alpha=A
    elif branch=='isotropic':
        psi=1+mass/(2*rho)
        bar_rr=(1+r)**2/(4*(1-r)**2)*psi**4
        K=Jet(0);Kr=Jet(0);alpha=omega/psi**2
    else:
        # h_R = P/F, P=s (R/a)/sqrt(F+(R/a)^2). Only evaluated outside horizon.
        H=R/a;P=s*H/(F+H*H)**.5
        bar_rr=W*W/(omega*omega*F)*(1-P*P)
        normal_R=P*(F/(1-P*P))**.5
        kr=-normal_R.d/R.d
        dkr=-(normal_R.dd*R.d-normal_R.d*R.dd)/(R.d*R.d)
        kt=-normal_R/R
        Kr=Jet(kr,dkr);K=Kr+2*kt
        psi=1+mass/(2*rho)
        alpha=(1-s)*omega/psi**2+s*A
    bar_tt=Y*Y/(r*r)
    chi=(bar_rr*bar_tt*bar_tt)**(-1/3)
    g=chi*bar_rr
    Arr=Jet(0) if branch in ('isotropic','hyperboloidal') else g/omega*(Kr-K/3)
    difference=math.expm1(1.5*math.log(g.v))/g.v
    V=g.d/g.v**2+2*difference/radius
    dV=(g.dd/g.v**2-2*g.d*g.d/g.v**3
        +2*(.5/g.v**.5+1/g.v**2)*g.d/radius-2*difference/radius**2)
    values=[chi-1,g-1,Arr,K+3,Jet(0),Jet(V,dV),alpha-A,Jet(0)]
    q=np.array([x.v for x in values]);d=np.array([x.d for x in values]);dd=np.array([x.dd for x in values])
    return q,d,dd,{'branch':branch,'R':Y.v/omega.v if omega.v else float('inf'),
        'physical_lapse':alpha.v/omega.v if omega.v else float('inf'),
        'physical_radial_metric':bar_rr.v/omega.v**2 if omega.v else float('inf'),
        'K_phys':K.v,'chi_total':chi.v}


def data(r,**kwargs):
    rows=[point(float(x),**kwargs) for x in r]
    return np.array([x[0] for x in rows]),np.array([x[1] for x in rows]),np.array([x[2] for x in rows]),[x[3] for x in rows]
