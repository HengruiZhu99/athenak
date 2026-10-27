"""Constraint-satisfying stationary CMC Schwarzschild trumpet, g=1, K=-3.

This is an already-trumpet control, never wormhole-to-trumpet evidence.
The numerical evolution reference remains hyperboloidal Minkowski.
"""
import math
import numpy as np
from wormhole_data import Jet
from wormhole_isotropic import sinh,log1p,expm1


class CMCTrumpet:
    def __init__(self,mass=.1,quadrature_order=96):
        if mass<=0:raise ValueError('Positive Schwarzschild mass required')
        self.mass=mass
        # The double root gives C=R0^2 sqrt(R0^2+1/3) and
        # M=R0^3+2R0/3-C. Rationalize the latter for large R0.
        lo,hi=1.5*mass,2*mass
        for _ in range(80):
            R=(lo+hi)/2
            found=2*R/3-R*R/(3*(math.sqrt(R*R+1/3)+R))
            if found<mass:lo=R
            else:hi=R
        self.R0=(lo+hi)/2;self.C=self.R0**2*math.sqrt(self.R0**2+1/3)
        self.y0=1/self.R0;self.gamma=math.sqrt(2+9*self.R0**2)
        self.coefficients=np.array([self.R0**2,2*self.R0**3,3*self.C**2/self.R0**2,2*self.C**2/self.R0,self.C**2])
        self.nodes,self.weights=np.polynomial.legendre.leggauss(quadrature_order)

    def P(self,y):
        c=self.coefficients;return c[0]+y*(c[1]+y*(c[2]+y*(c[3]+y*c[4])))

    def regular_integral(self,y):
        s=y[:,None]*(self.nodes+1)/2;c=self.coefficients;z=self.y0
        # (P(y0)-P(s))/(y0-s), avoiding subtraction near the double root.
        divided=c[1]+c[2]*(z+s)+c[3]*(z*z+z*s+s*s)+c[4]*(z**3+z*z*s+z*s*s+s**3)
        root=np.sqrt(self.P(s));integrand=divided/(root*self.gamma*(self.gamma+root))
        return y/2*np.sum(self.weights*integrand,axis=1)

    def inverse(self,r):
        target=-np.log(r);lo=np.zeros_like(r);hi=self.gamma*target
        v=np.minimum(target/self.y0,hi)
        for _ in range(60):
            y=-self.y0*np.expm1(-v);value=v/self.gamma+self.regular_integral(y)-target
            if max(abs(value)/target)<5e-15:break
            lo=np.where(value<0,v,lo);hi=np.where(value>=0,v,hi)
            trial=v-value*np.sqrt(self.P(y))
            v=np.where((trial<=lo)|(trial>=hi)|~np.isfinite(trial),(lo+hi)/2,trial)
        else:raise RuntimeError('Trumpet isotropic coordinate inverse did not converge')
        return -self.y0*np.expm1(-v),np.exp(-v)

    def delta(self,y):
        s=y*(self.nodes+1)/2
        root=(self.y0-s)*np.sqrt(self.P(s));flat=np.sqrt(1+s*s)
        numerator=2*(self.mass+self.C)*s**3-self.C**2*s**6
        return y/2*np.sum(self.weights*numerator/(root*flat*(root+flat)))

    def point(self,radius,y=None,z=None):
        if not 0<radius<=1:raise ValueError('A trumpet puncture is excluded; actual scri is included')
        if radius==1:
            q=np.zeros(8);d=q.copy();dd=q.copy();dd[2]=-4*self.C
            return q,d,dd,dict(R=math.inf,chi_total=1.,physical_lapse=math.inf,beta=-1.)
        if y is None:
            yy,zz=self.inverse(np.array([radius]));y=float(yy[0]);z=float(zz[0])
        root=self.y0*z*math.sqrt(self.P(y))
        c=self.coefficients;Py=c[1]+2*c[2]*y+3*c[3]*y*y+4*c[4]*y**3
        Dy=-2*self.y0*z*self.P(y)+(self.y0*z)**2*Py
        yp=-root/radius;ypp=(root+Dy/2)/(radius*radius)
        yj=Jet(y,yp,ypp);zj=Jet(z,-yp/self.y0,-ypp/self.y0)
        r=Jet(radius,1);O=(1-r*r)/2;A=(1+r*r)/2
        R=1/yj;chi=(r*yj/O)**2
        alpha=O*self.y0*zj*self.P(yj)**.5/yj
        cc=chi-1;u=alpha-A
        if radius>.5:
            delta=self.delta(y)
            flat=math.sqrt(1+y*y);flatp=y/flat;rootp=Dy/(2*root)
            numerator=2*(self.mass+self.C)*y**3-self.C**2*y**6
            numeratorp=6*(self.mass+self.C)*y*y-6*self.C**2*y**5
            dp=numerator/(root*flat*(root+flat))
            dpp=(numeratorp-numerator*(rootp/root+flatp/flat+(rootp+flatp)/(root+flat)))/(root*flat*(root+flat))
            dj=Jet(delta,dp*yp,dpp*yp*yp+dp*ypp)
            t=2*sinh(dj/2)**2+(1+yj*yj)**.5/yj*sinh(dj)
            cc=expm1(-2*log1p(t));chi=1+cc
            diff=-r*r*cc/chi-2*(self.mass+self.C)*O**3*chi**.5/r+self.C**2*O**6*chi**2/r**4
            u=diff/((A*A+diff)**.5+A);alpha=A+u
        Ar=-2*self.C*yj**3/O
        fields=[cc,Jet(0),Ar,Jet(0),Jet(0),Jet(0),u,Jet(0)]
        H=R-self.C/R**2;beta=-H*r/R
        return (np.array([f.v for f in fields]),np.array([f.d for f in fields]),np.array([f.dd for f in fields]),
            dict(R=R.v,chi_total=chi.v,physical_lapse=alpha.v/O.v,beta=beta.v))

    def data(self,r):
        r=np.asarray(r);positive=r<1;y,z=self.inverse(r[positive]);rows=[];j=0
        for x in r:
            if x<1:rows.append(self.point(float(x),float(y[j]),float(z[j])));j+=1
            else:rows.append(self.point(1.))
        return tuple(np.array([row[k] for row in rows]) for k in range(3))+([row[3] for row in rows],)

    def lapse_source(self,r,restoring=1.,form='multiplicative'):
        """Analytic stationary-reference lapse forcing, not an Einstein RHS.

        G=-r*v_ref-beta_ref*(Omega*u_ref'+r*u_ref)+restoring_source.
        No numerical derivative, stage field or numerical shift enters G.
        """
        q,d,_,meta=self.data(r);r=np.asarray(r);O=(1-r*r)/2;A=(1+r*r)/2
        alpha=np.ones_like(r);alpha[:-1]=O[:-1]*np.array([z['physical_lapse'] for z in meta[:-1]])
        if r[-1]!=1:alpha[-1]=O[-1]*meta[-1]['physical_lapse']
        beta=np.array([z['beta'] for z in meta])
        # Stable v_ref=C*r/R^3, including its exact zero at scri.
        radius=np.array([z['R'] for z in meta]);v=self.C*r/radius**3
        restoring_source=2*restoring*alpha*q[:,6] if form=='multiplicative' else restoring*(alpha+A)*q[:,6]
        return -r*v-beta*(O*d[:,6]+r*q[:,6])+restoring_source
