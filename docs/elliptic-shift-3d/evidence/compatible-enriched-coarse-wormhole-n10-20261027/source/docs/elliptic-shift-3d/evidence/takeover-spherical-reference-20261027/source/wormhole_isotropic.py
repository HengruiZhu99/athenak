"""Two-ended Schwarzschild wormhole with a hyperboloidal outer end, g=1.

An isotropic radial map is derived from the embedded slice, not imposed on
ordinary time-symmetric data. Initial data only; no regular-center extension.
"""
import math
import numpy as np
from wormhole_data import Jet


def sinh(z):
    return Jet(math.sinh(z.v),math.cosh(z.v)*z.d,
        math.cosh(z.v)*z.dd+math.sinh(z.v)*z.d*z.d)


def log1p(z):
    return Jet(math.log1p(z.v),z.d/(1+z.v),z.dd/(1+z.v)-z.d*z.d/(1+z.v)**2)


def expm1(z):
    return Jet(math.expm1(z.v),math.exp(z.v)*z.d,math.exp(z.v)*(z.dd+z.d*z.d))


class IsotropicWormhole:
    def __init__(self,mass=.1,quadrature_order=96):
        if mass<=0:raise ValueError('Positive Schwarzschild mass required')
        self.mass=mass;self.R0=4*mass;self.R1=8*mass
        self.nodes,self.weights=np.polynomial.legendre.leggauss(quadrature_order)
        self.y0=1/self.R0;self.y1=1/self.R1
        self.J0=float(self.integral(np.array([self.y0]))[0]);self.r0=math.exp(-self.J0)
        rho0=(self.R0-mass+math.sqrt(self.R0*(self.R0-2*mass)))/2
        self.k=self.r0/rho0

    def switch(self,R):
        R=np.asarray(R);result=np.zeros_like(R);result[R>=self.R1]=1
        use=(R>self.R0)&(R<self.R1);z=(R[use]-self.R0)/(self.R1-self.R0)
        E=1/z-1/(1-z);s=np.zeros_like(z);s[E< -700]=1
        middle=abs(E)<=700;s[middle]=1/(1+np.exp(E[middle]));result[use]=s
        return result

    def switch_derivatives(self,R):
        if R<=self.R0:return 0.,0.,0.,0.
        if R>=self.R1:return 1.,0.,0.,0.
        width=self.R1-self.R0;z=(R-self.R0)/width;E=1/z-1/(1-z)
        if E>700:return 0.,0.,0.,0.
        if E< -700:return 1.,0.,0.,0.
        s=1/(1+math.exp(E));E1=-1/z**2-1/(1-z)**2
        E2=2/z**3-2/(1-z)**3;E3=-6/z**4-6/(1-z)**4
        s1=-s*(1-s)*E1
        s2=s*(1-s)*((1-2*s)*E1**2-E2)
        s3=s*(1-s)*(-(1-6*s+6*s*s)*E1**3+3*(1-2*s)*E1*E2-E3)
        return s,s1/width,s2/width**2,s3/width**3

    def D(self,y):
        y=np.asarray(y);s=np.ones_like(y);use=y>0;s[use]=self.switch(1/y[use])
        return y*y-2*self.mass*y**3+s*s

    def segment(self,lo,hi,delta=False):
        lo=np.asarray(lo);hi=np.asarray(hi);y=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*self.nodes/2
        if delta:
            A=np.sqrt(1+y*y-2*self.mass*y**3);B=np.sqrt(1+y*y)
            integrand=2*self.mass*y**3/(A*B*(A+B))
        else:integrand=1/np.sqrt(self.D(y))
        return (hi-lo)/2*np.sum(self.weights*integrand,axis=1)

    def integral(self,y):
        y=np.asarray(y);split=np.minimum(y,self.y1)
        return self.segment(np.zeros_like(y),split)+self.segment(split,y)

    def inverse(self,r):
        target=-np.log(r);lo=np.zeros_like(r);hi=np.full_like(r,self.y0)
        y=self.y0*target/self.J0
        for _ in range(60):
            E=self.integral(y)-target
            if max(abs(E)/target)<3e-15:break
            lo=np.where(E<0,y,lo);hi=np.where(E>=0,y,hi)
            trial=y-E*np.sqrt(self.D(y))
            y=np.where((trial<lo)|(trial>hi)|~np.isfinite(trial),(lo+hi)/2,trial)
        else:raise RuntimeError('Isotropic coordinate quadrature/root tolerance not achieved')
        return y

    def point(self,radius,y=None):
        if not 0<radius<=1:raise ValueError('Puncture is excluded; actual scri is included')
        if radius==1:
            return np.zeros(8),np.zeros(8),np.zeros(8),dict(branch='hyperboloidal',chi_total=1.,R=math.inf,physical_lapse=math.inf)
        r=Jet(radius,1);O=(1-r*r)/2;A=(1+r*r)/2;M=self.mass
        if radius<=self.r0:
            rho=r/self.k;psi=1+M/(2*rho);R=rho*psi*psi
            chi=self.k*self.k/(O*O*psi**4);c=chi-1
            alpha=O/psi**2;u=alpha-A;K=Jet(0);Ar=Jet(0);branch='isotropic'
        else:
            if y is None:y=float(self.inverse(np.array([radius]))[0])
            Rvalue=1/y;s,sR,sRR,sRRR=self.switch_derivatives(Rvalue)
            D=y*y-2*M*y**3+s*s;Dy=2*y-6*M*y*y-2*s*sR/(y*y)
            yp=-math.sqrt(D)/radius;ypp=(math.sqrt(D)+Dy/2)/(radius*radius)
            yjet=Jet(y,yp,ypp);R=1/yjet
            if s==1:
                delta=float(self.segment(np.array([0.]),np.array([y]),delta=True)[0])
                aa=math.sqrt(1+y*y-2*M*y**3);bb=math.sqrt(1+y*y)
                ap=(y-3*M*y*y)/aa;bp=y/bb
                delta1=2*M*y**3/(aa*bb*(aa+bb))
                delta2=delta1*(3/y-ap/aa-bp/bb-(ap+bp)/(aa+bb))
                dj=Jet(delta,delta1*yp,delta2*yp*yp+delta1*ypp)
                t=2*sinh(dj/2)**2+(1+yjet*yjet)**.5/yjet*sinh(dj)
                c=expm1(-2*log1p(t));chi=1+c
                difference=-r*r*c/chi-2*M*O**3*chi**.5/r
                u=difference/((A*A+difference)**.5+A);alpha=A+u
                K=Jet(-3);Ar=Jet(0);branch='hyperboloidal'
            else:
                chi=(r/(O*R))**2;c=chi-1
                Kvalue=-3*s-Rvalue*sR;KR=-4*sR-Rvalue*sRR;KRR=-5*sRR-Rvalue*sRRR
                K=Jet(Kvalue,KR*R.d,KRR*R.d*R.d+KR*R.dd)
                Av=-2*Rvalue*sR/3;AR=-2*(sR+Rvalue*sRR)/3;ARR=-2*(2*sRR+Rvalue*sRRR)/3
                Ar=Jet(Av,AR*R.d,ARR*R.d*R.d+AR*R.dd)/O
                sj=Jet(s,sR*R.d,sRR*R.d*R.d+sR*R.dd)
                rho=(R-M+(R*(R-2*M))**.5)/2;psi=1+M/(2*rho)
                F=1-2*M/R;outer_lapse=O*(F+R*R)**.5
                alpha=(1-sj)*O/psi**2+sj*outer_lapse;u=alpha-A;branch='transition'
        variables=[c,Jet(0),Ar,K+3,Jet(0),Jet(0),u,Jet(0)]
        return np.array([z.v for z in variables]),np.array([z.d for z in variables]),np.array([z.dd for z in variables]),dict(
            branch=branch,chi_total=chi.v,R=R.v,physical_lapse=alpha.v/O.v,
            physical_radial_metric=(R.v/radius)**2,K_phys=K.v)

    def data(self,r):
        r=np.asarray(r);ys=np.zeros_like(r);use=(r>self.r0)&(r<1)
        ys[use]=self.inverse(r[use]);rows=[self.point(float(x),float(y) if mask else None) for x,y,mask in zip(r,ys,use)]
        return np.array([z[0] for z in rows]),np.array([z[1] for z in rows]),np.array([z[2] for z in rows]),[z[3] for z in rows]
