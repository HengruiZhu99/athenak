"""Analytic linear TT -> native Z4c target map; no PDE evolution or reduction.

The native curvature/trace/connection are retained. Radial tensor coefficients
and their Taylor series are target-evaluation algebra only.
"""
from functools import lru_cache
from math import factorial
import mpmath as m
from goal1_global_clock_oracle import Clock

pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]

def profile(u,order=0):
    s=(u+m.mpf('.35'))/m.mpf('.15')
    if abs(s)>=1:return m.mpf(0)
    return m.mpf('1e-8')*sum((-1)**k*m.binomial(16,k)*factorial(2*k)/factorial(2*k-order)*s**(2*k-order)for k in range((order+1)//2,17))/m.mpf('.15')**order

class TensorNative:
    def __init__(self,radius,height_cache):
        self.r0=m.mpf(radius);self.clock=Clock(radius,height_cache)

    def reference(self,r):
        o,L,A,b=self.clock.reference(r);G=(L/A)**2;B=-b*A/L;chi=G**(-m.mpf(1)/3)
        return o,L,A,G,B,chi

    @lru_cache(maxsize=24000)
    def _metric(self,r,t,precision):
        o,L,A,G,B,chi=self.reference(r);z=o/r;f=[profile(t+self.clock.uref(r),k)for k in range(5)]
        p=z*(f[4]+z*(2*f[3]+z*(3*f[2]+z*(3*f[1]+3*z*f[0]))))
        s=(p-6*z**3*(f[2]+3*z*f[1]+3*z*z*f[0]))/2
        b=-2*L*o/r**2*(f[3]+3*z*f[2]+6*z*z*f[1]+6*z**3*f[0])
        a=6*L*L*o/r**3*(f[2]+3*z*f[1]+3*z*z*f[0])
        da=-B*B*a/(2*A);dx=-chi*(-p+2*s+a/G)/3
        beta_a=-B*b;beta_c=-B*a/G
        return [p,s,b,a,dx,da,beta_a,beta_c,chi*p,chi*s+dx,chi*b,chi*a+dx*G]

    def metric(self,r,t):return self._metric(r,t,m.mp.prec)

    @lru_cache(maxsize=24000)
    def _coefficients(self,r,t,precision):
        o,L,alpha,G,B,chi=self.reference(r);p,s,b,a,dx,da,ba,bc,*g=self.metric(r,t)
        Gp=m.diff(lambda z:self.reference(z)[3],r);Bp=m.diff(lambda z:self.reference(z)[4],r)
        kr=(B*Gp+2*G*Bp)/(2*alpha);kt=B/(alpha*r);K=kr/G+2*kt
        dr=[m.diff(lambda z,i=i:self.metric(z,t)[i],r)for i in range(4)]
        dt=[m.diff(lambda tau,i=i:self.metric(r,tau)[i],t)for i in range(4)]
        bap=m.diff(lambda z:self.metric(z,t)[6],r);bcp=m.diff(lambda z:self.metric(z,t)[7],r)
        ke=(-dt[0]+B*dr[0]+2*B*p/r+2*ba/r)/(2*alpha)
        kp=(-dt[1]+B*dr[1]+2*B*s/r+2*(bc-ba)/r)/(2*alpha)-da*kt/alpha
        kb=(-dt[2]+B*dr[2]+(Bp+B/r)*b+bap+(2*G*bc-ba)/r)/(2*alpha)
        ka=(-dt[3]+B*dr[3]+2*Bp*a+bc*Gp+2*G*bcp)/(2*alpha)-da*kr/alpha
        dK=-ke+2*kp+ka/G-((-p+2*s)*kt+a*kr/G**2)
        op=m.diff(lambda z:self.reference(z)[0],r);w=-B*op/alpha;dw=-bc*op/alpha-w*da/alpha
        dP=o*dK+3*dw
        ar=[chi*(ke-p*K/3),dx*(kt-K/3)+chi*(kp-s*K/3-dK/3),chi*(kb-b*K/3),dx*(kr-G*K/3)+chi*(ka-a*K/3-G*dK/3)]
        def inverse_coeff(z):
            *_,gz,Bz,chiz=self.reference(z);ge,gp,gb,ga=self.metric(z,t)[8:]
            return [-ge/chiz**2,-gp/chiz**2,-gb/(chiz**2*gz),-ga/(chiz**2*gz**2)]
        ie,ip,ib,ia=inverse_coeff(r);ibp=m.diff(lambda z:inverse_coeff(z)[2],r);iap=m.diff(lambda z:inverse_coeff(z)[3],r)
        lv=-((-3*ie+2*ip+3*ib)/r+ibp);ln=-((ie-2*ip-3*ib+2*ia)/r+iap)
        # scalar chi/alpha/P, beta(vperp,en), Lambda(vperp,en), g4, A4.
        return [dx,da,dP,ba,bc,lv,ln,*g,*ar]

    def coefficients(self,r,t):return self._coefficients(r,t,m.mp.prec)

    @staticmethod
    def values_from_coefficients(x,c,E):
        r=m.sqrt(sum(v*v for v in x));n=m.matrix([v/r for v in x]);P=m.eye(3)-n*n.T;Ep=P*E*P;e=(n.T*E*n)[0];vp=E*n-e*n;cross=vp*n.T+n*vp.T
        dx,da,dP,ba,bc,lv,ln=c[:7]
        def tensor(k):return c[k]*Ep+c[k+1]*e*P+c[k+2]*cross+c[k+3]*e*(n*n.T)
        g=tensor(7);a=tensor(11);beta=ba*vp+bc*e*n;lam=lv*vp+ln*e*n
        return [dx*e]+[g[i,j]for i,j in pairs]+[dP*e]+[a[i,j]for i,j in pairs]+list(lam)+[m.mpf(0),da*e]+list(beta)

    def values(self,x,t,E):
        r=m.sqrt(sum(v*v for v in x));return self.values_from_coefficients(x,self.coefficients(r,t),E)

    def jet_fixture(self,x,t,E):
        """184 consumed native spatial jet entries and22 independent time targets."""
        r=self.r0
        if r==0:
            return [m.mpf(0)]*184,[m.mpf(0)]*22,{k:[m.mpf(0)]*15 for k in ["coefficients","first_radial","second_radial_consumed","time"]}
        c=self.coefficients(r,t)
        # Lambda DD is not consumed by actual finite RHS or its endpoint quotient.
        # A's second spatial derivatives are absent from native184 representation.
        dr=[m.diff(lambda z,i=i:self.coefficients(z,t)[i],r)for i in range(15)]
        dd=[m.diff(lambda z,i=i:self.coefficients(z,t)[i],r,2)if i not in [5,6,11,12,13,14]else m.mpf(0)for i in range(15)]
        dt=[m.diff(lambda tau,i=i:self.coefficients(r,tau)[i],t)for i in range(15)]
        def local(*y):
            z=m.sqrt(sum(v*v for v in y))-r;coeff=[c[i]+z*dr[i]+z*z*dd[i]/2 for i in range(15)]
            return self.values_from_coefficients(y,coeff,E)
        value=local(*x);first=[];second=[]
        for field in range(22):
            first.append([m.diff(lambda *y:local(*y)[field],x,tuple(int(i==d)for i in range(3)))for d in range(3)])
            if field in [*range(0,8),17,18,19,20,21]:
                second.append([[m.diff(lambda *y:local(*y)[field],x,tuple(int(i==d)+int(i==e)for i in range(3)))for e in range(3)]for d in range(3)])
            else:second.append([[m.mpf(0)]*3 for _ in range(3)])
        flat=[]
        def scalar(field):return [value[field],*first[field],*[second[field][d][e]for d,e in pairs]]
        for field in [0,18,7,17]:flat+=scalar(field)
        for i in range(3):flat+=scalar(19+i)+scalar(14+i)
        for k in range(6):flat+=scalar(1+k)+[value[8+k],*first[8+k]]
        time=self.values_from_coefficients(x,dt,E);assert len(flat)==184 and len(time)==22
        return flat,time,{'coefficients':c,'first_radial':dr,'second_radial_consumed':dd,'time':dt}
