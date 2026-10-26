"""Independent global radial harmonic-clock Minkowski oracle (native variables).

No native RHS or spatial-reduction code is imported. The implicit clock and all
jets here are analytic target data, never new evolved state variables.
"""
from functools import lru_cache
import math
import mpmath as m

class Clock:
    def __init__(self, radius, height_cache, sigma='.2'):
        self.r0=m.mpf(radius);self.sigma=m.mpf(sigma)
        self.constant=m.mpf(height_cache['outer_constant'])
        self.height_error=m.mpf(height_cache['declared_height_uncertainty'])
        if self.r0>=m.mpf('.95'):
            self.ucoeff=None
        else:
            saved={m.mpf(k):m.mpf(v) for k,v in height_cache['uref_values'].items()}
            u0=-self.r0 if self.r0<=m.mpf('.05') else saved[self.r0]
            self.ucoeff=[u0]+[m.diff(self.uref_prime,self.r0,k-1) for k in range(1,6)]

    def reference(self,r):
        if self.r0>=m.mpf('.95') or r>=m.mpf('.95'):
            o=1-r*r;b=2*r;L=1+r*r
        elif r<=m.mpf('.05'):
            o=m.mpf(1);b=m.mpf(0);L=m.mpf(1)
        else:
            def omega(z):
                s=(z-m.mpf('.05'))/m.mpf('.9');w=1/(1+m.exp(1/s-1/(1-s)))
                return 1-w*z*z
            def cutoff(z):
                s=(z-m.mpf('.05'))/m.mpf('.9');return 1/(1+m.exp(1/s-1/(1-s)))
            o=omega(r);b=2*r*cutoff(r);L=o-r*m.diff(omega,r)
        A=m.sqrt(o*o+b*b)
        return o,L,A,b

    def uref_prime(self,r):
        o,L,A,b=self.reference(r)
        return -L/(A*(A+b))

    def uref(self,r):
        if self.ucoeff is None:return self.constant+(1-r)/(2*(1+r))
        z=r-self.r0
        return sum(c*z**k/math.factorial(k) for k,c in enumerate(self.ucoeff))

    def F(self,s,n=0):
        x=s/self.sigma;h0=m.mpf(1)
        if n==0:return m.exp(-x*x)
        h1=2*x
        for k in range(1,n):h0,h1=h1,2*x*h1-2*k*h0
        return (-1)**n*h1*m.exp(-x*x)/self.sigma**n

    def wave(self,T,R):
        """H,H_T,H_R; analytic even series supplies origin jets, no radial floor."""
        if abs(R)<m.mpf('1e-15'):
            H=sum(-2*self.F(T,2*k+1)*R**(2*k)/math.factorial(2*k+1) for k in range(6))
            HT=sum(-2*self.F(T,2*k+2)*R**(2*k)/math.factorial(2*k+1) for k in range(6))
            HR=sum(-4*k*self.F(T,2*k+1)*R**(2*k-1)/math.factorial(2*k+1) for k in range(1,6))
            return H,HT,HR
        fu,fv=self.F(T-R),self.F(T+R);fpu,fpv=self.F(T-R,1),self.F(T+R,1)
        return (fu-fv)/R,(fpu-fpv)/R,-(fpu+fpv)/R-(fu-fv)/(R*R)

    @lru_cache(maxsize=12000)
    def _core(self,r,t,e,precision):
        o,L,A,b=self.reference(r);u0=t+self.uref(r)
        if self.ucoeff is None:
            z=m.mpf(0)
            for _ in range(8):
                U=u0+o*z
                if o==0:fv=fpv=m.mpf(0)
                else:fv=self.F(U+2*r/o);fpv=self.F(U+2*r/o,1)
                residual=z+e/r*(self.F(U)-fv)
                z-=residual/(1+e*o/r*(self.F(U,1)-fpv))
            U=u0+o*z
            if o==0:fv=fpv=advanced_div=m.mpf(0)
            else:
                fv=self.F(U+2*r/o);fpv=self.F(U+2*r/o,1);advanced_div=fpv/(o*r)
            I=1+e*o/r*(self.F(U,1)-fpv)
            D=(1/(A*(A+b))-e*((self.F(U)-fv)/(r*r)+2*advanced_div))/I
            G=L*L*D*(2-o*o*D);v=1-o*o*D
            return {'o':o,'L':L,'Aref':A,'bref':b,'G':G,'q':1/I,'v':v,'D':D,'U':U,'shift_T':o*z}
        R=r/o;T0=u0+R;delta=m.mpf(0)
        for _ in range(8):
            H,HT,HR=self.wave(T0+delta,R)
            delta-=(delta+e*H)/(1+e*HT)
        T=T0+delta;H,HT,HR=self.wave(T,R);I=1+e*HT;v=(b/A-e*HR)/I
        G=(L/o)**2*(1-v)*(1+v)
        return {'o':o,'L':L,'Aref':A,'bref':b,'G':G,'q':1/I,'v':v,'T':T,'shift_T':delta}

    def core(self,r,t,e):return self._core(r,t,e,m.mp.prec)

    def radial(self,r,t,e):
        c=self.core(r,t,e);o,L,G,q,v=[c[k] for k in ['o','L','G','q','v']]
        Gr=m.diff(lambda z:self.core(z,t,e)['G'],r)
        chi=G**(-m.mpf(1)/3);alpha=q*L/m.sqrt(G);beta=-q*v*L/G
        if self.ucoeff is None:
            D=c['D'];Dr=m.diff(lambda z:self.core(z,t,e)['D'],r);op=-2*r
            kr=L*L*(2*op*D+o*Dr)/G**m.mpf('1.5');kt=-v*L/(r*m.sqrt(G))
            shear=L*L*(Dr+2*D/r-3*L*o*D*D/r+L*o**3*D**3/r)/G**m.mpf('1.5')
        elif r==0:
            kr=kt=-m.diff(lambda z:self.core(z,t,e)['v'],r);shear=m.mpf(0)
        else:
            vr=m.diff(lambda z:self.core(z,t,e)['v'],r);kr=-o*o*vr/(L*(1-v*v)**m.mpf('1.5'));kt=-v*o/(r*m.sqrt(1-v*v));shear=(kr-kt)/o
        connection=0 if r==0 else m.mpf(2)/3*G**(-m.mpf(5)/3)*Gr+2*G**(-m.mpf(2)/3)*(G-1)/r
        return [chi,alpha,kr+2*kt,0,beta,connection,chi*G,chi,chi*shear*2*G/3,-chi*shear/3]

    def components(self,r,t,e):
        """chi,alpha,P,Theta,beta_r,Lambda_r,g_rr,g_tt,A_rr,A_tt."""
        return self.radial(r,t,e)

    def physical_T(self,r,t,e):
        c=self.core(r,t,e)
        return t+self.uref(r)+r/c['o']+c['shift_T']

    def values(self,x,t,e):
        r=m.sqrt(sum(v*v for v in x));f=self.radial(r,t,e);chi,alpha,P,theta,beta,lam,gr,gt,ar,at=f
        if r==0:n=[m.mpf(0)]*3
        else:n=[v/r for v in x]
        g=[[gt*(i==j)+(gr-gt)*n[i]*n[j] for j in range(3)] for i in range(3)]
        a=[[at*(i==j)+(ar-at)*n[i]*n[j] for j in range(3)] for i in range(3)]
        pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
        return [chi]+[g[i][j]for i,j in pairs]+[P]+[a[i][j]for i,j in pairs]+[lam*z for z in n]+[theta,alpha]+[beta*z for z in n]
