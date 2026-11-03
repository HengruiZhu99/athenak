"""Independent exact-Schwarzschild embedding prototype; not an evolution initializer."""
import sys,json
from pathlib import Path
import numpy as np
RULES=[np.polynomial.legendre.leggauss(n) for n in [16,32]]
def quad(f,lo,hi,epsabs=2e-13,epsrel=2e-13,depth=0):
 if lo==hi:return 0.,0.
 vals=[(hi-lo)/2*sum(wi*f((lo+hi)/2+(hi-lo)/2*xi) for xi,wi in zip(x,w)) for x,w in RULES]
 err=abs(vals[0]-vals[1])
 if err<epsabs+epsrel*abs(vals[1]):return vals[1],err
 assert depth<20
 mid=(lo+hi)/2
 a=quad(f,lo,mid,epsabs/2,epsrel,depth+1);b=quad(f,mid,hi,epsabs/2,epsrel,depth+1)
 return a[0]+b[0],a[1]+b[1]
def brentq(f,lo,hi,xtol=1e-13,rtol=1e-14):
 assert f(lo)<0<f(hi)
 for _ in range(100):
  m=(lo+hi)/2;v=f(m)
  if hi-lo<xtol+rtol*abs(m):return m
  if v>0:hi=m
  else:lo=m
 raise ValueError('root did not converge')
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole
class BoostedEmbedding:
 def __init__(self,v=.1):
  self.base=IsotropicWormhole(1/3,128,4,12,'poly6');self.m=self.base.mass;self.v=v;self.G=1/np.sqrt(1-v*v);self.rho0=self.base.r0/self.base.k
  self.C=-self.rstar(self.base.R0)-self.tail(self.base.R0)
 def rstar(self,R):return R+2*self.m*np.log(R/(2*self.m)-1)
 def tail(self,R):
  # Integral_R^infty 1/[A(A+R S)] dR, split at the slicing plateau.
  def f(y):
   if y==0:return .5
   S=float(self.base.switch(np.array([1/y]))[0]);A=np.sqrt(y*y-2*self.m*y**3+S*S)
   return 1/(A*(A+S))
  y=1/R;split=min(y,1/self.base.R1)
  return quad(f,0,split,epsabs=2e-13,epsrel=2e-13)[0]+quad(f,split,y,epsabs=2e-13,epsrel=2e-13)[0]
 @staticmethod
 def poly(z):
  if z<=0:return 0.,0.,0.
  if z>=1:return 1.,0.,0.
  a=min(z,1-z);p=a**7*(1716+a*(-9009+a*(20020+a*(-24024+a*(16380+a*(-6006+924*a))))))
  return (p if z<=.5 else 1-p),12012*(z*(1-z))**6,72072*(z*(1-z))**5*(1-2*z)
 def radial(self,rho):
  m=self.m;a=m/(2*rho);psi=1+a;al=(1-a)/psi;b=psi*psi;R=rho*psi**2;Rp=1-a*a;Rpp=2*a*a/rho
  width=.75*self.rho0;P,Pp,Ppp=self.poly((rho-.25*self.rho0)/width);B=1-P;Bp=-Pp/width;Bpp=-Ppp/width**2
  if rho<=self.rho0:return dict(a=al,b=b,ap=m/(rho*rho*psi*psi),bp=-m*psi/(rho*rho),h1=0.,h2=0.,u0=0.,u1=0.,u2=0.,B=B,Bp=Bp,Bpp=Bpp,Q0=1.,S=0.,sigma=rho,sigma1=1.,sigma2=0.)
  S,SR,SRR,_=self.base.switch_derivatives(R);Sp=SR*Rp;Spp=SRR*Rp*Rp+SR*Rpp
  f=al*al;fr=2*m/R**2;A=np.sqrt(f+R*R*S*S);AR=(fr+2*R*S*S+2*R*R*S*SR)/(2*A)
  hR=R*S/(f*A);hRR=(S+R*SR)/(f*A)-R*S*(fr*A+f*AR)/(f*A)**2
  h1=hR*Rp;h2=hRR*Rp*Rp+hR*Rpp
  tail=self.tail(R);g=1/(A*(A+R*S));gR=-g*(AR/A+(AR+S+R*SR)/(A+R*S))
  C=self.rstar(R)+self.C;Cp=Rp/f;Cpp=Rpp/f-fr*Rp*Rp/f**2
  sig=rho+S*(R-rho);sig1=1+Sp*(R-rho)+S*(Rp-1);sig2=Spp*(R-rho)+2*Sp*(Rp-1)+S*Rpp
  u0=tail+(1-S)*C+S*sig
  u1=-g*Rp+(1-S)*Cp+S*sig1+Sp*(sig-C)
  u2=-gR*Rp*Rp-g*Rpp+(1-S)*Cpp+2*Sp*(sig1-Cp)+Spp*(sig-C)+S*sig2
  return dict(a=al,b=b,ap=m/(rho*rho*psi*psi),bp=-m*psi/(rho*rho),h1=h1,h2=h2,u0=u0,u1=u1,u2=u2,B=B,Bp=Bp,Bpp=Bpp,Q0=f/(A*A),S=S,sigma=sig,sigma1=sig1,sigma2=sig2)
 def geometry(self,x):
  x=np.asarray(x,float);r=np.linalg.norm(x);n=x/r
  if r<=self.base.r0:rho=r/self.base.k;rp=1/self.base.k;rpp=0.
  else:
   y=float(self.base.inverse(np.array([r]))[0]);R=1/y;rho=(R-self.m+np.sqrt(R*(R-2*self.m)))/2
   a=self.m/(2*rho);S,SR,_,_=self.base.switch_derivatives(R);A=np.sqrt(1-2*self.m/R+R*R*S*S);dR=R*A/r;rp=dR/(1-a*a)
   AR=(2*self.m/R**2+2*R*S*S+2*R*R*S*SR)/(2*A);ddR=dR*(A+R*AR-1)/r
   rpp=(ddR-2*a*a/rho*rp*rp)/(1-a*a)
  lab=self.radial(rho);X=lab['sigma']*n;Y=np.array([X[0]/self.G,X[1],X[2]])
  I=np.eye(3);ex=I[0]
  for iteration in range(40):
   rr=np.linalg.norm(Y);nn=Y/rr;q=self.radial(rr);Z=q['sigma']*nn
   U0=q['u0']-self.v*Y[0]*q['B'];Ug=q['u1']*nn-self.v*(ex*q['B']+Y[0]*q['Bp']*nn)
   J=q['sigma']/rr*I+(q['sigma1']-q['sigma']/rr)*np.outer(nn,nn)
   J[0]=self.G*(J[0]+self.v*Ug)
   target=Z.copy();target[0]=self.G*(Z[0]+self.v*U0)
   residual=target-X
   if np.max(abs(residual))<2e-13*max(1.,np.linalg.norm(X)):break
   Y-=np.linalg.solve(J,residual)
  else:raise ValueError('boosted map inversion failed')
  rr=np.linalg.norm(Y);nn=Y/rr;q=self.radial(rr);Yx=Y[0]
  w=-self.v*(ex*q['B']+Yx*q['Bp']*nn);F=q['h1']*nn+w
  F2=q['h2']*np.outer(nn,nn)+q['h1']/rr*(I-np.outer(nn,nn))-self.v*(q['Bp']*(np.outer(ex,nn)+np.outer(nn,ex))+Yx*(q['Bpp']*np.outer(nn,nn)+q['Bp']/rr*(I-np.outer(nn,nn))))
  U=q['u1']*nn+w;J=q['sigma']/rr*I+(q['sigma1']-q['sigma']/rr)*np.outer(nn,nn);J[0]=self.G*(J[0]+self.v*U);inv=np.linalg.inv(J)
  P=lab['sigma']/r*I+(lab['sigma1']*rp-lab['sigma']/r)*np.outer(n,n);T=inv@P
  normal=nn@T;tangent=T-np.outer(nn,normal)
  gamma=q['b']**2*(tangent.T@tangent+q['Q0']*np.outer(normal,normal))-q['a']**2*(q['h1']*(np.outer(normal,w@T)+np.outer(w@T,normal))+np.outer(w@T,w@T))
  Q=q['Q0']-q['a']**2/q['b']**2*(2*q['h1']*np.dot(nn,w)+np.dot(w,w))
  assert Q>0,(r,rr,Q)
  # Signed static lapse gives the smooth wormhole extrinsic curvature through the throat.
  ap=q['ap']*nn;bp=q['bp']*nn
  GammF=(np.outer(F,bp)+np.outer(bp,F)-I*np.dot(bp,F))/q['b']
  KY=-(q['a']*F2+np.outer(ap,F)+np.outer(F,ap)-q['a']*GammF-q['a']**2*np.dot(ap,F)/q['b']**2*np.outer(F,F))/np.sqrt(Q)
  K=T.T@KY@T if q['S']<1 else -gamma
  om=(1-r*r)/2;bar=om*om*gamma;chi=np.linalg.det(bar)**(-1/3);gt=chi*bar;trace=np.trace(np.linalg.solve(gamma,K));At=chi*om*(K-gamma*trace/3)
  return dict(chi=chi,g=gt,A=At,K=trace,Q=Q,detJ=np.linalg.det(J),rho=rr,metric=gamma,curvature=K)
if __name__=='__main__':
 b=BoostedEmbedding();out={'v':b.v,'scale':b.base.k,'kinematic_exit_t_over_M':3*.18/(b.base.k*b.v),'core_inner_support_t_over_M':3*.16/(b.base.k*b.v),'points':[]}
 for r in [.001,.03,.05,.1,.2,.3,.5,.8,.95,.99,.999,.9999]:
  for n in [np.array([1.,0,0]),np.array([-1.,0,0]),np.array([.3,.4,np.sqrt(.75)])]:
   d=b.geometry(r*n);out['points'].append(dict(r=r,direction=n.tolist(),chi=d['chi'],metric_deviation=float(np.max(abs(d['g']-np.eye(3)))),trace=d['K'],Amax=float(np.max(abs(d['A']))),Q=d['Q'],detJ=d['detJ']))
 print(json.dumps(out,indent=2))
