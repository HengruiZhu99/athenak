"""Independent local advanced/retarded native gauge profiles from ADM pullback.

Analysis/oracle only; no new Cauchy variables. Harmonic target-coordinate
maps use Phi=exp(lambda*(T-R))/R and its spatial gradient. Native ADM quantities
are derived from the physical Minkowski metric before conformal/native conversion.
Positive-frequency advanced modes are local finite-boundary controls only; no global scri-admissibility or origin-regularity claim.
"""
import mpmath as mp
from functools import lru_cache
@lru_cache(maxsize=20000)
def components(o,lam,kind,precision=70,direction=-1,anchor=None):
 r=mp.sqrt(1-o);L=1+r*r
 def generators(x):
  O=1-x*x;LL=1+x*x;phase=(1-x)/(2*(1+x))+(direction+1)*x/O;phase0=mp.mpf(0)if anchor is None else (1-mp.sqrt(1-anchor))/(2*(1+mp.sqrt(1-anchor)))+(direction+1)*mp.sqrt(1-anchor)/anchor;F=mp.exp(lam*(phase-phase0));f=O/x*F if kind=='time'else mp.mpf(0);s=mp.mpf(0)if kind=='time'else(direction*lam*O/x-O*O/(x*x))*F
  return -f+2*x/LL*s,-O*O/LL*s
 def spatial(x):
  O=1-x*x;LL=1+x*x;xt,xr=generators(x);dxt=mp.diff(lambda z:generators(z)[0],x);dxr=mp.diff(lambda z:generators(z)[1],x);kR=4*x/O*xr-4*x*dxt+2*dxr;kT=2*LL/(x*O)*xr;tr=xr*(-2-8*x*x/O)-O*O*dxt-2*x*dxr-2*x*lam*xt+lam*xr;tt=-2*O*O*lam*xt-4*x*lam*xr;beta=tr+2*x*kR;alpha=(-tt+4*x*x*kR-4*x*beta)/(2*LL);return kR,kT,alpha,beta
 kR,kT,alpha,beta=spatial(r);kRp=mp.diff(lambda x:spatial(x)[0],r);kTp=mp.diff(lambda x:spatial(x)[1],r);bp=mp.diff(lambda x:spatial(x)[3],r);kr=o/(2*L)*(-lam*kR+4*r/o*beta-2*r*kRp+2*bp)+2*alpha/L;kt=o/(2*L)*(-lam*kT+2*L/(r*o)*beta-2*r*kTp)+2*alpha/L;chi=-(kR+2*kT)/3;h=mp.mpf(2)/3*(kR-kT);hp=mp.diff(lambda x:mp.mpf(2)/3*(spatial(x)[0]-spatial(x)[1]),r)
 return mp.matrix([chi,h,alpha,beta,mp.mpf(2)/3*(kr-kt),kr+2*kt,0,o*(hp+3*h/r)])

def jets(o,lam,kind,direction=-1,anchor=None):
 return [mp.matrix([mp.diff(lambda z:components(z,lam,kind,mp.mp.prec,direction,anchor)[i],o,k)for i in range(8)])for k in range(3)]
