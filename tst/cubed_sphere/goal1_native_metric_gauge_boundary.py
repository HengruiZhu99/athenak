"""First-order physical metric gauge conditions at finite inner boundary.

Flat outgoing-to-inner harmonic-coordinate controls motivate the conditions.
Off-shell/mass/nonlinear extension is a tested candidate, not an assumed identity.
All observables are existing native fields and the unchanged RWM RHS.
"""
import mpmath as mp
from functools import lru_cache
import goal1_native_constraint_finite as finite

def geometry(o):
 r=mp.sqrt(1-o);L=2-o;X=mp.matrix([[-2*r*r/o**2,2*r*r/o**2,-L/o**2,-2*r/o**2]]);U=mp.matrix([[L*L/(2*o*o),-L*L/(2*o*o),4*r*r/(L*o*o),2*r/(o*o)]]);S=mp.matrix([[r/(2*o),r/(4*o),0,0]]);return (L+2*r)/L,2*r*o*o/L,o/r,X,U,S
@lru_cache(None)
def operators(o,precision=100):
 with mp.workdps(precision):
  o=mp.mpf(o);F=finite.matrices(o)[0];A0,A1,A2=[f[:4,:]for f in F];assert mp.norm(A2,p=mp.inf)<mp.mpf('1e-70');a,b,rho,X,U,S=geometry(o);Xp=mp.diff(lambda z:geometry(z)[3],o);Up=mp.diff(lambda z:geometry(z)[4],o);select=mp.matrix(4,8)
  for k in range(4):select[k,k]=1
  time=[a*X*A0+(b*Xp-rho*X)*select,a*X*A1+b*X*select,mp.matrix(1,8)];space=[(a*U+2*rho*S)*A0+(b*Up-3*rho*U)*select,(a*U+2*rho*S)*A1+b*U*select,mp.matrix(1,8)];return time,space
