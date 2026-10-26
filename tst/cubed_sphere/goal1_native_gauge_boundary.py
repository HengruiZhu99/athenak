"""Physical inward-absorbing gauge boundary expressed on existing native fields.

Only finite inner boundary equations change. Their time derivatives are evaluated
through the unchanged native RWM/curvature RHS, yielding at most second spatial
order. No new wave momentum, gradient, coordinate potential or memory is evolved.
"""
import mpmath as mp
from functools import lru_cache
import goal1_native_constraint_finite as finite

def geometry(o):
 r=mp.sqrt(1-o);L=2-o;Kx=mp.matrix([[-2*r*r/o**2,2*r*r/o**2,-L/o**2,-2*r/o**2]]);Ks=mp.matrix([[r/(2*o),r/(4*o),0,0]]);return (L+2*r)/L,2*r*o*o/L,o/r,Kx,Ks

@lru_cache(None)
def operators(o,precision=100):
 with mp.workdps(precision):
  o=mp.mpf(o);F=finite.matrices(o)[0];Fp=[mp.diff(lambda z:finite.matrices(z)[0][j],o)for j in range(3)];A0,A1,A2=[f[:4,:]for f in F];assert mp.norm(A2,p=mp.inf)<mp.mpf('1e-70');A0p,A1p=[Fp[j][:4,:]for j in range(2)];Qtt=[A0*F[0]+A1*Fp[0],A0*F[1]+A1*(F[0]+Fp[1]),A0*F[2]+A1*(F[1]+Fp[2])];assert mp.norm(A1*F[2],p=mp.inf)<mp.mpf('1e-70')
  a,b,rho,X,S=geometry(o);ap=mp.diff(lambda z:geometry(z)[0],o);bp=mp.diff(lambda z:geometry(z)[1],o);Xp=mp.diff(lambda z:geometry(z)[3],o);Sp=mp.diff(lambda z:geometry(z)[4],o);Spp=mp.diff(lambda z:geometry(z)[4],o,2);select=mp.matrix(4,8)
  for k in range(4):select[k,k]=1
  Btime=[a*X*A0+(b*Xp-rho*X)*select,a*X*A1+b*X*select,mp.matrix(1,8)]
  st=[S*A0,S*A1,mp.matrix(1,8)];stt=[S*A for A in Qtt];stO=[Sp*A0+S*A0p,Sp*A1+S*(A0+A1p),S*A1];sO=[Sp*select,S*select,mp.matrix(1,8)];sOO=[Spp*select,2*Sp*select,S*select];ss=[S*select,mp.matrix(1,8),mp.matrix(1,8)];Bspace=[a*a*stt[j]+2*a*b*stO[j]+b*b*sOO[j]+b*ap*st[j]+b*bp*sO[j]-4*rho*(a*st[j]+b*sO[j])+2*rho*rho*ss[j]for j in range(3)]
  return Btime,Bspace
