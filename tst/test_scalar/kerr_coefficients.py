#!/usr/bin/env python3
"""Independent physical covariant Kerr trumpet -> normal scalar coefficients."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi,eval_jacobi
import mpmath as mp
import radial_coefficients as radial
mp.mp.dps=60

def components(rho,z,a):
 o,op=radial.compact(rho);lam=o-rho*op;c=mp.sqrt(1-a*a);Rb=rho+o;sigbar=Rb*Rb+a*a*o*o*z*z;sin2=1-z*z
 if abs(o)<mp.mpf('1e-45'):
  G=mp.matrix(4);G[0,0]=-(100+4*(1+c)-a*a*sin2)/(24*24);G[0,1]=G[1,0]=-1;G[2,2]=1/sigbar;G[3,3]=1/(sigbar*sin2)
 else:
  r=rho/o;R=r+1;sig=R*R+a*a*z*z
  # Eqs.12--18, R0=M. The chart includes both BL time and azimuth reslicing.
  g=mp.matrix(4);g[0,0]=-(sig-2*R)/sig;g[0,1]=g[1,0]=c/r;g[0,3]=g[3,0]=-2*a*R*sin2/sig
  g[1,1]=sig/(r*r);g[1,3]=g[3,1]=-a*c*sin2/r;g[2,2]=sig;g[3,3]=(R*R+a*a+2*a*a*R*sin2/sig)*sin2
  hp=radial.smooth((r-8)/8)*(r/mp.sqrt(r*r+100)+(2+c)/r)
  J=mp.eye(4);J[0,1]=hp*lam/o**2;J[1,1]=lam/o**2;G=(o*o*J.T*g*J)**-1
 alpha=1/mp.sqrt(-G[0,0]);br=-G[0,1]/G[0,0];bp=-G[0,3]/G[0,0]
 grr=G[1,1]+br*br/alpha**2;grp=G[1,3]+br*bp/alpha**2;gppcorr=G[3,3]+bp*bp/alpha**2-1/(sigbar*sin2)
 volume=lam*sigbar/alpha
 return alpha,br,bp,grr,grp,gppcorr,volume,sigbar

def derivative(values,rho):
 step=mp.mpf('1e-5')
 if abs(rho-24)<mp.mpf('1e-40'):
  samples=[values(rho-k*step) for k in range(5)];weights=[mp.mpf(25)/12,-4,3,-mp.mpf(4)/3,mp.mpf(1)/4]
 else:
  samples=[values(rho+k*step) for k in [-2,-1,1,2]];weights=[mp.mpf(1)/12,-mp.mpf(2)/3,mp.mpf(2)/3,-mp.mpf(1)/12]
 return [sum(w*s[k] for w,s in zip(weights,samples))/step for k in range(len(samples[0]))]

def main():
 p=argparse.ArgumentParser();p.add_argument('--nr',type=int,required=True);p.add_argument('--na',type=int,required=True);p.add_argument('--spin',type=str,default='.5');p.add_argument('--inner',default='.04');p.add_argument('--smooth-layer',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();radial.SMOOTH_LAYER=a.smooth_layer
 if a.output.exists():raise RuntimeError('Existing coefficient table is immutable')
 z,weight=roots_jacobi(a.na,2,2);V=np.stack([eval_jacobi(k,2,2,z) for k in range(a.na)],axis=1);L=V@np.diag([-((k+2)*(k+3)) for k in range(a.na)])@np.linalg.inv(V)
 assert np.max(abs(L@np.ones(a.na)+6))<1e-9
 lo=mp.mpf(a.inner);h=(24-lo)/(a.nr-1);spin=mp.mpf(a.spin);c=mp.sqrt(1-spin*spin)
 with a.output.open('w') as f:
  f.write(f'{a.nr} {a.na} {float(h):.17g}\n');f.write(' '.join(f'{x:.17g}' for x in z)+'\n');f.write(' '.join(f'{x:.17g}' for x in weight)+'\n')
  for row in L:f.write(' '.join(f'{x:.17g}' for x in row)+'\n')
  for i in range(a.nr):
   rho=lo+i*h;psi=radial.initial(rho);dp=mp.diff(radial.initial,rho) if i<a.nr-1 else 0
   o,op=radial.compact(rho);lam=o-rho*op
   def flux(r):
    v,vp=radial.compact(r);return (r*r-c*c*v*v)*vp/(v-r*vp)
   fluxprime=mp.diff(flux,rho)
   for zj in z:
    zz=mp.mpf(str(zj));C=components(rho,zz,spin);alpha,br,bp,grr,grp,gpp,volume,sig=C
    def jets(r):
     q=components(r,zz,spin);return [q[0],q[1],q[2],q[6]*q[1],q[6]*q[3],q[6]*q[4]]
    ap,brp,bpp,volume_br_p,volume_grr_p,volume_grp_p=derivative(jets,rho)
    aK=volume_br_p/volume;GammaR=-volume_grr_p/volume;GammaP=-volume_grp_p/volume
    potential=-o*fluxprime/(lam*sig)
    values=[rho,alpha,br,bp,grr,ap,brp,bpp,aK,alpha*GammaR-grr*ap,grp,alpha*GammaP-grp*ap,potential,1/sig,gpp,psi,dp]
    f.write(' '.join(f'{float(x):.17g}' for x in values)+'\n')
 meta=dict(nr=a.nr,na=a.na,spin=float(spin),inner=float(lo),smooth_physical_layer=a.smooth_layer,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),table_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),angular_constant_error=float(np.max(abs(L@np.ones(a.na)+6))))
 a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
