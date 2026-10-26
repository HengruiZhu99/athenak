#!/usr/bin/env python3
"""Independent Schwarzschild 1+1 oracle: value-only physical 2-metric pullback.

Does not import scalar C++ geometry. Uses high precision only to prepare fixed
coefficient tables, then a separate conservative two-field FD6 evolution.
"""
import argparse,hashlib,json
from pathlib import Path
import mpmath as mp
mp.mp.dps=60
SMOOTH_LAYER=False

def smooth(x):
 if x<=0:return mp.mpf(0)
 if x>=1:return mp.mpf(1)
 if SMOOTH_LAYER:
  a=mp.exp(-1/x);b=mp.exp(-1/(1-x));return a/(a+b)
 return x**3*(10-15*x+6*x*x)
def compact(rho):
 z=(rho-8)/8;w=smooth(z);wp=(w*(1-w)*(1/z**2+1/(1-z)**2) if SMOOTH_LAYER else 30*z*z*(1-z)**2)/8 if 0<z<1 else 0
 return 1-w*rho/24,-(w+rho*wp)/24

def coefficients(rho):
 o,op=compact(rho);lam=o-rho*op;Rb=rho+o;density=lam*Rb*Rb/(rho*rho)
 if abs(o)<mp.mpf('1e-50'):return -mp.mpf(108)/576,mp.mpf(-1),mp.mpf(0),density,mp.mpf(0),1/(Rb*Rb)
 r=rho/o;hp=smooth((r-8)/8)*(r/mp.sqrt(r*r+100)+3/r)
 # Physical trumpet covariant 2-metric in (T_tr,r). Its determinant is -1.
 g=mp.matrix([[-(r-1)/(r+1),1/r],[1/r,(r+1)**2/r**2]])
 J=mp.matrix([[1,hp*lam/o**2],[0,lam/o**2]]);bar=o*o*J.T*g*J;inv=bar**-1
 def flux(s):
  v,dv=compact(s);lv=v-s*dv
  return (s*s-v*v)*dv/lv
 potential=-o*mp.diff(flux,rho)/(lam*Rb*Rb)
 return inv[0,0],inv[0,1],inv[1,1],density,potential,1/(Rb*Rb)

def initial(rho):
 o,_=compact(rho)
 if abs(o)<mp.mpf('1e-50'):return mp.mpf(0)
 r=rho/o
 return mp.exp(-((r-6)/mp.mpf('1.2'))**2)*smooth((r-mp.mpf('1.2'))/mp.mpf('.8'))/rho

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--inner',type=str,default='.04');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise RuntimeError('Coefficient evidence already exists')
 lo=mp.mpf(a.inner);hi=mp.mpf(24);h=(hi-lo)/(a.n-1)
 with a.output.open('w') as f:
  f.write(f'{a.n} {float(h):.17g}\n')
  for i in range(a.n):
   rho=lo+i*h;c=coefficients(rho);psi=initial(rho);dp=mp.diff(initial,rho) if i<a.n-1 else mp.mpf(0);momentum=c[3]*c[1]*dp
   f.write(' '.join(f'{float(v):.17g}' for v in [rho,*c,psi,momentum])+'\n')
 meta=dict(method='60-digit covariant Schwarzschild 2-metric pullback; exact stationary R0=M trumpet and fixed hyperboloidal height',n=a.n,inner=float(lo),outer=24,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),table_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
 a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
