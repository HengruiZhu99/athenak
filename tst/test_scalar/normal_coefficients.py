#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import mpmath as mp
from radial_coefficients import coefficients,initial
import radial_coefficients
mp.mp.dps=60

def normal(rho):
 a,b,c,D,V,L=coefficients(rho);alpha=1/mp.sqrt(-a);beta=-b/a;ginv=c-b*b/a;J=rho*rho*D/alpha
 return alpha,beta,ginv,J,V,L

def derivative(function,rho):
 if abs(rho-24)<mp.mpf('1e-40'):
  h=mp.mpf('1e-5');return (mp.mpf(25)/12*function(rho)-4*function(rho-h)+3*function(rho-2*h)-mp.mpf(4)/3*function(rho-3*h)+function(rho-4*h)/4)/h
 return mp.diff(function,rho)

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--smooth-layer',action='store_true');p.add_argument('--inner',default='.04');p.add_argument('--output',type=Path,required=True);a=p.parse_args();radial_coefficients.SMOOTH_LAYER=a.smooth_layer;lo=mp.mpf(a.inner);h=(24-lo)/(a.n-1)
 if a.output.exists():raise RuntimeError('Existing table is immutable')
 with a.output.open('w') as f:
  f.write(f'{a.n} {float(h):.17g}\n')
  for i in range(a.n):
   x=lo+i*h;alpha,beta,ginv,J,V,L=normal(x);ap=derivative(lambda z:normal(z)[0],x);bp=derivative(lambda z:normal(z)[1],x)
   aK=derivative(lambda z:normal(z)[3]*normal(z)[1],x)/J
   Gamma=-derivative(lambda z:normal(z)[3]*normal(z)[2],x)/J
   psi=initial(x);q=mp.diff(initial,x) if i<a.n-1 else 0;Pi=beta*q/alpha
   values=[x,alpha,beta,ginv,ap,bp,aK,alpha*Gamma-ginv*ap,V,L,psi,Pi,q]
   f.write(' '.join(f'{float(z):.17g}' for z in values)+'\n')
 a.output.with_suffix('.json').write_text(json.dumps(dict(n=a.n,inner=float(lo),smooth_physical_layer=a.smooth_layer,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),table_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
