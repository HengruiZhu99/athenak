#!/usr/bin/env python3
"""Check the actual harmonic functions against the addition theorem and derivatives.

Compile a host numerical harness, replacing only Athena/Kokkos type declarations.
No Kokkos installation is needed for this mathematical regression check.
"""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / 'src/utils/spherical_harm.hpp').read_text()
source = source.replace('#include "athena.hpp"', '\n'.join([
    'using Real=double;', '#define KOKKOS_INLINE_FUNCTION inline',
    'namespace Kokkos { using std::max; using std::min; using std::pow; '
    'using std::sqrt; using std::sin; using std::cos; }']))
source += r'''
int main(){
 double worst=0,deriv=0;
 for(double th: {0.1,0.7,1.2,2.8}) for(int l: {0,1,8,16,32,64,96}) {
  double sum=0;
  for(int m=-l;m<=l;++m){double a,b;SphericalHarm(&a,&b,l,m,th,.3);sum+=a*a+b*b;}
  worst=std::max(worst,std::abs(sum/((2*l+1)/(4*M_PI))-1));
 }
 for(int l: {1,8,32,64}) for(int m: {0,1}) {
  double a,b,at,bt,ap,bp,att,btt,app,bpp,atp,btp;
  SphericalHarmSecondDerivs(&a,&b,&at,&bt,&ap,&bp,&att,&btt,&app,&bpp,&atp,&btp,l,m,1.2,.3);
  double u,v,w,z,h=1e-5;
  SphericalHarm(&u,&v,l,m,1.2+h,.3);SphericalHarm(&w,&z,l,m,1.2-h,.3);
  deriv=std::max(deriv,std::abs((u-w)/(2*h)-at)/(1+std::abs(at)));
  deriv=std::max(deriv,std::abs((u-2*a+w)/(h*h)-att)/(1+std::abs(att)));
 }
 std::cout<<"addition theorem relative error "<<worst<<"; derivative error "<<deriv<<"\n";
 return (worst<1e-12 && deriv<5e-5)?0:1;
}
'''
with tempfile.TemporaryDirectory() as tmp:
    cpp = Path(tmp) / 'check.cpp'
    exe = Path(tmp) / 'check'
    cpp.write_text(source)
    subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
                    str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
