// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_QUADRATURE_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_QUADRATURE_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include "mesh/chart_atlas.hpp"
namespace z4c_hyperboloidal {
// Composite midpoint quadrature for dOmega, not the dynamical metric area.
// Gnomonic x=(1,u,v)/sqrt(1+u*u+v*v); dOmega=du dv/(1+u*u+v*v)^(3/2).
inline double ScriSolidAngleWeight(const chart_atlas::BlockMap &map,int n,int j,int k) {
  const double pi=acos(-1.);
  double u=map.lo[1]+(j+.5)*(map.hi[1]-map.lo[1])/n;
  double v=map.lo[2]+(k+.5)*(map.hi[2]-map.lo[2])/n;
  double jac=1;
  if(map.equiangular) {u=tan(pi*u/4);v=tan(pi*v/4);jac=pi*pi*(1+u*u)*(1+v*v)/16;}
  return jac*(map.hi[1]-map.lo[1])*(map.hi[2]-map.lo[2])/(n*n*pow(1+u*u+v*v,1.5));
}
// Orthonormal complex scalar harmonics, every m; host cache only.
inline std::array<double,2> ScriScalarHarmonic(int l,int m,const Real *x) {
  const int a=std::abs(m);const double pi=acos(-1.);
  const double r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  const double z=std::max(-1.,std::min(1.,double(x[2]/r))),phi=atan2(x[1],x[0]);
  // Normalized recurrence avoids factorial overflow at larger lmax.
  double p=1/sqrt(4*pi);
  for(int k=1;k<=a;++k)p*=-sqrt((2.*k+1)/(2.*k))*sqrt(std::max(0.,1-z*z));
  if(l>a) {
    double prev=p;p=z*sqrt(2.*a+3)*p;
    for(int k=a+2;k<=l;++k) {
      const double aa=sqrt((4.*k*k-1)/(double(k*k-a*a)));
      const double bb=sqrt(((k-1.)*(k-1.)-a*a)/(4*(k-1.)*(k-1.)-1));
      double next=aa*(z*p-bb*prev);prev=p;p=next;
    }
  }
  double re=p*cos(a*phi),im=p*sin(a*phi);
  if(m<0) {double sign=a%2?-1:1;re*=sign;im*=-sign;}
  return {re,im};
}
}  // namespace z4c_hyperboloidal
#endif
