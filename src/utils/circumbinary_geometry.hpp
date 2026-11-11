#ifndef UTILS_CIRCUMBINARY_GEOMETRY_HPP_
#define UTILS_CIRCUMBINARY_GEOMETRY_HPP_

#include <cmath>
#include "athena.hpp"

namespace circumbinary {
// Equal-time inverse boost followed by the arbitrary-spin Kerr level set.
// sqrt_fn permits differentiation with the metric's existing dual-number type.
template<class T, class Sqrt>
void KerrQuadric(T mass, const T chi[3], const T velocity[3], T H[3][3],
                 Sqrt sqrt_fn) {
  T chi2=chi[0]*chi[0]+chi[1]*chi[1]+chi[2]*chi[2];
  T rp=mass*(T(1)+sqrt_fn(T(1)-chi2));
  T a[3]={mass*chi[0],mass*chi[1],mass*chi[2]};
  T a2=mass*mass*chi2;
  T v2=velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2];
  T gamma=T(1)/sqrt_fn(T(1)-v2);
  // (Gamma-1)/v^2 = Gamma^2/(Gamma+1), including v=0.
  T q=gamma*gamma/(gamma+T(1));
  T L[3][3], Q[3][3];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    L[i][j]=T(i==j)+q*velocity[i]*velocity[j];
    Q[i][j]=(T(i==j)+a[i]*a[j]/(rp*rp))/(rp*rp+a2);
  }
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    H[i][j]=T(0);
    for (int k=0;k<3;++k) for (int l=0;l<3;++l)
      H[i][j]=H[i][j]+L[k][i]*Q[k][l]*L[l][j];
  }
}

struct Quadric {
  Real center[3]={}, center_dot[3]={}, H[3][3]={}, H_dot[3][3]={};
  Real Radius(Real theta, Real phi) const {
    Real n[3]={sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)}, h=0;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) h+=n[i]*H[i][j]*n[j];
    return 1.0/sqrt(h);
  }
  void Motion(Real theta, Real phi, Real w[3]) const {
    Real n[3]={sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)}, hd=0;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) hd+=n[i]*H_dot[i][j]*n[j];
    Real r=Radius(theta,phi);
    for (int i=0;i<3;++i) w[i]=center_dot[i]-0.5*r*r*r*hd*n[i];
  }
};
}  // namespace circumbinary
#endif
