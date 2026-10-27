#ifndef MESH_CUBED_SPHERE_MAP_HPP_
#define MESH_CUBED_SPHERE_MAP_HPP_

#include "athena.hpp"

namespace cubed_sphere {
// All evolved components remain in the common Cartesian basis. J and H act only
// on derivative directions, including the non-tensorial Cartesian Gamma fields.
struct Point {
  Real x[3]{};
  Real J[3][3]{};       // dq^A / dx^a
  Real H[3][3][3]{};    // d^2 q^A / (dx^a dx^b)
};

// A second-order scalar jet used to differentiate the analytic inverse map.
struct Jet {
  Real f = 0, d[3]{}, h[3][3]{};
  KOKKOS_INLINE_FUNCTION Jet() = default;
  KOKKOS_INLINE_FUNCTION Jet(Real v) : f(v) {}
  KOKKOS_INLINE_FUNCTION static Jet Variable(Real v, int a) {
    Jet q(v); q.d[a] = 1; return q;
  }
};
KOKKOS_INLINE_FUNCTION Jet operator+(const Jet &a, const Jet &b) {
  Jet c(a.f+b.f);
  for (int i=0;i<3;++i) {
    c.d[i]=a.d[i]+b.d[i];
    for (int j=0;j<3;++j) c.h[i][j]=a.h[i][j]+b.h[i][j];
  }
  return c;
}
KOKKOS_INLINE_FUNCTION Jet operator-(const Jet &a, const Jet &b) {
  Jet c(-b.f);
  for (int i=0;i<3;++i) {
    c.d[i]=-b.d[i];
    for (int j=0;j<3;++j) c.h[i][j]=-b.h[i][j];
  }
  return a+c;
}
KOKKOS_INLINE_FUNCTION Jet operator*(const Jet &a, const Jet &b) {
  Jet c(a.f*b.f);
  for (int i=0;i<3;++i) {
    c.d[i]=a.d[i]*b.f+a.f*b.d[i];
    for (int j=0;j<3;++j)
      c.h[i][j]=a.h[i][j]*b.f+a.d[i]*b.d[j]+a.d[j]*b.d[i]+a.f*b.h[i][j];
  }
  return c;
}
KOKKOS_INLINE_FUNCTION Jet Power(const Jet &a, Real p) {
  Jet c(pow(a.f,p));
  const Real df=p*pow(a.f,p-1), ddf=p*(p-1)*pow(a.f,p-2);
  for (int i=0;i<3;++i) {
    c.d[i]=df*a.d[i];
    for (int j=0;j<3;++j) c.h[i][j]=df*a.h[i][j]+ddf*a.d[i]*a.d[j];
  }
  return c;
}
KOKKOS_INLINE_FUNCTION Jet operator/(const Jet &a, const Jet &b) {
  return a*Power(b,-1);
}
KOKKOS_INLINE_FUNCTION Jet Log(const Jet &a) {
  Jet c(log(a.f));
  for (int i=0;i<3;++i) {
    c.d[i]=a.d[i]/a.f;
    for (int j=0;j<3;++j)
      c.h[i][j]=a.h[i][j]/a.f-a.d[i]*a.d[j]/(a.f*a.f);
  }
  return c;
}

// Patch order +X,-X,+Y,-Y,+Z,-Z. q=(s,u,v), 0<=s<=1,
// -1<=u,v<=1. At s=0 the six faces match a Cartesian cube of half width L;
// at s=1 they tile the smooth sphere of radius R. R>sqrt(3)*L.
struct Map {
  Real L = 1, R = 4;
  Real radial_stretch = 0;
  KOKKOS_INLINE_FUNCTION Real RadialForward(Real s) const {
    return radial_stretch==0 ? s : (1-exp(-radial_stretch*s))/(1-exp(-radial_stretch));
  }
  KOKKOS_INLINE_FUNCTION Real RadialInverse(Real t) const {
    return radial_stretch==0 ? t : -log(1-(1-exp(-radial_stretch))*t)/radial_stretch;
  }
  KOKKOS_INLINE_FUNCTION void Forward(int patch, const Real q[3], Real x[3]) const {
    int a=patch/2, b=(a+1)%3, c=(a+2)%3;
    Real sign=(patch%2==0 ? 1 : -1);
    Real t=RadialForward(q[0]);
    Real d=(1-t)*L+t*R/sqrt(1+q[1]*q[1]+q[2]*q[2]);
    x[a]=sign*d; x[b]=d*q[1]; x[c]=d*q[2];
  }
  KOKKOS_INLINE_FUNCTION void Inverse(int patch, const Real x[3], Real q[3]) const {
    int a=patch/2, b=(a+1)%3, c=(a+2)%3;
    Real d=(patch%2==0 ? x[a] : -x[a]);
    q[1]=x[b]/d; q[2]=x[c]/d;
    q[0]=RadialInverse((d-L)/(R/sqrt(1+q[1]*q[1]+q[2]*q[2])-L));
  }
  KOKKOS_INLINE_FUNCTION int Owner(const Real x[3]) const {
    int a=0;
    for (int b=1;b<3;++b) if (fabs(x[b])>fabs(x[a])) a=b;
    return 2*a+(x[a]<0);
  }
  KOKKOS_INLINE_FUNCTION Point Geometry(int patch, const Real q[3]) const {
    Point p;
    Forward(patch,q,p.x);
    Jet x[3];
    for (int a=0;a<3;++a) x[a]=Jet::Variable(p.x[a],a);
    int a=patch/2, b=(a+1)%3, c=(a+2)%3;
    Jet d=Jet(patch%2==0 ? 1 : -1)*x[a];
    Jet u=x[b]/d, v=x[c]/d;
    Jet z[3]={(d-Jet(L))/(Jet(R)*Power(Jet(1)+u*u+v*v,-0.5)-Jet(L)),u,v};
    if (radial_stretch!=0)
      z[0]=Jet(-1/radial_stretch)*Log(Jet(1)-Jet(1-exp(-radial_stretch))*z[0]);
    for (int A=0;A<3;++A) for (int i=0;i<3;++i) {
      p.J[A][i]=z[A].d[i];
      for (int j=0;j<3;++j) p.H[A][i][j]=z[A].h[i][j];
    }
    return p;
  }
};
using GeometryView = DvceArray4D<Point>;
}  // namespace cubed_sphere
#endif
