#ifndef MESH_CUBED_SPHERE_CONFORMAL_BRIDGE_HPP_
#define MESH_CUBED_SPHERE_CONFORMAL_BRIDGE_HPP_

#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace cubed_sphere {
// Components remain in the global Cartesian basis. Only their derivative
// indices were loaded in logical coordinates. The inverse-map Hessian term is
// essential even for a Cartesian-linear component on a curved logical patch.
KOKKOS_INLINE_FUNCTION void CartesianDerivatives(Real d[3],Real dd[3][3],
                                               const Point &p) {
  Real first[3]{},second[3][3]{};
  for (int a=0;a<3;++a) for (int A=0;A<3;++A) {
    first[a]+=p.J[A][a]*d[A];
    for (int b=0;b<3;++b) {
      second[a][b]+=p.H[A][a][b]*d[A];
      for (int B=0;B<3;++B) second[a][b]+=p.J[A][a]*p.J[B][b]*dd[A][B];
    }
  }
  for (int a=0;a<3;++a) {
    d[a]=first[a];
    for (int b=0;b<3;++b) dd[a][b]=second[a][b];
  }
}
KOKKOS_INLINE_FUNCTION void CartesianJet(z4c::hyperboloidal::Z4cJet<Real> &u,
                                        const Point &p) {
  CartesianDerivatives(u.chi.d,u.chi.dd,p);
  CartesianDerivatives(u.alpha.d,u.alpha.dd,p);
  CartesianDerivatives(u.trace.d,u.trace.dd,p);
  CartesianDerivatives(u.theta.d,u.theta.dd,p);
  for (int a=0;a<3;++a) {
    for (int field=0;field<2;++field) {
      auto &v=field ? u.lambda : u.beta;
      Real d[3],dd[3][3];
      for (int A=0;A<3;++A) {
        d[A]=v.d[A][a];
        for (int B=0;B<3;++B) dd[A][B]=v.dd[A][B][a];
      }
      CartesianDerivatives(d,dd,p);
      for (int A=0;A<3;++A) {
        v.d[A][a]=d[A];
        for (int B=0;B<3;++B) v.dd[A][B][a]=dd[A][B];
      }
    }
    for (int b=0;b<3;++b) {
      Real d[3],dd[3][3],da[3]{};
      for (int A=0;A<3;++A) {
        d[A]=u.metric.dg[A][a][b];
        for (int B=0;B<3;++B) dd[A][B]=u.metric.ddg[A][B][a][b];
        for (int B=0;B<3;++B) da[A]+=p.J[B][A]*u.a.dk[B][a][b];
      }
      CartesianDerivatives(d,dd,p);
      for (int A=0;A<3;++A) {
        u.metric.dg[A][a][b]=d[A]; u.a.dk[A][a][b]=da[A];
        for (int B=0;B<3;++B) u.metric.ddg[A][B][a][b]=dd[A][B];
      }
    }
  }
}
}  // namespace cubed_sphere
#endif
