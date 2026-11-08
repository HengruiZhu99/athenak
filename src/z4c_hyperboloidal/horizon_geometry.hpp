// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_HORIZON_GEOMETRY_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_HORIZON_GEOMETRY_HPP_

#include "z4c/horizon_geometry.hpp"
#include "z4c_hyperboloidal/conformal_rhs.hpp"

namespace z4c_hyperboloidal {
// Physical Cartesian ADM first jet at a finite interior point. This preserves
// the predecessor's conversion exactly; scri is excluded, with no Omega floor.
template<typename T>
KOKKOS_INLINE_FUNCTION z4c::HorizonGeometry<T> PhysicalHorizonGeometry(
    const Z4cJet<T> &u,const OmegaJet<T> &o) {
  z4c::HorizonGeometry<T> q{};
  if (!(o.omega>0 && u.chi.value>0 && Kokkos::isfinite(o.omega)
        && Kokkos::isfinite(u.chi.value))) return q;
  const T scale=1/(u.chi.value*o.omega*o.omega);
  const T trace=u.trace.value+2*u.theta.value;
  for (int i=0;i<3;++i)for (int j=0;j<3;++j) {
    q.metric[i][j]=scale*u.metric.g[i][j];
    q.curvature[i][j]=scale*o.omega*u.a.k[i][j]+q.metric[i][j]*trace/3;
    for (int d=0;d<3;++d)
      q.derivative[d][i][j]=scale*(u.metric.dg[d][i][j]
          -u.metric.g[i][j]*(u.chi.d[d]/u.chi.value+2*o.gradient[d]/o.omega));
  }
  q.valid=true;
  q.valid=z4c::ValidHorizonGeometry(q);
  return q;
}
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_NATIVE_HORIZON_GEOMETRY_HPP_
