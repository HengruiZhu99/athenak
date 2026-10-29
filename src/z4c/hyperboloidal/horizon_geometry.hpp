// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_HORIZON_GEOMETRY_HPP_
#define Z4C_HYPERBOLOIDAL_HORIZON_GEOMETRY_HPP_

#include "z4c/horizon_geometry.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c {
namespace hyperboloidal {

// Reconstruct physical, not Penrose, geometry at a finite interior point.
// gamma_ij = gtilde_ij/(chi Omega^2),
// K_ij = Atilde_ij/(chi Omega) + gamma_ij (P+2Theta)/3.
// Omega is fixed in time but its spatial gradient is essential. No lapse,
// normal derivative of Omega, or subtraction of the CMC reference is needed:
// P already stores the full physical K-2Theta. Scri itself is excluded, with
// no Omega floor that could turn a singular physical geometry into a horizon.
template <typename T>
KOKKOS_INLINE_FUNCTION
HorizonGeometry<T> PhysicalHorizonGeometry(const Z4cJet<T> &u,
                                          const OmegaJet<T> &o) {
  HorizonGeometry<T> q{};
  if (!(o.omega > 0 && u.chi.value > 0 && Kokkos::isfinite(o.omega)
        && Kokkos::isfinite(u.chi.value))) return q;
  const T scale = 1/(u.chi.value*o.omega*o.omega);
  const T trace = u.trace.value+2*u.theta.value;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    q.metric[i][j] = scale*u.metric.g[i][j];
    q.curvature[i][j] = scale*o.omega*u.a.k[i][j]+q.metric[i][j]*trace/3;
    for (int d = 0; d < 3; ++d) {
      q.derivative[d][i][j] = scale*(u.metric.dg[d][i][j]
          - u.metric.g[i][j]*(u.chi.d[d]/u.chi.value+2*o.gradient[d]/o.omega));
    }
  }
  q.valid = true;
  q.valid = ValidHorizonGeometry(q);
  return q;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_HORIZON_GEOMETRY_HPP_
