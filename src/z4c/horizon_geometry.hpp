// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HORIZON_GEOMETRY_HPP_
#define Z4C_HORIZON_GEOMETRY_HPP_

#include <Kokkos_Core.hpp>

namespace z4c {

// Physical ADM tensors in a common Cartesian coordinate basis. The derivative
// index comes first. A sampler supplies both symmetric copies of each tensor.
template <typename T>
struct HorizonGeometry {
  bool valid = false;
  T metric[3][3]{}, curvature[3][3]{}, derivative[3][3][3]{};
};

template <typename T>
KOKKOS_INLINE_FUNCTION
bool ValidHorizonGeometry(const HorizonGeometry<T> &q) {
  if (!q.valid) return false;
  const auto &g = q.metric;
  const T det = g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
      - g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
      + g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
  if (!(g[0][0] > 0 && g[0][0]*g[1][1]-g[0][1]*g[1][0] > 0
        && det > 0 && Kokkos::isfinite(det))) return false;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    if (!Kokkos::isfinite(g[i][j]) || !Kokkos::isfinite(q.curvature[i][j])) {
      return false;
    }
    for (int d = 0; d < 3; ++d) {
      if (!Kokkos::isfinite(q.derivative[d][i][j])) return false;
    }
  }
  return true;
}

}  // namespace z4c
#endif  // Z4C_HORIZON_GEOMETRY_HPP_
