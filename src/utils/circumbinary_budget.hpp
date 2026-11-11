//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef UTILS_CIRCUMBINARY_BUDGET_HPP_
#define UTILS_CIRCUMBINARY_BUDGET_HPP_
#include "athena.hpp"

class MeshBlockPack;
namespace circumbinary {
constexpr int kRegions = 4;
// A disjoint partition inside outer: near1, near2, cavity, annulus. The actual
// puncture excision ellipsoids are excluded. No origin or axis motion is implied.
struct BudgetGeometry {
  Real inner = 0, outer = 0, radius[2] = {}, center[2][3] = {}, excision_H[2][3][3] = {};
  int holes = 0, subcells = 2;
  KOKKOS_INLINE_FUNCTION int Region(Real x, Real y, Real z) const {
    Real pos[3] = {x, y, z};
    Real r2 = x * x + y * y + z * z;
    if (r2 >= outer * outer) {
      return -1;
    }
    for (int h = 0; h < holes; ++h) {
      Real d[3], d2 = 0, f = 0;
      for (int a = 0; a < 3; ++a) {
        d[a] = pos[a] - center[h][a];
        d2 += d[a] * d[a];
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          f += d[a] * excision_H[h][a][b] * d[b];
        }
      }
      if (f < 1.0) {
        return -1;
      }
      if (d2 < radius[h] * radius[h]) {
        return h;
      }
    }
    return r2 < inner * inner ? 2 : 3;
  }
};
struct RegionMoments {
  Real J[kRegions] = {}, torque[kRegions] = {}, volume[kRegions] = {}, bad[kRegions] = {};
  KOKKOS_INLINE_FUNCTION RegionMoments& operator+=(const RegionMoments& s) {
    for (int r = 0; r < kRegions; ++r) {
      J[r] += s.J[r];
      torque[r] += s.torque[r];
      volume[r] += s.volume[r];
      bad[r] += s.bad[r];
    }
    return *this;
  }
};
}  // namespace circumbinary
namespace Kokkos {
template <>
struct reduction_identity<circumbinary::RegionMoments> {
  KOKKOS_INLINE_FUNCTION static circumbinary::RegionMoments sum() { return {}; }
};
}  // namespace Kokkos
// Local sums only. MPI reduction is deliberately outside retryable task functions.
circumbinary::RegionMoments ReduceBinaryRegions(MeshBlockPack*,
                                                circumbinary::BudgetGeometry,
                                                bool with_torque = false);
#endif  // UTILS_CIRCUMBINARY_BUDGET_HPP_
