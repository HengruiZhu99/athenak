#ifndef PC_GH_REDUCTION_PROFILE_HPP_
#define PC_GH_REDUCTION_PROFILE_HPP_

#include <cmath>
#include "athena.hpp"

namespace pc_gh {

// Compact C-infinity radial weight. Squared radius avoids a nonsmooth norm at
// the center. Both radii are fixed physical input parameters, never mesh scales.
KOKKOS_INLINE_FUNCTION
Real SmoothReductionWeight(Real r2, Real core2, Real taper2) {
  if (r2 <= core2) return 1.0;
  if (r2 >= taper2) return 0.0;
  Real const q = (r2-core2)/(taper2-core2);
  Real const left = std::exp(-1.0/q);
  Real const right = std::exp(-1.0/(1.0-q));
  // At least one exponential is >= exp(-2), including underflow near an edge.
  return right/(left+right);
}

template <typename Centers>
KOKKOS_INLINE_FUNCTION
Real ReductionUnionWeight(const Real position[3], Centers centers,
                          Real core2, Real taper2) {
  Real outside = 1.0;
  for (int n = 0; n < centers.extent_int(0); ++n) {
    Real r2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      Real const offset = position[a] - centers(n, a);
      r2 += offset*offset;
    }
    outside *= 1.0 - SmoothReductionWeight(r2, core2, taper2);
  }
  return 1.0 - outside;
}

KOKKOS_INLINE_FUNCTION
Real BlendReductionTarget(Real old_value, Real target, Real weight) {
  // Exact limits preserve both the legacy reset and untouched exterior bits.
  if (weight == 0.0) return old_value;
  if (weight == 1.0) return target;
  return old_value + weight*(target - old_value);
}

}  // namespace pc_gh
#endif  // PC_GH_REDUCTION_PROFILE_HPP_
