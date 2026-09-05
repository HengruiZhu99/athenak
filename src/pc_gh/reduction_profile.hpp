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

}  // namespace pc_gh
#endif  // PC_GH_REDUCTION_PROFILE_HPP_
