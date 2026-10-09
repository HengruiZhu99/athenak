// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_INTERIOR_DISSIPATION_HPP_
#define Z4C_HYPERBOLOIDAL_INTERIOR_DISSIPATION_HPP_

#include <Kokkos_Core.hpp>

namespace z4c {
namespace hyperboloidal {

// KO6 as -sum_d D3_d^T D3_d/(64 h_d), keeping only four-node lines entirely
// inside the active domain. Thus sum q*Qq=-sum_lines (D3 q)^2/(64 h_d)<=0
// in the uniform-grid l2 inner product, independently of ghost extrapolation.
// It agrees with centered KO6 away from the boundary and annihilates quadratics.
// Boundary consistency is only O(h^2); this is not an energy estimate for Z4c.
// q and active are flattened callable views, with allocated radius-three halos.
// s must be active, and stride/h must describe the same uniform Cartesian patch.
template <typename Scalar, typename Mask, typename T>
KOKKOS_INLINE_FUNCTION
T InteriorKOSixth(const Scalar &q, const Mask &active, int s,
                  const int stride[3], const T h[3]) {
  T result = 0;
  const T c[4] = {-1,3,-3,1};
  for (int d = 0; d < 3; ++d) {
    for (int l = 0; l < 4; ++l) {
      bool valid = true;
      for (int b = 0; b < 4; ++b) {
        if (!active(s+(b-l)*stride[d])) valid = false;
      }
      if (!valid) continue;
      T third = 0;
      for (int b = 0; b < 4; ++b) third += c[b]*q(s+(b-l)*stride[d]);
      result -= c[l]*third/(64*h[d]);
    }
  }
  return result;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_INTERIOR_DISSIPATION_HPP_
