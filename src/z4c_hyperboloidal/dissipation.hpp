// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_DISSIPATION_HPP_
#define Z4C_HYPERBOLOIDAL_DISSIPATION_HPP_
#include "athena.hpp"

namespace z4c_hyperboloidal {
// N12 physical-field KO8: -epsilon*n/256 * V^-1 B4^T W B4.
// Native block tasks supply complete weighted lines, including mapped donor
// values. These functions neither build a mesh nor own a flattened field array.
struct PhysicalKO8 {
  static constexpr int order=4, points=5, normalization=256;
  KOKKOS_INLINE_FUNCTION static int Coefficient(int i) {
    const int c[points]={1,-4,6,-4,1};return c[i];
  }
  KOKKOS_INLINE_FUNCTION static Real Line(const Real values[points],
      int target,Real weight,Real volume,Real inverse_logical_spacing,Real epsilon) {
    Real difference=0;
    for (int a=0;a<points;++a)difference+=Coefficient(a)*(values[a]-values[0]);
    return -epsilon*inverse_logical_spacing*Coefficient(target)*weight*difference
           /(normalization*volume);
  }
};
}  // namespace z4c_hyperboloidal
#endif
