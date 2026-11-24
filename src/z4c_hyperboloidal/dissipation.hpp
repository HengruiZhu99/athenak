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

// Lower bound on the radial taper of angular KO lines in the spherical shells.
// The bulk taper [max(0,1-r^2/0.64)]^6 leaves the cubed-sphere seams of the
// outer shells without dissipation; radial lines keep the bulk taper alone.
struct AngularKOFloor {
  Real strength = 0, begin = .55, full = .65, fade = .91, end = .96;
  static Real Smooth(Real x) {
    x = x < 0 ? 0 : (x > 1 ? 1 : x);
    return x * x * x * (10 + x * (-15 + 6 * x));
  }
  bool Valid() const {
    return strength >= 0 && strength <= 1 && 0 <= begin && begin < full && full <= fade &&
           fade < end;
  }
  Real operator()(Real r) const {
    if (!(strength > 0)) return 0;
    return strength * Smooth((r - begin) / (full - begin)) *
           Smooth((end - r) / (end - fade));
  }
};
}  // namespace z4c_hyperboloidal
#endif
