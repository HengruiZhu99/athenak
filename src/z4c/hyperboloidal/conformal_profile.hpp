// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CONFORMAL_PROFILE_HPP_
#define Z4C_HYPERBOLOIDAL_CONFORMAL_PROFILE_HPP_

#include <array>
#include <cmath>
#include <stdexcept>
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c::hyperboloidal {

// A conformal frame, independent of the physical spatial coordinate map.
// This small value object is copied into kernels; no device reads host globals.
struct ConformalProfile {
  bool flat = false;
  double flat_radius = 0, outer_radius = 0, plateau = .5;

  void Validate() const {
    if (!flat)
      return;
    if (!std::isfinite(flat_radius) || !std::isfinite(outer_radius) || !std::isfinite(plateau) ||
        !(0 < flat_radius && flat_radius < outer_radius && outer_radius < 1) ||
        plateau < (1 - flat_radius * flat_radius) / 2)
      throw std::invalid_argument(
          "flat Omega needs 0 < flat < outer < 1 and plateau >= legacy Omega(flat)");
    // The sufficient plateau bound makes Omega nonincreasing: Omega-r*Omega'
    // is strictly positive away from scri. No sampled positivity test is used
    // to admit a potentially degenerate radial compactification.
  }

  template <class T>
  struct RadialJet {
    T value, first, second;
  };

  template <class T>
  KOKKOS_INLINE_FUNCTION RadialJet<T> Radial(T r) const {
    if (!flat || r >= T(outer_radius))
      return {(1 - r * r) / 2, -r, -1};
    if (r <= T(flat_radius))
      return {T(plateau), 0, 0};
    const T width = T(outer_radius - flat_radius), z = (r - T(flat_radius)) / width;
    const T t = z <= T(.5) ? z : 1 - z;
    const T p =
        pow(t, 7) *
        (1716 + t * (-9009 + t * (20020 + t * (-24024 + t * (16380 + t * (-6006 + 924 * t))))));
    const T s = z <= T(.5) ? p : 1 - p;
    const T d = 12012 * pow(z * (1 - z), 6) / width;
    const T dd = 72072 * pow(z * (1 - z), 5) * (1 - 2 * z) / (width * width);
    const T old = (1 - r * r) / 2, delta = old - T(plateau);
    return {T(plateau) + s * delta, d * delta - s * r, dd * delta - 2 * d * r - s};
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION OmegaJet<T> At(const T x[3]) const {
    OmegaJet<T> o{};
    const T r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2], r = sqrt(r2);
    if (!flat || r >= T(outer_radius)) {
      // Preserve the legacy arithmetic, including all outer and scri derivatives.
      o.omega = (1 - r * r) / 2;
      for (int a = 0; a < 3; ++a) {
        o.gradient[a] = -x[a];
        o.hessian[a][a] = -1;
      }
      return o;
    }
    const auto q = Radial(r);
    o.omega = q.value;
    if (r <= T(flat_radius))
      return o;
    for (int a = 0; a < 3; ++a) {
      o.gradient[a] = q.first * x[a] / r;
      for (int b = 0; b < 3; ++b)
        o.hessian[a][b] =
            (q.second - q.first / r) * x[a] * x[b] / r2 + (a == b ? q.first / r : T(0));
    }
    return o;
  }

  // Existing storage offsets (alpha-A, beta+x, K+3) are unchanged. Physical
  // gamma, K_ij, lapse, beta, determinant-one metric and Lambda are preserved.
  void TransformFields(double r, std::array<double, 31>& q) const {
    if (!flat || r >= outer_radius)
      return;
    const double old = (1 - r * r) / 2, ratio = Radial(r).value / old, A = (1 + r * r) / 2;
    q[0] /= ratio * ratio;
    for (int f = 7; f < 13; ++f)
      q[f] /= ratio;
    q[18] = ratio * (q[18] + A) - A;
  }
};
}  // namespace z4c::hyperboloidal
#endif
