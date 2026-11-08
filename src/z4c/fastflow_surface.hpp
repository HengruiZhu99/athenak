// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_FASTFLOW_SURFACE_HPP_
#define Z4C_FASTFLOW_SURFACE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "athena.hpp"
#include "utils/spherical_harm.hpp"

// An accepted FastFlow surface in its original real-harmonic basis. This is a
// small diagnostic copy, not a spherical approximation or a copy of mesh fields.
struct FastFlowSurface {
  int lmax = -1;
  std::array<Real, 3> center{};
  std::vector<Real> a0, ac, as;
  Real lower = 0, upper = 0;

  bool Valid() const {
    return lmax >= 1 && a0.size() == static_cast<size_t>(lmax+1)
        && ac.size() == static_cast<size_t>((lmax+1)*(lmax+1))
        && as.size() == ac.size();
  }

  // The addition theorem bounds each complex harmonic by sqrt((2l+1)/(4pi)).
  // The triangle bound conservatively encloses every retained harmonic mode.
  void Bound() {
    Real nonconstant = 0;
    const Real mean = a0.at(0)/std::sqrt(4*M_PI);
    for (int l = 1; l <= lmax; ++l) {
      const Real bound = std::sqrt((2*l+1)/(4*M_PI));
      nonconstant += std::abs(a0[l])*bound;
      for (int m = 1; m <= l; ++m) {
        const int k = lmindex(l, m, lmax);
        nonconstant += std::sqrt(2.)*(std::abs(ac[k])+std::abs(as[k]))*bound;
      }
    }
    lower = std::max(Real(0), mean-nonconstant);
    upper = mean+nonconstant;
  }

  Real Radius(Real theta, Real phi) const {
    Real radius = 0;
    for (int l = 0; l <= lmax; ++l) {
      Real re, im;
      SphericalHarm(&re, &im, l, 0, theta, phi);
      radius += a0[l]*re;
      for (int m = 1; m <= l; ++m) {
        SphericalHarm(&re, &im, l, m, theta, phi);
        const int k = lmindex(l, m, lmax);
        radius += std::sqrt(2.)*(ac[k]*re+as[k]*im);
      }
    }
    return radius;
  }

  bool Outside(const Real *x) const {
    const Real dx = x[0]-center[0], dy = x[1]-center[1], dz = x[2]-center[2];
    const Real radius = std::sqrt(dx*dx+dy*dy+dz*dz);
    if (radius > upper) return true;
    if (radius <= lower || radius == 0) return false;
    return radius > Radius(std::acos(std::clamp(dz/radius, Real(-1), Real(1))),
                           std::atan2(dy, dx));
  }
};
#endif  // Z4C_FASTFLOW_SURFACE_HPP_
