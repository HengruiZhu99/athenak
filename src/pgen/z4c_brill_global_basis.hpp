//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_brill_global_basis.hpp
//! \brief Regular-origin helpers for the global axisymmetric Brill basis.

#ifndef PGEN_Z4C_BRILL_GLOBAL_BASIS_HPP_
#define PGEN_Z4C_BRILL_GLOBAL_BASIS_HPP_

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace z4c_irisk {

//! Evaluate the unique regular value of the global Brill expansion at r=0.
//!
//! The exported basis is
//!   1 + sum_{m,l} a_ml sin((2m+1) atan(L/r)) cos(2l phi).
//! At the origin, regularity requires the angular l>0 sums to vanish.  A finite
//! spectral solve satisfies that condition only to its residual tolerance, so
//! evaluating an arbitrary coordinate angle at exactly r=0 leaves a point defect
//! that a finite-difference second derivative amplifies as h^-2.  Projecting to
//! l=0 supplies the unique angle-independent limit; sin((2m+1) pi/2)=(-1)^m.
inline double RegularGlobalBrillOriginPsi(
    const std::size_t radial_points, const std::size_t angular_points,
    const std::vector<double> &values) {
  if (radial_points == 0 || angular_points == 0 ||
      values.size() != radial_points * angular_points) {
    throw std::invalid_argument("invalid global Brill coefficient dimensions");
  }
  double psi = 1.0;
  for (std::size_t m = 0; m < radial_points; ++m) {
    const double radial_limit = (m % 2 == 0) ? 1.0 : -1.0;
    psi += radial_limit * values[m * angular_points];
  }
  return psi;
}

}  // namespace z4c_irisk

#endif  // PGEN_Z4C_BRILL_GLOBAL_BASIS_HPP_
