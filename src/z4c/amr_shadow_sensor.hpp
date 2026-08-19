//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_shadow_sensor.hpp
//! \brief Nyquist-sensitive, diagnostic-only Z4c AMR sensor primitives.

#ifndef Z4C_AMR_SHADOW_SENSOR_HPP_
#define Z4C_AMR_SHADOW_SENSOR_HPP_

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "athena.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION
Real NormalizedFourthDifference(const Real um2, const Real um1, const Real u0,
                                const Real up1, const Real up2) {
  const Real numerator = Kokkos::abs(um2 - 4.0 * um1 + 6.0 * u0 - 4.0 * up1 + up2);
  const Real denominator = Kokkos::abs(um2) + Kokkos::abs(um1) + Kokkos::abs(u0) +
                           Kokkos::abs(up1) + Kokkos::abs(up2) + 1.0e-30;
  return numerator / denominator;
}

KOKKOS_INLINE_FUNCTION
Real FourthDifferenceShadow2D(const Real im2, const Real im1, const Real u0,
                              const Real ip1, const Real ip2, const Real jm2,
                              const Real jm1, const Real jp1, const Real jp2) {
  const Real radial = NormalizedFourthDifference(im2, im1, u0, ip1, ip2);
  const Real axial = NormalizedFourthDifference(jm2, jm1, u0, jp1, jp2);
  return Kokkos::sqrt(radial * radial + axial * axial);
}

}  // namespace z4c

#endif  // Z4C_AMR_SHADOW_SENSOR_HPP_
