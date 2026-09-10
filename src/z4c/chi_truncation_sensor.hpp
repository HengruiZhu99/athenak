#ifndef Z4C_CHI_TRUNCATION_SENSOR_HPP_
#define Z4C_CHI_TRUNCATION_SENSOR_HPP_

#include <limits>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "athena.hpp"

namespace z4c {
// Leading truncation-error magnitudes for fourth-order centered D1 and D2.
// D5 and D6 are second-order estimates using seven native samples. Scaling
// by a FIXED physical length makes both estimates dimensionless. This is
// a derivative-error proxy, not an estimate of the complete evolution error.
KOKKOS_INLINE_FUNCTION
Real ChiDerivativeTruncationError(const Real *u, const Real h,
                                  const Real length) {
  const Real fifth = (-u[0] + 4*u[1] - 5*u[2] + 5*u[4] - 4*u[5] + u[6])/2;
  const Real sixth = u[0] - 6*u[1] + 15*u[2] - 20*u[3] +
                     15*u[4] - 6*u[5] + u[6];
  // Do not amplify cancellation at very small h into an AMR runaway.
  Real magnitude = 0;
  for (int q = 0; q < 7; ++q) magnitude = Kokkos::fmax(magnitude, Kokkos::abs(u[q]));
  const Real roundoff = 256*std::numeric_limits<Real>::epsilon()*magnitude;
  return Kokkos::fmax(Kokkos::fmax(Real{0}, Kokkos::abs(fifth)-roundoff)*length/(30*h),
                     Kokkos::fmax(Real{0}, Kokkos::abs(sixth)-roundoff)*length*length/(90*h*h));
}

inline Real ResolutionScaledChiErrorThreshold(const Real threshold,
                                              const int reference_nx1,
                                              const int current_nx1) {
  const Real ratio = static_cast<Real>(reference_nx1)/current_nx1;
  return threshold*ratio*ratio*ratio*ratio;
}
} // namespace z4c
#endif
